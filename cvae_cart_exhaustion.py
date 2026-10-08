"""
Conditional VAE (CVAE) for CAR-T / T cell exhaustion trajectories from scRNA-seq.

Designed to drop into cart_exhaustion_pipeline.ipynb (Colab) after Phase 0 (QC).
Each numbered section can be pasted into its own notebook cell.

Pipeline:
  1. Prepare data   – AnnData -> log-normalized matrix + one-hot condition labels
  2. Model          – Encoder(x, c) -> (mu, logvar); Decoder(z, c) -> x_hat
  3. Loss           – reconstruction (MSE on log-normalized data) + beta * KL
  4. Train          – Adam, KL warm-up, early stopping on validation loss
  5. Embed          – latent means -> adata.obsm["X_cvae"], UMAP + diffusion pseudotime
  6. Trajectory     – interpolate functional -> exhausted in latent space, decode, rank genes
  7. Counterfactual – swap the condition label for the same cells, see which genes move
"""

# =============================================================================
# 0. Imports and config
# =============================================================================
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "condition_key": "patient",     # nuisance variable to remove from z. NOT day: day is held out for validation
    "latent_dim": 10,               # ~2,000 input genes -> 10-d latent space
    "hidden_dims": [256, 128],      # encoder layer sizes (decoder mirrors them)
    "dropout": 0.1,
    "batch_size": 512,
    "lr": 1e-3,
    "max_epochs": 300,
    "kl_warmup_epochs": 50,         # beta ramps 0 -> beta_max over these epochs
    "beta_max": 1.0,
    "patience": 30,                 # early stopping
    "val_fraction": 0.1,
}

# Marker sets used to define the two ends of the trajectory (only genes present in the data are used)
FUNCTIONAL_MARKERS = ["TCF7", "SELL", "IL7R", "CCR7", "LEF1"]
EXHAUSTED_MARKERS = ["PDCD1", "HAVCR2", "LAG3", "TOX", "TIGIT", "CTLA4", "ENTPD1"]


# =============================================================================
# 1. Prepare data
# =============================================================================
def prepare_data(adata, condition_key, already_log_normalized=False):
    """
    Returns
      X        : float32 array (cells x genes), log-normalized
      C        : float32 one-hot array (cells x n_conditions)
      cond_cats: list of condition category names (column order of C)
    Raw counts are kept in adata.layers["counts"] so you can switch to a count likelihood later.
    """
    adata = adata.copy()
    if not already_log_normalized:
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    X = X.astype(np.float32)

    cond = adata.obs[condition_key].astype("category")
    cond_cats = list(cond.cat.categories)
    C = np.eye(len(cond_cats), dtype=np.float32)[cond.cat.codes.to_numpy()]

    return adata, X, C, cond_cats


def make_loaders(X, C, val_fraction, batch_size, seed=SEED):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def loader(ix, shuffle):
        ds = TensorDataset(torch.from_numpy(X[ix]), torch.from_numpy(C[ix]))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return loader(train_idx, True), loader(val_idx, False)


# =============================================================================
# 2. Model
# =============================================================================
def mlp(in_dim, hidden_dims, dropout):
    layers, d = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    return nn.Sequential(*layers), d


class Encoder(nn.Module):
    """q(z | x, c): maps a cell + its condition to a Gaussian over latent space."""
    def __init__(self, n_genes, n_cond, hidden_dims, latent_dim, dropout):
        super().__init__()
        self.body, d = mlp(n_genes + n_cond, hidden_dims, dropout)
        self.mu = nn.Linear(d, latent_dim)
        self.logvar = nn.Linear(d, latent_dim)

    def forward(self, x, c):
        h = self.body(torch.cat([x, c], dim=1))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    """p(x | z, c): reconstructs log-expression from a latent point + condition."""
    def __init__(self, n_genes, n_cond, hidden_dims, latent_dim, dropout):
        super().__init__()
        self.body, d = mlp(latent_dim + n_cond, list(reversed(hidden_dims)), dropout)
        self.out = nn.Linear(d, n_genes)

    def forward(self, z, c):
        h = self.body(torch.cat([z, c], dim=1))
        return F.softplus(self.out(h))  # log1p expression is non-negative


class CVAE(nn.Module):
    def __init__(self, n_genes, n_cond, hidden_dims=(64, 32), latent_dim=5, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(n_genes, n_cond, list(hidden_dims), latent_dim, dropout)
        self.decoder = Decoder(n_genes, n_cond, list(hidden_dims), latent_dim, dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        # z = mu + sigma * eps  -> randomness lives in eps, so gradients flow through mu and sigma
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z, c)
        return x_hat, mu, logvar


# =============================================================================
# 3. Loss
# =============================================================================
def cvae_loss(x, x_hat, mu, logvar, beta):
    # Sum over genes / latent dims, mean over cells
    recon = F.mse_loss(x_hat, x, reduction="none").sum(dim=1).mean()
    # Closed-form KL( N(mu, sigma^2) || N(0, 1) )
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean()
    return recon + beta * kl, recon, kl


# =============================================================================
# 4. Train
# =============================================================================
def train_cvae(model, train_loader, val_loader, cfg):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    history = []
    best_val, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(cfg["max_epochs"]):
        beta = cfg["beta_max"] * min(1.0, (epoch + 1) / cfg["kl_warmup_epochs"])

        model.train()
        tr = np.zeros(3)
        for x, c in train_loader:
            x, c = x.to(DEVICE), c.to(DEVICE)
            x_hat, mu, logvar = model(x, c)
            loss, recon, kl = cvae_loss(x, x_hat, mu, logvar, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr += np.array([loss.item(), recon.item(), kl.item()]) * x.size(0)
        tr /= len(train_loader.dataset)

        model.eval()
        va = np.zeros(3)
        with torch.no_grad():
            for x, c in val_loader:
                x, c = x.to(DEVICE), c.to(DEVICE)
                x_hat, mu, logvar = model(x, c)
                loss, recon, kl = cvae_loss(x, x_hat, mu, logvar, cfg["beta_max"])
                va += np.array([loss.item(), recon.item(), kl.item()]) * x.size(0)
        va /= len(val_loader.dataset)

        history.append({"epoch": epoch, "beta": beta,
                        "train_loss": tr[0], "train_recon": tr[1], "train_kl": tr[2],
                        "val_loss": va[0], "val_recon": va[1], "val_kl": va[2]})
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d} | beta {beta:.2f} | train {tr[0]:.3f} "
                  f"(recon {tr[1]:.3f}, kl {tr[2]:.3f}) | val {va[0]:.3f}")

        # Only start early stopping once warm-up is done (val loss is not comparable before that)
        if epoch >= cfg["kl_warmup_epochs"]:
            if va[0] < best_val - 1e-4:
                best_val, bad_epochs = va[0], 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= cfg["patience"]:
                    print(f"Early stopping at epoch {epoch}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


# =============================================================================
# 5. Embed + pseudotime
# =============================================================================
@torch.no_grad()
def get_latent(model, X, C, batch_size=1024):
    """Use the latent MEANS (not samples) as each cell's coordinates."""
    model.eval()
    mus = []
    for i in range(0, X.shape[0], batch_size):
        x = torch.from_numpy(X[i:i + batch_size]).to(DEVICE)
        c = torch.from_numpy(C[i:i + batch_size]).to(DEVICE)
        mu, _ = model.encoder(x, c)
        mus.append(mu.cpu().numpy())
    return np.vstack(mus)


def marker_score(adata, genes):
    """Mean z-scored expression of the markers present in the panel."""
    present = [g for g in genes if g in adata.var_names]
    if not present:
        raise ValueError(f"None of {genes} found in adata.var_names")
    sub = adata[:, present].X
    sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    z = (sub - sub.mean(0)) / (sub.std(0) + 1e-8)
    return z.mean(1), present


def embed_and_pseudotime(adata, Z, root_mask=None):
    """
    Neighbors/UMAP on the CVAE latent space, marker scores, and diffusion pseudotime.
    root_mask: boolean array of candidate root cells (e.g. infusion-product cells).
      The root is the candidate closest to their centroid in diffusion space -- a "typical" cell.
      If None, falls back to the single most functional cell, which can land on an outlier
      (that happened on GSE125881 and broke pseudotime), so passing a root_mask is recommended.
    """
    adata.obsm["X_cvae"] = Z
    sc.pp.neighbors(adata, use_rep="X_cvae")   # graph built on latent z, not on PCA of genes
    sc.tl.umap(adata)

    func, _ = marker_score(adata, FUNCTIONAL_MARKERS)
    exh, _ = marker_score(adata, EXHAUSTED_MARKERS)
    adata.obs["functional_score"] = func
    adata.obs["exhaustion_score"] = exh
    adata.obs["exh_minus_func"] = exh - func

    sc.tl.diffmap(adata)
    if root_mask is not None:
        dm = adata.obsm["X_diffmap"][:, 1:]     # diffusion component 0 is trivial, skip it
        cand = np.where(np.asarray(root_mask))[0]
        centroid = dm[cand].mean(0)
        adata.uns["iroot"] = int(cand[np.argmin(((dm[cand] - centroid) ** 2).sum(1))])
    else:
        adata.uns["iroot"] = int(np.argmin(adata.obs["exh_minus_func"].to_numpy()))
    sc.tl.dpt(adata)
    return adata


# =============================================================================
# 6. Latent trajectory: functional -> exhausted
# =============================================================================
@torch.no_grad()
def latent_trajectory(model, adata, Z, cond_cats, condition, n_steps=20, quantile=0.1):
    """
    Start = centroid of the top `quantile` most functional cells,
    End   = centroid of the top `quantile` most exhausted cells.
    Linearly interpolate between them in latent space and decode every step
    under a single fixed condition, so only cell state changes along the path.
    """
    s = adata.obs["exh_minus_func"].to_numpy()
    lo, hi = np.quantile(s, [quantile, 1 - quantile])
    z_start = Z[s <= lo].mean(0)
    z_end = Z[s >= hi].mean(0)

    alphas = np.linspace(0, 1, n_steps, dtype=np.float32)
    z_path = np.stack([(1 - a) * z_start + a * z_end for a in alphas]).astype(np.float32)

    c_vec = np.zeros((n_steps, len(cond_cats)), dtype=np.float32)
    c_vec[:, cond_cats.index(condition)] = 1.0

    model.eval()
    x_path = model.decoder(torch.from_numpy(z_path).to(DEVICE),
                           torch.from_numpy(c_vec).to(DEVICE)).cpu().numpy()

    path_df = pd.DataFrame(x_path, columns=adata.var_names, index=np.round(alphas, 3))
    path_df.index.name = "alpha"

    change = path_df.iloc[-1] - path_df.iloc[0]
    ranking = pd.DataFrame({
        "start_expr": path_df.iloc[0],
        "end_expr": path_df.iloc[-1],
        "delta": change,
    }).sort_values("delta", ascending=False)
    return path_df, ranking


@torch.no_grad()
def decode_path(model, z_start, z_end, cond_cats, condition, var_names, n_steps=20):
    """
    General version of the trajectory above: decode a straight line between ANY two
    latent points (e.g. infusion-product centroid -> late-timepoint centroid).
    Returns the decoded expression at each step and the end-minus-start change per gene.
    """
    alphas = np.linspace(0, 1, n_steps, dtype=np.float32)
    z = np.stack([(1 - a) * z_start + a * z_end for a in alphas]).astype(np.float32)
    c = np.zeros((n_steps, len(cond_cats)), dtype=np.float32)
    c[:, cond_cats.index(condition)] = 1.0          # fixed condition: only cell state moves
    model.eval()
    x = model.decoder(torch.from_numpy(z).to(DEVICE), torch.from_numpy(c).to(DEVICE)).cpu().numpy()
    df = pd.DataFrame(x, columns=var_names, index=np.round(alphas, 2))
    return df, (df.iloc[-1] - df.iloc[0]).sort_values(ascending=False)


# =============================================================================
# 7. Counterfactual: swap condition
# =============================================================================
@torch.no_grad()
def counterfactual(model, X, C, cond_cats, from_cond, to_cond, var_names):
    """Encode cells from `from_cond`, decode them as if they were `to_cond`."""
    model.eval()
    mask = C[:, cond_cats.index(from_cond)] == 1
    x = torch.from_numpy(X[mask]).to(DEVICE)
    c_from = torch.from_numpy(C[mask]).to(DEVICE)
    mu, _ = model.encoder(x, c_from)

    c_to = torch.zeros_like(c_from)
    c_to[:, cond_cats.index(to_cond)] = 1.0

    x_orig = model.decoder(mu, c_from).cpu().numpy()
    x_cf = model.decoder(mu, c_to).cpu().numpy()
    return pd.DataFrame({
        "decoded_as_" + str(from_cond): x_orig.mean(0),
        "decoded_as_" + str(to_cond): x_cf.mean(0),
        "delta": (x_cf - x_orig).mean(0),
    }, index=var_names).sort_values("delta", ascending=False)


# =============================================================================
# Run (in the notebook, after QC):
# =============================================================================
if __name__ == "__main__":
    # adata = sc.read_h5ad("your_qc_filtered_tcells.h5ad")
    raise SystemExit("Load your QC'd AnnData as `adata`, then run the cells below in the notebook.")

    adata, X, C, cond_cats = prepare_data(adata, CONFIG["condition_key"])
    train_loader, val_loader = make_loaders(X, C, CONFIG["val_fraction"], CONFIG["batch_size"])

    model = CVAE(n_genes=X.shape[1], n_cond=C.shape[1],
                 hidden_dims=CONFIG["hidden_dims"], latent_dim=CONFIG["latent_dim"],
                 dropout=CONFIG["dropout"])
    model, history = train_cvae(model, train_loader, val_loader, CONFIG)
    history.plot(x="epoch", y=["train_recon", "val_recon", "train_kl"], logy=True)

    Z = get_latent(model, X, C)
    adata = embed_and_pseudotime(adata, Z)
    sc.pl.umap(adata, color=[CONFIG["condition_key"], "exhaustion_score",
                             "functional_score", "dpt_pseudotime"])

    path_df, ranking = latent_trajectory(model, adata, Z, cond_cats, condition=cond_cats[0])
    print(ranking.head(15))   # genes rising most from functional -> exhausted

    if len(cond_cats) > 1:
        cf = counterfactual(model, X, C, cond_cats, cond_cats[0], cond_cats[1], adata.var_names)
        print(cf.head(15))

    torch.save(model.state_dict(), "cvae_cart_exhaustion.pt")
