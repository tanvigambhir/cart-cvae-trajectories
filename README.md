# CAR-T Cell State Trajectories with a Conditional VAE

A conditional variational autoencoder (PyTorch) trained on single-cell RNA-seq from CD19 CAR-T cells. It learns a patient-corrected latent space and decodes paths through that space to see how CAR-T cell states change between the infusion product and months after infusion.

**Data:** [GSE125881](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE125881) ([Sheih et al., *Nat Commun* 2020](https://www.nature.com/articles/s41467-019-13880-1)). The dataset has 62,167 CD8⁺ CD19 CAR-T cells from 4 patients (2 CLL, 2 NHL), sampled at the infusion product (IP), ~day 12–21, ~day 28–38, and ~day 83–112. Platform: 10x 5′.

## Approach

| Step | What | Why |
|---|---|---|
| Load | Read the dense 20k × 62k GEO CSV in 1,000-gene chunks, converting each chunk to sparse before reading the next | Loading the whole dense table at once exceeds Colab RAM |
| Sample labels | Mapped barcode suffixes to patient/timepoint, then **verified the mapping** by overlapping each suffix with the barcodes in GEO's per-sample TCR files | The suffix order does **not** match GEO's sample order for 3 of the 4 patients; inferring it would have mislabeled ~70% of cells |
| QC | Removed Scrublet doublets (per sample); normalized and log-transformed over all genes | The authors had already applied mito and low-gene filters |
| Genes | 2,000 HVGs selected per patient, plus exhaustion and stemness markers; **TCR V/J genes removed** | TCR genes are clone- and therefore patient-specific. Keeping them produced patient-specific clusters that conditioning could not remove |
| Model | CVAE: encoder q(z \| x, patient), decoder p(x \| z, patient); 10-d latent space; MSE reconstruction + β·KL with warm-up | Conditioning on patient removes donor variation from z. **Day was kept out of the model** so the latent space can be checked against real time |
| Evaluation | Diffusion pseudotime on the latent space vs. actual day; genes decoded along latent paths vs. observed differential expression, checked within each patient | Tests whether the latent space and the decoded trajectories reflect the data, rather than artifacts of the model |

## Results

- **The latent space recovers time without being given it.** Pseudotime (rooted in the infusion product) correlates with days post-infusion at Spearman ρ = 0.54 overall and 0.58–0.61 within 3 of 4 patients (CLL-2: 0.27). Most of this signal is the infusion-product vs. post-infusion split. Within post-infusion timepoints alone, ρ = 0.26.
- **Decoded trajectories match real expression.** For the top 40 genes along the IP → late path (mitochondrial and ribosomal genes excluded), the direction predicted by the decoder matched observed differential expression for **40/40**. Of these, **36/40 changed in the same direction in all 4 patients**.
- **Biology: from a hypoxic, glycolytic, activated state to a quiescent, NK-like effector state.**
  - *Down after infusion (consistent in all 4 patients):* hypoxia and glycolysis genes (EGLN3, PGK1, LDHA, ENO1, TPI1, GPI, PKM, GAPDH); immediate-early activation genes (FOS, CD69, IER2); LTB, IL32, GZMA.
  - *Up after infusion:* NK-like cytotoxic genes (GNLY, NKG7, GZMH, CCL4, CMC1); quiescence and activation-restraint genes (CXCR4, CBLB, NFKBIA, TNFAIP3, ZFP36L2).
  - Classic exhaustion markers (PDCD1, HAVCR2, TOX) do **not** rise progressively after infusion. This agrees with the original study.
- A proliferating subpopulation (MKI67⁺/TOP2A⁺) sits between the infusion-product and early post-infusion cells.

## Limitations

- **Time and environment are confounded.** Infusion-product cells come from culture; later cells come from patient blood. The hypoxia signature (EGLN3 is down ~10 log₂ units in pooled differential expression) points to culture conditions, so the largest shift probably reflects leaving culture as much as time spent in the patient.
- **n = 4 patients.** The cell-level p-values are not meaningful because cells from the same patient are not independent. That is why consistency across patients is used as the main criterion.
- Late timepoints differ between patients and have few cells in some samples (NHL-7 d89: 197 cells).
- MSE on log-normalized data is a simplification. A negative binomial likelihood on raw counts (as in scVI) would model the counts more faithfully.

## Files

| File | Contents |
|---|---|
| `cart_exhaustion_pipeline.ipynb` | Full analysis (Colab): loading, sample-label verification, QC, training, latent space, trajectories, validation |
| `cvae_cart_exhaustion.py` | CVAE model, loss, training loop, embedding, pseudotime, latent trajectories, counterfactuals (the model sections of the notebook) |
| `requirements.txt` | Python packages (`pip install -r requirements.txt`) |

## Reproduce

Open `cart_exhaustion_pipeline.ipynb` in Colab with a GPU runtime and run the cells in order. The notebook downloads GSE125881 from GEO and saves processed data to Google Drive. The full run takes about 20–30 minutes on a T4.
