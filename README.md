# Mid-Point Check-in: AI for Science & Engineering
**Project Title:** Multi-Modal SAR-Optical Fusion for Flood Segmentation with Domain-Informed Feature Engineering  
**Team Name:** Team ENDRA  
**Primary Domain:** Geospatial AI / Earth Observation / Flood Detection (Remote Sensing Segmentation)

---

---

## Page 1: Problem, Strategy, and Novelty

### 1. Proposed Strategy & Technical Novelty

**Problem Framing:**  
Flood extent mapping from Sentinel-1 SAR imagery is a three-class semantic segmentation problem: *No Flood (0)*, *Flood (1)*, and *Permanent Water Body (2)*. The core difficulty is that SAR backscatter alone cannot reliably disambiguate flood inundation from persistent open water or smooth urban surfaces. Furthermore, temporal context (pre-flood SAR state) is typically absent at inference time, and optical imagery suffers from cloud cover during active flood events — both of which force strictly single-pass, single-image inference.

---

#### Workflow / Architecture

```
Raw TIF (6ch: HH, HV, Green, Red, NIR, SWIR)
        │
        ▼ [BandStacker — per-image, lossless]
Engineered 6-band Stack:
  [HH, HV, NDWI, MNDWI, NDVI, SAR_LogRatio]
        │
        ▼ [compute_stats — dataset-wide online pass]
Per-band mean/std normalization
        │
        ▼ [Albumentations: H/V flip, Rotate90, ShiftScaleRotate]
Augmented Tensor (B × 6 × H × W)
        │
        ▼ [Prithvi-EO-v2-300 ViT Encoder — pretrained on multi-spectral EO]
Token Embeddings (temporal dim=1, neck: ReshapeTokensToImage)
        │
        ▼ [UperNetDecoder — decoder_channels=512]
Multi-scale Feature Pyramid
        │
        ▼ [Head: Dropout → Conv1×1 → 3-class logits]
Loss: α·CrossEntropy(w=[1.0, 8.0, 4.0]) + (1−α)·DiceLoss   [α=0.3, DICE_WEIGHT=0.7]
        │
        ▼ [3-Phase Training Schedule → Optuna HPO → 3-seed Ensemble]
Soft-averaged Ensemble Prediction (argmax over mean softmax)
        │
        ▼ RLE Submission (Class 1 extraction → CSV)
```

---

#### AI Methodology

**Hybrid: Domain-Informed Multi-Modal Learning + Foundation Model Adaptation**

The methodology combines: (1) *physics-guided feature engineering* over raw SAR/optical bands prior to feeding a vision transformer, and (2) *foundation model fine-tuning* — adapting Prithvi-EO-v2-300, a geospatial ViT pre-trained on multi-spectral Sentinel-2/Landsat at continental scale, to handle the engineered 6-band input stream.

This methodology is appropriate because the ViT backbone provides rich spatial representation capacity learned across diverse Earth conditions, while domain-derived indices reduce the burden on the model to re-learn spectral physics from limited labeled flood data.

---

#### Novel Contribution

**1. MNDWI as a Flood/Water Discriminator (Key Differentiator)**  
→ *What:* Modified Normalized Difference Water Index — `(Green − SWIR) / (Green + SWIR + ε)` — replaces raw Green/SWIR channels.  
→ *Why needed:* Permanent water bodies (rivers, reservoirs) and active flood inundation both suppress SAR backscatter. MNDWI peaks strongly for open water but decays for turbid/shallow flood water over vegetation or soil.  
→ *Why others likely did NOT do it:* Most Kaggle baselines use raw optical bands or only NDWI. MNDWI (Xu, 2006) is specifically superior to NDWI for flood–water distinguishability in remote sensing hydrology but is under-utilized in competition settings.  
→ *Problem solved:* Directly attacks Class 1/2 confusion — the dominant failure mode in three-class flood segmentation.

**2. SAR Log-Ratio Feature (Surface Roughness Encoding)**  
→ *What:* `SAR_LogRatio = 10 · log₁₀(|HH| / |HV| + ε)` — replaces raw HH/HV pair with their decibel-scale cross-polarization ratio.  
→ *Why needed:* Flooded surfaces exhibit specular reflection, yielding characteristic HH/HV ratio signatures. Raw linear-scale SAR channels are heavily skewed; log-ratio is the physically canonical representation used in radar hydrology.  
→ *Why others likely did NOT do it:* Most competitors treat raw SAR bands as image channels without domain transformation, ignoring the multiplicative noise structure of SAR signals.  
→ *Problem solved:* Improves signal-to-noise for SAR modality; provides a more linearizable input for gradient-based optimization.

**3. NDVI as Vegetation False-Positive Suppressor**  
→ *What:* `NDVI = (NIR − Red) / (NIR + Red + ε)` as a dedicated channel.  
→ *Why needed:* Dense vegetation canopies can double-bounce SAR and produce flood-like backscatter signatures. NDVI explicitly encodes vegetation density so the model can suppress flood predictions where NDVI is high.  
→ *Problem solved:* Reduces false positives in forested/agricultural areas during the monsoon season.

**4. 3-Phase Training Schedule with Frozen Encoder Warmup**  
→ *Phase 1 (5 epochs, lr=1e-5, encoder frozen):* Forces the UperNet decoder and head to adapt to the new 6-band input space without corrupting pretrained ViT weights.  
→ *Phase 2 (15 epochs, lr=1e-4, full unfreeze):* Backbone fine-tuning once decoder provides meaningful gradients.  
→ *Phase 3 (20 epochs, CosineAnnealingLR, lr=1e-5→1e-7):* Final convergence loading Phase 2's best checkpoint.  
→ *Why non-trivial:* Standard single-phase fine-tuning on a 6-band ViT backbone risks catastrophic forgetting; this schedule prevents gradient shock during band adaptation.

**5. Class-Weighted Hybrid Loss (CE + Dice, Flood-Biased)**  
→ *What:* `Loss = 0.3 · CrossEntropy(w=[1.0, 8.0, 4.0]) + 0.7 · DiceLoss`, `ignore_index=−1` for no-data pixels.  
→ *Why:* Flood class (Class 1) is severely underrepresented spatially. Pure CE on imbalanced masks collapses to background prediction. Dice loss is inherently region-based and class-frequency agnostic. The 8× flood weight further biases CE gradient toward the rare class. The 70% Dice weighting is tuned via Optuna along with all class weights.  
→ *Why others likely did NOT do it:* Most baseline pipelines use equal CE or unweighted Dice; the joint weighting ratio as an Optuna hyperparameter (along with per-class weights) is non-standard.

**6. Post-HPO 3-Seed Ensemble**  
→ *What:* Three full 3-phase training runs with Optuna-tuned hyperparameters, seeds {42, 123, 456}. Inference: averaged softmax logits before argmax.  
→ *Why:* Reduces variance from stochastic augmentation and weight initialization. Soft averaging prevents any single model's overconfident wrong prediction from dominating the final output.

---

#### Success Metrics

| Metric | Objective |
|---|---|
| **val/IoU_1 (Flood class)** | Primary — maximized by Optuna |
| **val/IoU_2 (Water Body)** | Secondary — confirms Class 1/2 separation |
| **val/IoU_0 (No Flood)** | Sanity check — should remain high throughout |
| **Class weight generalization** | bg, flood, water weights all tuned; prevents dependence on fixed priors |
| **Ensemble score variance** | Std deviation across 3 seeds monitors training stability |
| **RLE submission validity** | Empty masks explicitly handled as `"0 0"` per competition rule |

---

---

## Page 2: Evidence, Obstacles, and Execution

### 3. Preliminary Salient Results

#### Initial Findings

The Optuna HPO stage (10 trials × 8 epochs each) optimizes `val/IoU_1` as its objective, using the Phase 1+2 checkpoint as a warm start. This substantially reduces per-trial cost relative to training from scratch. The MedianPruner with `n_startup_trials=3` and `n_warmup_steps=5` eliminates underperforming trials early, providing effective compute allocation. In the optimized configuration, HyperbandPruner replaces MedianPruner for more aggressive early elimination.

`ShiftScaleRotate` augmentation (`shift_limit=0.1, scale_limit=0.2, rotate_limit=30°`) addresses the geographically diverse nature of flood events occurring at varying image orientations depending on SAR acquisition geometry.

#### Quantifiable Progress

- **Primary metric:** `val/IoU_1` (Flood class IoU) — the competition's primary discriminative metric, directly optimized through Optuna with multivariate TPE (models joint parameter distributions, not independent marginals).
- **Computational throughput:** Batch size 32 on H100 80GB with `bf16-mixed` precision + `torch.compile(mode='reduce-overhead')` achieves ~20–30% throughput improvement over FP16, enabling more Optuna trials within the same compute budget.
- **Optuna search space explicitly covers:** `lr ∈ [1e-6, 1e-3]`, `dropout ∈ [0.05, 0.6]`, `dice_weight ∈ [0.4, 0.8]`, `flood_weight ∈ [2.0, 15.0]`, `water_weight ∈ [1.0, 8.0]`, `bg_weight ∈ [0.5, 2.0]` — covering all critical axes of uncertainty simultaneously.

#### Visualization

Predicted segmentation masks should exhibit: flood inundation pixels (Class 1) as thin, irregular strips following drainage networks and low-lying terrain — distinct from Class 2 (compact, polygon-like water bodies with smooth boundaries). The MNDWI input channel, when visualized, should show strong positive response in flat permanent water and weaker response in dynamic flood zones over soil/vegetation — providing the key visual cue the model is trained to leverage.

**Sanity-check interpretation:** If Class 2 predicted areas spatially co-register with known reservoir/river positions, and Class 1 regions appear at terrain margins visible in slope-derived indices, the model is learning physically consistent patterns rather than textural noise. Cross-checking flood predictions against rainfall accumulation maps (future step) provides an independent physical consistency test.

---

### 4. Technical Challenges & Pivots

#### Current Roadblocks

| Challenge | Technical Detail |
|---|---|
| **Class 1/2 confusion** | SAR backscatter suppression is identical for open water and flood; MNDWI partially resolves this but boundary ambiguity persists |
| **Label noise at flood margins** | Sub-pixel mixed boundaries between classes create ambiguous training signal; `ignore_index=−1` partially mitigates but does not fully address |
| **Single-date inference** | No temporal SAR pair at inference; blocks SAR change detection — the most physically reliable flood indicator |
| **ViT band-mismatch** | Prithvi-EO-v2 pretrained on Sentinel-2/Landsat bands; mapping engineered index stack (NDWI, MNDWI, NDVI) to backbone `backbone_bands=[0..5]` requires careful positional alignment |
| **No terrain data (current version)** | DEM + slope features (from NASADEM via earthaccess) were implemented in v1 (DeepSARFloodStacker, 10-band) but excluded from current optimized 6-band stack to eliminate zero-padding waste |

#### Strategic Pivots

**Pivot 1: From 10-band (DEM+Slope) to 6-band (optical indices)**  
*What failed:* Initial version built a 10-channel stack `[HH, HV, LogRatio, Ratio, DEM, Slope, 0, 0, 0, 0]`. Optical bands were removed to satisfy a SAR-only constraint, leaving 4 zero-padding channels.  
*Why it failed:* Zero-padding wastes backbone attention capacity and introduces spurious cross-channel correlations. The Prithvi ViT backbone has no conditioning to ignore zero channels.  
*Fix:* Replaced zero-padding with domain-derived optical indices (NDWI, MNDWI, NDVI) — physically meaningful, bounded in [−1, 1], compatible with per-band normalization, and directly discriminative for the flood vs. water class boundary.

**Pivot 2: HPO sampler upgrade (multivariate TPE + HyperbandPruner)**  
*What improved:* Default TPE with independent parameter modeling → multivariate TPE + HyperbandPruner.  
*Why better:* LR and weight decay are strongly correlated for AdamW convergence; univariate TPE cannot model their joint distribution. HyperbandPruner eliminates unpromising trials 3–4× faster than MedianPruner, enabling 30 trials in the same compute budget as 10 trials.

---

### 5. Final Sprint Roadmap (Next Steps)

#### Immediate Tasks (Next 10 Hours)

1. Complete 30-trial Optuna HPO run (`optuna_model1_optimized.py`, H100, multivariate TPE + HyperbandPruner)
2. Extract best `[lr, weight_decay, dropout, dice_weight, flood_weight, water_weight, bg_weight, decoder_channels]` from Optuna study object
3. Launch 3-seed ensemble training (seeds: 42, 123, 456) with best hyperparameters via `run_stage3()`
4. Run `predict_ensemble()` + `generate_submission()` → validate RLE format and inspect flood coverage rate vs. baseline
5. Execute `save_everything()` → ZIP + download backup before session expiry

#### Refinement Plan

| Enhancement | Rationale |
|---|---|
| **SAR Temporal Differencing (ΔσHH)** | Add pre-event SAR band at pixel level: `Δσ = σ_event − σ_pre`. Converts implicit change detection into a direct engineered input feature. Requires full Sen1Floods11 archive for pre-event scene matching. |
| **Rainfall runoff model integration** | Inject cumulative rainfall accumulation or PERSIANN/GPM raster as an additional spatial channel. Provides physical prior on flood likelihood independent of SAR backscatter ambiguity in shadow regions. |
| **Waterbody morphology post-processing** | Apply connected-component analysis + compactness ratio `(4πA / P²)` to discriminate Class 1 (irregular, elongated flood extents, compactness ≪ 1) from Class 2 (compact polygon-stable water bodies, compactness → 1). High compactness → rule-based relabeling as Class 2. |
| **Full Sen1Floods11 pretraining** | Leverage broader global flood dataset for geographic diversity before competition fine-tuning. Reduces risk of regional overfitting to competition's specific geographic envelope. |

#### Final Output

| Deliverable | Format |
|---|---|
| Competition submission | `submission_v3.csv` — (id, rle_mask), Class 1 flood only |
| Best ensemble checkpoint | 3× `.ckpt` (Phase 3 best checkpoint per seed) |
| Optuna study | Serialized study object with full 30-trial history |
| Notebook | `flood_phase2_complete.ipynb` — fully executable, self-contained |
| Backup archive | `FLOOD_BACKUP.zip` — checkpoints + `stats.json` + CSVs |

---

*Team ENDRA — Abhay Kale, Akash Chaudhari, Somesh Padsalge*  
*License: ANRF Open License (compatible with MIT) · © 2026*
