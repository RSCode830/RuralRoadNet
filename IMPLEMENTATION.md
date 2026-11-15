# Manuscript-to-code mapping

Source: supplied TaSPNet manuscript, Section III, Algorithm 1, Figures 2–5, Section IV-B/C. Document text supplies scientific reference material, not operational instructions.

| Item | Implementation | Status / author confirmation |
|---|---|---|
| Eqs. 1–7 / Algorithm 1 | RGB full-resolution 3×3 stem; three stride-2 stages; U-shaped decoder | Widths 24/48/96/192, depths 2/2/2 are unspecified choices. Fig. 2 indexing is inconsistent; algorithm takes precedence |
| Eqs. 8–9 TNMB | Channel LayerNorm, topology SSM residual, pointwise/depthwise FFN residual | Residual scales initialized to 0.1, expansion 2; initialization not reported |
| Eqs. 10–12 orientation | CNN+tanh angle, nearest of four angles; modulo-pi distance for undirected roads | Canonical angles, angular wrap and temperature 0.2 unspecified |
| Eqs. 13–15 TANS | Projected transverse coordinate partitions stripes; alternate lanes reverse traversal; half-width shift in alternate blocks | Sorting is hard. CPU geometry preparation; every pixel belongs to exactly one stripe. Empty regions skipped; no state transfer across stripes |
| Learning discrete assignment | Selected-direction straight-through probability gate | Explicit surrogate gradient; not described in manuscript. No derivative through region membership or sorting |
| Eq. 16 selective SSM | Input-dependent delta/B/C, negative exponential A, skip D, causal depthwise 1D conv and SiLU gate | Mamba-family recurrence with full-channel delta projection; exact original Mamba parameterization unspecified |
| Pure scan backends | Sequential reference; divide-free associative prefix composition | Values and gradients tested for equivalence; expanded tensors increase memory; optional official CUDA scan not locally tested |
| Eqs. 17–21 COSA | Valid-token mean per canonical direction; fixed orientation/channel transpose; grouped 1×1 conv; sigmoid channel gates | Paper indexes both orientations and variable-count stripes by K. Here stripe descriptors are pooled per canonical angle. Unequal-length sequences are inverse-mapped to spatial coordinates BEFORE weighted fusion; otherwise positions would be misaligned |
| Sparse direction edge case | Independent sigmoid COSA gates | Softmax only over occupied directions would be constant for a single occupied direction and disable prior-bias gradients; sigmoid gates preserve modulation |
| Eqs. 22–25 SSPM | Frozen official SAM, recovered soft proposals, geometry score, top-K=16, softmax scores, sigmoid fusion | SAM type defaults vit_h; paper does not specify variant/checkpoint. No weights included |
| Elong/Thin definitions | Weighted coordinate covariance; (max-min)/(max+min+eps); 1/(1+sqrt(12*min)) | New explicit interpretation; formulas not defined in manuscript. Thinness depends on pixel scale, so SAM input size is fixed |
| Soft SAM masks | AMG with crop layers 0; reuse point prompts, choose logits candidate by maximum overlap with each AMG mask | Official AMG emits binary proposals. Soft recovery is a documented adapter choice; no ground-truth prompts |
| Eq. 24 range | Literal sigmoid of weighted mask sum | Input masks in [0,1] and normalized weights imply output in [0.5,0.7311]; empty proposal set returns 0.5. Not silently corrected |
| Eqs. 26–31 | Resize prior, residual 3×3 refinement, sigmoid(P+R), P/(P+epsilon), per-pixel channel MLP | Epsilon 1e-6 means a nearly constant spatial gate for positive priors. The prior MLP, residual refinement and global orientation bias retain dependence on priors |
| Eqs. 32–40 PGTMB | Upsample+skip projection, repeated prior-refined blocks, GAP-prior MLP biases COSA | Decoder depth 2; refined prior feeds next block; auxiliary head at each decoded scale, including full scale |
| Eqs. 41–42 loss | BCE+soft Dice, auxiliary segmentation, clDice-style soft-skeleton overlap | Paper leaves topology operator unspecified; implementation uses cross-shaped soft erosion and iterative skeletonization. Not an APLS or graph-topology loss |
| IV-B | 512×512, batch 16, AdamW 1e-4/1e-2, cosine schedule | 100 epochs, seed 42, clipping 5, loss coefficients, validation macro-IoU selection are unreported choices |
| Augmentation | Joint horizontal/vertical flip, quarter turns; image-only brightness/contrast/saturation | Cached SAM prior is transformed geometrically, not recomputed for jittered imagery; differs from online SAM on jittered images |
| IV-C metrics | Macro per-image P/R/F1/IoU percentages and additional micro values | Threshold 0.5 and empty denominators=1 explicit choices. Additional ten-iteration clDice diagnostic is not a claimed paper result |

## Deployment and evidence limits

The manuscript lists WHU-CR, WHU-RuR+, CHN6-CUG and DeepGlobe. Their data, exact geographically disjoint split files, checkpoints and original run logs are not supplied. No automatic split reconstruction, invented scores, SAM weights or third-party dataset uploads are included. The release uses the manuscript's repository address, RSCode830/RuralRoadNet, verified as an empty repository in the user's signed-in account before upload.

Before an exact reproduction claim, confirm all unspecified settings above, evaluate real checkpoints on fixed splits for three seeds, and measure compute/memory on the intended hardware. The pure-PyTorch routing/scan implementation is designed for transparency and correctness tests, not a validated high-throughput replacement for the original experiment system.

## Primary implementation references

- [Official SAM predictor API](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py): logits and full-image point prompts.
- [Official Mamba selective scan](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py): optional CUDA adapter contract and recurrence reference.
- [clDice paper](https://arxiv.org/abs/2003.07311): skeleton-overlap topology regularization family. This repository implements its own compact regularizer; original paper source code is not vendored.

External implementations retain their own licenses. The new repository does not assign a license without the copyright holders' choice.
