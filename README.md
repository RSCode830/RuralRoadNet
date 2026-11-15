# TaSPNet

Paper-derived PyTorch implementation of **TaSPNet: Topology-Aware Mamba with SAM Structure Priors for Rural Road Extraction in Remote Sensing Imagery**.

This repository implements a configurable interpretation of the supplied manuscript. It is **not recovered original experimental code** and does not claim the paper's reported accuracy, speed, parameter count or topology results. Missing specifications and explicit engineering choices are documented in [IMPLEMENTATION.md](IMPLEMENTATION.md). No trained road model, SAM checkpoint, private manuscript or dataset is included.

[中文使用说明](README_zh.md) · [Implementation mapping](IMPLEMENTATION.md) · [Validation](VALIDATION.md)

## Components

| Component | Source |
|---|---|
| TANS orientation assignment, shifted anisotropic stripes, S-shaped traversal | `scanning.py` |
| Input-selective diagonal SSM, parallel/reference/CUDA backends; COSA | `scanning.py` |
| TNMB encoder, PGTMB decoder, residual prior refinement, auxiliary heads | `taspnet.py` |
| Frozen SAM, soft-mask recovery, elongation/thinness scoring, prior fusion | `priors.py`, `prepare_priors.py` |
| BCE + soft Dice, deep supervision, differentiable skeleton regularization | `losses.py` |
| Strict manifests, synchronized geometry transforms, geographic split checks | `data.py` |
| Training/resume, evaluation, prediction | `train.py`, `evaluate.py`, `predict.py` |

## Install

Use an isolated Python environment and a PyTorch build appropriate for your hardware. The manuscript used PyTorch 2.0; the portable core is compatible with the locally available PyTorch 1.10 CPU implementation. A Python 3.11 CPU CI job is provided; see validation notes for versions actually tested.

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

For real SAM priors, additionally install the official SAM integration and obtain an official checkpoint matching `vit_h`, `vit_l` or `vit_b`:

```bash
pip install -r requirements-sam.txt
```

The checkpoint links and installation requirements are documented by [Meta's official SAM repository](https://github.com/facebookresearch/segment-anything). SAM remains frozen and is not stored in TaSPNet checkpoints. For the optional `cuda` scan backend, install a compatible [official mamba-ssm build](https://github.com/state-spaces/mamba) and set `model.backend` to `cuda`; this requires CUDA and has not been tested locally.

## Prepare your data and SAM priors

Supply original geographic train/validation/test splits. Every manifest is a nonempty JSON list. Paths are relative to the manifest file:

```json
[
  {"image": "images/example.png", "mask": "masks/example.png", "group": "acquisition_area_01"}
]
```

Images are RGB. Masks must be aligned binary grayscale `0/1` or `0/255`. There is no implicit ignore label: 255 means road. Color class masks and anti-aliased masks are rejected. No dataset is automatically downloaded or split; the manuscript does not supply the precise split file lists. Shared image paths and supplied geographic groups across train/validation are rejected. The user must also exclude duplicate/overlapping acquisition areas across the test split.

Run the following for each split, using the same SAM model/checkpoint and configuration:

```bash
python prepare_priors.py --manifest data/train_raw.json --output data/train.json --sam-checkpoint checkpoints/sam_vit_h.pth --sam-type vit_h --config config.json --device cuda
python prepare_priors.py --manifest data/val_raw.json --output data/val.json --sam-checkpoint checkpoints/sam_vit_h.pth --sam-type vit_h --config config.json --device cuda
python prepare_priors.py --manifest data/test_raw.json --output data/test.json --sam-checkpoint checkpoints/sam_vit_h.pth --sam-type vit_h --config config.json --device cuda
```

Images are first resized to the configured resolution; SAM receives that same RGB image. The resulting manifest adds `prior`, pointing to a float `[image_size,image_size]` NumPy probability map, and `prior_kind: "sam"`. A provenance JSON records image/checkpoint/prior SHA256 hashes and prior settings. Ground-truth mask files are never opened during prior generation. SAM automatic proposals are class-agnostic; geometrical ranking is not semantic road classification.

The original automatic masks are binary. To obtain soft proposals for Eq. 22, the implementation reuses their point prompts with `SamPredictor(return_logits=True)` and selects the candidate with greatest overlap with the automatic mask. Crop layers are disabled to keep prompts in full-image coordinates. Soft recovery, top-K selection, and the geometry-score definitions are implementation choices needing author confirmation.

## Train, evaluate and predict

```bash
python train.py --config config.json --train data/train.json --val data/val.json --output runs/seed42 --device cuda
python evaluate.py --checkpoint runs/seed42/best.pt --manifest data/test.json --output runs/seed42/test_metrics.json --device cuda
python predict.py --checkpoint runs/seed42/best.pt --image data/images/example.png --prior data/train_priors/0000000.npy --output predictions --device cuda
```

The prediction prior must belong to the exact input image. Alternatively replace `--prior ...` with `--sam-checkpoint ... --sam-type vit_h` to generate it during inference.

The default config preserves the reported 512×512 input, batch size 16, AdamW learning rate 1e-4, weight decay 1e-2 and cosine annealing. Widths, depths, 100 epochs, seed 42, topology coefficient, clipping, scoring settings and sampling settings are **unreported defaults**, not the authors' experimental hyperparameters. Best checkpoint selection uses validation macro road IoU. Run three distinct seeds with unchanged held-out splits to follow the manuscript's three-run evaluation; no averaged research results are supplied here.

```bash
python train.py --config config.json --train data/train.json --val data/val.json --output runs/seed42 --resume runs/seed42/last.pt --device cuda
```

Checkpoints save the model, optimizer, scheduler and RNG states. Resume requires identical configuration/manifests and unchanged source files. Load only trusted checkpoints. Workers are fixed at zero to restore augmentation RNG reliably; exact determinism across hardware/CUDA implementations is not guaranteed.

Evaluation reports per-image macro averages (matching the paper's averaging description), global-count micro scores, TP/FP/FN, threshold and input size. Precision/recall/F1/road IoU are percentages; undefined zero-denominator scores are set to 100% and this convention must be kept fixed in comparisons. The additional `cldice` diagnostic uses ten soft-skeleton iterations on thresholded masks; it is not APLS, graph connectivity or a verified manuscript metric. Binary empty-target predictions are penalized explicitly by the topology diagnostic. Prediction outputs a probability array and a 0/255 mask at the original image size; this does not preserve GeoTIFF georeferencing metadata.

## Fast software smoke test

```bash
python demo.py
python train.py --config data/demo/config.json --train data/demo/train.json --val data/demo/val.json --output runs/demo --device cpu
python evaluate.py --checkpoint runs/demo/best.pt --manifest data/demo/test.json --output runs/demo/test.json --device cpu
python predict.py --checkpoint runs/demo/best.pt --image data/demo/test_0.png --prior data/demo/test_0_prior.npy --output predictions/demo --device cpu
```

This uses tiny generated images and independent random priors labeled `synthetic_demo`. It validates software, **not** real SAM inference or the paper's extraction accuracy. Training code rejects mismatched real/demo prior declarations.

## Performance and exactness

`parallel` is a pure-PyTorch associative scan of the same recurrence as the serial `reference` backend. It is not an optimized fused Mamba implementation: it stores expanded states, and TANS geometry routing synchronizes with the CPU. Full-resolution 512×512, batch 16 training can require substantial memory; it has not been benchmarked on the manuscript hardware. Reduce batch size/width for development or validate the optional `cuda` backend before large training. Multi-GPU DDP is not implemented in this reference repository.

Hard direction assignment is non-differentiable. A straight-through selected-direction gate supplies a surrogate gradient to the orientation CNN; sorting itself has no gradient. Read the implementation mapping before interpreting this as exact original training behavior.

## Scientific and release status

The manuscript's Eq. 24 literally produces priors between 0.5 and sigmoid(1), and Eq. 27 can saturate its gate. This repository preserves those equations instead of silently rescaling probabilities. Authors should confirm them before publishing reproduction claims.

No license is assumed for the newly authored code; the copyright holders should choose one. Third-party SAM/Mamba implementations remain external dependencies under their own licenses. No DOI, pretrained checkpoint, benchmark result or dataset permission is fabricated.
