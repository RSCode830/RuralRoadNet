# TaSPNet

PyTorch implementation of **TaSPNet: Topology-Aware Mamba with SAM Structure Priors for Rural Road Extraction in Remote Sensing Imagery**.

 [Implementation mapping](IMPLEMENTATION.md) · [Validation](VALIDATION.md)

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

```bash
python train.py --config config.json --train data/train.json --val data/val.json --output runs/seed42 --resume runs/seed42/last.pt --device cuda
```
