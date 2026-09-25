# Project: Multi-task Semantic Segmentation + Monocular Depth Estimation

Language rule: Write code, comments, docstrings, variable names and
commit messages in English.

## Goal

Train a multi-task convolutional network that, from a single RGB image, simultaneously
produces:
- a pixel-wise **semantic segmentation map** (road, vehicles, pedestrians, etc.);
- a **dense depth map** (metric depth, in meters).

The starting point is a **DeepLab** model as baseline (third-party code already present in
this repo, covering part of the project), which is extended to Multi-Task Learning (MTL).
This is a research project: what matters is fair, reproducible comparisons between the
variants, not just getting the code to run.

## Dataset: Miniscapes (a subset/derivative of Synscapes, arXiv 1810.08705, Wrenninge & Unger)

**We use the dataset already wired into the code** (`source/datasets/dataset_miniscapes.py`,
`DatasetMiniscapes`), not the raw Synscapes release. Synscapes is a photorealistic synthetic
street-scene dataset; the raw release (25,000 RGB PNGs at 1440x720, Cityscapes-convention class
PNGs, float OpenEXR planar depth) is background only and is NOT what the code reads.

Facts read from the code (NOT yet verified on the actual data files):
- Layout: `<dataset_root>/<split>/<modality>/<index>.<ext>` with split in `train|val|test` and
  modality in `rgb` (`.jpg`), `semseg` (`.png`), `depth` (`.png`); indices are `0..N-1`.
- Split sizes are hardcoded in `__len__`: 20000 train / 2500 val / 2500 test (25,000 total).
  The split is fixed by the provided directory layout; do not re-split or move files.
- Semantic segmentation: 19 Cityscapes train classes (torchvision `Cityscapes.classes` without
  `ignore_in_eval`); label 255 = void/ignore. The PNGs are assumed to already hold train ids.
- Depth: 8-bit PNG storing quantized **disparity**, decoded to meters (planar depth) over the
  range **4 m - 300 m**; pixel value 0 = invalid (out of range, e.g. sky). Quantization is uniform
  in disparity over 254 levels, so the depth error grows with distance (about 0.2% at 4 m, about
  2.4% at 50 m, about 15% at 300 m; derived from the constants, so treat as a GT precision limit).
- Depth statistics hardcoded in the dataset class: mean 27.0727 m, std 29.1264 m.
- RGB is normalized with ImageNet statistics.

To verify by inspecting the actual data (do not assume):
- [ ] image resolution, and that rgb / semseg / depth have identical size for every sample
- [ ] set of label values in `semseg` PNGs is exactly {0..18} + {255} (already train ids)
- [ ] actual depth range, fraction of invalid (0) pixels, the hardcoded mean/std and 4-300 m range
- [ ] effective split: counts per split, indices contiguous, no missing/extra files
- [ ] whether the test split has semseg/depth ground truth (it may be hidden for the course grader)
- [ ] ground truth is aligned with the RGB (visual overlay check on sampled images)
- [ ] data path: read it from config / environment variable (`DATASET_ROOT`), never hardcode it

Do not download the dataset and do not commit data or checkpoints (check `.gitignore`).

## Experiments to implement

Three MTL approaches, plus the single-task baselines. All of them use the same backbone,
the same split and the same data pipeline, otherwise the comparison is not valid.

0. **Single-task baselines**: DeepLab for segmentation only and for depth only (same
   encoder), used as reference to measure the gain/loss brought by MTL.
1. **Joint architecture**: shared encoder, ASPP and decoder. Output has n+1 channels:
   n class logits + 1 depth channel. Cross-entropy only on the first n channels,
   depth loss only on the last one.
2. **Branched architecture**: only the encoder is shared; each task has its own dedicated
   ASPP modules and decoder.
3. **Depth as classification with adaptive bins** (AdaBins idea, Bhat et al.,
   CVPR 2021): the depth range is discretized into bins whose widths/positions are
   **predicted per image** through an attention mechanism; the depth output becomes the
   per-pixel probabilities of falling into each bin, and the final depth is obtained as
   the linear combination of the bin centers weighted by those probabilities. Read the
   paper before implementing (module structure, additional loss on the bin centers) and
   document in `docs/` every choice where you deviate from it or from the DeepLab setup.

Number of bins, min/max depth range and the choice of attention module are configuration
parameters, not constants in the code.

## Metrics and losses

- **Segmentation**: mIoU. Accumulate the confusion matrix over the whole validation set
  and compute per-class IoU from it (do not average per-image IoUs). Ignore void pixels
  (`ignore_index`). Also report per-class IoU.
- **Depth**: **SI-logRMSE**. With d_i = log(pred_i) - log(gt_i) over valid pixels:
  `sqrt( mean(d_i^2) - lambda * mean(d_i)^2 )`. The value of lambda (1.0 = fully scale
  invariant; AdaBins uses 0.85 in its loss) must be an explicit parameter and reported
  together with the results. Always compute in meters, with a valid-pixel mask
  (gt finite, > 0 and <= chosen max depth): no `inf`/`nan` and no sky in the metric.
  Never take `log(0)`: clamp with an epsilon.
- **MTL loss**: start from a weighted sum with weights in the config; always log the
  per-task losses separately. More sophisticated weighting schemes (e.g. uncertainty
  weighting) are a later ablation, not the default.

## How to work on this repo

First thing in a new session: explore the repo (dependencies, entry points,
configs) and **update the "Commands" section below** with what you actually find.
Do not assume the framework (PyTorch/TensorFlow) or the DeepLab version: verify them.

Third-party code:
- Modify it as needed. Prefer extending/wrapping (new modules, subclasses)
  over rewriting existing files.
- Respect license and attributions; do not remove headers or credits.
  
Experiments:
- One experiment = one versioned config file. Seeds are fixed and logged.
- Save config, seed, commit hash and metrics together with the checkpoints. Never
  overwrite results from previous runs.
- Do not tune hyperparameters by looking at the test set. Report numbers as they are,
  even if unfavorable.

Before claiming "it works", verify:
- Shape tests for the three architectures (expected output: n+1 channels; two separate
  heads; bin probabilities summing to 1).
- Tests of the metrics against a simple reference (numpy) on small inputs.
- Smoke test on very few images (e.g. 8): the loss must go down and the model must be
  able to overfit a mini-batch.

Operating rules:
- Do not launch long or expensive training runs without asking me: first propose the
  command, estimated duration and resources.
- Keep in mind that this repo will be run on a hpc cluster of GPUs and not locally, if you need to test locally do not use the GPU.
- For large changes (new architecture, refactor) propose a short plan first.
- Small, atomic commits with clear messages.
- If something is ambiguous or cannot be verified from the data or the paper, say so and
  ask: do not invent values (depth range, split, classes, hyperparameters).

## Commands

Found by exploring the repo (PyTorch + PyTorch Lightning 2.x, DeepLabV3+ with ResNet18/34 encoder,
ETH CVAIAC course template; dataset class is `miniscapes`, a Synscapes subset). Full guide: `doc/training_system.md`.

- Environment setup: `pip install -r requirements.txt` (Python >= 3.11; versions pinned as of 2026-09-19).
  Locally, the conda env `industry` has CPU torch 2.14 / lightning 2.6.6 / wandb / pytest.
- Configuration: `config.yaml` (template; copy to `configs/<experiment>.yaml`). All training parameters, input/output
  dirs and W&B identity live there; schema in `source/utils/config.py`. Values are overridable with `section.key=value`.
  `dataset_root` / `output_dir` read the env vars `DATASET_ROOT` / `SAVEDIR`.
- Training (any model/task set): `python -m source.scripts.train --config config.yaml [section.key=value ...]`.
  Each run writes to `<output_dir>/<name>_seed<seed>_<timestamp>/` (config, run_info.json with commit, checkpoints,
  csv metrics, predictions) and never overwrites another run. `wandb.mode=disabled|offline|online`.
- Experiments: one generic `ExperimentMultiTask` (`source/experiments/experiment_multitask.py`); the experiment is
  chosen by `experiment.tasks` + `model.model_name` (see the table in `doc/training_system.md`).
- Data checks: `python -m source.datasets.dataset_miniscapes <root>` (integrity), `source/scripts/compute_statistics.py`.
- Evaluation (mIoU, SI-logRMSE): computed in validation by `source/utils/metrics.py`; the SI-log metric
  currently hardcodes lambda = 1 and a x100 scale. `SILogLoss` in `source/losses/si_log.py` is a TODO stub; the
  depth loss in use is `MaskedDepthRegressionLoss` (L1/L2).
- Tests: `python -m pytest -q` (CPU, ~1 min, synthetic dataset, needs ~1 GB temp disk).
- Lint/format: TODO (no tooling configured)

Not implemented in the template (TODO stubs): `ModelDeepLabV3PlusMultiTask` (branched), the adaptive-bins part of
`ModelAdaptiveDepth`, `SelfAttention`, `SILogLoss`.

## Open decisions (to settle together with me)

- Framework and DeepLab version (v3 / v3+) and backbone
- Input resolution (native 1440x720, crop, or 2k) and data augmentation
- Number of bins (the depth range 4-300 m is fixed by the dataset encoding; confirm on the data)
- Number of classes (19 train classes per the code; confirm on the data) and void handling (255)
- Initial weights of the multi-task loss
