# Validation record

Local verification on 2026-11-15 used Windows, Python 3.7.6 and PyTorch 1.10.1+cpu. Tests were run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to avoid unrelated global plugins.

- **11 tests passed.** Coverage: associative/serial SSM values and gradients; orientation stripe partition and inverse mapping with/without shifts; network backward on square/odd shapes; finite parameter gradients and nonzero orientation/prior-bias gradients; batch isolation despite variable padded sequences; geometry scoring; exact Eq. 24 sigmoid; empty proposals; skeleton-overlap behavior; hand-computed metrics; strict data/prior contract and split rejection; checkpoint prediction equality.
- **Two synthetic training epochs completed.** Held-out synthetic evaluation and original-size probability/mask prediction completed. The model predicts background after this tiny smoke run; these diagnostic numbers are not research evidence and do not establish training convergence.
- **Interrupted/resumed training matched uninterrupted training exactly.** A two-epoch schedule was stopped after epoch one with `--stop-after 1`, then resumed; every final model tensor equaled the uninterrupted two-epoch run.
- **Default architecture forward checked at 32×32.** Output `[1,1,32,32]` was finite; 2,729,419 learnable parameters excluding frozen SAM. This number describes this implementation only, not a reproduced paper complexity value.
- The only pytest warning came from the legacy environment's deprecated `imp` import in Windows utilities.

Not verified locally: actual SAM inference (no SAM checkpoint supplied), optional mamba-ssm CUDA backend, 512×512 batch-16 memory/speed, multi-GPU operation, real dataset convergence or manuscript accuracy. GitHub Actions provides a separate modern Python CPU check; consult its run status rather than assuming success from configuration alone.

Generated images, cached priors, checkpoints and diagnostic predictions remain outside the source package. The test data can be recreated with `python demo.py`.
