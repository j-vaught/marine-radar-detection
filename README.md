# Marine radar detection

This repository contains classical and learned detectors for marine radar imagery, together with the revision experiment and analysis code. The revision trains each learned model on one source site and evaluates the selected checkpoint on fixed destination holdouts. Classical methods select one source configuration and apply it to the same holdouts.

Start with the [revision reproduction guide](docs/review-reproduction.md) for the frozen inputs, worker releases, experiment matrix, archive replay and supplemental audits. The [Python environment guide](docs/revision-environment.md) describes the locked dependencies and native build requirements. The [common-GPU timing guide](docs/revision-inference-timing.md) and [Wavelet extension workflow](src/revision/wavelet_study.md) document those separate studies.

The processed dataset, pretrained weights and full experiment archive are separate inputs. A checkout alone does not reproduce the reported results. Preserve their hashes and the per-attempt records described in the reproduction guide.
