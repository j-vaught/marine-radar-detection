# Revision Python environment

The revision environment uses Python 3.11 and the package versions in `uv.lock`. The four primary dependencies are PyTorch 2.5.1, torchvision 0.20.1, Ultralytics 8.4.14, and pycocotools 2.0.11. On Linux x86-64, `uv` selects the CUDA 12.1 PyTorch and torchvision wheels from the explicit PyTorch index. Other platforms use their standard PyPI wheels. CUDA execution also requires a compatible NVIDIA driver on the host.

From the repository root, create a new environment and install the locked dependencies.

```sh
uv sync --locked
```

The default development group includes pytest, Ruff, and ty. Use `uv run pytest` to run tests. The lock fixes transitive package versions for this environment. It does not by itself reproduce the input data, model weights, driver, hardware, random seeds, or all packages from earlier experiment environments. Record those separately when reproducing a run.

The lock can be refreshed deliberately with `uv lock`; review and commit the resulting changes before using `uv sync --locked` again. Native Rust and CUDA builds have additional toolchain requirements documented with those components.
