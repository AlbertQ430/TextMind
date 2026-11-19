"""Smoke test：确认设备与矩阵运算可用。

独立运行：``python test/test_device.py``

本模块在导入时不执行任何计算或设备检测，只定义 ``main()``，
因此可以安全地被 ``python -m unittest discover`` 导入。
"""

import platform


def main() -> int:
    import torch

    print("系统:", platform.platform())
    print("架构:", platform.machine())
    print("PyTorch:", torch.__version__)
    print("MPS 已编译:", torch.backends.mps.is_built())
    print("MPS 可用:", torch.backends.mps.is_available())

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    x = torch.randn(1000, 1000, device=device)
    y = x @ x.T

    print("当前设备:", device)
    print("计算测试:", y.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
