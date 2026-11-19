import torch
import platform

print("系统:", platform.platform())
print("架构:", platform.machine())
print("PyTorch:", torch.__version__)
print("MPS 已编译:", torch.backends.mps.is_built())
print("MPS 可用:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

x = torch.randn(1000, 1000, device=device)
y = x @ x.T

print("当前设备:", device)
print("计算测试:", y.shape)
