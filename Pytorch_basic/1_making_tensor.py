'''
file name: first tensor code
modified: 2023.02.23
'''
import torch

#M1 맥북에서 GPU 사용 여부 파악
device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
print(f"PyTorch version:{torch.__version__}") # 1.12.1 이상
print(f"MPS 장치를 지원하도록 build 되었는지: {torch.backends.mps.is_built()}") # True 여야 합니다.
print(f"MPS 장치가 사용 가능한지: {torch.backends.mps.is_available()}") # True 여야 합니다.

# Making New tensor
print(torch.tensor([[1,2],[3,4]]))    # 2차원 tensor
print(torch.tensor([[1,2],[3,4]], device = device))    # GPU에 텐서 생성
print(torch.tensor([[1,2],[3,4]], dtype=torch.float64))    # dtype으로 텐서 생성

# Tensor to ndarray
temp = torch.tensor([[1,2],[3,4]])
print(temp.numpy())

temp = torch.tensor([[1,2],[3,4]], device = device)
print(temp.to("cpu").numpy())   # GPU 텐서를 CPU 텐서로 변환 -> ndarray로 변환
