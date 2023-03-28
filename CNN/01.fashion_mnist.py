import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# transform: image -> tensor
train_dataset = torchvision.datasets.FashionMNIST("../CNN/data", download=True, transform = transforms.Compose([transforms.ToTensor()]))
test_dataset = torchvision.datasets.FashionMNIST("../CNN/data", download=True, train=False,transform = transforms.Compose([transforms.ToTensor()]))

# 데이터를 데이터로더에 전달
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100)   # 배치 사이즈 100으로 받아옴
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)

# 분류 클래스 정의
labels_map = {0: 'T-Shirt', 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}
fig  = plt.figure(figsize = (8,8));
columns = 4;
rows = 5;
for i in range (1, columns * rows + 1):
    img_xy = np.random.randint(len(train_dataset));

