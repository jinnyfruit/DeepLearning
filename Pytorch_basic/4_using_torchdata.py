'''
file name: Using dataset from Pytorch
modified: 2023.02.24
'''
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import requests

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(1.0,))     # mean = 0.5, Standard Deviation = 1.0
])

download_root = '/Users/jinnyfruit/PycharmProjects/DeepLearning/Pytorch_basic'

train_dataset = MNIST(download_root, transform = mnist_transform, train = True, download = True)
validation_dataset = MNIST(download_root, transform = mnist_transform, train = False, download = True)
test_dataset = MNIST(download_root, transform = mnist_transform, train = False, download = True)

