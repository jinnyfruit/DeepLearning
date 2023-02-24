'''
file name: Using custom dataset
modified: 2023.02.24
notification: 실제로 돌아가는 코드가 아니라 Custom data를 어떻게 구현하는가에 대한 코드임을 명시함
'''
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    # 필요한 변수 선언, dataset 전처리
    def __init__(self, csv_file):
        self.label = pd.read_csv(csv_file)

    # 전체 dataset의 길이 반환 (총 sample수)
    def __len__(self):
        return len(self.label)

    # dataset에서 특정 데이터를 가져오는 함수
    def __getitem__(self, idx):
        sample = torch.tensor(self.label.iloc[idx,0:3]).int()
        label = torch.tensor(self.label.iloc[idx,3]).int()
        return sample, label

tensor_dataset = CustomDataset('../covtype.csv')
dataset = DataLoader(tensor_dataset, batch_size=4, shuffle=True)