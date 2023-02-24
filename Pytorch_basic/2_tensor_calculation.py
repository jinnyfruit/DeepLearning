'''
file name: torch calculation
modified: 2023.02.23
'''
import torch
mps_device = torch.device("mps")

temp = torch.FloatTensor([1,2,3,4,5,6,7,])  # 1차원 벡터 생성
print("\n---------- Index approach ----------")
print(temp[0],temp[1],temp[-1])

print("\n---------- Slice approach ----------")
print(temp[2:5],temp[4:-1])

v = torch.tensor([1,2,3])   #길이가 3인 vector
w = torch.tensor([3,4,6])
print("\n---------- Result of vector reduction ----------")
print(w-v)  #길이가 같은 벡터간 연산

temp = torch.tensor([
    [1,2], [3,4]    # 2by2 행렬
])
print("\n---------- Original tensor shape ----------")
print(temp.shape)
print("\n---------- Transformed tensor shape using View() ----------")
print(temp.view(4,1))
print(temp.view(-1))
print(temp.view(1,-1))  # 여기서 -1은 앞에 나온 숫자 * 모르는 값이 모든 원수의 개수가 나오면 된다는 것을 의미
print(temp.view(-1,1))

