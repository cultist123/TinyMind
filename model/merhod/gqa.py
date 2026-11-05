import torch
import torch.nn as nn

dropout_layer = nn.Dropout(p=0.5)

t1=torch.Tensor([1,2,3])
t2=dropout_layer(t1)
#dropout丢弃为了保持期望值不变，会对保留的元素扩大2倍

print(t2) 

#linear层
linear = nn.Linear(in_features=3, out_features=5,bias=True)
t1=torch.Tensor([1,2,3])

t2=torch.Tensor([[1,2,3]])
output = linear(t2)

print(output)
#线性变换对应用的张亮乘w矩阵+bias

#维度变换
t=torch.Tensor([[1,2,3,4,5,6],[7,8,9,10,11,12]])
t_view1=t.view(3,4)
print(t_view1)
t_view2=t.view(4,3)
print(t_view2)

#维度交换
t1=torch.Tensor([[1,2,3],[4,5,6]])
t1=t1.transpose(0,1)
print(t1)

#掩码计算
x=torch.Tensor([[1,2,3],[4,5,6],[7,8,9]])
print(torch.triu(x))
print(torch.triu(x, diagonal=1))
#输出上三角矩阵
print(torch.tril(x))
print(torch.tril(x, diagonal=-1))
#输出下三角矩阵

#reshape和view比较近似
x=torch.arange(1,7)
print(x)
y=torch.reshape(x,(2,3))
print(y)
z=torch.reshape(x,(3,-1))#-1表示自动计算该维度的大小
print(z)