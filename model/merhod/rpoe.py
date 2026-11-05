import torch

x=torch.tensor([1, 2, 3, 4, 5])
y=torch.tensor([10, 20, 30, 40, 50])

condition = x> 3

result = torch.where(condition, x, y)

print(result)  # Output: tensor([10, 20, 30,  4,  5])



t=torch.arange(0, 10, 2)
print(t)

t2=torch.arange(5, 0, -1)
print(t2)


v1=torch.tensor([1, 2, 3])
v2=torch.tensor([4, 5, 6])
result=torch.outer(v1, v2)
print(result)  # Output: tensor([[ 4,  5,  6],[ 8, 10, 12],[12, 15, 18]])

#拼接操作cat
t1=torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
t2=torch.tensor([[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]])
print(t1.shape) 
result=torch.cat((t1, t2), dim=0)#拼接操作,在第0个维度拼接
print(result.shape)  
print(result)  


result=torch.cat((t1, t2), dim=-1)#拼接操作,在第2个维度拼接
print(result.shape)  
print(result)  


#自适应增加维度
t1=torch.tensor([1, 2, 3])
t2=t1.unsqueeze(0) #在第0个维度增加一个维度
print(t1.shape)
print(t2)
print(t2.shape)
