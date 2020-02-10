import torch
import random
import string
from debug import ptf_tensor

## 创建新的张量：
t2=torch.tensor([[0,1,2],[3,4,5],[6,7,8]])
ptf_tensor(t2,'common')

## 创建新的张量（指定）：
t=torch.zeros(3,3,1)
ptf_tensor(t,'zeros') # size(3,3,1)

t=torch.full((3,3),5.)
ptf_tensor(t,'full') # size(3,3), All elements are 5.

t=torch.eye(5)
ptf_tensor(t,'eye') # 对角线是1，其余是0 ，单位矩阵

## 创建随机的张量：
t=torch.randint(low=0,high=4,size=(3,3)) #指定最大最小值
ptf_tensor(t,'randint') #创建离散分布的int类型随机数

t=torch.rand(3,3)
ptf_tensor(t,'rand') #创建 0-1之间的随机数

## 创建数列：
t=torch.arange(0,8,step=2) #差是2 的等差数列
ptf_tensor(t,'arange array') 

t=torch.arange(24)
ptf_tensor(t,'default step=1') # 默认0-23 step=1


## squeeze(del) 和 unsqueeze(add)

t=torch.arange(24)
ptf_tensor(t)
t2=t.unsqueeze(dim=0)
ptf_tensor(t2,tag='t add dim=0')
t3=t.unsqueeze(dim=1)
ptf_tensor(t3,tag='t add dim=1')
t4=t3.squeeze(dim=1)
ptf_tensor(t4,'t del dim=1')
t=torch.arange(24).reshape(3,8)
ptf_tensor(t,'after reshpae')



# 张量部分元素的选取
t=torch.arange(24).reshape(2,6,2)
index = torch.tensor([3,4]) # 选择第index=3和4的元素
ptf_tensor(t,'t')
ptf_tensor(t.index_select(1,index)) #在 dim=1 上选取
# 提示：观察【2，6，2】-->【2，2，2】

t=torch.arange(0,50,step=6)
ptf_tensor(t,'raw t')
ptf_tensor(t[3:6]) #选择index from 3 to 6
ptf_tensor(t[-2:],'from back select')


# 张量的拓展和拼接
tp=torch.arange(12).reshape(3,4)
ptf_tensor(tp,'tp')
tn=-tp
ptf_tensor(tn,'tn')
tc0=torch.cat([tp,tn],dim=0)
ptf_tensor(tc0,'tc0 dim=0 cat')
tc1=torch.cat([tp,tn],dim=1)
ptf_tensor(tc1,'tc1 dim=1 cat')
