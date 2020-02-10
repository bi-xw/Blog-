from math import pi
import torch
import torch.optim
from debug import ptf_tensor, draw3D_func


# 示计算定点梯度
x=torch.tensor([pi/3,pi/6],requires_grad=True)
f = - ((x.cos() ** 2).sum()) ** 2
ptf_tensor(f)
f.backward() # 计算梯度
ptf_tensor(x.grad) # 点在【pi/3,pi/6】的梯度值


optimizer=torch.optim.SGD([x,],lr=0.1,momentum=0)

for step in range(20):
    if step:
        optimizer.zero_grad() #清空优化器上一次迭代产生的数据
        f.backward() # 计算梯度
        optimizer.step() #更新x[0],x[1]的值
    f = - ((x.cos() ** 2).sum()) ** 2
    print ('step {}: x = {}, f(x) = {}, grad = {}'.format(step, x.tolist(), f, x.grad))




# 示例： Himmelblau函数的优化
# 这是一个四个坑的函数，从不同的点出发可以收敛到不同位置的地方

def himmelblau(x):
    return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2

draw3D_func(himmelblau) # 绘制函数图像
x=torch.tensor([0.,0.],requires_grad=True)
optimizer=torch.optim.Adam([x,])

for step in range(20001):
    if step:
        optimizer.zero_grad()
        f.backward() # f.backward() 找局部最小值，我们只需要利用相反（-f）.backward()就能求局部最大值
        optimizer.step()
    f=himmelblau(x)
    if step % 1000 ==0:
      print ('step {}: x = {}, f(x) = {}, grad = {}'.format(step, x.tolist(), f, x.grad))  
