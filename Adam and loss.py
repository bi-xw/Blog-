import torch
import torch.nn
import torch.optim

x = torch.tensor([[1., 1., 1.], [2., 3., 1.],[3., 5., 1.], [4., 2., 1.], [5., 4., 1.]])
y = torch.tensor([-10., 12., 14., 16., 18.])
w = torch.zeros(3, requires_grad=True)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam([w,],)

for step in range(30001):
    if step:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    pred = torch.mv(x, w) # 矩阵和向量相乘
    loss = criterion(pred, y)
    if step % 1000 == 0:
        print('step = {}, loss = {:g}, W = {}, grad = {}'.format(step, loss, w.tolist(),w.grad))