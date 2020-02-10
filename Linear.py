import torch
'''
#单个因变量
x=torch.tensor([[1,1,1],[2,2,1],[3,4,1],[4,2,1],[5,4,1]],dtype=torch.float32)
y = torch.tensor([-10,12,14,16,18],dtype=torch.float32)
wr,_=torch.lstsq(y, x) # _ 废弃变量承接的是QR分解结果，用于对最小二乘法进一步分析
w=wr[:3] #[0][1][2]表示三个parameters的值，后面[3][4]表示残差量，他们的平方和是MSE的值
print(w)

#多个因变量
x = torch.tensor([[1, 1, 1], [2, 3, 1],[3, 5, 1], [4, 2, 1], [5, 4, 1]],dtype=torch.float32)
y = torch.tensor([[-10, -3], [12, 14], [14, 12], [16, 16], [18, 16]],dtype=torch.float32)
wr, _ = torch.lstsq(y, x)
w = wr[:3, :]
print(w)
'''
########################################
#利用 Linear 类实现线性回归
x=torch.tensor([[1,1,1],[2,2,1],[3,4,1],[4,2,1],[5,4,1]],dtype=torch.float32)
y = torch.tensor([-10,12,14,16,18],dtype=torch.float32)

fc = torch.nn.Linear(3,1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(fc.parameters())
weights,bias=fc.parameters()

pred=fc(x)
loss=criterion(pred,y)

for step in range(30001):
    if step:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pred=fc(x)
    loss=criterion(pred,y)
    if step % 1000 == 0:
        print('step = {}, loss = {:g}, weights = {}, bias={}'.format(step, loss, weights[0, :].tolist(), bias.item()))
