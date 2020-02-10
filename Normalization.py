import torch
import torch.nn
import torch.optim

x = torch.tensor([[1000000, 0.0001], [2000000, 0.0003],
        [3000000, 0.0005], [4000000, 0.0002], [5000000, 0.0004]])
y = torch.tensor([-1000., 1200., 1400., 1600., 1800.]).reshape(-1, 1)

x_mean, x_std = torch.mean(x,dim=0), torch.std(x,dim=0)
x_norm = (x-x_mean)/x_std

y_mean, y_std = torch.mean(y,dim=0), torch.std(y,dim=0)
y_norm=(y-y_mean)/y_std

fc=torch.nn.Linear(2,1)
criterion=torch.nn.MSELoss()
optimizer=torch.optim.Adam(fc.parameters())


for step in range(10001):
    if step:
        optimizer.zero_grad()
        loss_norm.backward()
        optimizer.step()
    
    pred_norm=fc(x_norm)
    loss_norm=criterion(pred_norm,y_norm)
    pred=pred_norm * y_std + y_mean
    loss = criterion (pred,y)
    if step % 1000 ==0:
        print('step = {}, loss = {:g}'.format(step, loss))

