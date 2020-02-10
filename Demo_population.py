import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn
import torch.optim

from debug import ptf_tensor

url='C:/Users/HUAWEI/Desktop/深度学习/Blog附带代码/population.csv'
df = pd.read_csv(url, index_col=0)
#print(df) #数据预览

years=torch.tensor(df.iloc[:,0],dtype=torch.float32) # 取出第0列数据作为Years
populations=torch.tensor(df.iloc[:,1],dtype=torch.float32) #取出第1列数据作为人口数据

ptf_tensor(years,'years',data_only=True) #预览数据
ptf_tensor(populations,'populations',data_only=True) #预览数据

x=years.reshape(-1,1) # reshpae(-1,1)等价于reshape(n,1),因为大多时候我们都不知道所有元素的总数
ptf_tensor(x,'years reshape',)
y=populations
'''
这是我们需要的数据形式:
    X=[ [,,] , [,,] , [,,] ] # X=m*n m是样本数量，n是特征数
    y=[,,,] # Y=m*1
    W=[,,] # W=n*1
Y=XW
'''
def get_mean_std_norm(x):
    x_mean,x_std=torch.mean(x),torch.std(x)
    x_norm=(x-x_mean)/x_std
    return x_mean,x_std,x_norm

x_mean,x_std,x_norm=get_mean_std_norm(x)
y_mean,y_std,y_norm=get_mean_std_norm(y)

fc=torch.nn.Linear(1,1) #Y=XW式子中y的数量和x的数量
criterion=torch.nn.MSELoss()
optimizer=torch.optim.Adam(fc.parameters())

weights_norm,bias_norm=fc.parameters() #每次结果生成器

for step in range(20001):
    if step:
        optimizer.zero_grad()
        loss_norm.backward()
        optimizer.step()
    
    output_norm=fc(x_norm)
    pred_norm=output_norm.squeeze()
    loss_norm=criterion(pred_norm,y_norm) # 统一size才能计算loss

    weights=y_std/x_std *weights_norm
    bias=(weights_norm*(0-x_mean)/x_std+bias_norm)*y_std+y_mean
    if step % 5000==0:
        print('step={}, weights={}, loss={}'.format(step,weights.item(),loss_norm))


plt.scatter(years, populations, s=0.1, label='actual', color='k')
plt.plot(years.tolist(), (years*weights + bias).squeeze(dim=0).tolist(), label='result', color='k') 
#注意：这里years.size=[67],result.size=[1,67],所以要对result进行squeeze

plt.xlabel('Year')
plt.ylabel('Population')
plt.legend()
plt.show()






