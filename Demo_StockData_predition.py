import pandas as pd

import torch
import torch.nn
import torch.optim

from debug import ptf_tensor

url = 'C:/Users/HUAWEI/Desktop/深度学习/Blog附带代码/FB.csv'
df = pd.read_csv(url, index_col=0)

#数据集的处理
'''
因为数据是日期新的占index靠前
'''
train_start, train_end=sum(df.index>='2017'),sum(df.index>='2013') 
test_start, test_end=sum(df.index>='2018'),sum(df.index>='2017')

n_total_train = train_end -train_start
n_total_test = test_end -test_start

s_mean=df[train_start:train_end].mean()
s_std=df[train_start:train_end].std()

n_features=5 # 五个特征量

df_feature=((df-s_mean)/s_std).iloc[:,:n_features] #选取col from 0-4 也就是Open，High，Low，Close，Volume
s_labels=(df['Volume']<df['Volume'].shift(1)).astype(int)
##.shift(1)把数据下移一位
#用法参见：https://www.zhihu.com/question/264963268

#label建立的标准：假如今天次日的成交量大于当日的成交量，标签=1，反之=0

#print(df_feature)
#print(s_labels)

DEVICE='cuda:0'

fc=torch.nn.Linear(n_features,1)
fc.to(DEVICE)

weights,bias=fc.parameters()
criterion=torch.nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(fc.parameters())

x=torch.tensor(df_feature.values,dtype=torch.float32) # size: [m,5]
ptf_tensor(x,'x')
y=torch.tensor(s_labels.values.reshape(-1,1),dtype=torch.float32) # size [m,1]
ptf_tensor(y,'y')

x=x.to(DEVICE)
y=y.to(DEVICE)

n_steps=20001

for step in range(n_steps):
    if step:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    pred=fc(x)
    loss=criterion(pred[train_start:train_end],y[train_start:train_end])

    if step % 500==0:
        #print('#{}, 损失 = {:g}'.format(step, loss))        
        output = (pred > 0)
        correct = (output == y.bool())
        n_correct_train = correct[train_start:train_end].sum().item()
        n_correct_test = correct[test_start:test_end].sum().item()
        accuracy_train = n_correct_train / n_total_train
        accuracy_test = n_correct_test / n_total_test
        print('训练集准确率 = {}, 测试集准确率 = {}'.format(accuracy_train, accuracy_test))

