import torchvision.datasets
import torchvision.transforms
import torch.utils.data
import matplotlib.pyplot as plt
import torch.nn
import torch.optim

from debug import ptf_tensor

# Hyperparameters超参数
BATCH_SIZE=100
NUM_EPOCHS=5
DEVICE='cuda:0'

########################## 训练集的准备 ##############################################

train_dataset=torchvision.datasets.MNIST(root='D:/DataTmp/mnist',train=True,transform=torchvision.transforms.ToTensor(),download=True)
#root:下载数据存放到哪里，train:下载训练集还是测试集，transfrom:数据转化的形式

test_dataset=torchvision.datasets.MNIST(root='D:/DataTmp/mnist',train=False, transform=torchvision.transforms.ToTensor(),download=True)

#由于数据集里面有上万条数据，我们需要分批从数据集读取数据
train_dataloader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE)
print('The len of train dataset={}'.format(len(train_dataset)))

test_dataloader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE)
print('The len of test dataset={}'.format(len(test_dataset)))

for images,labels in train_dataloader:
    print('The images size is {}',format(images.size())) 
    print('The labels size is {}'.format(labels.size())) 
    break

#plt.imshow(images[0,0],cmap=['gray'])
#plt.title('label = {}'.format(labels[0]))


fc=torch.nn.Linear(28*28,10) #只使用一层线性分类器
fc.to(DEVICE)#如果用CPU去掉


criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(fc.parameters())

for epoch in range(NUM_EPOCHS):
    for idx, (images,labels) in enumerate(train_dataloader):
        x =images.reshape(-1,28*28)

        x=x.to(DEVICE)# 如果用CPU去掉
        labels=labels.to(DEVICE)# 如果用CPU去掉

        optimizer.zero_grad()
        preds=fc(x)
        loss=criterion(preds,labels)
        loss.backward()
        optimizer.step()

        if idx % 100 ==0:
            print('epoch={}:idx={},loss={:g}'.format(epoch,idx,loss))


correct=0
total=0

for idx,(images,labels) in enumerate(test_dataloader):
    x =images.reshape(-1,28*28)
    x=x.to(DEVICE)
    labels=labels.to(DEVICE)

    preds=fc(x)
    predicted=torch.argmax(preds,dim=1) #在dim=1中选取max值的索引
    if idx ==0:
        print('x size:{}'.format(x.size()))
        print('preds size:{}'.format(preds.size()))
        print('predicted size:{}'.format(predicted.size()))

    total+=labels.size(0)
    correct+=(predicted == labels).sum().item()
    #print('##########################\nidx:{}\npreds:{}\nactual:{}\n##########################\n'.format(idx,predicted,labels))

accuracy=correct/total
print('{:1%}'.format(accuracy))

#torch.save(fc.state_dict(), 'D:/DataTmp/mnist/tst.pth')
#fc=torch.nn.Linear(28*28,10) #只使用一层线性分类器
#fc.to(DEVICE)#如果用CPU去掉
#fc.load_state_dict(torch.load('D:/DataTmp/mnist/tst.pth'))