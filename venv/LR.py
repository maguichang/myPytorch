# -*- coding: utf-8 -*-
# @Time: 2019/9/8 15:59
# @Emali: mgc5320@163.com
# @Author: Ma Gui Chang
"""
pytorch 实现线性回归
"""
import torch
from torch import nn
# 导入pytorch的激励函数
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())
# 画图
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()
plt.ion()
plt.show()

"""
创建神经网络,
方法一: 通过定义一个Net类来创建神经网络
"""
class Net(torch.nn.Module):

    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self, x):
        # 前向传递
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
"""
方法二: 通过torch.nn.Sequential快速建立神经网络
"""
net2 = nn.Sequential(
    nn.Linear(2,10),
    nn.ReLU(),
    nn.Linear(10,2)
)

net = Net(n_feature=1, n_hidden=10, n_output=1)
# print(net)
"""
训练神经网络
"""
# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(),lr=0.2)
loss_func = torch.nn.MSELoss()

for t in range(100):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
        plt.show()