import torch
import torch.nn as nn
import torchvision
import numpy
import torch.nn.functional as F
device = torch.device("cuda:0")
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.emb = nn.Embedding(320, 128, padding_idx=0)
    self.conv1 = nn.Conv1d(128, 256, 2, padding=1)
    self.conv2 = nn.Conv1d(128, 256, 2, padding=1)
    self.conv3 = nn.Conv1d(128, 128, 2, padding=1)
    self.li1 = nn.Linear(896,256)
    self.li2 = nn.Linear(256,64)
    self.li3 = nn.Linear(64,1)
    self.relu = nn.ReLU(True)
  def forward(self,x):
    x = self.emb(x)
    x = x.transpose(2, 1)
    x = self.conv1(x)
    x = F.glu(x,dim=1)
    x = self.conv2(x)
    x = F.glu(x,dim=1)
    x = self.conv3(x)
    x = self.relu(x)
    x = x.view(x.size(0),-1)
    x = self.li1(x)
    x = self.relu(x)
    x = self.li2(x)
    x = self.relu(x)
    x = self.li3(x)
    x = self.relu(x)
    return x
net = Net().to(device)
pri = [123,234,112,233]
pri = torch.LongTensor(pri).view(1,-1).to(device)
result = net(pri)
print(result)
