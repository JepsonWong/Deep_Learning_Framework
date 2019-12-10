# https://ptorch.com/news/172.html
# http://www.360doc.com/content/19/0726/16/32196507_851161514.shtml

import torch as t
import torch.nn as nn
import torch.nn.functional as F

def forward_hook(module, input, output):
    print(module)
    print(type(input), type(output))
    for val in input:
        print("input val:",val.shape)
    for out_val in output:
        print("output val:", out_val.shape)

def backward_hook(module, input, output):
    print(module)
    print(type(input), type(output))
    for val in input:
        print("input val:", val.shape)
    for out_val in output:
        print("output val:", out_val.shape)
 
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
   
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def run_net():
    net = LeNet()
    img = t.autograd.Variable((t.arange(32*32*1).view(1,1,32,32))).type('torch.FloatTensor')
    print(net(img))

def run_net_forward_hook():
    net = LeNet()
    img = t.autograd.Variable((t.arange(32*32*1).view(1,1,32,32))).type('torch.FloatTensor')
    handle = net.conv1.register_forward_hook(forward_hook)
    out = net(img)
    print(out)
    handle.remove()
    print("-----after remove handle-----")
    out = net(img)
    print(out)

'''
Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
(<type 'tuple'>, <class 'torch.Tensor'>)
('input val:', (1, 1, 32, 32))
('output val:', (6, 28, 28))
tensor([[-15.3069,  12.1838,  32.2389, -25.1271, -17.6846, -45.1112,  82.4961,
          34.3775,   3.4116, -15.0286]], grad_fn=<ThAddmmBackward>)
-----after remove handle-----
tensor([[-15.3069,  12.1838,  32.2389, -25.1271, -17.6846, -45.1112,  82.4961,
          34.3775,   3.4116, -15.0286]], grad_fn=<ThAddmmBackward>)
'''

def run_net_backward_hook():
    net = LeNet()
    img = t.autograd.Variable((t.arange(32*32*1).view(1,1,32,32))).type('torch.FloatTensor')
    #handle = net.fc3.register_backward_hook(backward_hook)
    handle = net.register_backward_hook(backward_hook)
    output = net(img)
    print(output)
    output = output.sum()
    output.backward()
    handle.remove()
    print("-----after remove handle-----")
    output = net(img)
    print(output) 
    output = output.sum()
    output.backward()

'''
tensor([[-0.6023, -4.7758, -3.8702, -0.6611,  3.5404,  0.1636,  2.5151,  1.2280,
          4.2266, -0.1469]], grad_fn=<ThAddmmBackward>)
LeNet(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
(<type 'tuple'>, <type 'tuple'>)
('input val:', (1, 10))
('input val:', (1, 84))
('input val:', (84, 10))
('output val:', (1, 10))
-----after remove handle-----
tensor([[-0.6023, -4.7758, -3.8702, -0.6611,  3.5404,  0.1636,  2.5151,  1.2280,
          4.2266, -0.1469]], grad_fn=<ThAddmmBackward>)
'''

#run_net_forward_hook()
run_net_backward_hook()

