import torch.nn as nn
import torch.nn.functional as F

nclasses = 20

class Net(nn.Module):

    def __init__(self,num_shape):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.fc1 = nn.Linear(num_shape, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 20)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

