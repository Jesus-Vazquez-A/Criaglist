import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()

        self.fc1 = nn.Linear(in_features = 425,out_features = 64)
        self.fc2 = nn.Linear(in_features = 64,out_features = 32)
        self.fc3 = nn.Linear(in_features= 32,out_features=16)
        self.output = nn.Linear(in_features = 16,out_features = 1)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.output(x)
