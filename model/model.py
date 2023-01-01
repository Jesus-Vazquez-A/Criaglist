import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()

        self.fc1 = nn.Linear(in_features = 6646,out_features = 128)
        self.fc2 = nn.Linear(in_features = 128,out_features = 64)
        self.fc3 = nn.Linear(in_features= 64,out_features=32)
        self.output = nn.Linear(in_features = 32,out_features = 1)
    
    def forward(self,x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.dropout(x,0.3)
    
        
        return self.output(x)