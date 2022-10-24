"""

"""
import torch.nn as nn
import torch.nn.functional as F

class NNet(nn.Module):
    def __init__(self,n_in,n_out,hlayer=(128,256,128)):
        super(NNet,self).__init__()
        self.n_hlayers = len(hlayer)
        #print("n_in",n_in)
        self.fcs = nn.ModuleList([nn.Linear(n_in,hlayer[i]) if i==0 else
                                  nn.Linear(hlayer[i-1],n_out) if i ==self.n_hlayers else
                                  nn.Linear(hlayer[i-1],hlayer[i]) for i in range(self.n_hlayers+1)])
    def forward(self,x):
        ####x = e_vw
        #print("x",x.shape)
        #print("x.size",x.size()[1:])
        x = x.contiguous().view(-1,self.num_flat_features(x)) #batch*flatten size
        #print("modified_x",x.shape)
        for i in range(self.n_hlayers):
            x = F.relu(self.fcs[i](x))
        x = self.fcs[-1](x)
        return x
    def num_flat_features(self,x):
        size = x.size()[1:]# all dimeansions except the batch dimension
        num_features = 1
        for s in size:
            num_features *=s
        return num_features
