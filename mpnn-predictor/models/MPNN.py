"""
Create on Sep 10,2021
"""
from .MessageFunction import MessageFunction
from .UpdateFunction import UpdateFunction
from .ReadoutFunction import ReadoutFunction
import torch
import torch.nn as nn
from torch.autograd import Variable
class MPNN(nn.Module):
    """
            MPNN as proposed by Gilmer et al..

            This class implements the whole Gilmer et al. model following the functions Message, Update and Readout.

            Parameters
            ----------
            in_n : int list
                Sizes for the node and edge features.
            hidden_state_size : int
                Size of the hidden states (the input will be padded with 0's to this size).
            message_size : int
                Message function output vector size.
            n_layers : int
                Number of iterations Message+Update (weight tying).
            l_target : int
                Size of the output.
            type : str (Optional)
                Classification | [Regression (default)]. If classification, LogSoftmax layer is applied to the output vector.
        """
    def __init__(self,in_n,hidden_state_size,mesage_size,n_layer,l_target,type = "Regression"):
        super(MPNN,self).__init__()
        self.message_model = nn.ModuleList([MessageFunction("mpnn",args = {"edge_feat":in_n[1],"in":hidden_state_size,"out":mesage_size})])
        self.update_model = nn.ModuleList([UpdateFunction("mpnn",args = {"in_m":mesage_size,"out":hidden_state_size})])
        self.Readout_model = ReadoutFunction("mpnn",args = {"in":hidden_state_size,"target":l_target})
        self.type = type
        self.args = {}
        self.args["out"] = hidden_state_size ####
        self.n_layer = n_layer
    def forward(self,g,h_in,e):
        #print("g.shape",g.shape)
        #print("h_in_shape",h_in.shape)
        #print("e_shape",e.shape)
        h = []
        ###Paddong to same large dimension d
        h_t = torch.cat([h_in,Variable(torch.zeros(h_in.size(0),h_in.size(1),self.args["out"]-h_in.size(2)).type_as(h_in.data))],2)
        #print("h_t_shape",h_t.shape)
        h.append(h_t.clone())
        ###layer
        for t in range(0,self.n_layer):
            #print("t",t)
            e_aux = e.view(-1,e.size(3))
            #print("e_aux",e_aux.shape)
            h_aux = h[t].view(-1,h[t].size(2))
            #print("h_aux",h_aux.shape)
            m = self.message_model[0].forward(h[t],h_aux,e_aux)
            #import IPython
            #IPython.embed()
            m = m.view(h[0].size(0),h[0].size(1),-1,m.size(1))
            #import IPython
            #IPython.embed()
            ###Nodes without edge set message to 0
            m = torch.unsqueeze(g,3).expand_as(m) *m
            m = torch.squeeze(torch.sum(m,1))
            #import IPython
            #IPython.embed()
            ####update nodes
            h_t = self.update_model[0].forward(h[t],m)
            #import IPython
            #IPython.embed()
            #Delete virtual nodes
            h_t = (torch.sum(h_in,2)[...,None].expand_as(h_t)>0).type_as(h_t)*h_t
            #import IPython
            #IPython.embed()
            h.append(h_t)
        ###Reaout
        res = self.Readout_model.forward(h)
        #import IPython
        #IPython.embed()
        if self.type =="classification":
            res = nn.LogSoftnax()(res)
        return res








