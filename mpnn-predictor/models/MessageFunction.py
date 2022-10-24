"""
Create on sep 10,2021
"""
from __future__ import print_function
import utils
from .nnet import NNet
import numpy as np
import os
import argparse
import time
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
class MessageFunction(nn.Module):
    def __init__(self,message_def="mpnn",args={}):
        super(MessageFunction,self).__init__()
        #print("args",args)
        self.m_definition = ""
        self.m_function = None
        self.args = {}
        self.__set_message(message_def,args)

    ### Message from h_v to h_w through e_vw
    def forward(self,h_v,h_w,e_vw,args = None):
        return self.m_function(h_v,h_w,e_vw,args)
    ### Set a message function
    def __set_message(self,message_def,args = {}):
        self.m_definition = message_def.lower()
        self.m_function = {
            "mpnn":self.m_mpnn
        }.get(self.m_definition,None)
        if self.m_function is None:
            print("WARNING!:Message Function has not been set correctly\n\t Incorrect definition")
            quit()
        init_parameters = {"mpnn":self.init_mpnn
                           }.get(self.m_definition,lambda x:(nn.ParameterList([]),nn.ModuleList([]),{}))
        #import IPython
        #IPython.embed()
        self.learn_args,self.learn_modules,self.args = init_parameters(args)
        #print("--init_parameters",self.learn_args,self.learn_modules,self.args)
        self.m_size = {
            "mpnn":self.out_mpnn
        }.get(self.m_definition,None)
    def out_mpnn(self,size_h,size_e,args):
        return self.args["out"]
    def m_mpnn(self,h_v,h_w,e_vw,opt = {}):
        edge_output = self.learn_modules[0](e_vw)
        #import IPython
        #IPython.embed()
        edge_output = edge_output.view(-1,self.args["out"],self.args["in"])
        h_w_rows = h_w[...,None].expand(h_w.size(0),h_w.size(1),h_v.size(1)).contiguous()
        #import IPython
        #IPython.embed()
        h_w_rows = h_w_rows.view(-1,self.args["in"])
        #import IPython
        #IPython.embed()
        h_multiply = torch.bmm(edge_output,torch.unsqueeze(h_w_rows,2))
        #import IPython
        #IPython.embed()
        m_new = torch.squeeze(h_multiply)
        return m_new

    def init_mpnn(self,params):
        learn_args = []
        learn_modules = []
        args = {}
        args["in"] = params["in"]
        args["out"] = params["out"]
        ###Define a parameter matrix A for each edge label
        learn_modules.append(NNet(n_in = params["edge_feat"],n_out=(params["in"]*params["out"])))
        #print("init_mpnn",learn_args,learn_modules,args)
        return nn.ParameterList(learn_args),nn.ModuleList(learn_modules),args

