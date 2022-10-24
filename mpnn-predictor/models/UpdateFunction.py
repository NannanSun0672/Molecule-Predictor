# !/usr/bin/python
# -*- coding: utf-8 -*-

"""
    UpdateFunction.py: Updates the nodes using the previous state and the message.

    Usage:

"""

from __future__ import print_function

# Own modules
import utils
from .MessageFunction import MessageFunction
from .nnet import NNet

import numpy as np
import time
import os
import argparse
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable

# dtype = torch.cuda.FloatTensor
dtype = torch.FloatTensor


class UpdateFunction(nn.Module):

    # Constructor
    def __init__(self, update_def='nn', args={}):
        super(UpdateFunction, self).__init__()
        self.u_definition = ''
        self.u_function = None
        self.args = {}
        self.__set_update(update_def, args)

    # Update node hv given message mv
    def forward(self, h_v, m_v, opt={}):
        return self.u_function(h_v, m_v, opt)

    # Set update function
    def __set_update(self, update_def, args):
        self.u_definition = update_def.lower()
        #print("UpdataFunction_args",args)

        self.u_function = {
            'mpnn': self.u_mpnn
        }.get(self.u_definition, None)

        if self.u_function is None:
            print('WARNING!: Update Function has not been set correctly\n\tIncorrect definition ' + update_def)

        init_parameters = {
            'mpnn': self.init_mpnn
        }.get(self.u_definition, lambda x: (nn.ParameterList([]), nn.ModuleList([]), {}))

        self.learn_args, self.learn_modules, self.args = init_parameters(args)
        #print("UpdataFunction",self.learn_args,self.learn_modules,self.args)

    # Get the name of the used update function
    def get_definition(self):
        return self.u_definition

    # Get the update function arguments
    def get_args(self):
        return self.args


    def u_mpnn(self, h_v, m_v, opt={}):
        h_in = h_v.view(-1, h_v.size(2))
        m_in = m_v.view(-1, m_v.size(2))
        h_new = self.learn_modules[0](m_in[None, ...], h_in[None, ...])[0]  # 0 or 1???
        return torch.squeeze(h_new).view(h_v.size())

    def init_mpnn(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        args['in_m'] = params['in_m']
        args['out'] = params['out']

        # GRU
        learn_modules.append(nn.GRU(params['in_m'], params['out']))

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args

