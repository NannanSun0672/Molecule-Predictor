#from __future__ import print_function
# !/usr/bin/python
# -*- coding: utf-8 -*-

"""
    MessageFunction.py: Propagates a message depending on two nodes and their common edge.

    Usage:

"""

from __future__ import print_function

# Own modules
import utils
from .MessageFunction import MessageFunction
from .UpdateFunction import UpdateFunction
from .nnet import NNet

import time
import torch
import torch.nn as nn
import os
import argparse
import numpy as np

from torch.autograd.variable import Variable

# dtype = torch.cuda.FloatTensor
dtype = torch.FloatTensor


class ReadoutFunction(nn.Module):

    # Constructor
    def __init__(self, readout_def='nn', args={}):
        super(ReadoutFunction, self).__init__()
        self.r_definition = ''
        self.r_function = None
        self.args = {}
        self.__set_readout(readout_def, args)

    # Readout graph given node values at las layer
    def forward(self, h_v):
        return self.r_function(h_v)

    # Set a readout function
    def __set_readout(self, readout_def, args):
        #print("Readout",args)
        self.r_definition = readout_def.lower()

        self.r_function = {
            'mpnn': self.r_mpnn
        }.get(self.r_definition, None)

        if self.r_function is None:
            print('WARNING!: Readout Function has not been set correctly\n\tIncorrect definition ' + readout_def)
            quit()

        init_parameters = {
            'mpnn': self.init_mpnn
        }.get(self.r_definition, lambda x: (nn.ParameterList([]), nn.ModuleList([]), {}))

        self.learn_args, self.learn_modules, self.args = init_parameters(args)
        #print("Readout",self.learn_args,self.learn_modules,self.args)

    # Get the name of the used readout function
    def get_definition(self):
        return self.r_definition

    def r_mpnn(self, h):

        aux = Variable(torch.Tensor(h[0].size(0), self.args['out']).type_as(h[0].data).zero_())
        # For each graph
        for i in range(h[0].size(0)):
            nn_res = nn.Sigmoid()(self.learn_modules[0](torch.cat([h[0][i, :, :], h[-1][i, :, :]], 1))) * \
                     self.learn_modules[1](h[-1][i, :, :])
            #import IPython
            #IPython.embed()

            # Delete virtual nodes
            nn_res = (torch.sum(h[0][i, :, :], 1)[..., None].expand_as(nn_res) > 0).type_as(nn_res) * nn_res
            #import IPython
            #IPython.embed()
            aux[i, :] = torch.sum(nn_res, 0)

            #import IPython
            #IPython.embed()

        return aux

    def init_mpnn(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        # i
        learn_modules.append(NNet(n_in=2 * params['in'], n_out=params['target']))

        # j
        learn_modules.append(NNet(n_in=params['in'], n_out=params['target']))

        args['out'] = params['target']

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args
