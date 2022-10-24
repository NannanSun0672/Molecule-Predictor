"""
Create on Sep 3,2021

@Function: Train for Predictor
"""
import os
import sys
import torch
import torch.optim  as opt
import torch.nn as nn
from torch.autograd import Variable
import pandas  as pd
import argparse
import utils
import numpy as np
from utils import Data_process
from models.MPNN import MPNN
from utils.LogMetric import AverageMeter,Logger
import time
def restricted_float(x,inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5,1e-2]"%(x,))
    return x
Parser = argparse.ArgumentParser(description="Message passing neural network")
Parser.add_argument("--datasetPath",default="./data/3CL_activity.csv",help="dataset path")
Parser.add_argument("--logPath",default="./log",help="log path")
Parser.add_argument("--resume",default="./checkpoint",help="path to latest checkpoint")
Parser.add_argument("--no-cuda",action="store_true",default=False,help="Enables CUDA training")
Parser.add_argument("--batch_size",type=int,default=16,metavar="N",help="Input batch size for training")
Parser.add_argument("--epochs",type= int,default=100,metavar="N",help="Mumber of epochs to train")
Parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 1e-2]), default=1e-3, metavar='LR',
                    help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
Parser.add_argument('--lr-decay', type=lambda x: restricted_float(x, [.01, 1]), default=0.6, metavar='LR-DECAY',
                    help='Learning rate decay factor [.01, 1] (default: 0.6)')
Parser.add_argument('--schedule', type=list, default=[0.1, 0.9], metavar='S',
                    help='Percentage of epochs to start the learning rate decay [0, 1] (default: [0.1, 0.9])')
Parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
# i/o
Parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='How many batches to wait before logging training status')
# Accelerating
Parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')

# Model modification
Parser.add_argument('--model', type=str,help='MPNN model name MPNN',
                        default='MPNN')
args = Parser.parse_args()
#logger = Logger(args.logPath)
def main(args):
    best_er1 = 0
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("Prepare Data--------.")
    datasets = utils.Data_process.reader_data(args.datasetPath)
    #print(datasets)
    keys = list(datasets.keys())
    idx = np.random.permutation(len(keys))
    idx = idx.tolist()
    valid_ids = [keys[i]for i in idx[0:10]]
    test_ids = [keys[i] for i in idx[10:20]]
    train_ids = [keys[i] for i in idx[20:]]
    data_train = utils.dataset(datasets,train_ids,edge_transform = Data_process.edges_trans,e_representation = "chem_graph")
    data_valid =  utils.dataset(datasets,valid_ids,edge_transform = Data_process.edges_trans,e_representation = "chem_graph")
    data_test = utils.dataset(datasets,test_ids,edge_transform = Data_process.edges_trans,e_representation = "chem_graph")
    #import IPython
    #IPython.embed()
    #print(test_data)
    print("select one graph of data_train")
    g_tuple, label = data_train[0]
    g,h_t,e = g_tuple
    ####Data Loader
    train_loader = torch.utils.data.DataLoader(data_train,batch_size = args.batch_size,
                                               shuffle = True,collate_fn = Data_process.collate_g,
                                               num_workers = args.prefetch,pin_memory = True)
    valid_loader = torch.utils.data.DataLoader(data_valid,batch_size = args.batch_size,
                                               shuffle = True,collate_fn = Data_process.collate_g,
                                               num_workers = args.prefetch,pin_memory = True)
    test_loader = torch.utils.data.DataLoader(data_test,batch_size = args.batch_size,
                                              shuffle = True,collate_fn = Data_process.collate_g,
                                              num_workers = args.prefetch,pin_memory = True)
    #import IPython
    #IPython.embed()
    ####Create model
    print("Creating model--------")
    in_n = [len(h_t[0]),len(list(e.values())[0])]###node_num, dim for each edge
    print("in_n",in_n)
    hidden_state_size = 90
    message_size = 90
    n_layes = 3
    l_target = len(label)
    type = "Regression"
    if args.model == "MPNN":
        model = MPNN(in_n,hidden_state_size,message_size,n_layes,l_target,type = type)
    else:
        raise EOFError
    optimizer = opt.Adam(model.parameters(),lr = args.lr)
    criterion = nn.MSELoss()
    evaluation = lambda output,target:torch.mean(torch.abs(output-target) / torch.abs(target))
    print("logger")
    logger = Logger(args.logPath)
    lr_step = (args.lr - args.lr*args.lr_decay)/(args.epochs*args.schedule[1] - args.epochs*args.schedule[0])

    ###get the best checkpoint if available without training

    if args.resume:
        checkpoint_dir = args.resume
        best_model_file = os.path.join(checkpoint_dir,"model_best.pth")
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        if os.path.isfile(best_model_file):
            print("---loading best model '{}'".format(best_model_file))
            checkpoint = torch.load(best_model_file)
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_er1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("----loaded best model '{}' (epoch {})".format(best_model_file,checkpoint["epoch"]))
        else:
            print("----no best model found at '{}'".format(best_model_file))
    print("-----Check cuda")
    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    #print("logger",logger)
    for epoch in range(0,args.epochs):
        if epoch > args.epochs*args.schedule[0] and epoch < args.epochs*args.schedule[1]:
            args.lr -= lr_step
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
        print("epoch learning rate",epoch,args.lr)
        ####Training
        train(train_loader,model,criterion,optimizer,epoch,evaluation,logger)
        er1 = validate(valid_loader, model, criterion, evaluation, logger)

        is_best = er1 > best_er1
        best_er1 = min(er1, best_er1)
        Data_process.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_er1': best_er1,
                               'optimizer': optimizer.state_dict(), }, is_best=is_best, directory=args.resume)

        # Logger step
        logger.log_lr('learning_rate', args.lr).step()
    if args.resume:
        checkpoint_dir = args.resume
        best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if os.path.isfile(best_model_file):
            print("=> loading best model '{}'".format(best_model_file))
            checkpoint = torch.load(best_model_file)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_er1']
            model.load_state_dict(checkpoint['state_dict'])
            if args.cuda:
                model.cuda()
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded best model '{}' (epoch {})".format(best_model_file, checkpoint['epoch']))
        else:
            print("=> no best model found at '{}'".format(best_model_file))

    # For testing
    validate(test_loader, model, criterion, evaluation)
def train(train_loader,model,criterion,optimizer,epoch,evaluation,logger):
    #switch to train mode
    losses = AverageMeter()
    data_time = AverageMeter()
    error_ratio = AverageMeter()
    batch_time = AverageMeter()
    model.train()
    start = time.time()
    for i, (g,h,e,target) in enumerate(train_loader):
        #print("i",i,g,h,e,target)
        #import IPython
        #IPython.embed()
        if args.cuda:
            g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
        g,h,e,target = Variable(g),Variable(h),Variable(e),Variable(target)
        #Measure data loading time
        data_time.update(time.time()-start)
        optimizer.zero_grad()
        ###Compute output
        output = model(g,h,e)
        #import IPython
        #IPython.embed()
        train_loss = criterion(output,target)
        losses.update(train_loss.item(),g.size(0))
        error_ratio.update(evaluation(output,target).item(),g.size(0))
        #import IPython 
        #IPython.embed()
        ####compute gradient and do SGD step
        train_loss.backward()
        optimizer.step()
        batch_time.update(time.time()-start)
        end  = time.time()
        if i % args.log_interval ==0 and i >0:
            print("Epoch:[{0}][{1}/{2}]\t"
                  "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                  "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                  "Error Ratio {err.val:.4f} ({err.avg:.4f})".format(epoch,i,len(train_loader),batch_time= batch_time,data_time= data_time,
                                                                      loss = losses,err = error_ratio))
    logger.log_value("train_epoch_loss",losses.avg)
    logger.log_value("train_epoch_error_ratio",error_ratio.avg)
    print("Epoch:[{0}] Avg Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}"
          .format(epoch,err=error_ratio,loss = losses,b_time = batch_time))
def validate(val_loader,model,criterion,evaluation,logger = None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error_ratio = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (g, h, e, target) in enumerate(val_loader):

        # Prepare input data
        if args.cuda:
            g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
        g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)

        # Compute output
        output = model(g, h, e)

        # Logs
        losses.update(criterion(output, target).item(), g.size(0))
        error_ratio.update(evaluation(output, target).item(), g.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0 and i > 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error Ratio {err.val:.4f} ({err.avg:.4f})'
                  .format(i, len(val_loader), batch_time=batch_time,
                          loss=losses, err=error_ratio))

    print(' * Average Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}'
          .format(err=error_ratio, loss=losses))

    if logger is not None:
        logger.log_value('test_epoch_loss', losses.avg)
        logger.log_value('test_epoch_error_ratio', error_ratio.avg)

    return error_ratio.avg
if __name__ == "__main__":
    main(args)




















