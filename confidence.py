import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from utils import accuracy, normalize_adj, sparse_mx_to_torch_sparse_tensor
from models_calibration import CaGCN
from utils import *
from util_calibration import _ECELoss
from util_calibration import *
import os
import copy
import scipy.sparse as sp
from sample import Sampler

global result
result = []
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--dataset', type=str, default="CoraFull",
                    help='dataset for training')
parser.add_argument('--stage', type=int, default=1,
                    help='times of retraining')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--epoch_for_st', type=int, default=200,
                    help='Number of epochs to calibration for self-training')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--lr_for_cal', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--l2_for_cal', type=float, default=5e-3,
                    help='Weight decay (L2 loss on parameters) for calibration.')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--labelrate', type=int, default=60)
parser.add_argument('--n_bins', type=int, default=20)
parser.add_argument('--Lambda', type=float, default=0.5,
                    help='the weight for ranking loss')
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--threshold', type=float, default=0.85)
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
criterion = torch.nn.CrossEntropyLoss().cuda()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(device)
conf_histogram = None


def train(epoch, model, optimizer, train_adj, val_adj, features, pseudo_labels, labels, idx_train, idx_val, idx_test, sign=False):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(features, train_adj)
    ece_criterion = _ECELoss(args.n_bins).cuda()
    ece = ece_criterion(output[idx_train], pseudo_labels[idx_train])

    if not sign:
        loss_train = criterion(output[idx_train], pseudo_labels[idx_train])
    else:
        loss_train = criterion(output[idx_train], pseudo_labels[idx_train]) + \
                     args.Lambda * intra_distance_loss(output[idx_train], pseudo_labels[idx_train])

    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    with torch.no_grad():
        model.eval()
        output = model(features, val_adj)
        loss_val = criterion(output[idx_val], labels[idx_val])
        loss_test = criterion(output[idx_test], labels[idx_test])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        acc_test = accuracy(output[idx_test], labels[idx_test])

    return acc_val, loss_val, acc_train, loss_test, acc_test, output

@torch.no_grad()
def val(model, adj, features, labels, idx_val):
        model.eval()
        output = model(features, adj)
        loss_val = criterion(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        return acc_val, loss_val

@torch.no_grad()
def test(model, adj, features, labels, idx_test):
        model.eval()
        output = model(features, adj)
        loss_test = criterion(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return acc_test, loss_test

def func_scaling(adj, features, pseudo_labels, labels, nclass, idx_val, idx_train,
                 idx_test, model_b_scaling, model_a_scaling, epochs, sign):
    state_dict = torch.load(model_b_scaling)
    base_model = get_models(args, features.shape[1], nclass)
    base_model.load_state_dict(state_dict)
    base_model.to(device)
    model = CaGCN(args, nclass, base_model=base_model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr_for_cal, weight_decay=args.l2_for_cal)
    model.to(device)
    best = 100
    # Train model
    for epoch in range(epochs):
        acc_val, loss_val, acc_train, loss_test, acc_test, output = train(epoch, model, optimizer, adj, adj,
                                                                 features, pseudo_labels, labels, idx_train,
                                                                 idx_val, idx_test, True)

        if epoch == (epochs-1):
            torch.save(model.state_dict(), model_a_scaling)
            best = loss_val
            continue

    print("Scaling:", f'acc_train: {acc_train:.4f}', f'loss_val: {best:.4f}', f'acc_test: {acc_test:.4f}')
    return best

@torch.no_grad()
def generate_pesudo_label(output, idx_train, pseudo_labels, idx_test, idx_val, act_labels, adj, model, features):

    train_index = torch.where(idx_train==True)
    test_index = torch.where(idx_test==True)
    val_index = torch.where(idx_val==True)
    confidence, pred_label = get_confidence(output)
    index = torch.where(confidence>args.threshold)[0]
    
    for i in range(10):
      new_adj = randomedge_sampler(adj.clone(), 0.9)
      if i == 0:
        preds = get_confidence(model(features, new_adj))[1].clone()
      elif i == 1:
        robustness = (preds == get_confidence(model(features, new_adj))[1])
      else:
        robustness = torch.logical_and(robustness, (preds == get_confidence(model(features, new_adj))[1]))
   
    for i in index:
        if i not in train_index[0]:

            #pseudo_labels[i] = pred_label[i]
            #idx_train[i] = True
           
            if robustness[i] == True:  
              pseudo_labels[i] = pred_label[i]
              idx_train[i] = True

    return idx_train, pseudo_labels 

@torch.no_grad()
def prune_graph(adj, model, features, idx_train):
    adj = adj.to_dense()
    output = model(features, adj)
    confidence, pred_label = get_confidence(output)
    train_index = torch.where(idx_train==True)[0]
    for i in train_index: 
        edges = torch.where(adj[i] != 0)[0]
        for edge in edges:
          if edge in train_index:
            if pred_label[i] != pred_label[edge]:
               adj[i,edge] = 0
    adj = adj.to_sparse()
    return adj

@torch.no_grad()
def prune_graph_temp(adj, model, features):
    adj = adj.to_dense()
    output = model(features, adj)
    confidence, pred_label = get_confidence(output)
    for i in range(adj.size()[0]): 
        edges = torch.where(adj[i] != 0)[0]
        for edge in edges:
            if pred_label[i] != pred_label[edge]:
               adj[i,edge] *= (1 - (confidence[i] * confidence[edge]))
    adj = adj.to_sparse()
    return adj

@torch.no_grad()
def randomedge_sampler(train_adj, percent):
  nnz = train_adj._nnz() 
  perm = np.random.permutation(nnz)   
  preserve_nnz = int(nnz*(1-percent)) 
  perm = perm[:preserve_nnz]
  train_adj._values()[perm] = 0 
  return train_adj 

def main(dataset):
    global criterion
    data, adj, features, labels, idx_train, idx_val, idx_test, nxg= load_data(dataset, args.labelrate)
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    for i in range(adj.size()[0]):
       if idx_train[i] == True:
         idx_test[i] = False
       else:
         idx_test[i] = True
    nclass = labels.max().item() + 1

    pseudo_labels = labels.clone()
    pseudo_labels = pseudo_labels.to(device)
    acc_test_times_list = list()
    n_time = 0
    seed = 42
    model_b_scaling = './save_model/%s-%s-%s-%d-w_o-s.pth'%(args.model, args.dataset, args.threshold, args.labelrate)
    model_a_scaling = './save_model/%s-%s-%s-%d-w-s.pth'%(args.model, args.dataset, args.threshold, args.labelrate)
    t_total = time.time()
    t_train = 0
    curr_val_acc = 0
    prev_val_acc = 0
    stage = 0
    adj = adj.coalesce()
    while stage <= 6:
        stage += 1
        print("########################### Iteration:", stage, "Labels:", labels[idx_train].size()[0], "###########################")  

        #sample_rate = 0.9
        #args.model = 'ML_GCN'
        
        if stage == 1:
           args.model = 'ML_GCN'
           sample_rate = 0.9
        else:
           args.model = 'GCN'
           sample_rate = 0.8
      
        n_time += 1
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Model and optimizer
        model = get_models(args, features.shape[1], nclass)
        optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
        model.to(device)
        
        # Train model
        best = 100
        train_start = time.time()
        for epoch in range(args.epochs):
            train_adj = randomedge_sampler(adj.clone(), sample_rate)
            acc_val, loss_val, acc_train, loss_test, acc_test, output = train(epoch, model, optimizer, train_adj, adj,
                                                                     features, pseudo_labels, labels, idx_train, idx_val, idx_test)
            if epoch == (args.epochs-1):
                torch.save(model.state_dict(), model_b_scaling)
                best = loss_val
        train_end = time.time()
        t_train += (train_end - train_start)
        
        print("Training:", f'acc_train: {acc_train:.4f}', f'loss_val: {best:.4f}', f'acc_test: {acc_test:.4f}')
        print('Training Time:', t_train)
        
        model.eval()

        new_adj = prune_graph_temp(adj.clone(), model, features)
        #new_adj = adj
 
        output = model(features, new_adj)
        idx_train, pseudo_labels = generate_pesudo_label(output, idx_train, pseudo_labels, idx_test, idx_val, labels, new_adj.clone(), model, features)
    
        # Testing
        acc_test, loss_test = test(model, adj, features, labels, idx_test)
        print("Test set results:", f'acc_test: {acc_test:.4f}', f'loss_test: {loss_test:.4f}')
        print('Time:', time.time() - t_total)
        adj = prune_graph(adj.clone(), model, features, idx_train)

if __name__ == '__main__':
    main(dataset=args.dataset)




