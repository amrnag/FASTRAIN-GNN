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
parser.add_argument('--sampling_rate', type=float, default=1.0, help='Sampling rate for training.')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
criterion = torch.nn.CrossEntropyLoss().cuda()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(device)
conf_histogram = None


def train(epoch, model, optimizer, train_adj, val_adj, features, pseudo_labels, labels, idx_train, idx_test, sign=False):
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
        loss_test = criterion(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
    '''print(f'epoch: {epoch}',
          f'loss_train: {loss_train.item():.4f}',
          f'acc_train: {acc_train:.4f}',
          f'loss_test: {loss_test.item():4f}',
          f'acc_test: {acc_test:.4f}',
          f'time: {time.time() - t:.4f}s')'''

    return acc_train, loss_test, acc_test, output

@torch.no_grad()
def test(adj, features, labels, idx_test, nclass, idx_train,
         model_b_scaling, model_a_scaling):

    nfeat = features.shape[1]
    state_dict = torch.load(model_a_scaling)
    model = CaGCN(args, nclass, base_model = get_models(args, nfeat, nclass))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    output = model(features, adj)
    loss_test = criterion(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    confidence_t = torch.softmax(output[idx_test], dim=1).cpu()
    confidence_t = torch.max(confidence_t, 1)[0]
    pred_label = torch.max(output[idx_test], 1)[1]
    correct_index_t = labels[idx_test] == pred_label
    correct_index_t = correct_index_t.cpu()
    # Calculate ECE after temperature scaling in test set
    ece_criterion = _ECELoss(args.n_bins).cuda()
    ece = ece_criterion(output[idx_test], labels[idx_test]).item()
    brier_score = brier_score_criterion(output[idx_test], labels[idx_test], nclass).item()
    #plot_acc_calibration(idx_test, output, labels, args.n_bins, 'Ours - %s - %d - %s'%(args.dataset, args.labelrate, args.model))
    print(f"Test set results with CaGCN:",
          f"loss= {loss_test.item():.4f}",
          f"accuracy= {acc_test:.4f}",
          f"ece = {ece:.4f}",
          f"brier_score {brier_score:.4f}")  
    #plot_histograms(confidence_t[correct_index_t], confidence_t[np.invert(correct_index_t)], 'Ours - %s - %d - %s'%(args.dataset, args.labelrate, args.model), ['Correct', "InCorrect"])   
    return acc_test, loss_test, ece, brier_score


def func_scaling(adj, features, pseudo_labels, labels, nclass, idx_train,
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
        acc_train, loss_test, acc_test, output = train(epoch, model, optimizer, adj, adj,
                                                                 features, pseudo_labels, labels, idx_train,
                                                                 idx_test, True)

        if epoch == (epochs-1):
            torch.save(model.state_dict(), model_a_scaling)
            continue

    print("Scaling:", f'acc_train: {acc_train:.4f}', f'acc_test: {acc_test:.4f}')
    return best

@torch.no_grad()
def generate_pseudo_label(output, idx_train, pseudo_labels, idx_test, act_labels, adj, model, features):

    train_index = torch.where(idx_train==True)
    test_index = torch.where(idx_test==True)
    confidence, pred_label = get_confidence(output)
    index = torch.where(confidence>args.threshold)[0]
   
    for i in index:
      if i not in train_index[0]:
        pseudo_labels[i] = pred_label[i]
        idx_train[i] = True

    return idx_train, pseudo_labels 

@torch.no_grad()
def pgp(adj, model, features, idx_train):
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
    idx_test = idx_test.to(device)

    # test on all unlabeled nodes
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
    adj = adj.coalesce()

    for stage in range(args.stage):
        print("########################### Stage:", stage + 1, "Training set size:", labels[idx_train].size()[0], "###########################")    
  
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
        for epoch in range(args.epochs):
            train_adj = randomedge_sampler(adj.clone(), 0.9)
            acc_train, loss_test, acc_test, output = train(epoch, model, optimizer, train_adj, adj,
                                                                     features, pseudo_labels, labels, idx_train, idx_test)
            if epoch == (args.epochs-1):
                torch.save(model.state_dict(), model_b_scaling)
        
        print("Training:", f'acc_train: {acc_train:.4f}', f'acc_test: {acc_test:.4f}')

        # Scaling using training set        
        curr_cal_loss = func_scaling(adj, features, pseudo_labels, labels, nclass, 
                         idx_train, idx_test, model_b_scaling, model_a_scaling, args.epoch_for_st, True)
   
        ######  self-training to find pesudo label  ########
        state_dict = torch.load(model_a_scaling)
        model = CaGCN(args, nclass, base_model=get_models(args, features.shape[1], nclass))
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
 
        output = model(features, adj)

        # Generate pseudolabels 
        idx_train, pseudo_labels = generate_pseudo_label(output, idx_train, pseudo_labels, idx_test, labels, adj.clone(), model, features)

        # Testing
        acc_test, loss_test, ece, brier_score = test(adj, features, labels, idx_test, nclass, idx_train,
                        model_b_scaling, model_a_scaling)
        acc_test_times_list.append(acc_test)


if __name__ == '__main__':
    main(dataset=args.dataset)




