# FASTRAIN-GNN: Fast and Accurate Self-training for Graph Neural Networks
This repository contains the source code for FASTRAIN-GNN: Fast and Accurate Self-training for Graph Neural Networks. This repository builds on the implementation of [CaGCN](https://github.com/BUPT-GAMMA/CaGCN).

## Dependencies
+ python == 3.8.8
+ pytorch == 1.8.1
+ dgl -cuda11.1 == 0.6.1
+ networkx == 2.5
+ numpy == 1.20.2

Tested on RTX 2080 Ti GPU with CUDA Version 11.4


## Commands
We follow the exact hyperparameter settings described [here]((https://github.com/BUPT-GAMMA/CaGCN) for experiments on confidence-calibrated (CaGCN/ CaGAT) models. Sample commands are given below.

+ To obtain the baseline (self-trained) CaGCN model, use `python CaGCN.py --model <GCN/ML_GCN> --hidden 64 --dataset <Cora/Citeseer/Pubmed> --labelrate <labels/class> --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8`.
+ To obtain the FASTRAIN-CaGCN model, use `python fastrain.py --model GCN --hidden 64 --dataset <Cora/Citeseer/Pubmed> --labelrate <labels/class> --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8`.


+ To obtain the baseline (self-trained) CaGAT model, use `python CaGCN.py --model <GAT/ML_GAT> --hidden 8 --dataset <Cora/Citeseer/Pubmed> --labelrate <labels/class> --dropout 0.6 --lr 0.005 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8`.
+ To obtain the FASTRAIN-CaGAT model, use `python CaGCN.py --model GAT --hidden 8 --dataset <Cora/Citeseer/Pubmed> --labelrate <labels/class> --dropout 0.6 --lr 0.005 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8`.

