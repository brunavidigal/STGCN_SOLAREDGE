# @Time     : Jan. 28, 2021 21:45
# @Author   : Bruna Rodrigues Vidigal
# @FileName : main.py
# @Version  : 1.0
# @Project  : Orion
# @IDE      : PyCharm

# LIBRARY

from scripts_aux.utils.data_utils import *
from scripts_aux.utils.math_graph import *
from trainer import model_train
from tester import model_test
# from utils.math_utils import *
# from base_model import build_model

# from sklearn.model_selection import train_test_split
from os.path import join as pjoin

# import pandas as pd
# import networkx as nx
# import numpy as np
# import time
import argparse

import tensorflow.compat.v1 as tf
import os

tf.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Config GPU to process tensorflow
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.Session(config=config)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=420)
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pred', type=int, default=6)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--epoch', type=int, default=2)
parser.add_argument('--save', type=int, default=10)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='merge')
parser.add_argument('--metric', type=str, default='Corrente_modulo')

args = parser.parse_args()
print(f'Training configs: {args}')

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
metric = args.metric
# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[1, 10, 20], [20, 10, 40]] #[[1, 32, 64], [64, 32, 128]] | [[1, 10, 20], [20, 10, 40]]

# Load adjacency matrix W
if args.graph == 'default':
    # W = weight_matrix(pjoin('./Database/Topologias_PaineisFotovoltaicos', 'CPID_LayoutLogico_InversorOut.graphml'))
    W = weight_matrix(pjoin('./Database/Topologias_PaineisFotovoltaicos', 'CPID_LayoutLogico_InversorOut.tgf'))
else:
    # load customized graph weight matrix
    W = weight_matrix(pjoin('./Database/Topologias_PaineisFotovoltaicos', args.graph))

print(W)

# Calculate graph kernel
L = scaled_laplacian(W)
# Alternative approximation method: 1st approx - first_approx(W, n).
Lk = cheb_poly_approx(L, Ks, n)
tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))

# Data Preprocessing
data_file = f'{metric}.csv'
n_train, n_val, n_test = 30, 5, 10
PeMS = data_gen(pjoin('Database/pre-processing/dados_rede', data_file), (n_train, n_val, n_test), n, n_his + n_pred)
print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')

if __name__ == '__main__':
    model_train(PeMS, blocks, args)
    model_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode)
