# @Time     : Nov. 16, 2021 15:10
# @Author   : Bruna Rodrigues Vidigal
# @FileName : refactor.py
# @Version  : 1.0
# @Project  : Orion
# @IDE      : PyCharm

# ------------------------ LIBRARY ------------------------------------------
# ---------------------------------------------------------------------------

from scripts_aux.utils.math_graph import *
from scripts_aux.utils.data_utils import *
# from scripts_aux.models.layers import *
from tester import *
from base_model import *

from os.path import join as pjoin
import argparse
import tensorflow.compat.v1 as tf
import os
import shutil

tf.debugging.set_log_device_placement(True)

# ---------------------- DELETE FILES OF MODELS -----------------------------
# ---------------------------------------------------------------------------

folder = './output/_tensorboard/train'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

# ------------------------ GLOBAL VARIABLES ---------------------------------
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--n_modules', type=int, default=420)  # number of modules
parser.add_argument('--graph', type=str, default='default')  # adjacency matrix A ou adjacency matrix W
parser.add_argument('--n_his', type=int, default=12)  # historical time window - 3 hrs
parser.add_argument('--n_pred', type=int, default=6)  # prediction window - 1:30 hrs
parser.add_argument('--metric', type=str, default='Corrente_modulo')  # choose metric - current or voltage
parser.add_argument('--ks', type=int, default=3)  # kernel size of spatial convolution
parser.add_argument('--kt', type=int, default=3)  # kernel size of temporal convolution
parser.add_argument('--batch_size', type=int, default=20)  # batch_size - 5
parser.add_argument('--epoch', type=int, default=50)  # epochs
parser.add_argument('--lr', type=float, default=1e-3)  # learning rate
parser.add_argument('--opt', type=str, default='RMSProp')  # define optimization function
parser.add_argument('--inf_mode', type=str, default='merge')  # define type of step index
parser.add_argument('--act_func', type=str, default='sigmoid')  # define activation function
parser.add_argument('--save', type=int, default=10)  # save model each 10 epochs

args = parser.parse_args()

blocks = [[1, 16, 64], [32, 16, 64]]  # [[1, 32, 64], [64, 32, 128]]

# Paths
sum_path = './output/_tensorboard'
load_path = './output/models/'

# ------------------------ READ DATASET -------------------------------------
# ---------------------------------------------------------------------------

# Load adjacency matrix W - Graph
if args.graph == 'default':
    # W = weight_matrix(pjoin('./Database/Topologias_PaineisFotovoltaicos', 'CPID_LayoutLogico_InversorOut.graphml'))
    W = weight_matrix(pjoin('./Database/Topologias_PaineisFotovoltaicos', 'CPID_LayoutLogico_InversorOut.tgf'))
else:
    # load customized graph weight matrix
    W = weight_matrix(pjoin('./Database/Topologias_PaineisFotovoltaicos', args.graph))
# Calculate graph kernel
L = scaled_laplacian(W)
# Alternative approximation method: 1st approx - first_approx(W, n).
# Lk = first_approx(W, args.n_modules)
Lk = cheb_poly_approx(L, args.ks, args.n_modules)

# Load characteristic matrix X - current and voltage
data_file = f'{args.metric}.csv'
n_train, n_val, n_test = 30, 5, 10
measures = data_gen(pjoin('Database/pre-processing/dados_rede', data_file), (n_train, n_val, n_test), args.n_modules, args.n_his + args.n_pred)
print(f'>> Loading dataset with Mean: {measures.mean:.2f}, STD: {measures.std:.2f}')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
input('Enter')

# ------------------------ MODEL INFORMATION --------------------------------
# ---------------------------------------------------------------------------

# Configure Tensorflow
tf.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

# Create a new collection - Graph Collection
tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))
# print('Graph:', Lk.shape, [item for item in tf.get_collection('graph_kernel')])

# ------------------------ MODEL TRAIN --------------------------------------
# ---------------------------------------------------------------------------

# Placeholder for model training
''' Insere um espaço reservado para um tensor que sempre será alimentado.'''
x = tf.placeholder(tf.float32, [None, args.n_his + 1, args.n_modules, 1], name='data_input')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# Define model loss
train_loss, pred = build_model(x, args.n_his, args.ks, args.kt, blocks, keep_prob, args.act_func)
tf.summary.scalar('train_loss', train_loss)
copy_loss = tf.add_n(tf.get_collection('copy_loss'))  # train_loss manually
tf.summary.scalar('copy_loss', copy_loss)

# Learning rate settings
global_steps = tf.Variable(0, trainable=False)
len_train = measures.get_len('train')
if len_train % args.batch_size == 0:
    epoch_step = len_train / args.batch_size
else:
    epoch_step = int(len_train / args.batch_size) + 1
# Learning rate decay with rate 0.7 every 5 epochs.
lr = tf.train.exponential_decay(args.lr, global_steps, decay_steps=5 * epoch_step, decay_rate=0.7, staircase=True)
tf.summary.scalar('learning_rate', lr)
step_op = tf.assign_add(global_steps, 1)
with tf.control_dependencies([step_op]):
    if args.opt == 'RMSProp':
        train_op = tf.train.RMSPropOptimizer(lr).minimize(train_loss)
    elif args.opt == 'ADAM':
        train_op = tf.train.AdamOptimizer(lr).minimize(train_loss)
    else:
        raise ValueError(f'ERROR: optimizer "{args.opt}" is not defined.')

merged = tf.summary.merge_all()

# Training model
with tf.Session() as sess:
    writer = tf.summary.FileWriter(pjoin(sum_path, 'train'), sess.graph)
    sess.run(tf.global_variables_initializer())

    if args.inf_mode == 'sep':
        # for inference mode 'sep', the type of step index is int.
        step_idx = args.n_pred - 1
        tmp_idx = [step_idx]
        min_val = min_va_val = np.array([4e1, 1e5, 1e5])
    elif args.inf_mode == 'merge':
        # for inference mode 'merge', the type of step index is np.ndarray.
        step_idx = tmp_idx = np.arange(3, args.n_pred + 1, 3) - 1
        min_val = min_va_val = np.array([4e1, 1e5, 1e5] * len(step_idx))
        print('[SESSION] Step_idx and Min_val: \n', step_idx, min_val)
    else:
        raise ValueError(f'ERROR: test mode "{args.inf_mode}" is not defined.')

    for i in range(args.epoch):
        start_time_epoch = time.time()  # epoch processing start time
        print('Epoch:', i, start_time_epoch)
        ''' j: index batch | x_batch: batch'''
        for j, x_batch in enumerate(gen_batch(measures.get_data('train'), args.batch_size, dynamic_batch=True, shuffle=True)):
            summary, _ = sess.run([merged, train_op], feed_dict={x: x_batch[:, 0:args.n_his + 1, :, :], keep_prob: 1.0})
            writer.add_summary(summary, i * epoch_step + j)
            if j % 50 == 0:
                loss_value = \
                    sess.run([train_loss, copy_loss],
                             feed_dict={x: x_batch[:, 0:args.n_his + 1, :, :], keep_prob: 1.0})
                print(f'Epoch {i:2d}, Step {j:3d}: [{loss_value[0]:.3f}, {loss_value[1]:.3f}]')
        print(f'Epoch {i:2d} Training Time {time.time() - start_time_epoch:.3f}s')

        # validation - model inference
        start_time_inference = time.time()
        min_va_val, min_val = \
            model_inference(sess, pred, measures, args.batch_size, args.n_his, args.n_pred, step_idx, min_va_val, min_val)
        for ix in tmp_idx:
            va, te = min_va_val[ix - 2:ix + 1], min_val[ix - 2:ix + 1]
            print(f'Time Step {ix + 1}: '
                  f'MAPE {va[0]:7.3%}, {te[0]:7.3%}; '
                  f'MAE  {va[1]:4.3f}, {te[1]:4.3f}; '
                  f'RMSE {va[2]:6.3f}, {te[2]:6.3f}.')
        print(f'Epoch {i:2d} Inference Time {time.time() - start_time_inference:.3f}s')

        # save model each 10 epochs
        if (i + 1) % args.save == 0:
            models_save(sess, global_steps, 'STGCN')

    writer.close()
print('Training model finished!')

# ------------------------ MODEL TEST ---------------------------------------
# ---------------------------------------------------------------------------

start_time_test = time.time()

model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path
print(model_path)

test_graph = tf.Graph()

# makes test_graph the default graph
with test_graph.as_default():
    saver = tf.train.import_meta_graph(pjoin(f'{model_path}.meta'))

with tf.Session(graph=test_graph) as test_sess:
    saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
    print(f'>> Loading saved model from {model_path} ...')

    pred = test_graph.get_collection('y_pred')

    if args.inf_mode == 'sep':
        # for inference mode 'sep', the type of step index is int.
        step_idx = args.n_pred - 1
        tmp_idx = [step_idx]
    elif args.inf_mode == 'merge':
        # for inference mode 'merge', the type of step index is np.ndarray.
        step_idx = tmp_idx = np.arange(3, args.n_pred + 1, 3) - 1
    else:
        raise ValueError(f'ERROR: test mode "{args.inf_mode}" is not defined.')

    x_test, x_stats = measures.get_data('test'), measures.get_stats()

    y_test, len_test = multi_pred(test_sess, pred, x_test, measures.get_len('test'), args.n_his, args.n_pred, step_idx)
    evl = evaluation(x_test[0:len_test, step_idx + args.n_his, :, :], y_test, x_stats)

    for ix in tmp_idx:
        te = evl[ix - 2:ix + 1]
        print(f'Time Step {ix + 1}: MAPE {te[0]:7.3%}; MAE  {te[1]:4.3f}; RMSE {te[2]:6.3f}.')
    print(f'Model Test Time {time.time() - start_time_test:.3f}s')

print('Testing model finished!')

# tensorboard --logdir=.\output\_tensorboard\train
