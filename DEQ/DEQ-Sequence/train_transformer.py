import argparse
import time
from os import times

import math
import os, sys
import itertools
import numpy as np
import random

import subprocess
import torch

_bootstrap_parser = argparse.ArgumentParser(add_help=False)
_bootstrap_parser.add_argument('--gpu-count', type=int, default=3)
_bootstrap_args, _ = _bootstrap_parser.parse_known_args()


def get_free_gpu():
    """获取空闲GPU设备号，按照可用内存排序"""
    try:
        # 使用nvidia-smi命令获取GPU信息
        gpu_stats = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free,index", "--format=csv,nounits,noheader"],
                                          universal_newlines=True)
        gpu_df = []
        for line in gpu_stats.strip().split("\n"):
            values = line.split(",")
            gpu_df.append((int(values[0]), int(values[1])))

        # 按照空闲内存排序（降序）
        gpu_df.sort(reverse=True)
        return [str(gpu_idx) for _, gpu_idx in gpu_df]
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
        return ["0"]  # 默认使用0号GPU

GPU_n = _bootstrap_args.gpu_count
# 自动设置可用GPU，选择GPU_n个空闲内存最大的GPU
available_gpus = get_free_gpu()
selected_gpus = available_gpus[:GPU_n] if len(available_gpus) >= GPU_n else available_gpus
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpus)
print(f"自动选择GPU设备: {os.environ['CUDA_VISIBLE_DEVICES']}")
torch.cuda.set_device(0)  # 设置主GPU为第一个设备

# 选取GPU_ids为GPU_n个空闲内存最大的GPU
device_ids = list(range(len(selected_gpus)))  # torch.cuda.device_count() - 1


import torch.nn as nn
import torch.optim as optim

sys.path.append('../')

from data_utils import get_lm_corpus
from models.deq_transformer import DEQTransformerLM
from lib.solvers import anderson, broyden
from lib import radam
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel
# from torch.utils.tensorboard import SummaryWriter

# new
from torch.utils.data import DataLoader, TensorDataset
from models.deq_transformer_CD import ConsistencyFunction
from tqdm import tqdm, trange
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch DEQ Sequence Model')
parser.add_argument('--data', type=str, default='./data/wikitext-103',
                    help='location of the data corpus (default to the WT103 path)')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103'],
                    help='dataset name')
parser.add_argument('--n_layer', type=int, default=12,
                    help='number of total layers')
parser.add_argument('--eval_n_layer', type=int, default=12,
                    help='number of total layers at evaluation')
parser.add_argument('--n_head', type=int, default=10,
                    help='number of heads (default: 10)')
parser.add_argument('--d_head', type=int, default=70,
                    help='head dimension (default: 50)')
parser.add_argument('--d_embed', type=int, default=-1,
                    help='embedding dimension (default: match d_model [-1])')
parser.add_argument('--d_model', type=int, default=700,
                    help='model dimension (default: 500)')
parser.add_argument('--d_inner', type=int, default=48000,
                    help='inner dimension in the position-wise feedforward block (default: 8000)')

# Dropouts
parser.add_argument('--dropout', type=float, default=0.05,
                    help='global dropout rate (default: 0.05)')
parser.add_argument('--dropatt', type=float, default=0.0,
                    help='attention map dropout rate (default: 0.0)')

# Initializations
# Note: Generally, to make sure the DEQ model is stable initially, we should constrain the range
#       of initialization.
parser.add_argument('--init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--emb_init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--init_range', type=float, default=0.05,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--emb_init_range', type=float, default=0.01,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--init_std', type=float, default=0.01,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--proj_init_std', type=float, default=0.01,
                    help='parameters initialized by N(0, init_std)')

# Optimizers
parser.add_argument('--optim', default='Adam', type=str,
                    choices=['Adam', 'SGD', 'Adagrad', 'RMSprop', 'RAdam'],
                    help='optimizer to use.')
parser.add_argument('--lr', type=float, default=0.00025,
                    help='initial learning rate (0.00025|5 for adam|sgd)')
parser.add_argument('--scheduler', default='cosine', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                    help='lr scheduler to use.')
parser.add_argument('--warmup_step', type=int, default=0,
                    help='the number of steps to warm up the learning rate to its lr value')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min', type=float, default=0.0,
                    help='minimum learning rate during annealing')

# Gradient updates
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--clip_nonemb', action='store_true',
                    help='only clip the gradient of non-embedding params')
parser.add_argument('--max_step', type=int, default=300000,
                    help='upper epoch limit (at least 200K for WT103 or PTB)')
parser.add_argument('--batch_size', type=int, default=28,  # 56
                    help='batch size')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='split batch into chunks to save memory')

# Sequence logistics
parser.add_argument('--tgt_len', type=int, default=150,
                    help='number of tokens to predict')
parser.add_argument('--eval_tgt_len', type=int, default=150,
                    help='number of tokens to predict for evaluation')
parser.add_argument('--mem_len', type=int, default=300,
                    help='length of the retained previous heads')
parser.add_argument('--local_size', type=int, default=0,
                    help='local horizon size')

# DEQ related [Bai et al. 2019]
parser.add_argument('--f_solver', default='anderson', type=str,
                    choices=['anderson', 'broyden'],
                    help='forward solver to use (only anderson and broyden supported now)')
parser.add_argument('--b_solver', default='broyden', type=str,
                    choices=['anderson', 'broyden', 'None'],
                    help='backward solver to use (if None, then set it to f_solver)')
parser.add_argument('--stop_mode', type=str, default="rel",
                    choices=['abs', 'rel'],
                    help='stop criterion absolute or relative')
parser.add_argument('--rand_f_thres_delta', type=int, default=0,
                    help='use (f_thres + U(-delta, 0)) for forward threshold (delta default to 0)')
parser.add_argument('--f_thres', type=int, default=40,
                    help='forward pass Broyden threshold')
parser.add_argument('--b_thres', type=int, default=40,
                    help='backward pass Broyden threshold')

# Jacobian regularization related [Bai et al. 2021]
parser.add_argument('--jac_loss_weight', type=float, default=0.0,
                    help='jacobian regularization loss weight (default to 0)')
parser.add_argument('--jac_loss_freq', type=float, default=0.0,
                    help='the frequency of applying the jacobian regularization (default to 0)')
parser.add_argument('--jac_incremental', type=int, default=0,
                    help='if positive, increase jac_loss_weight by 0.1 after this many steps')
parser.add_argument('--spectral_radius_mode', action='store_true',
                    help='compute spectral radius at validation time')

# Training techniques
parser.add_argument('--not_tied', action='store_true',
                    help='do not tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--eval', action='store_true', default=True,
                    help='evaluation mode')
parser.add_argument('--adaptive', action='store_true', default=True,
                    help='use adaptive softmax')
parser.add_argument('--div_val', type=int, default=4,
                    help='divident value for adapative input and softmax')
parser.add_argument('--pre_lnorm', action='store_true',
                    help='apply LayerNorm to the input instead of the output')
parser.add_argument('--wnorm', action='store_true', default=True,
                    help='apply WeightNorm to the weights')
parser.add_argument('--varlen', action='store_true',
                    help='use variable length')
parser.add_argument('--multi_gpu', action='store_true', default=False,  # False
                    help='use multiple GPU')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--eval-interval', type=int, default=5000,
                    help='evaluation interval')
parser.add_argument('--work_dir', default='LM-TFM', type=str,
                    help='experiment directory.')
parser.add_argument('--restart', action='store_true',
                    help='restart training from the saved checkpoint')
parser.add_argument('--restart_dir', type=str, default='',
                    help='restart dir')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--same_length', action='store_true',
                    help='use the same attn length for all tokens')
parser.add_argument('--attn_type', type=int, default=0,
                    help='attention type. 0 for ours, 1 for Shaw et al,'
                         '2 for Vaswani et al, 3 for Al Rfou et al. (Only 0 supported now)')
parser.add_argument('--eta_min', type=float, default=0.0,
                    help='min learning rate for cosine scheduler')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--gpu0_bsz', type=int, default=4,  # 7
                    help='batch size on gpu 0')
parser.add_argument('--gpu-count', type=int, default=GPU_n,
                    help='number of free GPUs to select automatically')
parser.add_argument('--max_eval_steps', type=int, default=-1,
                    help='max eval steps')
parser.add_argument('--pretrain_steps', type=int, default=0,
                    help='number of pretrain steps (default to 0')
parser.add_argument('--start_train_steps', type=int, default=0,
                    help='starting training step count (default to 0)')
parser.add_argument('--patience', type=int, default=0,
                    help='patience')
parser.add_argument('--load', type=str, default='pretrained_wt103_deqtrans_v3.pkl',
                    help='path to load weight')
parser.add_argument('--name', type=str, default='ljc',
                    help='name of the trial')
parser.add_argument('--save-trajectory', action='store_true',
                    help='save DEQ solver trajectories on the validation set and exit')
parser.add_argument('--trajectory-prefix', type=str, default='traj_all',
                    help='prefix for saved/loaded trajectory files, e.g. traj_all_1.pt')
parser.add_argument('--train-CM', '--train-cm', dest='train_CM', action='store_true',
                    help='train the consistency model before evaluation')
parser.add_argument('-CM', '--CM', dest='CM', action='store_true',
                    help='use the trained consistency model during evaluation')
parser.add_argument('--cm-load', type=str, default='best_CM_model.pth',
                    help='path to a trained consistency model')
parser.add_argument('--cm-save', type=str, default='best_CM_model.pth',
                    help='path to save the best trained consistency model')
parser.add_argument('--cm-checkpoint', type=str, default='cm_checkpoint/cm_checkpoint.pt',
                    help='path to save/load CM training checkpoint')
parser.add_argument('--deq-func-load', type=str, default='./models/pretrained_deq_func.pth',
                    help='path to the pretrained DEQ function weights for CM training')
parser.add_argument('--plot-CM', '--plot-cm', dest='plot_CM', action='store_true',
                    help='plot CM training curves')
parser.add_argument('--cm-start-file-idx', type=int, default=1,
                    help='first traj_all_N.pt file index for CM training')
parser.add_argument('--cm-max-file-idx', type=int, default=3,
                    help='last traj_all_N.pt file index for CM training')
parser.add_argument('--cm-max-traj-per-file', type=int, default=15,
                    help='max trajectories to use from each traj_all_N.pt file')
parser.add_argument('--cm-num-samples', type=int, default=100,
                    help='number of trajectories sampled for CM training')
parser.add_argument('--cm-epochs', type=int, default=50,
                    help='epochs per sampled trajectory for CM training')
parser.add_argument('--cm-batch-size', type=int, default=16,
                    help='batch size for CM trajectory training')

args = parser.parse_args()
args.tied = not args.not_tied
args.pretrain_steps += args.start_train_steps
assert args.mem_len > 0, "For now you must set mem_len > 0 when using deq"
args.work_dir += "deq"
args.cuda = torch.cuda.is_available()

if args.d_embed < 0:
    args.d_embed = args.d_model

assert args.batch_size % args.batch_chunk == 0

args.work_dir = '{}-{}'.format(args.work_dir, args.dataset)
timestamp = time.strftime('%Y%m%d-%H%M%S')
if args.restart_dir:
    timestamp = args.restart_dir.split('/')[1]
args.work_dir = os.path.join(args.work_dir, timestamp)
if args.name == "N/A" and not args.debug:
    # If you find this too annoying, uncomment the following line and use timestamp as name.
    # args.name = timestamp
    raise ValueError("Please give a name to your run!")
print(f"Experiment name: {args.name}")
logging = create_exp_dir(args.work_dir,
                         scripts_to_save=['train_transformer.py', 'models/deq_transformer.py', '../lib/solvers.py'],
                         debug=args.debug)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(args.seed)



device = torch.device('cuda' if args.cuda else 'cpu')



###############################################################################
# Load data
###############################################################################
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)
args.n_token = ntokens  # 267735

eval_batch_size = max(16, torch.cuda.device_count())
# eval_batch_size = 1

tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len, device=device)
va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len, device=device)
te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len, device=device)  # 7674*32 if eval_batch_size=32

# adaptive softmax / embedding
cutoffs, tie_projs = [], [False]
if args.adaptive:
    assert args.dataset in ['wt103']
    cutoffs = [20000, 40000, 200000]
    tie_projs += [True] * len(cutoffs)


###############################################################################
# Build the model
###############################################################################
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv1d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i].weight, 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i].weight, 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('WeightShareSelfAttention') != -1:
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)

model = DEQTransformerLM(ntokens, args.n_layer, args.eval_n_layer, args.n_head, args.d_model, args.d_head,
                         args.d_inner,
                         args.dropout, args.dropatt, tie_weights=args.tied, d_embed=args.d_embed,
                         div_val=args.div_val, tie_projs=tie_projs, pre_lnorm=args.pre_lnorm,
                         wnorm=args.wnorm, local_size=args.local_size, pretrain_steps=args.pretrain_steps,
                         tgt_len=args.tgt_len, mem_len=args.mem_len, cutoffs=cutoffs, load=args.load,
                         f_solver=eval(args.f_solver), b_solver=eval(args.b_solver), stop_mode=args.stop_mode,
                         logging=logging)
if len(args.load) == 0:
    model.apply(weights_init)  # Note: This applies weight_init recursively to modules in model
    model.word_emb.apply(weights_init)

args.n_all_param = sum([p.nelement() for p in model.parameters() if p.requires_grad])

# para_model <- model
if args.multi_gpu:
    model = model.to(device)
    if args.gpu0_bsz >= 0 and args.batch_size != args.gpu0_bsz * torch.cuda.device_count():
        para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk, model, dim=1).to(device)
    else:
        para_model = nn.DataParallel(model, device_ids=device_ids, dim=1).to(device)
else:
    para_model = model.to(device)

#### optimizer
optimizer = getattr(optim if args.optim != 'RAdam' else radam, args.optim)(model.parameters(), lr=args.lr,
                                                                           weight_decay=args.weight_decay)
if not args.debug and not args.eval:
    writer = SummaryWriter(log_dir=f'log/{args.dataset}/deq_F{args.f_thres}_B{args.b_thres}', flush_secs=5)
else:
    writer = None

#### scheduler
if args.scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_step, eta_min=args.eta_min)
elif args.scheduler == 'inv_sqrt':
    # originally used for Transformer (in Attention is all you need)
    def lr_lambda(step):
        # return a multiplier instead of a learning rate
        if step == 0 and args.warmup_step == 0:
            return 1.
        else:
            return 1. / (step ** 0.5) if step > args.warmup_step else step / (args.warmup_step ** 1.5)


    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
elif args.scheduler == 'dev_perf':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)

if args.restart:
    # E.g., When you want to resume from a checkpoint from the same machine, where things should
    #       be stored in `args.restart_dir`
    if os.path.exists(os.path.join(args.restart_dir, 'optimizer.pt')):
        with open(os.path.join(args.restart_dir, 'optimizer.pt'), 'rb') as f:
            opt_state_dict = torch.load(f)
            optimizer.load_state_dict(opt_state_dict)
    else:
        print('Optimizer was not saved. Start from scratch.')

if args.start_train_steps > 0 and not args.restart:
    # E.g., When you want to directly load a state_dict (e.g., trained on another machine),
    #       You may want to manually adjust the optimizer's steps. On command line, you
    #       should run `bash ... --load [PATH] --start_train_steps N --pretrain_steps 0`
    #       in order to start the training from step N
    diff_from_warmup = args.start_train_steps - args.warmup_step
    # Speed up the scheduler
    if args.scheduler in ['cosine', 'constant', 'dev_perf']:
        if diff_from_warmup < 0:
            # Hasn't finished warmup yet
            curr_lr = args.lr * args.start_train_steps / args.warmup_step
            optimizer.param_groups[0]['lr'] = curr_lr
        else:
            if args.scheduler == 'cosine':
                for i in range(args.warmup_step, args.start_train_steps):
                    optimizer.step()
                    scheduler.step(i)
    elif args.scheduler == 'inv_sqrt':
        for i in range(args.warmup_step, args.start_train_steps):
            optimizer.step()
            scheduler.step(i)

logging('=' * 100)
for k, v in args.__dict__.items():
    logging('    - {} : {}'.format(k, v))
logging('=' * 100)


###############################################################################
# Training code
###############################################################################
def evaluate(eval_iter):
    global train_step
    model.eval()
    model.reset_length(args.eval_tgt_len, args.mem_len)

    # Evaluation
    total_len, total_loss = 0, 0.
    rho_list = []
    if args.spectral_radius_mode:
        print("WARNING: You are evaluating with the power method at val. time. This may make things extremely slow.")
    with torch.no_grad():
        mems = []
        traj_all = []
        for i, (data, target, seq_len) in enumerate(eval_iter):  # 90(for val) or 102(for test) sentences
            if 0 < args.max_eval_steps <= i:
                break
            rel_diff = None
            t = time.time()
            # data.shape = target.shape = torch.Size([150, 16])
            if save_trajectory:
                ret = para_model(data, target, mems, train_step=train_step, f_thres=args.f_thres,
                                 b_thres=args.b_thres, compute_jac_loss=False,
                                 spectral_radius_mode=args.spectral_radius_mode, writer=writer,
                                 save_trajectory=save_trajectory)

                loss, _, sradius, trajectory, mems = ret[0], ret[1], ret[2], ret[3], ret[4:]

                # 将trajectory堆叠到一个张量traj中
                traj_all.append(trajectory)

                # 每30个batch保存一次，避免内存溢出，以i/30为文件名
                if i % 10 == 0 and i != 0:
                    filename = f'{args.trajectory_prefix}_{i // 10}.pt'
                    torch.save(traj_all, filename)
                    print(f"Saved trajectory data!, traj_all_len={len(traj_all)}")
                    traj_all = []

            else:  # 推理模式
                if CM_mode is False:  # 原始DEQ
                    ret = para_model(data, target, mems, train_step=train_step, f_thres=args.f_thres,
                                     b_thres=args.b_thres, compute_jac_loss=False,
                                     spectral_radius_mode=args.spectral_radius_mode, writer=writer)
                else:  # with CM
                    ret = para_model(data, target, mems, train_step=train_step, f_thres=args.f_thres,
                                     b_thres=args.b_thres, compute_jac_loss=False,
                                     spectral_radius_mode=args.spectral_radius_mode, writer=writer,
                                     CM_load = CM_load)

                loss, _, sradius, _, mems = ret[0], ret[1], ret[2], ret[3], ret[4:]
                rel_diff = None

            loss = loss.mean()
            if args.spectral_radius_mode:
                rho_list.append(sradius.mean().item())
            total_loss += seq_len * loss.float().item()
            total_len += seq_len
            message = f"i:{i}, Time: {(time.time() - t):.4f}, loss:{loss.float():.4f}"
            if rel_diff is not None:
                message += f", rel_diff:{rel_diff.float():.4f}"
            print(message)  # 0.27s


    if rho_list:
        logging(f"(Estimated) Spectral radius over validation set: {np.mean(rho_list)}")
    model.train()
    return total_loss / total_len


def _train_on_trajectory(CD, CD_ema, params_ema, CD_optimizer, dataloader, N_EPOCHS, T, EPSILON):
    rel_diff_append = []
    loss_append = []
    tot_loss = 0

    with trange(N_EPOCHS) as pbar:
        for epoch in range(N_EPOCHS):
            epoch_loss = 0.0
            N_steps = 38
            t_steps = [(EPSILON ** (1 / 7) + (j / (N_steps - 1)) * (T ** (1 / 7) - EPSILON ** (1 / 7))) ** 7
                       for j in range(0, N_steps)]
            # n_steps = int(19 + 19 * epoch / N_EPOCHS)
            n_steps = 8  # 点太多容易爆显存

            # 从N_steps个点中，等距取出n_steps个点
            indices = torch.linspace(1, N_steps - 1, steps=n_steps).round().long().tolist()

            # 在indices中随机取数量为n_steps的点
            n_1 = []
            for _ in range(n_steps):
                n_1.append(indices[random.randint(0, len(indices) - 1)])

            for data in dataloader:
                x_batch = data[0]  # 获取第一个元素 x_traj
                batch_size = x_batch.size(0)
                current_func_args = data[1:]  # 获取剩余元素作为 func_args

                tn_1 = torch.tensor([t_steps[i] for i in n_1]).to(x_batch.device)
                tn = torch.tensor([t_steps[i - 1] for i in n_1]).to(x_batch.device)

                x_tn_1 = x_batch[:,n_1]
                x_tn = x_batch[:,(np.array(n_1) - 1).tolist()]

                # 计算损失
                with torch.no_grad():
                    out_tn = CD_ema(x_tn, tn.unsqueeze(0).expand(x_tn.size(0), -1), current_func_args)
                loss_1 = F.mse_loss(CD(x_tn_1, tn_1.unsqueeze(0).expand(x_tn.size(0), -1), current_func_args), out_tn)

                x_fixed = x_batch[:,-1].unsqueeze(1).expand(-1, n_steps, -1, -1)
                loss_2_x = CD(x_tn, tn.unsqueeze(0).expand(x_tn.size(0), -1), current_func_args)
                loss_2 = F.smooth_l1_loss(loss_2_x, x_fixed)  # F.mse_loss: loss_2_x 和 x_fixed 这两个张量之间的均方误差
                loss = 0.1 * loss_1 + 0.9 * loss_2

                # 更新模型
                loss.backward()
                CD_optimizer.step()
                CD_optimizer.zero_grad()

                # 更新EMA参数
                params_ema = {k: 0.98 * params_ema[k] + 0.02 * v for k, v in CD.module.state_dict().items()}
                CD_ema.module.load_state_dict(params_ema)

                epoch_loss += loss.item()

                # 计算相对误差
                with torch.no_grad():
                    x_ini = x_batch[:,0:1]
                    T = t_steps[-1] if isinstance(t_steps[-1], torch.Tensor) else torch.tensor(t_steps[-1], device=x_batch.device)
                    x1 = CD_ema(x_ini, T.unsqueeze(0).expand(batch_size, -1), current_func_args)
                    rel_diff = (x1 - x_batch[:, -1:]).norm() / x_batch[:,-1:].norm()
                    rel_diff_append.append(rel_diff.item())
                    loss_append.append(loss.item())

            tot_loss = epoch_loss / len(dataloader)
            pbar.set_postfix(epoch=epoch, loss=tot_loss, rel_diff=rel_diff.item())
            pbar.update()

    return tot_loss, rel_diff_append, loss_append


def CM_train(CM_training):  # 仅针对func in self.f_solver
    # 设置设备
    device = torch.device('cuda:0')

    '''构造Consistency Distrillation模型'''
    CD = ConsistencyFunction(n_head=args.n_head, d_model=args.d_model, d_head=args.d_head, d_inner=args.d_inner,
                             dropout=args.dropout, n_layer=args.n_layer, func_args=None).to(device)
    # 给ConsistencyFunction.func赋予pretrained parameters
    CD.func.load_state_dict(func_params_dict)
    CD_optimizer = torch.optim.AdamW(CD.parameters(), lr=4e-3)
    CD_ema = ConsistencyFunction(n_head=args.n_head, d_model=args.d_model, d_head=args.d_head, d_inner=args.d_inner,
                                 dropout=args.dropout, n_layer=args.n_layer, func_args=None).to(device)

    CD_ema.load_state_dict(CD.state_dict())  # 等价于deepcopy
    params_ema = CD_ema.state_dict()

    # 并行计算
    CD = nn.DataParallel(CD, device_ids=device_ids, dim=0).to(device)
    CD_ema = nn.DataParallel(CD_ema, device_ids=device_ids, dim=0).to(device)

    '''检查是否有断点续训的检查点文件'''
    checkpoint_path = args.cm_checkpoint

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        CD.module.load_state_dict(checkpoint['model'])
        CD_ema.module.load_state_dict(checkpoint['model_ema'])
        params_ema = checkpoint['params_ema']
        CD_optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        # 如果没有找到检查点文件，则默认从头开始训练
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")


    # Define global constants
    T = 5  # The maximum time, which is equal to the maximum noise standard deviation
    EPSILON = 0.002  # The minimum time
    N_EPOCHS = args.cm_epochs  # Number of epochs

    # 收集所有可用的轨迹
    max_file_idx = args.cm_max_file_idx
    max_traj_file = args.cm_max_traj_per_file
    all_trajectories = []

    # 首先收集所有可用的轨迹位置信息
    for file_idx in range(args.cm_start_file_idx, max_file_idx + 1):
        try:
            traj_path = f'{args.trajectory_prefix}_{file_idx}.pt'
            traj_file = torch.load(traj_path, map_location='cpu')
            print(f"找到文件 {traj_path}，共有{len(traj_file)}条轨迹")

            # 收集该文件中的所有轨迹索引
            for traj_idx in range(min(len(traj_file), max_traj_file)):
                all_trajectories.append((file_idx, traj_idx))
        except:
            print(f"无法加载文件 {args.trajectory_prefix}_{file_idx}.pt")

    print(f"共收集到 {len(all_trajectories)} 条可用轨迹")
    if not all_trajectories:
        raise FileNotFoundError("No traj_all_N.pt files found for CM training")

    # 设置随机种子以确保可重复性
    random.seed(42)  # 使用固定种子

    # 确保总是采样指定数量轨迹，即使需要重复采样
    num_samples = args.cm_num_samples
    if len(all_trajectories) >= num_samples:
        # 如果轨迹数量足够，直接随机抽样
        sampled_trajectories = random.sample(all_trajectories, num_samples)
    else:
        # 如果轨迹数量不足，则有放回地随机采样
        sampled_trajectories = [random.choice(all_trajectories) for _ in range(num_samples)]

    print(f"随机抽取了 {num_samples} 条轨迹进行训练（可能包含重复轨迹）")

    best_loss = 1e9
    best_epoch = 0

    # 对随机抽取的轨迹进行训练
    for idx, (file_idx, traj_idx) in enumerate(sampled_trajectories):
        print(f"正在处理第 {idx+1}/{num_samples} 条轨迹，来自文件 {args.trajectory_prefix}_{file_idx}.pt 中的第 {traj_idx} 条")

        traj = torch.load(f'{args.trajectory_prefix}_{file_idx}.pt', map_location=device)
        x_list = [traj[i]['x_traj'] for i in range(len(traj))]
        func_args = []
        for i in range(len(traj)):
            func_args.append([traj[i]['func_args'][0], traj[i]['func_args'][1], traj[i]['func_args'][2]])

        # 取指定文件中的指定轨迹
        x_list = x_list[traj_idx]  # [39, 16, 700, 150]
        # x_list = torch.flip(x_list, [0])  # 将x_traj逆向排列
        x_traj = x_list.permute(1, 0, 2, 3)  # 将x_traj第0和1维交换
        func_args = func_args[traj_idx]  # list:3

        '''对于某一条轨迹进行训练'''
        # 有n_samples=39个样本点，每个样本点的shape为16*700*150  bsz x d_model x qlen
        bsz, n_samples, d_model, qlen = x_traj.shape[0], x_traj.shape[1], x_traj.shape[2], x_traj.shape[3]

        # Create dataset and dataloader
        func_args[2] = func_args[2].unsqueeze(0).expand(bsz, *func_args[2].shape)
        dataset = TensorDataset(x_traj, *func_args)
        dataloader = DataLoader(dataset, batch_size=args.cm_batch_size, shuffle=True, drop_last=True,
                                generator=torch.Generator(device=device))

        '''Training'''
        if CM_training:
            # 训练当前轨迹
            tot_loss, rel_diff_append, loss_append = _train_on_trajectory(CD, CD_ema, params_ema, CD_optimizer, dataloader,
                                            N_EPOCHS, T, EPSILON)

            # plot the relative error and loss
            if args.plot_CM:
                import matplotlib.pyplot as plt
                plt.plot(rel_diff_append)
                plt.plot(loss_append)
                plt.show()

            # 保存检查点
            cm_check = True
            if cm_check:
                checkpoint = {
                    'model': CD.module.state_dict(),
                    'model_ema': CD_ema.module.state_dict(),
                    'params_ema': params_ema,
                    'optimizer': CD_optimizer.state_dict(),
                }
                checkpoint_dir = os.path.dirname(checkpoint_path)
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(checkpoint, checkpoint_path)

            # 更新最佳模型
            if tot_loss < best_loss:
                best_loss = tot_loss
                torch.save(CD.module.state_dict(), args.cm_save)
                print(f"新的最佳模型{args.cm_save}已保存")

        else:
            print("Training skipped. Loading the best model...")

        # Evaluation
        with torch.no_grad():
            x_ini = x_list[0].unsqueeze(1)  # torch.Size([16, 1, 700, 150])
            t = torch.tensor(T).to(device).unsqueeze(0).expand(bsz, -1)  # 直接取最后时间步：T == 5 torch.Size([16, 1])

            CD.module.load_state_dict(torch.load(args.cm_save))
            x = CD(x_ini, t, func_args).squeeze(1)  # torch.Size([16, 700, 150])
            x_rel = x_traj[:,-1]

            rel_diff = (x - x_rel).norm() / x_rel.norm()  # relative error
            print(f"Relative error: {rel_diff.item()}")

    print(f"所有 {num_samples} 条轨迹训练完成，最佳模型已保存")


# Loop over epochs.
train_step = 0
train_loss = 0
train_jac_loss = []
best_val_loss = None

log_start_time = time.time()
eval_start_time = time.time()
train_step = 1e9
epoch = -1

# Command modes replace the old manual comment switches.
save_trajectory = args.save_trajectory
CM_mode = args.CM
CM_load = args.cm_load if args.CM else None

if args.save_trajectory:
    valid_loss = evaluate(va_iter)
    logging('=' * 100)
    logging('| End of evaluating on validation set | valid loss {:5.2f} | valid ppl {:9.3f}'.format(
        valid_loss, math.exp(valid_loss)))
    logging('=' * 100)
    print("Trajectory Data Saved!")
    sys.exit(0)

if args.train_CM:
    func_params_dict = torch.load(args.deq_func_load)
    CM_train(CM_training=True)


valid_loss = evaluate(va_iter)
logging('=' * 100)
logging('| End of training | valid loss {:5.2f} | valid ppl {:9.3f}'.format(valid_loss, math.exp(valid_loss)))
logging('=' * 100)

# test_loss = evaluate(te_iter)
# logging('=' * 100)
# logging('| End of training | test loss {:5.2f} | test ppl {:9.3f}'.format(test_loss, math.exp(test_loss)))
# logging('=' * 100)

sys.exit(0)






# 训练DEQ
# At any point you can hit Ctrl + C to break out of training early.
# try:
#     for epoch in itertools.count(start=1):
#         train()
#         if train_step == args.max_step:
#             logging('-' * 100)
#             logging('End of training')
#             break
# except KeyboardInterrupt:
#     logging('-' * 100)
#     logging('Exiting from training early')
#
# # Load the best saved model.
# with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
#     model = torch.load(f)
# para_model = model.to(device)
#
# # Run on test data.
# test_loss = evaluate(te_iter)
# logging('=' * 100)
# logging('| End of training | test loss {:5.2f} | test ppl {:9.3f}'.format(test_loss, math.exp(test_loss)))
# logging('=' * 100)

