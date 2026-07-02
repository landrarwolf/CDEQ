import argparse
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange


SEQ_DIR = Path(__file__).resolve().parent
DEQ_DIR = SEQ_DIR.parent
for path in (str(SEQ_DIR), str(DEQ_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from data_utils import get_lm_corpus
from lib.solvers import anderson
from models.deq_transformer import DEQTransformerLM
from models.deq_transformer_CD import ConsistencyFunction
from utils.data_parallel import BalancedDataParallel
from utils.exp_utils import create_exp_dir


SOLVERS = {
    "anderson": anderson,
}


def build_parser():
    parser = argparse.ArgumentParser(description="PyTorch DEQ Sequence Model")
    parser.add_argument("--data", type=str, default="./data/wikitext-103",
                        help="location of the data corpus")
    parser.add_argument("--dataset", type=str, default="wt103", choices=["wt103"],
                        help="dataset name")
    parser.add_argument("--n_layer", type=int, default=12,
                        help="number of total layers")
    parser.add_argument("--eval_n_layer", type=int, default=12,
                        help="number of total layers at evaluation")
    parser.add_argument("--n_head", type=int, default=10,
                        help="number of heads")
    parser.add_argument("--d_head", type=int, default=70,
                        help="head dimension")
    parser.add_argument("--d_embed", type=int, default=-1,
                        help="embedding dimension")
    parser.add_argument("--d_model", type=int, default=700,
                        help="model dimension")
    parser.add_argument("--d_inner", type=int, default=48000,
                        help="inner dimension in the feedforward block")
    parser.add_argument("--dropout", type=float, default=0.05,
                        help="global dropout rate")
    parser.add_argument("--dropatt", type=float, default=0.0,
                        help="attention map dropout rate")
    parser.add_argument("--init", default="normal", type=str,
                        help="parameter initializer")
    parser.add_argument("--emb_init", default="normal", type=str,
                        help="embedding initializer")
    parser.add_argument("--init_range", type=float, default=0.05,
                        help="uniform init range")
    parser.add_argument("--emb_init_range", type=float, default=0.01,
                        help="embedding init range")
    parser.add_argument("--init_std", type=float, default=0.01,
                        help="normal init std")
    parser.add_argument("--proj_init_std", type=float, default=0.01,
                        help="projection init std")
    parser.add_argument("--optim", default="Adam", type=str,
                        choices=["Adam", "SGD", "Adagrad", "RMSprop", "RAdam"],
                        help="optimizer to use for historical training commands")
    parser.add_argument("--lr", type=float, default=0.00025,
                        help="initial learning rate")
    parser.add_argument("--scheduler", default="cosine", type=str,
                        choices=["cosine", "inv_sqrt", "dev_perf", "constant"],
                        help="historical scheduler option")
    parser.add_argument("--warmup_step", type=int, default=0,
                        help="historical warmup steps")
    parser.add_argument("--decay_rate", type=float, default=0.5,
                        help="historical scheduler decay")
    parser.add_argument("--lr_min", type=float, default=0.0,
                        help="historical minimum learning rate")
    parser.add_argument("--clip", type=float, default=0.25,
                        help="historical gradient clipping")
    parser.add_argument("--clip_nonemb", action="store_true",
                        help="historical non-embedding clipping")
    parser.add_argument("--max_step", type=int, default=300000,
                        help="historical max step")
    parser.add_argument("--batch_size", type=int, default=28,
                        help="batch size")
    parser.add_argument("--batch_chunk", type=int, default=1,
                        help="split batch into chunks")
    parser.add_argument("--tgt_len", type=int, default=150,
                        help="tokens to predict")
    parser.add_argument("--eval_tgt_len", type=int, default=150,
                        help="tokens to predict at evaluation")
    parser.add_argument("--mem_len", type=int, default=300,
                        help="retained memory length")
    parser.add_argument("--local_size", type=int, default=0,
                        help="local horizon size")
    parser.add_argument("--f_solver", default="anderson", choices=sorted(SOLVERS),
                        help="forward fixed-point solver")
    parser.add_argument("--b_solver", default="None",
                        choices=["anderson", "None"],
                        help="backward fixed-point solver")
    parser.add_argument("--stop_mode", type=str, default="rel", choices=["abs", "rel"],
                        help="solver stop criterion")
    parser.add_argument("--rand_f_thres_delta", type=int, default=0,
                        help="historical forward threshold jitter")
    parser.add_argument("--f_thres", type=int, default=40,
                        help="forward solver threshold")
    parser.add_argument("--b_thres", type=int, default=40,
                        help="backward solver threshold")
    parser.add_argument("--jac_loss_weight", type=float, default=0.0,
                        help="historical jacobian loss weight")
    parser.add_argument("--jac_loss_freq", type=float, default=0.0,
                        help="historical jacobian loss frequency")
    parser.add_argument("--jac_incremental", type=int, default=0,
                        help="historical jacobian increment")
    parser.add_argument("--spectral_radius_mode", action="store_true",
                        help="compute spectral radius at validation time")
    parser.add_argument("--not_tied", action="store_true",
                        help="do not tie word embedding and softmax weights")
    parser.add_argument("--seed", type=int, default=1111,
                        help="random seed")
    parser.add_argument("--cuda", action="store_true",
                        help="kept for command compatibility; CUDA is auto-detected")
    parser.add_argument("--eval", action="store_true", default=True,
                        help="evaluation mode")
    parser.add_argument("--adaptive", action="store_true", default=True,
                        help="use adaptive softmax")
    parser.add_argument("--div_val", type=int, default=4,
                        help="adaptive input/softmax div value")
    parser.add_argument("--pre_lnorm", action="store_true",
                        help="pre layer norm")
    parser.add_argument("--wnorm", action="store_true", default=True,
                        help="weight norm")
    parser.add_argument("--varlen", action="store_true",
                        help="historical variable length option")
    parser.add_argument("--multi_gpu", action="store_true", default=False,
                        help="use multiple GPUs")
    parser.add_argument("--log-interval", type=int, default=200,
                        help="historical report interval")
    parser.add_argument("--eval-interval", type=int, default=5000,
                        help="historical eval interval")
    parser.add_argument("--work_dir", default="LM-TFM", type=str,
                        help="experiment directory")
    parser.add_argument("--restart", action="store_true",
                        help="historical restart flag")
    parser.add_argument("--restart_dir", type=str, default="",
                        help="historical restart dir")
    parser.add_argument("--debug", action="store_true",
                        help="do not create experiment dir")
    parser.add_argument("--same_length", action="store_true",
                        help="historical same length option")
    parser.add_argument("--attn_type", type=int, default=0,
                        help="attention type; only 0 is supported")
    parser.add_argument("--eta_min", type=float, default=0.0,
                        help="historical cosine min lr")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="historical weight decay")
    parser.add_argument("--gpu0_bsz", type=int, default=4,
                        help="batch size on GPU 0")
    parser.add_argument("--gpu-count", type=int, default=3,
                        help="number of free GPUs to select automatically")
    parser.add_argument("--gpu-ids", type=str, default="",
                        help="comma-separated physical GPU ids to consider; empty means all")
    parser.add_argument("--max_eval_steps", type=int, default=-1,
                        help="max eval batches; <=0 means all")
    parser.add_argument("--pretrain_steps", type=int, default=0,
                        help="historical pretrain steps")
    parser.add_argument("--start_train_steps", type=int, default=0,
                        help="historical starting training step")
    parser.add_argument("--patience", type=int, default=0,
                        help="historical scheduler patience")
    parser.add_argument("--load", type=str, default="pretrained_wt103_deqtrans_v3.pkl",
                        help="path to load DEQ weights")
    parser.add_argument("--name", type=str, default="ljc",
                        help="trial name")
    parser.add_argument("--save-trajectory", action="store_true",
                        help="save DEQ solver trajectories and exit")
    parser.add_argument("--trajectory-prefix", type=str, default="traj_all",
                        help="prefix for saved/loaded trajectory files")
    parser.add_argument("--trajectory-solver", choices=["picard", "anderson"], default="picard",
                        help="solver used for saved trajectories and CM training/inference")
    parser.add_argument("--force-trajectory-regen", action="store_true",
                        help="overwrite existing trajectory cache for this prefix")
    parser.add_argument("--train-CM", "--train-cm", dest="train_CM", action="store_true",
                        help="train the consistency model")
    parser.add_argument("-CM", "--CM", dest="CM", action="store_true",
                        help="use the trained consistency model during evaluation")
    parser.add_argument("--cm-load", type=str, default="best_CM_model.pth",
                        help="path to a trained consistency model")
    parser.add_argument("--cm-compare-teacher", action="store_true",
                        help="also run the DEQ teacher solver during CM evaluation and print relative error")
    parser.add_argument("--cm-save", type=str, default="best_CM_model.pth",
                        help="path to save the best trained consistency model")
    parser.add_argument("--cm-checkpoint", type=str, default="cm_checkpoint/cm_checkpoint.pt",
                        help="path to save/load CM training checkpoint")
    parser.add_argument("--deq-func-load", type=str, default="./models/pretrained_deq_func.pth",
                        help="path to pretrained DEQ function weights")
    parser.add_argument("--plot-CM", "--plot-cm", dest="plot_CM", action="store_true",
                        help="plot CM training curves")
    parser.add_argument("--cm-start-file-idx", type=int, default=1,
                        help="first trajectory file index")
    parser.add_argument("--cm-max-file-idx", type=int, default=3,
                        help="last trajectory file index")
    parser.add_argument("--cm-max-traj-per-file", type=int, default=15,
                        help="max trajectories per file")
    parser.add_argument("--cm-num-samples", type=int, default=100,
                        help="number of trajectories sampled for CM training")
    parser.add_argument("--cm-epochs", type=int, default=50,
                        help="epochs per sampled trajectory")
    parser.add_argument("--cm-batch-size", type=int, default=16,
                        help="CM trajectory batch size")
    parser.add_argument("--cm-train-points", type=int, default=8,
                        help="trajectory points sampled per CM training batch")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    modes = [args.save_trajectory, args.train_CM, args.CM]
    if sum(bool(mode) for mode in modes) > 1:
        parser.error("--save-trajectory, --train-CM, and -CM/--CM are mutually exclusive")
    if args.mem_len <= 0:
        parser.error("--mem_len must be > 0 for this DEQ path")
    if args.batch_size % args.batch_chunk != 0:
        parser.error("--batch_size must be divisible by --batch_chunk")
    if args.attn_type != 0:
        parser.error("only --attn_type 0 is supported")

    args.tied = not args.not_tied
    args.pretrain_steps += args.start_train_steps
    if args.d_embed < 0:
        args.d_embed = args.d_model
    args.work_dir = f"{args.work_dir}deq-{args.dataset}"
    timestamp = Path(args.restart_dir).name if args.restart_dir else time.strftime("%Y%m%d-%H%M%S")
    args.work_dir = os.path.join(args.work_dir, timestamp)
    return args


def parse_gpu_ids(gpu_ids):
    if not gpu_ids:
        return None
    ids = []
    for raw_id in gpu_ids.split(","):
        gpu_id = raw_id.strip()
        if not gpu_id:
            continue
        if not gpu_id.isdigit():
            raise ValueError(f"Invalid GPU id: {gpu_id}")
        ids.append(gpu_id)
    return ids or None


def get_free_gpus(allowed_gpu_ids=None):
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,index", "--format=csv,nounits,noheader"],
            universal_newlines=True,
        )
    except Exception as exc:
        print(f"GPU query failed: {exc}")
        return ["0"]

    gpus = []
    for line in output.strip().splitlines():
        free_mem, index = line.split(",")
        index = index.strip()
        if allowed_gpu_ids is not None and index not in allowed_gpu_ids:
            continue
        gpus.append((int(free_mem), int(index)))
    gpus.sort(reverse=True)
    return [str(index) for _, index in gpus]


def select_gpus(gpu_count, gpu_ids=""):
    allowed_gpu_ids = parse_gpu_ids(gpu_ids)
    available = get_free_gpus(allowed_gpu_ids)
    if allowed_gpu_ids and not available:
        raise RuntimeError(f"No requested GPUs are available: {','.join(allowed_gpu_ids)}")
    selected = available[:gpu_count] if gpu_count > 0 else []
    if selected:
        print(f"自动选择GPU设备: {','.join(selected)}")
    if torch.cuda.is_available():
        device_ids = [int(gpu_id) for gpu_id in selected] or [0]
        torch.cuda.set_device(device_ids[0])
        return device_ids
    return []


def set_seed(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        torch.cuda.manual_seed_all(args.seed)


def init_experiment(args):
    if args.name == "N/A" and not args.debug:
        raise ValueError("Please give a name to your run")
    print(f"Experiment name: {args.name}")
    return create_exp_dir(
        args.work_dir,
        scripts_to_save=["train_transformer.py", "models/deq_transformer.py", "../lib/solvers.py"],
        debug=args.debug,
    )


def load_corpus(args, device, device_ids):
    corpus = get_lm_corpus(args.data, args.dataset)
    args.n_token = len(corpus.vocab)
    eval_batch_size = max(16, len(device_ids))
    return corpus.get_iterator("valid", eval_batch_size, args.eval_tgt_len, device=device)


def init_weight(args, weight):
    if args.init == "uniform":
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == "normal":
        nn.init.normal_(weight, 0.0, args.init_std)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def make_weights_init(args):
    def weights_init(module):
        classname = module.__class__.__name__
        if "Linear" in classname or "Conv1d" in classname:
            if getattr(module, "weight", None) is not None:
                init_weight(args, module.weight)
            if getattr(module, "bias", None) is not None:
                init_bias(module.bias)
        elif "AdaptiveEmbedding" in classname and hasattr(module, "emb_projs"):
            for proj in module.emb_projs:
                if proj is not None:
                    nn.init.normal_(proj.weight, 0.0, args.proj_init_std)
        elif "Embedding" in classname and hasattr(module, "weight"):
            init_weight(args, module.weight)
        elif "ProjectedAdaptiveLogSoftmax" in classname:
            if getattr(module, "cluster_weight", None) is not None:
                init_weight(args, module.cluster_weight)
            if getattr(module, "cluster_bias", None) is not None:
                init_bias(module.cluster_bias)
            if hasattr(module, "out_projs"):
                for proj in module.out_projs:
                    if proj is not None:
                        nn.init.normal_(proj.weight, 0.0, args.proj_init_std)
        elif "LayerNorm" in classname:
            if getattr(module, "weight", None) is not None:
                nn.init.normal_(module.weight, 1.0, args.init_std)
            if getattr(module, "bias", None) is not None:
                init_bias(module.bias)
        elif "WeightShareSelfAttention" in classname:
            if hasattr(module, "r_w_bias"):
                init_weight(args, module.r_w_bias)
            if hasattr(module, "r_r_bias"):
                init_weight(args, module.r_r_bias)

    return weights_init


def build_model(args, device, device_ids, logging):
    cutoffs, tie_projs = [], [False]
    if args.adaptive:
        cutoffs = [20000, 40000, 200000]
        tie_projs += [True] * len(cutoffs)

    b_solver = None if args.b_solver == "None" else SOLVERS[args.b_solver]
    model = DEQTransformerLM(
        args.n_token,
        args.n_layer,
        args.eval_n_layer,
        args.n_head,
        args.d_model,
        args.d_head,
        args.d_inner,
        args.dropout,
        args.dropatt,
        tie_weights=args.tied,
        d_embed=args.d_embed,
        div_val=args.div_val,
        tie_projs=tie_projs,
        pre_lnorm=args.pre_lnorm,
        wnorm=args.wnorm,
        local_size=args.local_size,
        pretrain_steps=args.pretrain_steps,
        tgt_len=args.tgt_len,
        mem_len=args.mem_len,
        cutoffs=cutoffs,
        load=args.load,
        f_solver=SOLVERS[args.f_solver],
        b_solver=b_solver,
        stop_mode=args.stop_mode,
        logging=logging,
    )
    if not args.load:
        weights_init = make_weights_init(args)
        model.apply(weights_init)
        model.word_emb.apply(weights_init)

    args.n_all_param = sum(param.nelement() for param in model.parameters() if param.requires_grad)
    if args.multi_gpu and len(device_ids) > 1:
        model = model.to(device)
        if args.gpu0_bsz >= 0 and args.batch_size != args.gpu0_bsz * len(device_ids):
            return model, BalancedDataParallel(args.gpu0_bsz // args.batch_chunk, model, dim=1).to(device)
        return model, nn.DataParallel(model, device_ids=device_ids, dim=1).to(device)
    return model, model.to(device)


def log_args(args, logging):
    logging("=" * 100)
    for key, value in args.__dict__.items():
        logging(f"    - {key} : {value}")
    logging("=" * 100)


def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, list):
        return [to_cpu(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(to_cpu(item) for item in obj)
    if isinstance(obj, dict):
        return {key: to_cpu(value) for key, value in obj.items()}
    return obj


def find_trajectory_files(prefix):
    prefix_path = Path(prefix)
    parent = prefix_path.parent if str(prefix_path.parent) else Path(".")
    files = list(parent.glob(f"{prefix_path.name}_*.pt"))

    def suffix_num(path):
        try:
            return int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return 10**12

    return sorted(files, key=suffix_num)


def trajectory_cache_expected(args):
    return {
        "dataset": args.dataset,
        "trajectory_solver": args.trajectory_solver,
        "f_thres": args.f_thres,
        "max_eval_steps": args.max_eval_steps,
    }


def trajectory_cache_hit(args):
    if args.force_trajectory_regen:
        return False
    files = find_trajectory_files(args.trajectory_prefix)
    if not files:
        return False

    expected = trajectory_cache_expected(args)
    sample_files = [files[0]]
    if files[-1] != files[0]:
        sample_files.append(files[-1])

    for path in sample_files:
        try:
            traj_file = torch.load(path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(
                f"Existing trajectory cache is unreadable: {path}. "
                "Use --force-trajectory-regen to overwrite it."
            ) from exc
        if not isinstance(traj_file, list) or not traj_file or not isinstance(traj_file[0], dict):
            raise RuntimeError(
                f"Existing trajectory cache has invalid format: {path}. "
                "Use --force-trajectory-regen to overwrite it."
            )
        item = traj_file[0]
        for key, expected_value in expected.items():
            actual_value = item.get(key)
            if actual_value != expected_value:
                raise RuntimeError(
                    f"Existing trajectory cache mismatch in {path}: "
                    f"{key}={actual_value!r}, expected {expected_value!r}. "
                    "Use --force-trajectory-regen to overwrite it."
                )

    print(
        f"Trajectory cache hit for dataset={args.dataset}, "
        f"solver={args.trajectory_solver}: {len(files)} files under {args.trajectory_prefix}_*.pt"
    )
    return True


def clear_trajectory_cache(args):
    if not args.force_trajectory_regen:
        return
    for path in find_trajectory_files(args.trajectory_prefix):
        path.unlink()
        print(f"Removed existing trajectory cache file: {path}")


def evaluate(args, eval_iter, model, para_model, logging, save_trajectory=False, cm_load=None):
    train_step = int(1e9)
    model.eval()
    model.reset_length(args.eval_tgt_len, args.mem_len)

    total_len, total_loss = 0, 0.0
    rho_list = []
    if args.spectral_radius_mode:
        print("WARNING: spectral radius evaluation is very slow.")

    with torch.no_grad():
        mems = []
        traj_all = []
        last_i = -1
        for i, (data, target, seq_len) in enumerate(eval_iter):
            last_i = i
            if 0 < args.max_eval_steps <= i:
                break
            start = time.time()
            if save_trajectory:
                ret = para_model(
                    data,
                    target,
                    mems,
                    train_step=train_step,
                    f_thres=args.f_thres,
                    b_thres=args.b_thres,
                    compute_jac_loss=False,
                    spectral_radius_mode=args.spectral_radius_mode,
                    writer=None,
                    save_trajectory=True,
                    trajectory_solver=args.trajectory_solver,
                )
                loss, _, sradius, trajectory, mems = ret[0], ret[1], ret[2], ret[3], ret[4:]
                trajectory["dataset"] = args.dataset
                trajectory["f_thres"] = args.f_thres
                trajectory["max_eval_steps"] = args.max_eval_steps
                traj_all.append(to_cpu(trajectory))
                if i % 10 == 0 and i != 0:
                    filename = f"{args.trajectory_prefix}_{i // 10}.pt"
                    torch.save(traj_all, filename)
                    print(f"Saved trajectory data!, traj_all_len={len(traj_all)}")
                    traj_all = []
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                kwargs = {}
                if cm_load is not None:
                    kwargs["CM_load"] = cm_load
                    kwargs["CM_solver"] = args.trajectory_solver
                    kwargs["CM_compare_teacher"] = args.cm_compare_teacher
                ret = para_model(
                    data,
                    target,
                    mems,
                    train_step=train_step,
                    f_thres=args.f_thres,
                    b_thres=args.b_thres,
                    compute_jac_loss=False,
                    spectral_radius_mode=args.spectral_radius_mode,
                    writer=None,
                    **kwargs,
                )
                loss, _, sradius, _, mems = ret[0], ret[1], ret[2], ret[3], ret[4:]

            loss = loss.mean()
            if args.spectral_radius_mode:
                rho_list.append(sradius.mean().item())
            total_loss += seq_len * loss.float().item()
            total_len += seq_len
            print(f"i:{i}, Time: {(time.time() - start):.4f}, loss:{loss.float():.4f}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if save_trajectory and traj_all:
            filename = f"{args.trajectory_prefix}_{last_i // 10 + 1}.pt"
            torch.save(traj_all, filename)
            print(f"Saved final trajectory data!, traj_all_len={len(traj_all)}")

    if rho_list:
        logging(f"(Estimated) Spectral radius over validation set: {np.mean(rho_list)}")
    model.train()
    return total_loss / total_len


def module_of(model):
    return model.module if hasattr(model, "module") else model


def train_on_trajectory(args, cd, cd_ema, params_ema, optimizer, dataloader, n_epochs, max_t, epsilon):
    rel_diff_append = []
    loss_append = []
    tot_loss = 0.0

    with trange(n_epochs) as pbar:
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_total_steps = dataloader.dataset.tensors[0].size(1)
            if n_total_steps < 2:
                raise ValueError("Need at least two trajectory points for CM training")
            t_steps = [
                (epsilon ** (1 / 7) + (j / (n_total_steps - 1)) * (max_t ** (1 / 7) - epsilon ** (1 / 7))) ** 7
                for j in range(n_total_steps)
            ]
            n_steps = min(args.cm_train_points, n_total_steps)
            indices = torch.linspace(1, n_total_steps - 1, steps=n_steps).round().long().tolist()
            n_1 = [indices[random.randint(0, len(indices) - 1)] for _ in range(n_steps)]
            rel_diff = torch.tensor(0.0)

            for data in dataloader:
                x_batch = data[0]
                batch_size = x_batch.size(0)
                current_func_args = data[1:]

                tn_1 = torch.tensor([t_steps[i] for i in n_1]).to(x_batch.device)
                tn = torch.tensor([t_steps[i - 1] for i in n_1]).to(x_batch.device)
                x_tn_1 = x_batch[:, n_1]
                x_tn = x_batch[:, (np.array(n_1) - 1).tolist()]
                x_tn_prev = x_batch[:, np.maximum(np.array(n_1) - 2, 0).tolist()]

                with torch.no_grad():
                    out_tn = cd_ema(x_tn, x_tn_prev, tn.unsqueeze(0).expand(x_tn.size(0), -1), current_func_args)
                loss_1 = F.mse_loss(
                    cd(x_tn_1, x_tn, tn_1.unsqueeze(0).expand(x_tn.size(0), -1), current_func_args),
                    out_tn,
                )
                x_fixed = x_batch[:, -1].unsqueeze(1).expand(-1, n_steps, -1, -1)
                loss_2_x = cd(x_tn, x_tn_prev, tn.unsqueeze(0).expand(x_tn.size(0), -1), current_func_args)
                loss_2 = F.smooth_l1_loss(loss_2_x, x_fixed)
                loss = 0.1 * loss_1 + 0.9 * loss_2

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                params_ema = {
                    key: 0.98 * params_ema[key] + 0.02 * value
                    for key, value in module_of(cd).state_dict().items()
                }
                module_of(cd_ema).load_state_dict(params_ema)
                epoch_loss += loss.item()

                with torch.no_grad():
                    x_ini = x_batch[:, 0:1]
                    t_final = torch.tensor(t_steps[-1], device=x_batch.device)
                    x1 = cd_ema(x_ini, x_ini, t_final.unsqueeze(0).expand(batch_size, -1), current_func_args)
                    rel_diff = (x1 - x_batch[:, -1:]).norm() / x_batch[:, -1:].norm()
                    rel_diff_append.append(rel_diff.item())
                    loss_append.append(loss.item())

            tot_loss = epoch_loss / len(dataloader)
            pbar.set_postfix(epoch=epoch, loss=tot_loss, rel_diff=rel_diff.item())
            pbar.update()

    return tot_loss, rel_diff_append, loss_append, params_ema


def train_consistency_model(args, device, device_ids, func_params_dict):
    cd = ConsistencyFunction(
        n_head=args.n_head,
        d_model=args.d_model,
        d_head=args.d_head,
        d_inner=args.d_inner,
        dropout=args.dropout,
        n_layer=args.n_layer,
        func_args=None,
        solver=args.trajectory_solver,
    ).to(device)
    cd.func.load_state_dict(func_params_dict)
    optimizer = torch.optim.AdamW(cd.parameters(), lr=4e-3)
    cd_ema = ConsistencyFunction(
        n_head=args.n_head,
        d_model=args.d_model,
        d_head=args.d_head,
        d_inner=args.d_inner,
        dropout=args.dropout,
        n_layer=args.n_layer,
        func_args=None,
        solver=args.trajectory_solver,
    ).to(device)
    cd_ema.load_state_dict(cd.state_dict())
    params_ema = cd_ema.state_dict()

    if device.type == "cuda" and len(device_ids) > 1:
        cd = nn.DataParallel(cd, device_ids=device_ids, dim=0).to(device)
        cd_ema = nn.DataParallel(cd_ema, device_ids=device_ids, dim=0).to(device)

    if os.path.exists(args.cm_checkpoint):
        checkpoint = torch.load(args.cm_checkpoint, map_location=device)
        module_of(cd).load_state_dict(checkpoint["model"])
        module_of(cd_ema).load_state_dict(checkpoint["model_ema"])
        params_ema = checkpoint["params_ema"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Loaded checkpoint from {args.cm_checkpoint}")
    else:
        print(f"No checkpoint found at {args.cm_checkpoint}, starting from scratch.")

    all_trajectories = []
    for file_idx in range(args.cm_start_file_idx, args.cm_max_file_idx + 1):
        traj_path = f"{args.trajectory_prefix}_{file_idx}.pt"
        try:
            traj_file = torch.load(traj_path, map_location="cpu")
        except Exception as exc:
            print(f"无法加载文件 {traj_path}: {exc}")
            continue
        print(f"找到文件 {traj_path}，共有{len(traj_file)}条轨迹")
        for traj_idx in range(min(len(traj_file), args.cm_max_traj_per_file)):
            all_trajectories.append((file_idx, traj_idx))

    print(f"共收集到 {len(all_trajectories)} 条可用轨迹")
    if not all_trajectories:
        raise FileNotFoundError("No trajectory files found for CM training")

    random.seed(42)
    if len(all_trajectories) >= args.cm_num_samples:
        sampled_trajectories = random.sample(all_trajectories, args.cm_num_samples)
    else:
        sampled_trajectories = [random.choice(all_trajectories) for _ in range(args.cm_num_samples)]
    print(f"随机抽取了 {args.cm_num_samples} 条轨迹进行训练（可能包含重复轨迹）")

    best_loss = 1e9
    max_t, epsilon = 5, 0.002
    for idx, (file_idx, traj_idx) in enumerate(sampled_trajectories):
        print(f"正在处理第 {idx + 1}/{args.cm_num_samples} 条轨迹，来自文件 {args.trajectory_prefix}_{file_idx}.pt 中的第 {traj_idx} 条")
        traj = torch.load(f"{args.trajectory_prefix}_{file_idx}.pt", map_location=device)
        item = traj[traj_idx]
        saved_solver = item.get("trajectory_solver")
        if saved_solver is not None and saved_solver != args.trajectory_solver:
            raise ValueError(
                f"Trajectory solver mismatch: file uses {saved_solver}, "
                f"but --trajectory-solver is {args.trajectory_solver}"
            )
        if saved_solver is None:
            print(f"轨迹未记录solver，按 --trajectory-solver={args.trajectory_solver} 处理")
        saved_dataset = item.get("dataset")
        if saved_dataset is not None and saved_dataset != args.dataset:
            raise ValueError(
                f"Trajectory dataset mismatch: file uses {saved_dataset}, "
                f"but --dataset is {args.dataset}"
            )
        if saved_dataset is None:
            print(f"轨迹未记录dataset，按 --dataset={args.dataset} 处理")
        x_list = item["x_traj"]
        func_args = [item["func_args"][0], item["func_args"][1], item["func_args"][2]]
        x_traj = x_list.permute(1, 0, 2, 3)
        bsz = x_traj.shape[0]

        func_args[2] = func_args[2].unsqueeze(0).expand(bsz, *func_args[2].shape)
        dataset = TensorDataset(x_traj, *func_args)
        dataloader = DataLoader(
            dataset,
            batch_size=args.cm_batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator(device=device),
        )

        tot_loss, rel_diff_append, loss_append, params_ema = train_on_trajectory(
            args,
            cd,
            cd_ema,
            params_ema,
            optimizer,
            dataloader,
            args.cm_epochs,
            max_t,
            epsilon,
        )
        if args.plot_CM:
            import matplotlib.pyplot as plt
            plt.plot(rel_diff_append)
            plt.plot(loss_append)
            plt.show()

        checkpoint_dir = os.path.dirname(args.cm_checkpoint)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            {
                "model": module_of(cd).state_dict(),
                "model_ema": module_of(cd_ema).state_dict(),
                "params_ema": params_ema,
                "optimizer": optimizer.state_dict(),
            },
            args.cm_checkpoint,
        )
        if tot_loss < best_loss:
            best_loss = tot_loss
            torch.save(module_of(cd).state_dict(), args.cm_save)
            print(f"新的最佳模型{args.cm_save}已保存")

        with torch.no_grad():
            x_ini = x_list[0].unsqueeze(1)
            t = torch.tensor(max_t).to(device).unsqueeze(0).expand(bsz, -1)
            module_of(cd).load_state_dict(torch.load(args.cm_save, map_location=device))
            x = cd(x_ini, x_ini, t, func_args).squeeze(1)
            rel_diff = (x - x_traj[:, -1]).norm() / x_traj[:, -1].norm()
            print(f"Relative error: {rel_diff.item()}")

    print(f"所有 {args.cm_num_samples} 条轨迹训练完成，最佳模型已保存")


def log_valid_loss(logging, label, valid_loss):
    try:
        ppl = math.exp(valid_loss)
    except OverflowError:
        ppl = float("inf")
    logging("=" * 100)
    logging(f"| {label} | valid loss {valid_loss:5.2f} | valid ppl {ppl:9.3f}")
    logging("=" * 100)


def run(argv=None):
    args = parse_args(argv)
    if args.save_trajectory and trajectory_cache_hit(args):
        print("Trajectory generation skipped.")
        return 0
    device_ids = select_gpus(args.gpu_count, args.gpu_ids)
    args.cuda = torch.cuda.is_available()
    logging = init_experiment(args)
    set_seed(args)
    device = torch.device(f"cuda:{device_ids[0]}" if args.cuda and device_ids else "cpu")
    valid_iter = load_corpus(args, device, device_ids)
    model, para_model = build_model(args, device, device_ids, logging)
    log_args(args, logging)

    if args.save_trajectory:
        clear_trajectory_cache(args)
        if args.trajectory_solver == "picard":
            model.func.load_state_dict(torch.load(args.deq_func_load, map_location=device))
            print(f"Loaded Picard trajectory func weights from {args.deq_func_load}")
        valid_loss = evaluate(args, valid_iter, model, para_model, logging, save_trajectory=True)
        log_valid_loss(logging, "End of evaluating on validation set", valid_loss)
        print("Trajectory Data Saved!")
        return 0

    if args.train_CM:
        func_params_dict = torch.load(args.deq_func_load, map_location=device)
        train_consistency_model(args, device, device_ids, func_params_dict)
        valid_loss = evaluate(args, valid_iter, model, para_model, logging)
        log_valid_loss(logging, "End of training", valid_loss)
        return 0

    valid_loss = evaluate(args, valid_iter, model, para_model, logging, cm_load=args.cm_load if args.CM else None)
    log_valid_loss(logging, "End of evaluation", valid_loss)
    return 0


def main(argv=None):
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
