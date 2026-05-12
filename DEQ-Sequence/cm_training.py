import os
import random
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
import matplotlib.pyplot as plt

from models.deq_transformer_CD import ConsistencyFunction
from cm_plugin.core import CMPluginConfig, CMPathManager


def build_cm_config(args, task="sequence"):
    return CMPluginConfig(
        enabled=getattr(args, "cm_enable", False),
        dataset=getattr(args, "dataset", "default"),
        task=task,
        base_dir=getattr(args, "cm_save_dir", "cm_checkpoints"),
        load_path=getattr(args, "cm_load", None),
        save_dir=getattr(args, "cm_save_dir", None),
    )


def _train_on_trajectory(CD, CD_ema, params_ema, CD_optimizer, dataloader, N_EPOCHS, T, EPSILON, AA=True):
    rel_diff_append = []
    loss_append = []
    tot_loss = 0
    AA = True

    with trange(N_EPOCHS) as pbar:
        for epoch in range(N_EPOCHS):
            epoch_loss = 0.0
            N_steps = 38
            t_steps = [(EPSILON ** (1 / 7) + (j / (N_steps - 1)) * (T ** (1 / 7) - EPSILON ** (1 / 7))) ** 7
                       for j in range(0, N_steps)]
            n_steps = 8

            if AA:
                indices = torch.linspace(2, N_steps - 1, steps=n_steps).round().long().tolist()

            n_1 = []
            for _ in range(n_steps):
                n_1.append(indices[random.randint(0, len(indices) - 1)])

            for data in dataloader:
                x_batch = data[0]
                batch_size = x_batch.size(0)
                current_func_args = data[1:]

                tn_1 = torch.tensor([t_steps[i] for i in n_1]).to(x_batch.device)
                tn = torch.tensor([t_steps[i - 1] for i in n_1]).to(x_batch.device)

                z_tn_1 = x_batch[:, n_1]
                z_tn = x_batch[:, [i - 1 for i in n_1]]
                z_tn_minus_1 = x_batch[:, [i - 2 for i in n_1]]

                tn_1_exp = tn_1.unsqueeze(0).expand(batch_size, -1)
                tn_exp = tn.unsqueeze(0).expand(batch_size, -1)

                with torch.no_grad():
                    if AA:
                        out_tn = CD_ema(z_tn, z_tn_minus_1, tn_exp, current_func_args)
                    else:
                        out_tn = CD_ema(z_tn, tn_exp, current_func_args)

                if AA:
                    pred_tn_1 = CD(z_tn_1, z_tn, tn_1_exp, current_func_args)
                else:
                    pred_tn_1 = CD(z_tn_1, tn_1_exp, current_func_args)

                loss_1 = F.mse_loss(pred_tn_1, out_tn)

                if AA:
                    loss_2_x = CD(z_tn, z_tn_minus_1, tn_exp, current_func_args)
                else:
                    loss_2_x = CD(z_tn, tn_exp, current_func_args)

                x_fixed = x_batch[:, -1].unsqueeze(1).expand(-1, n_steps, -1, -1)
                loss_2 = F.mse_loss(loss_2_x, x_fixed)
                loss = 0.2 * loss_1 + 0.8 * loss_2

                loss.backward()
                CD_optimizer.step()
                CD_optimizer.zero_grad()

                params_ema = {k: 0.98 * params_ema[k] + 0.02 * v for k, v in CD.module.state_dict().items()}
                CD_ema.module.load_state_dict(params_ema)
                epoch_loss += loss.item()

                with torch.no_grad():
                    x_ini = x_batch[:, 0:1]
                    T_end = t_steps[-1] if isinstance(t_steps[-1], torch.Tensor) else torch.tensor(t_steps[-1], device=x_batch.device)
                    x1 = CD_ema(x_ini, T_end.unsqueeze(0).expand(batch_size, -1), current_func_args)
                    rel_diff = (x1 - x_batch[:, -1:]).norm() / x_batch[:, -1:].norm()
                    rel_diff_append.append(rel_diff.item())
                    loss_append.append(loss.item())

            tot_loss = epoch_loss / len(dataloader)
            pbar.set_postfix(epoch=epoch, loss=tot_loss, rel_diff=rel_diff.item())
            pbar.update()

    return tot_loss, rel_diff_append, loss_append


def train_cm(args, func_params_dict, device_ids, cm_paths: CMPathManager, CM_training=True):
    device = torch.device('cuda:0')

    CD = ConsistencyFunction(n_head=args.n_head, d_model=args.d_model, d_head=args.d_head, d_inner=args.d_inner,
                             dropout=args.dropout, n_layer=args.n_layer, func_args=None).to(device)
    CD.func.load_state_dict(func_params_dict)
    CD_optimizer = torch.optim.AdamW(CD.parameters(), lr=4e-3)
    CD_ema = ConsistencyFunction(n_head=args.n_head, d_model=args.d_model, d_head=args.d_head, d_inner=args.d_inner,
                                 dropout=args.dropout, n_layer=args.n_layer, func_args=None).to(device)

    CD_ema.load_state_dict(CD.state_dict())
    params_ema = CD_ema.state_dict()

    CD = nn.DataParallel(CD, device_ids=device_ids, dim=0).to(device)
    CD_ema = nn.DataParallel(CD_ema, device_ids=device_ids, dim=0).to(device)

    checkpoint_path = cm_paths.checkpoint_path()

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        CD.module.load_state_dict(checkpoint['model'])
        CD_ema.module.load_state_dict(checkpoint['model_ema'])
        params_ema = checkpoint['params_ema']
        CD_optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")

    T = 5
    EPSILON = 0.002
    N_EPOCHS = 50

    with open('traj.json', 'r') as f:
        traj_info = json.load(f)

    all_trajectories = []
    for item in traj_info["trajectories"]:
        file_name = item["file_name"]
        file_index = int(file_name.split("_")[-1].split(".")[0])
        traj_count = item["count"]

        for traj_idx in range(traj_count):
            all_trajectories.append((file_index, traj_idx))

    random.seed(42)
    num_samples = 100
    AA = True

    if len(all_trajectories) >= num_samples:
        sampled_trajectories = random.sample(all_trajectories, num_samples)
    else:
        sampled_trajectories = [random.choice(all_trajectories) for _ in range(num_samples)]

    print(f"Total available trajectories: {len(all_trajectories)}")
    print(f"Sampled {num_samples} trajectories for training (may include duplicates)")

    best_loss = 1e9

    for idx, (file_idx, traj_idx) in enumerate(sampled_trajectories):
        print(
            f"Processing trajectory {idx + 1}/{num_samples} from traj_all_{file_idx}.pt (index {traj_idx})"
        )

        traj = torch.load(f'traj_all_{file_idx}.pt', map_location=device)
        x_list = [traj[i]['x_traj'] for i in range(len(traj))]
        func_args = []
        for i in range(len(traj)):
            func_args.append([traj[i]['func_args'][0], traj[i]['func_args'][1], traj[i]['func_args'][2]])

        x_list = x_list[traj_idx]
        x_traj = x_list.permute(1, 0, 2, 3)
        func_args = func_args[traj_idx]

        x_traj[:, 0] = torch.randn_like(x_traj[:, 0])

        if True:
            bsz, n_samples, d_model, qlen = x_traj.shape[0], x_traj.shape[1], x_traj.shape[2], x_traj.shape[3]
            for i in range(bsz):
                for j in range(d_model):
                    for k in range(1, n_samples - 3):
                        if random.random() < 0.1:
                            x_traj[i, k, j] = x_traj[i, -1, j]

        bsz, n_samples, d_model, qlen = x_traj.shape[0], x_traj.shape[1], x_traj.shape[2], x_traj.shape[3]

        func_args[2] = func_args[2].unsqueeze(0).expand(bsz, *func_args[2].shape)
        dataset = TensorDataset(x_traj, *func_args)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True,
                                generator=torch.Generator(device=device))

        if CM_training:
            tot_loss, rel_diff_append, loss_append = _train_on_trajectory(
                CD, CD_ema, params_ema, CD_optimizer, dataloader, N_EPOCHS, T, EPSILON, AA
            )

            plotting = True
            if plotting:
                plt.plot(rel_diff_append)
                plt.plot(loss_append)
                plt.show()

            cm_paths.ensure_dir()
            checkpoint = {
                'model': CD.module.state_dict(),
                'model_ema': CD_ema.module.state_dict(),
                'params_ema': params_ema,
                'optimizer': CD_optimizer.state_dict(),
            }
            torch.save(checkpoint, str(checkpoint_path))

            if tot_loss < best_loss:
                best_loss = tot_loss
                best_path = cm_paths.best_model_path()
                torch.save(CD.module.state_dict(), str(best_path))
                print(f"Saved new best model: {best_path}")

        else:
            print("Training skipped. Loading the best model...")

        with torch.no_grad():
            x_ini = x_list[0].unsqueeze(1)
            t = torch.tensor(T).to(device).unsqueeze(0).expand(bsz, -1)
            best_path = cm_paths.best_model_path()
            CD.module.load_state_dict(torch.load(best_path, map_location=device))

            if AA:
                x = CD(x_ini, x_ini, t, func_args).squeeze(1)
            else:
                x = CD(x_ini, t, func_args).squeeze(1)
            x_rel = x_traj[:, -1]

            rel_diff = (x - x_rel).norm() / x_rel.norm()
            print(f"Relative error: {rel_diff.item()}")

    print(f"Finished training on {num_samples} trajectories. Best model saved.")
