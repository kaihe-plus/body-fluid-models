import os
import random
import time
from math import ceil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils.dataset import SingleProData, BodyFluidDataset, fluid_list
from utils.metrics import eval_mean_print, select_threshold_by_mcc


class PUNet(nn.Module):

    def __init__(
            self,
            in_features: int = 20,
            num_filters: int = 64,
            num_conv_layers: int = 4,
            num_pool: int = 16,
            fc_dim: int = 32,
            num_out_layers: int = 20,
    ) -> None:
        super(PUNet, self).__init__()

        self.in_features = in_features
        self.num_pool = num_pool
        self.num_out_layers = num_out_layers

        self.kernels = nn.ModuleList([
            nn.Conv1d(in_features, num_filters, (3,), padding=1)
        ])

        in_channels = num_filters
        for i in range(1, num_conv_layers):
            self.kernels.append(
                nn.Conv1d(in_channels, in_channels // 2, (3,), padding=1)
            )
            in_channels //= 2
            num_filters += in_channels

        self.fc_linear = nn.Linear(num_filters * num_pool, fc_dim)
        self.out_linear = nn.Linear(fc_dim, 2 * num_out_layers)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.transpose(x, 1, 2)
        out_list = []
        for kernel in self.kernels:
            out = F.relu(
                kernel(out)
            )
            out_list.append(
                out
            )
        out = torch.concat(out_list, dim=1)
        out, _ = torch.topk(out, self.num_pool, dim=-1, sorted=False)
        out = torch.flatten(out, 1)
        out = F.relu(
            self.fc_linear(out)
        )
        out = self.out_linear(out)
        out = torch.reshape(out, [-1, self.num_out_layers, 2])
        return out


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    num_tasks = len(fluid_list)

    pro_data = SingleProData(args.folder, False, False, True)

    train_ds_list = []
    train_dl_list = []
    for fluid in fluid_list:
        p_ds = BodyFluidDataset(args.folder, 'train1', fluid, class_type='P')
        p_dl = DataLoader(p_ds, args.bs, shuffle=True, pin_memory=True)

        train_ds_list.append(p_ds)
        train_dl_list.append(p_dl)

        n_index = pro_data.splits_data[fluid]['tr_neg']
        unknown_index = pro_data.splits_data[fluid]['tr_unknown']
        u_index = np.concatenate([n_index, unknown_index], axis=0)

        u_len = len(u_index)
        num_samples = ceil(float(u_len) / args.num_splits)
        for _ in range(args.num_times):
            tmp_index = np.copy(u_index)
            np.random.shuffle(tmp_index)
            for i in range(args.num_splits):
                start = i * num_samples
                end = start + num_samples
                if end > u_len:
                    end = u_len
                select_index = tmp_index[start:end]

                u_ds = BodyFluidDataset(args.folder, 'train1', fluid, class_type='U')
                u_ds.index = select_index
                u_ds.label = np.zeros_like(select_index, dtype=u_ds.label.dtype)

                u_dl = DataLoader(u_ds, args.bs, shuffle=True, pin_memory=True)

                train_ds_list.append(u_ds)
                train_dl_list.append(u_dl)

    train_ds = [BodyFluidDataset(args.folder, 'train1', fluid, class_type='PU') for fluid in fluid_list]
    valid_ds = [BodyFluidDataset(args.folder, 'train2', fluid, class_type='PU') for fluid in fluid_list]
    test_ds = [BodyFluidDataset(args.folder, 'test', fluid, class_type='PU') for fluid in fluid_list]

    y_true_va_list = [ds.get_label() for ds in valid_ds]

    train_dl = [DataLoader(ds, args.bs, shuffle=False, pin_memory=True) for ds in train_ds]
    valid_dl = [DataLoader(ds, args.bs, shuffle=False, pin_memory=True) for ds in valid_ds]
    test_dl = [DataLoader(ds, args.bs, shuffle=False, pin_memory=True) for ds in test_ds]

    net = PUNet(
        num_filters=args.num_filters,
        num_conv_layers=args.num_conv_layers,
        num_pool=args.num_pool,
        fc_dim=args.fc_dim,
        num_out_layers=num_tasks * args.num_times * args.num_splits
    ).to(device)

    optimizer = Adam(
        net.parameters(),
        args.lr
    )
    loss_fn = nn.CrossEntropyLoss()

    best_ap_list = [0.] * num_tasks
    y_score_va_list = [np.nan] * num_tasks
    y_score_te_list = [np.nan] * num_tasks

    num_data = args.num_times * args.num_splits + 1
    output_shape = [-1, num_tasks, args.num_times * args.num_splits, 2]

    t0 = time.time()
    iter_idx = 1
    net.train()
    while iter_idx <= args.num_iter:
        for data_list in zip(*train_dl_list):

            loss_list = []
            for task_idx in range(num_tasks):
                start = task_idx * num_data
                end = start + num_data

                p_data, p_label = data_list[start]
                p_data, p_label = p_data.to(device), p_label.to(device)
                p_outputs = torch.reshape(
                    net(p_data),
                    output_shape
                )

                n_data_list_i = data_list[start+1: end]
                for neg_idx, (n_data, n_label) in enumerate(n_data_list_i):
                    n_data, n_label = n_data.to(device), n_label.to(device)
                    n_outputs = torch.reshape(
                        net(n_data),
                        output_shape
                    )

                    output = torch.concat(
                        [p_outputs[:, task_idx, neg_idx], n_outputs[:, task_idx, neg_idx]],
                        dim=0
                    )
                    label = torch.concat([p_label, n_label], dim=0)
                    loss = loss_fn(output, label)
                    loss_list.append(loss)

            loss_tensor = torch.stack(loss_list)
            mean_loss = torch.mean(loss_tensor)

            optimizer.zero_grad(set_to_none=True)
            mean_loss.backward()
            optimizer.step()
            if iter_idx > args.num_iter:
                break

            if iter_idx % args.eval_size == 0:
                net.eval()
                with torch.no_grad():
                    # evaluate on train dataset
                    train_loss = torch.zeros(num_tasks, dtype=torch.float32, device=device)
                    train_acc = torch.zeros(num_tasks, dtype=torch.float32, device=device)

                    for task_idx, dl in enumerate(train_dl):
                        for data, label in dl:
                            data, label = data.to(device), label.to(device)
                            outputs = torch.reshape(
                                net(data),
                                output_shape
                            )[:, task_idx]
                            outputs_reshape = torch.reshape(outputs, [-1, 2])
                            probs = torch.softmax(outputs_reshape, dim=1)
                            probs_reshape = torch.reshape(
                                probs,
                                [len(data), -1, 2]
                            )
                            mean_probs = torch.mean(probs_reshape, dim=1)
                            predict = torch.argmax(mean_probs, dim=1)

                            loss = loss_fn(mean_probs, label)
                            train_loss[task_idx] += loss

                            correct = torch.eq(predict, label)
                            train_acc[task_idx] += torch.sum(correct)

                        train_loss[task_idx] /= len(dl)
                        train_acc[task_idx] /= len(train_ds[task_idx])
                    
                    mean_train_loss = torch.mean(train_loss).item()
                    mean_train_acc = torch.mean(train_acc).item()

                    # evaluate on valid dataset
                    valid_loss = torch.zeros(num_tasks, dtype=torch.float32, device=device)
                    valid_acc = torch.zeros(num_tasks, dtype=torch.float32, device=device)
                    valid_ap_list = []

                    tmp_score_list = []
                    for task_idx, dl in enumerate(valid_dl):
                        prob_list = []
                        for data, label in dl:
                            data, label = data.to(device), label.to(device)
                            outputs = torch.reshape(
                                net(data),
                                output_shape
                            )[:, task_idx]
                            outputs_reshape = torch.reshape(outputs, [-1, 2])
                            probs = torch.softmax(outputs_reshape, dim=1)
                            probs_reshape = torch.reshape(
                                probs,
                                [len(data), -1, 2]
                            )
                            mean_probs = torch.mean(probs_reshape, dim=1)
                            predict = torch.argmax(mean_probs, dim=1)

                            prob_list.append(mean_probs)

                            loss = loss_fn(mean_probs, label)
                            valid_loss[task_idx] += loss

                            correct = torch.eq(predict, label)
                            valid_acc[task_idx] += torch.sum(correct)

                        prob_tensor = torch.concat(prob_list, dim=0)
                        valid_loss[task_idx] /= len(dl)
                        valid_acc[task_idx] /= len(valid_ds[task_idx])

                        tmp_score = prob_tensor[:, 1].cpu().numpy()
                        tmp_score_list.append(tmp_score)

                        ap_score = average_precision_score(
                            y_true_va_list[task_idx],
                            tmp_score
                        )
                        valid_ap_list.append(ap_score)

                    mean_valid_loss = torch.mean(valid_loss).item()
                    mean_valid_acc = torch.mean(valid_acc).item()
                    mean_valid_ap = np.array(valid_ap_list).mean()

                    for task_idx in range(num_tasks):
                        if valid_ap_list[task_idx] > best_ap_list[task_idx]:
                            best_ap_list[task_idx] = valid_ap_list[task_idx]
                            y_score_va_list[task_idx] = tmp_score_list[task_idx]
                            prob_list = []
                            for data, _ in test_dl[task_idx]:
                                data = data.to(device)
                                outputs = torch.reshape(
                                    net(data),
                                    output_shape
                                )[:, task_idx]
                                outputs_reshape = torch.reshape(outputs, [-1, 2])
                                probs = torch.softmax(outputs_reshape, dim=1)
                                probs_reshape = torch.reshape(
                                    probs,
                                    [len(data), -1, 2]
                                )
                                mean_probs = torch.mean(probs_reshape, dim=1)
                                prob_list.append(mean_probs)

                            prob_tensor = torch.concat(prob_list, dim=0)
                            y_score_te_list[task_idx] = prob_tensor[:, 1].cpu().numpy()

                    t = time.time() - t0
                    t0 = time.time()
                    net.train()
                    print('[iter {:05d} {:.0f}s] Train mean loss({:.4f}), ACC({:.4f}); Valid mean loss({:.4f}), '
                          'ACC({:.4f}), AP({:.4f})'
                          .format(iter_idx, t, mean_train_loss, mean_train_acc, mean_valid_loss, mean_valid_acc, mean_valid_ap))

            iter_idx += 1

    # evaluate the performance on test dataset
    y_true_te_list = [ds.get_label() for ds in test_ds]

    # threshold
    thd_list = [
        select_threshold_by_mcc(y_true, y_score)
        for y_true, y_score in zip(y_true_va_list, y_score_va_list)
    ]

    y_pred_va_list = [np.asarray(score > thd, int) for thd, score in zip(thd_list, y_score_va_list)]
    y_pred_te_list = [np.asarray(score > thd, int) for thd, score in zip(thd_list, y_score_te_list)]

    eval_mean_print(y_true_va_list, y_pred_va_list, y_score_va_list, 'Mean valid')
    eval_mean_print(y_true_te_list, y_pred_te_list, y_score_te_list, 'Mean test ')

    save_dict = {}
    for fluid, y_score_te, y_pred_te in zip(fluid_list, y_score_te_list, y_pred_te_list):
        save_dict[fluid + '-score'] = y_score_te
        save_dict[fluid + '-pred'] = y_pred_te

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, 'mpusec.npz')
    np.savez_compressed(save_path, **save_dict)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default=r'D:\datasets\BodyFluidData-20220912', type=str)
    parser.add_argument('--save-dir', default='tmp', type=str)
    parser.add_argument('--num_times', default=1, type=int)
    parser.add_argument('--num_splits', default=4, type=int)
    parser.add_argument('--num_conv_layers', default=4, type=int)
    parser.add_argument('--num_pool', default=8, type=int)
    parser.add_argument('--num_filters', default=128, type=int)
    parser.add_argument('--fc_dim', default=32, type=int)
    parser.add_argument('--num_iter', default=40000, type=int)
    parser.add_argument('--bs', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--eval_size', default=2000, type=int)
    parser.add_argument('--seed', default=43215, type=int)

    args = parser.parse_args()
    main(args)
