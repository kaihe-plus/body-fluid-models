import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List
from cvxopt import matrix
from cvxopt.solvers import qp, options
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils.balanced_sampling import BalancedSampling
from utils.dataset import BodyFluidDataset, fluid_list
from utils.metrics import eval_print


class MultiSecNet(nn.Module):

    def __init__(self, in_features: int = 20, filter_sizes: List[int] = None,
                 num_filters: int = 128, fc_dim: int = 64) -> None:
        super(MultiSecNet, self).__init__()
        if filter_sizes is None:
            filter_sizes = [3, 5, 7, 9]

        self.in_features = in_features
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.fc_dim = fc_dim

        self.kernels = nn.ModuleList([
            nn.Conv1d(in_features, num_filters, (kernel_size,), padding=kernel_size // 2)
            for kernel_size in filter_sizes
        ])
        self.fc_linear = nn.Linear(num_filters * len(filter_sizes), fc_dim)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.transpose(x, 1, 2)
        out = [kernel(out) for kernel in self.kernels]
        out = torch.cat(out, dim=1)
        out = F.relu(out)
        out = F.max_pool1d(out, out.shape[2])
        out = torch.squeeze(out, dim=-1)
        out = self.fc_linear(out)
        out = F.relu(out)
        return out


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    options['show_progress'] = False

    num_tasks = len(fluid_list)

    tr_pos_ds = [BodyFluidDataset(args.folder, 'train1', fluid, class_type='P') for fluid in fluid_list]
    tr_neg_ds = [BodyFluidDataset(args.folder, 'train1', fluid, class_type='N') for fluid in fluid_list]
    train_ds = [BodyFluidDataset(args.folder, 'train1', fluid, class_type='PN') for fluid in fluid_list]
    valid_ds = [BodyFluidDataset(args.folder, 'train2', fluid, class_type='PN') for fluid in fluid_list]
    test_ds = [BodyFluidDataset(args.folder, 'test', fluid, class_type='PN') for fluid in fluid_list]

    train_dl = BalancedSampling(tr_pos_ds, tr_neg_ds, args.bs, pin_memory=True, drop_last=True)
    train_dl_ = [DataLoader(ds, args.bs, shuffle=False, pin_memory=True) for ds in train_ds]
    valid_dl = [DataLoader(ds, args.bs, shuffle=False, pin_memory=True) for ds in valid_ds]
    test_dl = [DataLoader(ds, args.bs, shuffle=False, pin_memory=True) for ds in test_ds]

    num_samples_tr = torch.tensor([len(ds) for ds in train_ds], device=device)
    num_samples_va = torch.tensor([len(ds) for ds in valid_ds], device=device)

    num_batches_tr = torch.tensor([len(dl) for dl in train_dl_], device=device)
    num_batches_va = torch.tensor([len(dl) for dl in valid_dl], device=device)

    feature_fn = MultiSecNet(
        filter_sizes=args.filter_sizes,
        num_filters=args.num_filters,
        fc_dim=args.fc_dim
    )
    nets = nn.ModuleList([
        nn.Sequential(
            feature_fn,
            nn.Linear(feature_fn.fc_dim, 2)
        ) for _ in range(num_tasks)
    ]).to(device)
    optimizer = Adam(
        nets.parameters(),
        args.lr
    )
    loss_fn = nn.CrossEntropyLoss()
    feature_params = list(feature_fn.parameters())

    # MGDA optimization
    opt_h = opt_q = matrix(
        np.zeros(num_tasks)
    )
    opt_G = matrix(
        -np.eye(num_tasks)
    )
    opt_A = matrix(
        np.ones([1, num_tasks])
    )
    opt_b = matrix([1.])

    best_loss_list = [10.] * num_tasks
    valid_prob_list = [np.nan] * num_tasks
    test_prob_list = [np.nan] * num_tasks
    t0 = time.time()
    iter_idx = 1
    nets.train()
    while iter_idx <= args.num_iter:
        for data_list in train_dl:
            nets.train()
            loss_list = []
            grad_list = []
            for task_idx, (data, label) in enumerate(data_list):
                data, label = data.to(device), label.to(device)
                output = nets[task_idx](data)
                loss = loss_fn(output, label)
                loss_list.append(loss)
                grad = torch.cat(
                    [torch.ravel(g) for g in torch.autograd.grad(loss, feature_params, retain_graph=True)],
                    dim=0
                )
                grad_list.append(grad)
            loss_tensor = torch.stack(loss_list)
            grad_tensor = torch.stack(grad_list)
            grad_mat = grad_tensor @ grad_tensor.T
            mat_array = grad_mat.cpu().numpy()
            # solve optimization
            opt_P = matrix(mat_array.astype(np.double))
            res = qp(opt_P, opt_q, opt_G, opt_h, opt_A, opt_b)
            if res is None:
                continue
            sol = res['x']
            sol_tensor = torch.as_tensor(
                np.array(sol),
                dtype=torch.float32,
                device=device
            )
            dot_loss = torch.dot(
                torch.squeeze(sol_tensor, 1),
                loss_tensor
            )
            optimizer.zero_grad(set_to_none=True)
            dot_loss.backward()
            optimizer.step()
            iter_idx += 1

            if iter_idx % args.eval_size == 0:
                nets.eval()
                with torch.no_grad():
                    # evaluate on train dataset
                    train_loss = torch.zeros(num_tasks, dtype=torch.float32, device=device)
                    train_acc = torch.zeros(num_tasks, dtype=torch.float32, device=device)
                    for task_idx, dl in enumerate(train_dl_):
                        for data, label in dl:
                            data, label = data.to(device), label.to(device)
                            output = nets[task_idx](data)
                            probs = torch.softmax(output, dim=-1)
                            predict = torch.argmax(probs, dim=-1)
                            loss = loss_fn(output, label)
                            train_loss[task_idx] += loss
                            correct = torch.eq(predict, label)
                            train_acc[task_idx] += correct.sum()
                    train_loss /= num_batches_tr
                    train_acc /= num_samples_tr
                    mean_train_loss = torch.mean(train_loss).item()
                    mean_train_acc = torch.mean(train_acc).item()

                    # evaluate on valid dataset
                    valid_loss = torch.zeros(num_tasks, dtype=torch.float32, device=device)
                    valid_acc = torch.zeros(num_tasks, dtype=torch.float32, device=device)
                    temp_prob_list = []
                    for task_idx, dl in enumerate(valid_dl):
                        valid_prob = []
                        for data, label in dl:
                            data, label = data.to(device), label.to(device)
                            output = nets[task_idx](data)
                            probs = torch.softmax(output, dim=-1)
                            predict = torch.argmax(probs, dim=-1)
                            loss = loss_fn(output, label)
                            valid_loss[task_idx] += loss
                            correct = torch.eq(predict, label)
                            valid_acc[task_idx] += correct.sum()
                            valid_prob.append(probs[:, 1])
                        valid_prob = torch.cat(valid_prob, dim=0).cpu()
                        temp_prob_list.append(valid_prob)
                    valid_loss /= num_batches_va
                    valid_acc /= num_samples_va
                    mean_valid_loss = torch.mean(valid_loss).item()
                    valid_loss_list = valid_loss.tolist()
                    valid_acc_list = valid_acc.tolist()

                    for task_idx in range(num_tasks):
                        valid_loss = valid_loss_list[task_idx]
                        if valid_loss < best_loss_list[task_idx]:
                            best_loss_list[task_idx] = valid_loss
                            valid_prob_list[task_idx] = temp_prob_list[task_idx]
                            test_prob = torch.cat(
                                [
                                    torch.softmax(nets[task_idx](data.to(device)), dim=-1)[:, 1]
                                    for data, _ in test_dl[task_idx]
                                ],
                                dim=0
                            ).cpu()
                            test_prob_list[task_idx] = test_prob

                    t = time.time() - t0
                    t0 = time.time()
                    nets.train()
                    print_str = '[iter {:05d} {:.0f}s] train mean loss({:.4f}), mean acc({:.4f}); ' \
                                'valid mean loss({:.4f}), acc({:.4f}' + ',{:.4f}' * (num_tasks - 1) + ')'
                    print(print_str.format(
                        iter_idx, t, mean_train_loss, mean_train_acc, mean_valid_loss, *valid_acc_list
                    ))

    valid_score_list = []
    test_score_list = []
    for task_idx in range(num_tasks):
        valid_prob = valid_prob_list[task_idx].numpy()
        valid_predict = (valid_prob > 0.5).astype(np.int32)
        valid_label = valid_ds[task_idx].get_label()
        valid_score = eval_print(valid_label, valid_predict, valid_prob, '{:s}-valid'.format(fluid_list[task_idx]))
        valid_score_list.append(valid_score)

        test_prob = test_prob_list[task_idx].numpy()
        test_predict = (test_prob > 0.5).astype(np.int32)
        test_label = test_ds[task_idx].get_label()
        test_score = eval_print(test_label, test_predict, test_prob, '{:s}-test '.format(fluid_list[task_idx]))
        test_score_list.append(test_score)
    valid_score_tensor = np.array(valid_score_list)
    test_score_tensor = np.array(test_score_list)
    mean_valid_score = np.mean(valid_score_tensor, axis=0)
    mean_test_score = np.mean(test_score_tensor, axis=0)

    print('Valid performance: Mean ACC({:.6f}), Mean F1({:.6f}), Mean MCC({:.6f}), Mean AUC({:.6f})'
          .format(*mean_valid_score))
    print('Test  performance: Mean ACC({:.6f}), Mean F1({:.6f}), Mean MCC({:.6f}), Mean AUC({:.6f})'
          .format(*mean_test_score))

    # Save test prediction for all body fluids
    save_dict = {}

    for fluid, test_prob in zip(fluid_list, test_prob_list):

        test_score = test_prob.numpy()
        test_pred = (test_score > 0.5).astype(np.int32)

        save_dict[fluid + '-score'] = test_score
        save_dict[fluid + '-pred'] = test_pred

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, 'multisec.npz')
    np.savez_compressed(save_path, **save_dict)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default=r'D:\datasets\BodyFluidData-20220912', type=str)
    parser.add_argument('--save-dir', default='tmp', type=str)
    parser.add_argument('--filter-sizes', default=[3, 5, 7, 9], type=int, nargs='+')
    parser.add_argument('--num-filters', default=128, type=int)
    parser.add_argument('--fc-dim', default=32, type=int)
    parser.add_argument('--num-iter', default=20000, type=int)
    parser.add_argument('--bs', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--eval-size', default=1000, type=int)
    parser.add_argument('--seed', default=43215, type=int)

    args = parser.parse_args()
    main(args)
