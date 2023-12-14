import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils.dataset import BodyFluidDataset, fluid_list
from utils.metrics import eval_print


class DeepSec(nn.Module):
    filter_sizes = [1, 5, 7]
    num_filters = 50
    gru_hidden = 32
    gru_layer = 2
    fc_hidden = 16

    def __init__(self) -> None:
        super(DeepSec, self).__init__()
        self.filters = nn.ModuleList([
            nn.Conv1d(20, self.num_filters, (filter_size,), padding=(filter_size // 2,))
            for filter_size in self.filter_sizes
        ])
        self.gru = nn.GRU(
            self.num_filters * len(self.filter_sizes),
            self.gru_hidden,
            self.gru_layer,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(self.gru_hidden * 2, self.fc_hidden)
        self.out = nn.Linear(self.fc_hidden, 2)

    def forward(self, pssm, seq_len):
        pssm_t = torch.transpose(pssm, 1, 2)
        features = F.relu(
            torch.cat([filter(pssm_t) for filter in self.filters], dim=1)
        )
        features_t = torch.transpose(features, 1, 2)
        packed_features = nn.utils.rnn.pack_padded_sequence(
            features_t,
            seq_len,
            batch_first=True,
            enforce_sorted=False
        )
        gru, final = self.gru(packed_features, None)
        final_t = torch.transpose(final[-2:], 0, 1)
        context = torch.reshape(final_t, [final_t.shape[0], -1])
        fc = F.relu(
            self.fc(context)
        )
        return self.out(fc)


def train_and_predict(train_dl, train_dl_, valid_dl, test_dl, lr, num_iter, eval_size, device):
    net = DeepSec().to(device)
    optimizer = Adam(
        net.parameters(),
        lr=lr
    )
    loss_fn = nn.CrossEntropyLoss()

    best_loss = 10.
    valid_prob = None
    test_prob = None
    t0 = time.time()
    iter_idx = 1
    net.train()
    while iter_idx <= num_iter:
        for pssm, seq_len, label in train_dl:
            net.train()
            pssm, label = pssm.to(device), label.to(device)
            output = net(pssm, seq_len)
            loss = loss_fn(output, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            iter_idx += 1

            if iter_idx % eval_size == 0:
                net.eval()
                with torch.no_grad():
                    # evaluate on train dataset
                    train_loss = torch.zeros(1, dtype=torch.float32, device=device)
                    train_acc = torch.zeros(1, dtype=torch.float32, device=device)
                    for [pssm, seq_len, label] in train_dl_:
                        pssm, label = pssm.to(device), label.to(device)
                        output = net(pssm, seq_len)
                        probs = torch.softmax(output, dim=-1)
                        predict = torch.argmax(probs, dim=-1)
                        loss = loss_fn(output, label)
                        train_loss += loss
                        correct = torch.eq(predict, label)
                        train_acc += correct.sum()
                    train_loss /= len(train_dl_)
                    train_acc /= len(train_dl_.dataset)
                    train_loss = train_loss.item()
                    train_acc = train_acc.item()

                    # evaluate on test dataset
                    valid_loss = torch.zeros(1, dtype=torch.float32, device=device)
                    valid_acc = torch.zeros(1, dtype=torch.float32, device=device)
                    tmp_prob = []
                    for [pssm, seq_len, label] in valid_dl:
                        pssm, label = pssm.to(device), label.to(device)
                        output = net(pssm, seq_len)
                        probs = torch.softmax(output, dim=-1)
                        predict = torch.argmax(probs, dim=-1)
                        loss = loss_fn(output, label)
                        valid_loss += loss
                        correct = torch.eq(predict, label)
                        valid_acc += correct.sum()
                        tmp_prob.append(probs[:, 1])
                    valid_loss /= len(valid_dl)
                    valid_acc /= len(valid_dl.dataset)
                    tmp_prob = torch.cat(tmp_prob, dim=0).cpu()
                    valid_loss = valid_loss.item()
                    valid_acc = valid_acc.item()

                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        valid_prob = tmp_prob
                        test_prob = torch.cat(
                            [
                                torch.softmax(net(pssm.to(device), seq_len), dim=-1)[:, 1]
                                for pssm, seq_len, label in test_dl
                            ],
                            dim=0
                        ).cpu()

                t = time.time() - t0
                t0 = time.time()
                net.train()
                print('[iter {:05d} {:.0f}s] train loss({:.4f}) acc({:.4f}); valid loss({:.4f}), acc({:.4f})'
                      .format(iter_idx, t, train_loss, train_acc, valid_loss, valid_acc))
    return valid_prob, test_prob


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # lr parameters for different body fluids
    lr_list = [1e-4] * 17
    lr_list[4] = 1e-5
    lr_list[10] = 1e-5
    lr_list[12] = 1e-5
    lr_list[13] = 1e-5

    fluid_list14 = ['Plasma', 'Saliva', 'Urine', 'CSF', 'Seminal', 'Amniotic', 'Tear',
                    'BALF', 'Milk', 'Synovial', 'NAF', 'PE', 'Sputum', 'Sweat']

    save_dict = {}
    for fluid, lr in zip(fluid_list14, lr_list):
        print('Train DeepSec model for', fluid)
        torch.cuda.empty_cache()

        tr_pos_ds = BodyFluidDataset(args.folder, 'train1', fluid, class_type='P', with_len=True)
        tr_neg_ds = BodyFluidDataset(args.folder, 'train1', fluid, class_type='N', with_len=True)
        base_ds = BodyFluidDataset(args.folder, 'train1', fluid, class_type='PN', with_len=True)
        train_ds = BodyFluidDataset(args.folder, 'train1', fluid, class_type='PN', with_len=True)
        valid_ds = BodyFluidDataset(args.folder, 'train2', fluid, class_type='PN', with_len=True)
        test_ds = BodyFluidDataset(args.folder, 'test', fluid, class_type='PN', with_len=True)

        train_dl = DataLoader(base_ds, args.bs, shuffle=True, pin_memory=True)
        train_dl_ = DataLoader(train_ds, args.bs, shuffle=False, pin_memory=True)
        valid_dl = DataLoader(valid_ds, args.bs, shuffle=False, pin_memory=True)
        test_dl = DataLoader(test_ds, args.bs, shuffle=False, pin_memory=True)

        pos_index = tr_pos_ds.index
        neg_index = tr_neg_ds.index
        num_pos_samples = len(pos_index)
        num_neg_samples = len(neg_index)

        pn_rate = float(num_pos_samples) / num_neg_samples
        np_rate = 1 / pn_rate
        pn_rate_r = round(pn_rate)
        np_rate_r = round(np_rate)

        assert 0 <= pn_rate_r < 11 and 0 <= np_rate_r < 11

        if pn_rate_r == 1 and np_rate_r == 1:
            valid_prob, test_prob = train_and_predict(
                train_dl,
                train_dl_,
                valid_dl,
                test_dl,
                lr,  # learning rate for each body fluid
                args.num_iter,
                args.eval_size,
                device
            )
        else:
            pos_index = np.random.permutation(pos_index)
            neg_index = np.random.permutation(neg_index)

            split_pos = pn_rate_r > 1
            n_estimators = pn_rate_r if split_pos else np_rate_r
            each_samples = num_pos_samples // pn_rate_r if split_pos else num_neg_samples // np_rate_r

            valid_prob_list = []
            test_prob_list = []
            for i in range(n_estimators):
                start_idx = i * each_samples
                end_idx = start_idx + each_samples if i < n_estimators - 1 else -1
                if split_pos:
                    each_pos_index = pos_index[start_idx:end_idx]
                    each_neg_index = neg_index
                else:
                    each_pos_index = pos_index
                    each_neg_index = neg_index[start_idx:end_idx]
                each_index = np.concatenate([each_pos_index, each_neg_index], axis=0)
                each_label = np.concatenate(
                    [
                        np.ones_like(each_pos_index, dtype=np.int64),
                        np.zeros_like(each_neg_index, dtype=np.int64)
                    ],
                    axis=0
                )
                each_train_ds = BodyFluidDataset(args.folder, 'train1', args.fluid, 'all', with_len=True)
                each_train_ds.index = each_index
                each_train_ds.label = each_label
                each_train_dl = DataLoader(each_train_ds, args.bs, shuffle=True, pin_memory=True)
                each_valid_prob, each_test_prob = train_and_predict(
                    each_train_dl,
                    train_dl_,
                    valid_dl,
                    test_dl,
                    args.lr,
                    args.num_iter,
                    args.eval_size,
                    device
                )
                valid_prob_list.append(each_valid_prob)
                test_prob_list.append(each_test_prob)
            valid_prob = torch.mean(
                torch.stack(valid_prob_list),
                dim=0
            )
            test_prob = torch.mean(
                torch.stack(test_prob_list),
                dim=0
            )

        valid_prob = valid_prob.numpy()
        test_prob = test_prob.numpy()
        valid_predict = (valid_prob > 0.5).astype(np.int32)
        test_predict = (test_prob > 0.5).astype(np.int32)
        valid_label = valid_ds.get_label()
        test_label = test_ds.get_label()
        eval_print(valid_label, valid_predict, valid_prob, '{:s}-valid'.format(args.fluid))
        eval_print(test_label, test_predict, test_prob, '{:s}-test '.format(args.fluid))

        # Save the prediction
        save_dict[fluid + '-score'] = test_prob
        save_dict[fluid + '-pred'] = test_predict

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, 'deepsec.npz')
    np.savez_compressed(save_path, **save_dict)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default=r'D:\datasets\BodyFluidData-20220912', type=str)
    parser.add_argument('--save-dir', default='tmp', type=str)
    parser.add_argument('--num-iter', default=20000, type=int)
    parser.add_argument('--bs', default=256, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--eval-size', default=1000, type=int)
    parser.add_argument('--seed', default=43215, type=int)

    args = parser.parse_args()
    main(args)
