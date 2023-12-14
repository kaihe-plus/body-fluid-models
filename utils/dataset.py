import os.path as osp
import pickle

import numpy as np
from sklearn.preprocessing import scale
from torch.utils.data import Dataset

num_attr_features = 1610
fluid_list = [
    'Plasma', 'Saliva', 'Urine', 'CSF', 'Seminal', 'Amniotic',
    'Tear', 'BALF', 'Milk', 'Synovial', 'NAF', 'CVF', 'PE',
    'Sputum', 'EBC', 'PJ', 'Sweat'
]


class SingleProData:
    _instance = None
    attr_data = None
    seq_data = None
    pssm_data = None
    acc_data = None
    aa_dict = {
        'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
        'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
        'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
        'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
    }

    def __new__(cls, root: str, load_attr: bool = True, load_seq: bool = False, load_pssm: bool = False):
        if cls._instance is None:
            cls._instance = object.__new__(cls)

            # load data only once
            if load_attr:
                with open(osp.join(root, 'pro-attr.pkl'), 'rb') as f:
                    attr_data = pickle.load(f)
                    attr_data = scale(attr_data)
                    cls.attr_data = attr_data.astype(np.float32)

            if load_seq:
                with open(osp.join(root, 'pro-seq.pkl'), 'rb') as f:
                    seq_data = pickle.load(f)

                    aa_map = np.eye(20, 20, dtype=np.float32)

                    seq_list = []
                    for iter_seq in seq_data:
                        iter_seq_list = np.array([cls.aa_dict[aa] for aa in iter_seq])
                        # iter_seq_array = np.stack(iter_seq_list, axis=0)
                        iter_seq_array = np.stack(aa_map[iter_seq_list - 1], axis=0)
                        seq_list.append(iter_seq_array)
                    cls.seq_data = seq_list

            if load_pssm:
                with open(osp.join(root, 'pro-pssm.pkl'), 'rb') as f:
                    pssm_data = pickle.load(f)
                pssm_list = []
                for iter_pssm in pssm_data:
                    iter_pssm_trans = 1 / (1 + np.exp(- iter_pssm.astype(np.float32)))
                    pssm_list.append(iter_pssm_trans)
                cls.pssm_data = pssm_list

            with open(osp.join(root, 'fluid-splits.pkl'), 'rb') as f:
                cls.splits_data = pickle.load(f)
                cls.acc_data = pickle.load(f)
            print('load', root, 'success')
        return cls._instance

    def get_attr_dataset(self, fluid: str, pu: bool = False):
        assert self.attr_data is not None
        assert fluid in fluid_list

        split_data = self.splits_data[fluid]
        indices_tr_pos = split_data['tr_pos']
        indices_tr_neg = split_data['tr_neg']
        indices_tr_unknown = split_data['tr_unknown']
        indices_va_pos = split_data['va_pos']
        indices_va_neg = split_data['va_neg']
        indices_va_unknown = split_data['va_unknown']
        indices_te_pos = split_data['te_pos']
        indices_te_neg = split_data['te_neg']
        indices_te_unknown = split_data['te_unknown']

        if pu:
            indices_tr = np.concatenate([indices_tr_pos, indices_tr_neg, indices_tr_unknown], axis=0)
            label_tr = np.concatenate(
                [
                    np.ones_like(indices_tr_pos, dtype=np.int32),
                    np.zeros([len(indices_tr_neg) + len(indices_tr_unknown)], dtype=np.int32)
                ],
                axis=0
            )

            indices_va = np.concatenate([indices_va_pos, indices_va_neg, indices_va_unknown], axis=0)
            indices_te = np.concatenate([indices_te_pos, indices_te_neg, indices_te_unknown], axis=0)

            label_va = np.concatenate(
                [
                    np.ones_like(indices_va_pos, dtype=np.int32),
                    np.zeros(len(indices_va_neg) + len(indices_va_unknown), dtype=np.int32)
                ],
                axis=0
            )
            label_te = np.concatenate(
                [
                    np.ones_like(indices_te_pos, dtype=np.int32),
                    np.zeros(len(indices_te_neg) + len(indices_te_unknown), dtype=np.int32)
                ],
                axis=0
            )

        else:
            indices_tr = np.concatenate([indices_tr_pos, indices_tr_neg], axis=0)
            label_tr = np.concatenate(
                [
                    np.ones_like(indices_tr_pos, dtype=np.int32),
                    np.zeros_like(indices_tr_neg, dtype=np.int32)
                ],
                axis=0
            )

            indices_va = np.concatenate([indices_va_pos, indices_va_neg], axis=0)
            indices_te = np.concatenate([indices_te_pos, indices_te_neg], axis=0)

            label_va = np.concatenate(
                [
                    np.ones_like(indices_va_pos, dtype=np.int32),
                    np.zeros_like(indices_va_neg, dtype=np.int32)
                ],
                axis=0
            )
            label_te = np.concatenate(
                [
                    np.ones_like(indices_te_pos, dtype=np.int32),
                    np.zeros_like(indices_te_neg, dtype=np.int32)
                ],
                axis=0
            )

        data_tr = self.attr_data[indices_tr]
        data_va = self.attr_data[indices_va]
        data_te = self.attr_data[indices_te]
        return data_tr, label_tr, data_va, label_va, data_te, label_te


class BodyFluidDataset(Dataset):

    def __init__(self, root: str, mode: str = 'train', fluid: str = 'Plasma', data_type: str = 'pssm',
                 class_type: str = 'PN', max_len: int = 1000, with_len: bool = False) -> None:
        super(BodyFluidDataset, self).__init__()
        self.mode = mode
        self.data_type = data_type
        self.max_len = max_len

        self.left_len = max_len // 2
        self.right_len = max_len - self.left_len
        self.with_len = with_len

        assert fluid in fluid_list
        assert mode in ['train', 'train1', 'train2', 'test', 'eval']
        assert data_type in ['feat', 'seq', 'pssm']
        assert class_type in ['P', 'N', 'U', 'PN', 'PU']

        if data_type == 'attr':
            self.pro_data = SingleProData(root, True, False, False)
        elif data_type == 'seq':
            self.pro_data = SingleProData(root, False, True, False)
        else:
            self.pro_data = SingleProData(root, False, False, True)

        split_data = self.pro_data.splits_data[fluid]

        if mode != 'eval':
            if mode == 'train':
                pos_index = np.concatenate(
                    [split_data['tr_pos'], split_data['va_pos']],
                    axis=0
                )
                neg_index = np.concatenate(
                    [split_data['tr_neg'], split_data['va_neg']],
                    axis=0
                )
                unknown_index = np.concatenate(
                    [split_data['tr_unknown'], split_data['va_unknown']],
                    axis=0
                )
            elif mode == 'train1':
                pos_index = split_data['tr_pos']
                neg_index = split_data['tr_neg']
                unknown_index = split_data['tr_unknown']
            elif mode == 'train2':
                pos_index = split_data['va_pos']
                neg_index = split_data['va_neg']
                unknown_index = split_data['va_unknown']
            else:
                pos_index = split_data['te_pos']
                neg_index = split_data['te_neg']
                unknown_index = split_data['te_unknown']

            pos_label = np.ones_like(pos_index, dtype=np.int64)
            neg_label = np.zeros_like(neg_index, dtype=np.int64)
            unknown_label = np.zeros_like(unknown_index, dtype=np.int64)

            if class_type == 'PN':
                index = np.concatenate([pos_index, neg_index], axis=0)
                label = np.concatenate([pos_label, neg_label], axis=0)
            elif class_type == 'P':
                index = pos_index
                label = pos_label
            elif class_type == 'N':
                index = neg_index
                label = neg_label
            elif class_type == 'U':
                index = unknown_index
                label = unknown_label
            elif class_type == 'PU':
                index = np.concatenate([pos_index, neg_index, unknown_index], axis=0)
                label = np.concatenate([pos_label, neg_label, unknown_label], axis=0)
            else:
                raise ValueError('class_type')
        else:
            index = split_data['eval']
            label = - np.ones_like(index, dtype=np.int64)
        self.index = index
        self.label = label

    def _cut_and_pad(self, data):
        data_len = len(data)
        if data_len < self.max_len:
            data2 = np.pad(
                data,
                ((0, self.max_len - data_len), (0, 0)) if len(data.shape) == 2 else ((0, self.max_len - data_len),)
            )
            return data2, data_len
        else:
            data2 = np.concatenate(
                [data[:self.left_len], data[-self.right_len:]],
                axis=0
            )
            return data2, 1000

    def __getitem__(self, index):
        idx = self.index[index]
        label_item = self.label[index]

        if self.data_type == 'feat':
            attr_item = self.pro_data.attr_data[idx]
            return attr_item, label_item

        seq_item, len_item = self._cut_and_pad(self.pro_data.seq_data[idx] if self.data_type == 'seq'
                                               else self.pro_data.pssm_data[idx])
        if self.with_len:
            return seq_item, len_item, label_item
        else:
            return seq_item, label_item

    def __len__(self):
        return len(self.index)

    def get_label(self):
        return self.label


if __name__ == '__main__':

    folder = r'D:\dataset\PUSec17-20220912'
    pro_data = SingleProData(folder, True, True, True)

    for my_fluid in fluid_list:
        for my_mode in ['train', 'train1', 'train2', 'test', 'eval']:
            for my_data in ['feat', 'seq', 'pssm']:
                for my_class in ['P', 'N', 'U', 'PN', 'PU']:
                    ds = BodyFluidDataset(folder, my_mode, my_fluid, my_data, class_type=my_class)
                    print(
                        my_fluid, my_mode, my_data, my_class,
                        [data.shape for data in ds[0]],
                        len(ds)
                    )

    for fluid in fluid_list:
        data_list = pro_data.get_attr_dataset(fluid)
        print(fluid, [data.shape for data in data_list])
