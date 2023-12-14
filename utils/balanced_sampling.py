from typing import List

import torch
from torch.utils.data import DataLoader, Dataset


class BalancedSampling:

    def __init__(self, pos_datasets: List[Dataset], neg_datasets: List[Dataset], batch_size: int = 32,
                 pin_memory: bool = False, drop_last: bool = True) -> None:
        assert len(pos_datasets) == len(neg_datasets)
        self.pos_ds = pos_datasets
        self.neg_ds = neg_datasets
        self.batch_size = batch_size

        self.num_ds = len(pos_datasets)
        mean_batch_size = batch_size // 2
        if mean_batch_size < 1:
            mean_batch_size = 1
        self.pos_dls = [
            DataLoader(ds, mean_batch_size, True, pin_memory=pin_memory, drop_last=drop_last)
            for ds in pos_datasets
        ]
        self.neg_dls = [
            DataLoader(ds, mean_batch_size, True, pin_memory=pin_memory, drop_last=drop_last)
            for ds in neg_datasets
        ]
        self.dls = self.pos_dls + self.neg_dls

    def __iter__(self):
        for data_list in zip(*self.dls):
            yield [
                [
                    torch.cat([pdata, ndata], dim=0)
                    for pdata, ndata in zip(data_list[i], data_list[i + self.num_ds])
                ]
                for i in range(self.num_ds)
            ]

    def __len__(self):
        return min([len(dl) for dl in self.dls])


if __name__ == '__main__':
    from torch.utils.data import TensorDataset

    pos_ds = [
        TensorDataset(
            torch.randn(100, 10),
            torch.randn(100, 1)
        ) for _ in range(4)
    ]
    neg_ds = [
        TensorDataset(
            torch.randn(150, 10),
            torch.randn(150, 1)
        ) for _ in range(4)
    ]
    dl = BalancedSampling(pos_ds, neg_ds, batch_size=64)
    for i in range(5):
        for j, data_list in enumerate(dl):
            for data, label in data_list:
                print(i, j, data.shape, label.shape)
