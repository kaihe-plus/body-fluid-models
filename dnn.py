import os
import random

import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.dnn_model import DNNModel
from utils.dataset import SingleProData, fluid_list
from utils.t_test import TTestSelect
from utils.metrics import eval_print


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    save_dict = {}
    for fluid in fluid_list:
        print('Train DNN model for', fluid)

        pro_data = SingleProData(args.folder)
        X_tr, y_tr, X_va, y_va, X_te, y_te = pro_data.get_attr_dataset(fluid, pu=False)

        dnn = DNNModel(50, 500, 4,
                       0.0001, 32, device=torch.device('cuda'))
        fs_dnn = Pipeline([
            ('select', TTestSelect(k=50, alpha=0.05)),
            ('scale', StandardScaler()),
            ('dnn', dnn)
        ])
        fs_dnn.fit(X_tr, y_tr)

        prob_va = fs_dnn.predict_proba(X_va)
        pred_va = fs_dnn.predict(X_va)
        eval_print(y_va, pred_va, prob_va[:, 1], '{:8s} valid'.format(fluid))

        prob_te = fs_dnn.predict_proba(X_te)
        pred_te = fs_dnn.predict(X_te)
        eval_print(y_te, pred_te, prob_te[:, 1], '{:8s} test '.format(fluid))

        save_dict[fluid + '-score'] = prob_te[:, 1]
        save_dict[fluid + '-pred'] = pred_te

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, 'dnn.npz')
    np.savez_compressed(save_path, **save_dict)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default=r'D:\datasets\BodyFluidData-20220912', type=str)
    parser.add_argument('--save-dir', default='tmp', type=str)
    parser.add_argument('--seed', type=int, default=43215)

    args = parser.parse_args()
    main(args)
