import os
import random

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from utils.dataset import SingleProData, fluid_list
from utils.t_test import TTestSelect
from utils.metrics import eval_print


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    depth_list = [3, 5, 7, 7, 7, 7, 9, 7, 5, 9, 7, 7, 9, 5, 9, 7, 11]
    min_samples_split_list = [3, 7, 5, 7, 11, 5, 7, 11, 11, 9, 5, 3, 3, 3, 5, 3, 7]
    save_dict = {}
    for depth, min_samples_split, fluid in zip(depth_list, min_samples_split_list, fluid_list):
        print('Train DT model for', fluid)

        pro_data = SingleProData(args.folder)
        X_tr, y_tr, X_va, y_va, X_te, y_te = pro_data.get_attr_dataset(fluid, pu=False)

        dt = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_split=min_samples_split
        )
        fs_dt = Pipeline([
            ('select', TTestSelect(k=50, alpha=0.05)),
            ('dt', dt)
        ])
        fs_dt.fit(X_tr, y_tr)

        prob_va = fs_dt.predict_proba(X_va)[:, 1]
        pred_va = fs_dt.predict(X_va)
        eval_print(y_va, pred_va, prob_va, '{:8s} valid'.format(fluid))

        prob_te = fs_dt.predict_proba(X_te)[:, 1]
        pred_te = fs_dt.predict(X_te)
        eval_print(y_te, pred_te, prob_te, '{:8s} test '.format(fluid))

        # save the prediction
        save_dict[fluid + '-score'] = prob_te
        save_dict[fluid + '-pred'] = pred_te

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, 'dt.npz')
    np.savez_compressed(save_path, **save_dict)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default=r'D:\datasets\BodyFluidData-20220912', type=str)
    parser.add_argument('--save-dir', default='tmp', type=str)
    parser.add_argument('--seed', type=int, default=43215)

    args = parser.parse_args()
    main(args)
