import os
import random

import numpy as np
from pulearn.bagging import BaggingPuClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from utils.dataset import SingleProData, fluid_list
from utils.metrics import eval_print, eval_mean_print, select_threshold_by_mcc
from utils.t_test import TTestSelect


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    pro_data = SingleProData(args.folder)

    true_va_list = []
    pred_va_list = []
    score_va_list = []

    true_te_list = []
    pred_te_list = []
    score_te_list = []

    for fluid in fluid_list:
        X_tr, y_tr, X_va, y_va, X_te, y_te = pro_data.get_attr_dataset(fluid, pu=True)

        dt = DecisionTreeClassifier(
            max_depth=args.depth,
            min_samples_split=args.min_samples_split
        )
        pu_dt = BaggingPuClassifier(
            dt,
            n_estimators=args.n_estimators,
            max_samples=args.max_samples,
            max_features=args.max_features,
            bootstrap_features=args.bs_features,
            n_jobs=-1,
            random_state=args.seed
        )
        fs_dt = Pipeline([
            ('select', TTestSelect(k=50, alpha=0.05)),
            ('dt', pu_dt)
        ])
        fs_dt.fit(X_tr, y_tr)

        score_va = fs_dt.predict_proba(X_va)[:, 1]
        score_te = fs_dt.predict_proba(X_te)[:, 1]

        threshold = select_threshold_by_mcc(y_va, score_va)
        pred_va = np.asarray(score_va > threshold, dtype=int)
        pred_te = np.asarray(score_te > threshold, dtype=int)

        eval_print(y_va, pred_va, score_va, '{:8s} valid'.format(fluid))
        eval_print(y_te, pred_te, score_te, '{:8s} test '.format(fluid))

        true_va_list.append(y_va)
        pred_va_list.append(pred_va)
        score_va_list.append(score_va)

        true_te_list.append(y_te)
        pred_te_list.append(pred_te)
        score_te_list.append(score_te)

    eval_mean_print(true_va_list, pred_va_list, score_va_list, 'Mean valid')
    eval_mean_print(true_te_list, pred_te_list, score_te_list, 'Mean test ')

    save_dict = {}
    for fluid, y_pred, y_score in zip(fluid_list, pred_te_list, score_te_list):
        save_dict[fluid + '-pred'] = y_pred
        save_dict[fluid + '-score'] = y_score

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, 'pu-dt.npz')
    np.savez_compressed(save_path, **save_dict)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default=r'D:\datasets\BodyFluidData-20220912', type=str)
    parser.add_argument('--save-dir', default='tmp', type=str)
    parser.add_argument('--n_estimators', type=int, default=16)
    parser.add_argument('--max_samples', type=int, default=3000)
    parser.add_argument('--max_features', type=float, default=0.5)
    parser.add_argument('--bs_features', default=False, action='store_true')
    parser.add_argument('--depth', type=int, default=7)
    parser.add_argument('--min_samples_split', type=int, default=20)
    parser.add_argument('--seed', type=int, default=43215)

    args = parser.parse_args()
    main(args)
