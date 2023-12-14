import os
import random

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from utils.dataset import SingleProData, fluid_list
from utils.t_test import TTestSelect
from utils.metrics import eval_print


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    C_list = [1.0, 1.0, 0.1, 0.001, 0.01, 0.001, 1.0, 1.0, 1.0, 0.01, 1.0, 0.01, 0.1, 100, 1.0, 100, 1.0]
    save_dict = {}
    for C, fluid in zip(C_list, fluid_list):
        print('Train SVM model for', fluid)

        pro_data = SingleProData(args.folder)
        X_tr, y_tr, X_va, y_va, X_te, y_te = pro_data.get_attr_dataset(fluid, pu=False)

        svm = LinearSVC(C=C, class_weight='balanced', max_iter=300)
        fs_svm = Pipeline([
            ('select', TTestSelect(k=50, alpha=0.05)),
            ('svm', svm)
        ])
        fs_svm.fit(X_tr, y_tr)

        prob_va = fs_svm.decision_function(X_va)
        pred_va = fs_svm.predict(X_va)
        eval_print(y_va, pred_va, prob_va, '{:8s} valid'.format(fluid))

        prob_te = fs_svm.decision_function(X_te)
        pred_te = fs_svm.predict(X_te)
        eval_print(y_te, pred_te, prob_te, '{:8s} test '.format(fluid))

        # Save the prediction
        save_dict[fluid + '-score'] = prob_te
        save_dict[fluid + '-pred'] = pred_te

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, 'svm.npz')
    np.savez_compressed(save_path, **save_dict)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default=r'D:\datasets\BodyFluidData-20220912', type=str)
    parser.add_argument('--save-dir', default='tmp', type=str)
    parser.add_argument('--seed', type=int, default=43215)

    args = parser.parse_args()
    main(args)
