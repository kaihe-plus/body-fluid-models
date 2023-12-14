import numpy as np
from scipy.stats import ttest_ind
from sklearn.base import BaseEstimator, TransformerMixin


# t test for rfe
def ttest_ind_classif(X, y):
    y = y.ravel()
    y_uni = np.unique(y)
    if y_uni.shape[0] > 2:
        raise ValueError('y must have 2 labels')
    else:
        mask_a = y == y_uni[0]
        mask_b = y == y_uni[1]
        X_a = X[mask_a]
        X_b = X[mask_b]
        scores, p_values = ttest_ind(X_a, X_b, equal_var=False, nan_policy='omit')
        return scores, p_values


def fdr_adjust(scores, alpha=0.05):
    order = np.argsort(np.abs(scores) * -1)
    q_values = alpha / scores.shape[0] * np.arange(1, scores.shape[0] + 1)
    q_values = q_values[order.argsort()]
    return q_values


class TTestSelect(BaseEstimator, TransformerMixin):
    """Feature selection with t test."""

    def __init__(self, k=None, alpha=0.05):
        self.k = k
        self.alpha = alpha
        self.n_features = None
        self.score = None
        self.p_value = None
        self.q_value = None
        self.fdr_select = None
        self.k_select = None

    def fit(self, X, y=None, **fit_params):
        self.score, self.p_value = ttest_ind_classif(X, y)
        self.q_value = fdr_adjust(self.score, alpha=self.alpha)
        self.fdr_select = self.p_value < self.q_value
        n_features = np.sum(self.fdr_select)
        self.n_features = n_features
        k_select = n_features if self.k is None or self.k > n_features else self.k
        score_abs = np.abs(self.score[self.fdr_select])
        k_indices = np.argsort(score_abs)[-k_select:]
        k_select_mask = np.zeros_like(score_abs, dtype=bool)
        k_select_mask[k_indices] = True
        self.k_select = k_select_mask

    def transform(self, X, y=None, **fit_params):
        X = X[:, self.fdr_select]
        X = X[:, self.k_select]
        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
