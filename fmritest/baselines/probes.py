"""Linear site probes with subject-level aggregation and a label-permutation null."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _fit(X, y, C, seed):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, random_state=seed))
    return clf.fit(X, y)


def subject_accuracy(clf, X, subj, site_of_subj):
    """Average log-probabilities over each subject's samples, then take the argmax."""
    logp = clf.predict_log_proba(X)
    subs = np.unique(subj)
    pred = np.array([clf.classes_[logp[subj == s].mean(0).argmax()] for s in subs])
    truth = np.array([site_of_subj[s] for s in subs])
    return float((pred == truth).mean())


def site_probe(X_tr, subj_tr, X_te, subj_te, site, C=1.0, n_perm=200, seed=0) -> dict:
    """Train on training subjects' samples, evaluate on held-out subjects.

    X_*: (n_samples, d) features; subj_*: subject index of each sample; site: (N,) labels indexed by subject.
    The permutation null shuffles site labels across *training subjects* (keeping each subject's
    samples together), refits, and evaluates on the untouched test set.
    """
    y_tr = site[subj_tr]
    clf = _fit(X_tr, y_tr, C, seed)
    acc = subject_accuracy(clf, X_te, subj_te, site)
    win_acc = float((clf.predict(X_te) == site[subj_te]).mean())
    rng = np.random.default_rng(seed + 1)
    tr_subs = np.unique(subj_tr)
    null = []
    for _ in range(n_perm):
        perm_site = site.copy()
        perm_site[tr_subs] = rng.permutation(site[tr_subs])
        null.append(subject_accuracy(_fit(X_tr, perm_site[subj_tr], C, seed), X_te, subj_te, site))
    null = np.array(null)
    n_classes = len(np.unique(y_tr))
    return {"acc": acc, "window_acc": win_acc, "chance": 1.0 / n_classes,
            "null_mean": float(null.mean()) if n_perm else float("nan"),
            "null_q95": float(np.quantile(null, 0.95)) if n_perm else float("nan"),
            "p_perm": float((1 + (null >= acc).sum()) / (1 + n_perm)) if n_perm else float("nan"),
            "n_test_subjects": int(len(np.unique(subj_te)))}


def fc_features(X: np.ndarray) -> np.ndarray:
    """Upper-triangle of each subject's correlation matrix. X: (N, T, R)."""
    iu = np.triu_indices(X.shape[-1], 1)
    return np.stack([np.corrcoef(x.T)[iu] for x in X])
