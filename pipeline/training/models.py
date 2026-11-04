"""Candidate model families with Optuna search spaces and native (non-pickle) persistence.

All wrappers expose fit(X, y, X_val=None, y_val=None), predict(X) -> P(make), best_iteration,
save(dir) and a matching load_model(dir). Boosted models use early stopping on X_val when given;
otherwise they train exactly `n_estimators` rounds.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORICAL_CODES = ["zone_code", "family_code", "poss_code"]
N_JOBS = max(1, min(8, (os.cpu_count() or 2) - 1))


# --------------------------------------------------------------------------- search spaces


def suggest(trial, name: str) -> dict:
    if name == "logistic":
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True), "n_knots": trial.suggest_int("n_knots", 3, 7)}
    if name == "xgboost":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
    if name == "lightgbm":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 4, 48),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        }
    if name == "catboost":
        return {
            "depth": trial.suggest_int("depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        }
    raise ValueError(name)


# --------------------------------------------------------------------------- wrappers


class LogisticModel:
    """Regularized logistic GAM-style model: B-splines on continuous inputs, one-hot codes, L2 penalty."""

    family = "logistic"

    def __init__(self, features: list[str], params: dict, seed: int, n_estimators: int | None = None):
        self.features, self.params, self.seed = features, dict(params), seed
        self.best_iteration = None

    def _pipeline(self, X: pd.DataFrame):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
        cats = [c for c in self.features if c in CATEGORICAL_CODES]
        binary = [c for c in self.features if c not in cats and X[c].dropna().isin([0, 1]).all()]
        cont = [c for c in self.features if c not in cats and c not in binary]
        steps = []
        if cont:
            steps.append(("cont", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                                            ("spl", SplineTransformer(n_knots=self.params["n_knots"], degree=3, knots="quantile",
                                                                      extrapolation="constant")),
                                            ("sc", StandardScaler())]), cont))
        if binary:
            steps.append(("bin", SimpleImputer(strategy="most_frequent"), binary))
        if cats:
            steps.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats))
        return Pipeline([("pre", ColumnTransformer(steps)),
                         ("lr", LogisticRegression(C=self.params["C"], solver="newton-cholesky", max_iter=500))])

    def fit(self, X, y, X_val=None, y_val=None):
        self.model = self._pipeline(X).fit(X[self.features], np.asarray(y, int))
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict_proba(X[self.features])[:, 1]

    def save(self, d: Path) -> list[str]:
        import joblib
        joblib.dump(self.model, d / "model.joblib")
        return ["model.joblib"]


class XGBModel:
    family = "xgboost"

    def __init__(self, features, params, seed, n_estimators=2000, early_stopping=50):
        self.features, self.params, self.seed = features, dict(params), seed
        self.n_estimators, self.early_stopping = n_estimators, early_stopping
        self.best_iteration = None

    def fit(self, X, y, X_val=None, y_val=None):
        import xgboost as xgb
        kw = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", n_estimators=self.n_estimators,
                  random_state=self.seed, n_jobs=N_JOBS, **self.params)
        if X_val is not None:
            kw["early_stopping_rounds"] = self.early_stopping
        self.model = xgb.XGBClassifier(**kw)
        ev = [(X_val[self.features], np.asarray(y_val, int))] if X_val is not None else None
        self.model.fit(X[self.features], np.asarray(y, int), eval_set=ev, verbose=False)
        self.best_iteration = int(self.model.best_iteration + 1) if X_val is not None else self.n_estimators
        return self

    def predict(self, X):
        it = (0, self.best_iteration) if self.best_iteration else None
        return self.model.predict_proba(X[self.features], iteration_range=it)[:, 1]

    def save(self, d: Path) -> list[str]:
        self.model.get_booster().save_model(d / "model.xgb.json")
        return ["model.xgb.json"]


class LGBMModel:
    family = "lightgbm"

    def __init__(self, features, params, seed, n_estimators=2000, early_stopping=50):
        self.features, self.params, self.seed = features, dict(params), seed
        self.n_estimators, self.early_stopping = n_estimators, early_stopping
        self.best_iteration = None

    def fit(self, X, y, X_val=None, y_val=None):
        import lightgbm as lgb
        self.model = lgb.LGBMClassifier(objective="binary", n_estimators=self.n_estimators, subsample_freq=1,
                                        random_state=self.seed, n_jobs=N_JOBS, verbose=-1, **self.params)
        if X_val is not None:
            self.model.fit(X[self.features], np.asarray(y, int), eval_X=(X_val[self.features],), eval_y=(np.asarray(y_val, int),),
                           callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)])
            self.best_iteration = int(self.model.best_iteration_ or self.n_estimators)
        else:
            self.model.fit(X[self.features], np.asarray(y, int))
            self.best_iteration = self.n_estimators
        return self

    def predict(self, X):
        return self.model.predict_proba(X[self.features], num_iteration=self.best_iteration)[:, 1]

    def save(self, d: Path) -> list[str]:
        self.model.booster_.save_model(str(d / "model.lgb.txt"), num_iteration=self.best_iteration)
        return ["model.lgb.txt"]


class CatModel:
    family = "catboost"

    def __init__(self, features, params, seed, n_estimators=2000, early_stopping=50):
        self.features, self.params, self.seed = features, dict(params), seed
        self.n_estimators, self.early_stopping = n_estimators, early_stopping
        self.best_iteration = None

    def fit(self, X, y, X_val=None, y_val=None):
        from catboost import CatBoostClassifier
        self.model = CatBoostClassifier(iterations=self.n_estimators, loss_function="Logloss", random_seed=self.seed,
                                        thread_count=N_JOBS, verbose=0, allow_writing_files=False, **self.params)
        if X_val is not None:
            self.model.fit(X[self.features], np.asarray(y, int), eval_set=(X_val[self.features], np.asarray(y_val, int)),
                           early_stopping_rounds=self.early_stopping, use_best_model=True)
            self.best_iteration = int(self.model.get_best_iteration() + 1)
        else:
            self.model.fit(X[self.features], np.asarray(y, int))
            self.best_iteration = self.n_estimators
        return self

    def predict(self, X):
        return self.model.predict_proba(X[self.features])[:, 1]

    def save(self, d: Path) -> list[str]:
        self.model.save_model(str(d / "model.cbm"))
        return ["model.cbm"]


FAMILIES = {"logistic": LogisticModel, "xgboost": XGBModel, "lightgbm": LGBMModel, "catboost": CatModel}


def make_model(name: str, features: list[str], params: dict, seed: int, n_estimators: int = 2000, early_stopping: int = 50):
    cls = FAMILIES[name]
    if name == "logistic":
        return cls(features, params, seed)
    return cls(features, params, seed, n_estimators=n_estimators, early_stopping=early_stopping)


class LoadedModel:
    """Inference-only model restored from a saved artifact directory."""

    def __init__(self, d: Path, spec: dict):
        self.family, self.features = spec["family"], spec["features"]
        f = d / spec["files"][0]
        if self.family == "xgboost":
            import xgboost as xgb
            self.booster = xgb.Booster()
            self.booster.load_model(f)
        elif self.family == "lightgbm":
            import lightgbm as lgb
            self.booster = lgb.Booster(model_file=str(f))
        elif self.family == "catboost":
            from catboost import CatBoostClassifier
            self.booster = CatBoostClassifier()
            self.booster.load_model(str(f))
        elif self.family == "logistic":
            from pipeline.utils import require, sha256_file
            require(sha256_file(f) == spec["sha256"][spec["files"][0]], f"{f} sha256 mismatch; refusing to unpickle")
            import joblib
            self.booster = joblib.load(f)
        else:
            raise ValueError(self.family)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xf = X[self.features]
        if self.family == "xgboost":
            import xgboost as xgb
            return self.booster.predict(xgb.DMatrix(Xf.astype(float), feature_names=self.features))
        if self.family == "lightgbm":
            return self.booster.predict(Xf.astype(float))
        return self.booster.predict_proba(Xf)[:, 1]


def write_spec(d: Path, model, files: list[str], extra: dict) -> dict:
    from pipeline.utils import sha256_file
    spec = {"family": model.family, "features": model.features, "params": model.params,
            "n_estimators": model.best_iteration, "files": files,
            "sha256": {f: sha256_file(d / f) for f in files}, **extra}
    (d / "model_spec.json").write_text(json.dumps(spec, indent=2, default=str))
    return spec
