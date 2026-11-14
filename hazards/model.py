"""The dust devil model: expected vortex encounters per hour from the local climate.

A small neural network (two hidden layers of 8 units, tanh) maps climate inputs (by default
heat flux and boundary layer height, see DEFAULT_FEATURES) to a correction of the physics index;
the logarithm of the encounter rate is log(index) + network. It is trained on counts with a
Poisson likelihood and the listening time as exposure, so an hour with 3 minutes of data counts
less than an hour recorded in full. Several networks with different starting weights are
averaged. Plain numpy: the table has a few hundred rows, so nothing bigger is needed.

    model = DustDevilModel().fit(train)          # train: rows of dataset.build_table
    rate = model.predict(test)                   # encounters per Earth hour

Baselines to beat (same inputs available, fitted on the same rows):
    DiurnalTable   average rate by local hour, the same everywhere and in every season
    PhysicsIndex   rate = c * heat flux * boundary layer height (Renno et al. 1998 idea)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


# Chosen by training on one station and testing on the other (InSight <-> Perseverance), daytime
# only. A network that predicts the rate from scratch failed in one direction (it extrapolated
# wildly on heat flux values Jezero never had); with all seven inputs it did worse than a plain
# table, because wind and ground-air temperature difference differ between stations. A network
# that only corrects the physics index, on heat flux and boundary layer height, with strong L2,
# was never worse than the physics index and better from InSight to Jezero. Curiosity was not
# used for this choice.
DEFAULT_FEATURES = ["heat_up_wm2", "pbl_km"]


class DustDevilModel:
    """`residual=True` (default): the network only corrects the physics index
    (log rate = log index + net). Strong L2 pulls it back to plain physics wherever the training
    data says nothing, so it cannot run away on a climate it has not seen."""

    def __init__(self, hidden: int = 8, l2: float = 0.1, epochs: int = 2500, lr: float = 0.01,
                 members: int = 8, features: list[str] | None = None, seed: int = 0,
                 residual: bool = True):
        self.hidden, self.l2, self.epochs, self.lr, self.members = hidden, l2, epochs, lr, members
        self.features = features or list(DEFAULT_FEATURES)
        self.seed, self.residual = seed, residual

    def _offset(self, table: pd.DataFrame) -> np.ndarray:
        return np.log(PhysicsIndex.index(table)) if self.residual else np.zeros(len(table))

    # ----- training -----

    def fit(self, table: pd.DataFrame) -> "DustDevilModel":
        X = table[self.features].to_numpy(float)
        self.mean_, self.std_ = X.mean(0), X.std(0) + EPS
        X = (X - self.mean_) / self.std_
        y = table["count"].to_numpy(float)
        log_exp = np.log(table["exposure_h"].to_numpy(float)) + self._offset(table)
        base = np.log((y.sum() + 0.5) / np.exp(log_exp).sum())
        self.nets_ = [self._train_one(X, y, log_exp, base, np.random.default_rng(self.seed + k))
                      for k in range(self.members)]
        return self

    def _train_one(self, X, y, log_exp, base, rng):
        n_in, h = X.shape[1], self.hidden
        p = {"W1": rng.normal(0, 1 / np.sqrt(n_in), (n_in, h)), "b1": np.zeros(h),
             "W2": rng.normal(0, 1 / np.sqrt(h), (h, h)), "b2": np.zeros(h),
             "W3": rng.normal(0, 0.1 / np.sqrt(h), (h, 1)), "b3": np.array([base])}
        m = {k: np.zeros_like(v) for k, v in p.items()}
        v = {k: np.zeros_like(v) for k, v in p.items()}
        n = len(y)
        for step in range(1, self.epochs + 1):
            a1 = np.tanh(X @ p["W1"] + p["b1"])
            a2 = np.tanh(a1 @ p["W2"] + p["b2"])
            f = (a2 @ p["W3"] + p["b3"]).ravel()
            mu = np.exp(np.clip(f + log_exp, -30, 30))
            g = ((mu - y) / n)[:, None]                                # d loss / d f
            grads = {"W3": a2.T @ g, "b3": g.sum(0)}
            d2 = (g @ p["W3"].T) * (1 - a2 ** 2)
            grads["W2"], grads["b2"] = a1.T @ d2, d2.sum(0)
            d1 = (d2 @ p["W2"].T) * (1 - a1 ** 2)
            grads["W1"], grads["b1"] = X.T @ d1, d1.sum(0)
            for k in ("W1", "W2", "W3"):
                grads[k] = grads[k] + 2 * self.l2 * p[k]
            for k in p:                                                # Adam
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                p[k] -= self.lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
        return p

    # ----- use -----

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        """Encounters per Earth hour for each row."""
        X = (table[self.features].to_numpy(float) - self.mean_) / self.std_
        offset = self._offset(table)
        rates = []
        for p in self.nets_:
            a2 = np.tanh(np.tanh(X @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"])
            rates.append(np.exp(np.clip((a2 @ p["W3"] + p["b3"]).ravel() + offset, -30, 10)))
        return np.mean(rates, axis=0)


class DiurnalTable:
    """Average rate by local hour over the training rows: no place, no season."""

    def fit(self, table):
        g = table.groupby("hour").agg(c=("count", "sum"), e=("exposure_h", "sum"))
        self.rate_ = ((g["c"] + 0.05) / g["e"]).to_dict()
        return self

    def predict(self, table):
        return table["hour"].map(self.rate_).fillna(min(self.rate_.values())).to_numpy(float)


class PhysicsIndex:
    """rate = c * max(heat flux, 0) * boundary layer height, with one constant c."""

    @staticmethod
    def index(table):
        return np.clip(table["heat_up_wm2"].to_numpy(float), 0, None) * table["pbl_km"].to_numpy(float) + 1e-3

    def fit(self, table):
        self.c_ = table["count"].sum() / (table["exposure_h"] * self.index(table)).sum()
        return self

    def predict(self, table):
        return self.c_ * self.index(table)


# ---------- scores ----------

def poisson_deviance(count, expected) -> float:
    y, mu = np.asarray(count, float), np.asarray(expected, float) + EPS
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return float(2 * np.sum(log_term - (y - mu)))


def _explained(y, e, expected) -> float:
    null = poisson_deviance(y, e * y.sum() / e.sum())
    return 1 - poisson_deviance(y, expected) / null if null > 0 else np.nan


def score(table: pd.DataFrame, rate: np.ndarray) -> dict:
    """How well predicted rates explain the counts of `table`.

    explained: 1 - deviance / deviance of the best single constant rate for this very table
    (which the model never sees). 1 is perfect, 0 is no better than knowing the average, below 0
    is worse than the average. explained_day: the same on hours 8 to 17 only, where the easy
    part (no vortices at night) is gone. total_ratio: predicted / observed number of vortices.
    """
    y, e = table["count"].to_numpy(float), table["exposure_h"].to_numpy(float)
    expected = rate * e
    day = table["hour"].between(8, 17).to_numpy()
    return {"explained": _explained(y, e, expected),
            "explained_day": _explained(y[day], e[day], expected[day]),
            "observed": int(y.sum()), "predicted": float(expected.sum()),
            "total_ratio": float(expected.sum() / max(y.sum(), EPS))}
