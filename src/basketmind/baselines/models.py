"""Baseline recommenders. All expose fit(...) and score(task, batch, device) -> (B, n_items) scores."""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from basketmind.baselines.task import Task
from basketmind.utils.seed import seed_everything

Evaluate = Callable[["Recommender"], float]  # returns the validation selection metric


class Recommender:
    name = "base"
    stochastic = False  # True: results depend on the seed, so run several seeds

    def fit(self, task: Task, device: torch.device, seed: int, evaluate: Evaluate) -> dict[str, Any]:
        return {}

    def score(self, task: Task, batch: pd.DataFrame, device: torch.device) -> torch.Tensor:
        raise NotImplementedError


class Popularity(Recommender):
    """Same ranking for everyone: items ordered by training-basket frequency."""

    name = "popularity"

    def fit(self, task, device, seed, evaluate):
        self.pop = torch.as_tensor(task.popularity, dtype=torch.float32, device=device)
        return {}

    def score(self, task, batch, device):
        return self.pop.expand(len(batch), -1)


class ItemItemCF(Recommender):
    """Cosine item-item similarity from the training customer x item matrix; score = history @ similarity.

    The diagonal is kept (an item is similar to itself), so previously bought items rank highly
    under the `full` policy. Under `exclude_history` they are removed anyway.
    """

    name = "item_item_cf"

    def fit(self, task, device, seed, evaluate):
        m = torch.as_tensor(task.train_matrix.toarray(), device=device)
        co = m.T @ m
        norm = co.diagonal().clamp(min=1).sqrt()
        self.sim = co / (norm[:, None] * norm[None, :])
        return {}

    def score(self, task, batch, device):
        return task.history_hot(batch, device) @ self.sim


class MatrixFactorisation(Recommender):
    """Customer/item embeddings trained with full-matrix weighted BCE on the training baskets.

    Customers with no training baskets fall back to the learned item bias (a popularity-like score).
    The epoch count is chosen on validation Recall@10.
    """

    name = "matrix_factorisation"
    stochastic = True

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

    def fit(self, task, device, seed, evaluate):
        seed_everything(seed)
        c = self.cfg
        y = torch.as_tensor(task.train_matrix.toarray(), device=device)
        self.cust = task.train_customers
        self.P = nn.Parameter(0.1 * torch.randn(y.shape[0], c["dim"], device=device))
        self.Q = nn.Parameter(0.1 * torch.randn(y.shape[1], c["dim"], device=device))
        self.b = nn.Parameter(torch.zeros(y.shape[1], device=device))
        opt = torch.optim.AdamW([self.P, self.Q, self.b], lr=c["lr"], weight_decay=c["weight_decay"])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(c["pos_weight"], device=device))
        best, best_state, bad, history = -1.0, None, 0, []
        for epoch in range(1, c["epochs"] + 1):
            opt.zero_grad()
            loss = loss_fn(self.P @ self.Q.T + self.b, y)
            loss.backward()
            opt.step()
            if epoch % c["eval_every"] == 0:
                with torch.no_grad():
                    val = evaluate(self)
                history.append({"epoch": epoch, "loss": loss.item(), "val": val})
                if val > best:
                    best, bad = val, 0
                    best_state = [p.detach().clone() for p in (self.P, self.Q, self.b)]
                    best_epoch = epoch
                else:
                    bad += 1
                    if bad >= c["patience"]:
                        break
        self.P, self.Q, self.b = best_state
        return {"best_epoch": best_epoch, "best_val": best, "history": history}

    @torch.no_grad()
    def score(self, task, batch, device):
        out = self.b.expand(len(batch), -1).clone()
        rows = [(i, self.cust[c]) for i, c in enumerate(batch["CustomerID"]) if c in self.cust]
        if rows:
            pos, idx = map(lambda v: torch.as_tensor(v, device=device), zip(*rows, strict=True))
            out[pos] = self.P[idx] @ self.Q.T + self.b
        return out


class _GRUNet(nn.Module):
    def __init__(self, n_items: int, item_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.bag = nn.EmbeddingBag(n_items, item_dim, mode="mean")
        # nn.GRU's backward fails on ROCm for Windows (MIOpen cannot compile its reduction kernel,
        # miopenStatusUnknownError), so the recurrence is an explicit GRUCell loop: same maths, no MIOpen RNN.
        self.cell = nn.GRUCell(item_dim, hidden)
        self.hidden = hidden
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_items)

    def forward(self, histories: list[list[np.ndarray]], device: torch.device) -> torch.Tensor:
        lengths = torch.tensor([len(h) for h in histories], device=device)
        flat = [a for h in histories for a in h]
        sizes = torch.tensor([len(a) for a in flat])
        ids = torch.as_tensor(np.concatenate(flat), device=device)
        offsets = (torch.cumsum(sizes, 0) - sizes).to(device)
        vecs = self.bag(ids, offsets)  # one vector per basket (zeros for an empty basket)
        padded = pad_sequence(list(vecs.split(lengths.tolist())), batch_first=True)  # (B, L, d)
        h = torch.zeros(len(histories), self.hidden, device=device)
        for t in range(padded.shape[1]):
            h = torch.where((t < lengths)[:, None], self.cell(padded[:, t], h), h)  # padding keeps h
        return self.out(self.drop(h))


class GRURecommender(Recommender):
    """GRU over the customer's basket history (each basket = mean item embedding) -> catalogue logits."""

    name = "gru"
    stochastic = True

    def __init__(self, cfg: dict[str, Any], train_cfg: dict[str, Any]):
        self.cfg, self.tc = cfg, train_cfg

    def _histories(self, task: Task, batch: pd.DataFrame) -> list[list[np.ndarray]]:
        return [[task.basket_items[i] for i in h] for h in batch["history_invoices"]]

    def fit(self, task, device, seed, evaluate):
        seed_everything(seed)
        c, t = self.cfg, self.tc
        self.net = _GRUNet(task.n_items, c["item_dim"], c["hidden_dim"], c["dropout"]).to(device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=2)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(c["pos_weight"], device=device))
        train = task.examples["train"]
        gen = np.random.default_rng(seed)
        best, best_state, bad, history = -1.0, None, 0, []
        for epoch in range(1, t["max_epochs"] + 1):
            self.net.train()
            order = gen.permutation(len(train))
            total = 0.0
            for s in range(0, len(order), t["batch_size"]):
                batch = train.iloc[order[s : s + t["batch_size"]]]
                logits = self.net(self._histories(task, batch), device)
                loss = loss_fn(logits, task.target_hot(batch, device).float())
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), t["gradient_clip"])
                opt.step()
                total += loss.item() * len(batch)
            self.net.eval()
            val = evaluate(self)
            sched.step(val)
            history.append({"epoch": epoch, "loss": total / len(train), "val": val})
            if val > best:
                best, bad, best_epoch = val, 0, epoch
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
                if bad >= t["early_stopping_patience"]:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        return {"best_epoch": best_epoch, "best_val": best, "history": history}

    @torch.no_grad()
    def score(self, task, batch, device):
        return self.net(self._histories(task, batch), device)
