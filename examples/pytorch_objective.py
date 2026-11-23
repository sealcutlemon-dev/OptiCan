import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from optican import Bool, Choice, Float, OptiCan


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.layers(x)


torch.manual_seed(7)
X_train = torch.randn(512, 10)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).long()
X_valid = torch.randn(128, 10)
y_valid = (X_valid[:, 0] + X_valid[:, 1] > 0).long()


def objective(config, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    optimizer_class = getattr(torch.optim, config["optimizer"])
    optimizer = optimizer_class(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
    )

    scheduler = None
    if config["use_scheduler"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    best_val_loss = float("inf")
    for _ in range(15):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(
                model(X_valid.to(device)),
                y_valid.to(device),
            ).item()
        best_val_loss = min(best_val_loss, val_loss)

    return best_val_loss


if __name__ == "__main__":
    tuner = OptiCan(
        objective,
        config={
            "learning_rate": Float(1e-5, 3e-3, log=True),
            "weight_decay": Float(1e-7, 1e-2, log=True, allow_zero=True),
            "batch_size": Choice([32, 64, 128]),
            "optimizer": Choice(["AdamW", "Adam"]),
            "use_scheduler": Bool(),
        },
        rounds=3,
        initial_candidates=8,
        candidates_per_round=4,
        seeds=3,
        ubest=True,
    )
    result = tuner.optimize()
    print(result.best_config)
    print(result.best_score)
