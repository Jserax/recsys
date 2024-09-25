from typing import Any, Dict, Tuple

import lightning as pl
import torch


class MFModel(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        train_metrics: torch.nn.ModuleDict,
        val_metrics: torch.nn.ModuleDict,
        test_metrics: torch.nn.ModuleDict,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics

    def forward(self, users: torch.Tensor) -> torch.Tensor:
        return self.model.forward(users)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        users, ratings, weights = batch
        batch_size, item_size = ratings.size()
        preds = self.model(users)
        train_loss = (weights * self.criterion(preds, ratings)).mean()
        for metric_name, metric in self.train_metrics.items():
            if metric_name.startswith("recall"):
                metric.update(
                    preds,
                    ratings > 0,
                    indexes=users.repeat_interleave(item_size).view(
                        (batch_size, item_size)
                    ),
                )
            elif metric_name.startswith("ndcg"):
                metric.update(
                    preds,
                    ratings,
                    indexes=users.repeat_interleave(item_size).view(
                        (batch_size, item_size)
                    ),
                )
            else:
                metric.update(preds, ratings)
            self.log(
                f"train_{metric_name}",
                metric,
                on_epoch=True,
                on_step=False,
                logger=True,
            )
        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
        )
        return train_loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        users, ratings, weights, prev_weights = batch
        batch_size, item_size = ratings.size()
        preds = self.model(users)
        val_loss = (weights * self.criterion(preds, ratings)).mean()
        for metric_name, metric in self.val_metrics.items():
            if metric_name.startswith("recall"):
                metric.update(
                    prev_weights * preds,
                    ratings > 0,
                    indexes=users.repeat_interleave(item_size).view(
                        (batch_size, item_size)
                    ),
                )
            elif metric_name.startswith("ndcg"):
                metric.update(
                    prev_weights * preds,
                    ratings,
                    indexes=users.repeat_interleave(item_size).view(
                        (batch_size, item_size)
                    ),
                )
            else:
                metric.update(preds, ratings)
            self.log(
                f"val_{metric_name}",
                metric,
                on_epoch=True,
                on_step=False,
                logger=True,
            )
        self.log(
            "val_loss",
            val_loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        users, ratings, weights, prev_weights = batch
        batch_size, item_size = ratings.size()
        preds = self.model(users)
        test_loss = (weights * self.criterion(preds, ratings)).mean()
        for metric_name, metric in self.test_metrics.items():
            if metric_name.startswith("recall"):
                metric.update(
                    prev_weights * preds,
                    ratings > 0,
                    indexes=users.repeat_interleave(item_size).view(
                        (batch_size, item_size)
                    ),
                )
            elif metric_name.startswith("ndcg"):
                metric.update(
                    prev_weights * preds,
                    ratings,
                    indexes=users.repeat_interleave(item_size).view(
                        (batch_size, item_size)
                    ),
                )
            else:
                metric.update(preds, ratings)
            self.log(
                f"test_{metric_name}",
                metric,
                on_epoch=True,
                on_step=False,
                logger=True,
            )
        self.log(
            "test_loss",
            test_loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.optimizer(
            [
                {
                    "params": filter(
                        lambda p: p.requires_grad, self.model.parameters()
                    ),
                    "name": "model",
                }
            ]
        )
        scheduler = self.scheduler(optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
