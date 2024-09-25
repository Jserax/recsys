from typing import Any, Callable, Dict, Optional, Tuple, Union

import lightning as pl
import torch


class RankingModel(pl.LightningModule):
    def __init__(
        self,
        ranking_model: torch.nn.Module,
        candidate_model: torch.nn.Module,
        criterion: Union[torch.nn.Module, Callable],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        train_metrics: torch.nn.ModuleDict,
        val_metrics: torch.nn.ModuleDict,
        test_metrics: torch.nn.ModuleDict,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ranking_model = ranking_model
        self.candidate_model = candidate_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.save_hyperparameters()

    def forward(
        self,
        users: torch.Tensor,
        top_candidates: int = 256,
        top_ranking: int = 10,
        attn_mask: Optional[torch.Tensor] = None,
        exclude_prev: Optional[list[int]] = None,
    ) -> torch.Tensor:
        candidates = self.candidate_model(users)
        if exclude_prev is not None:
            candidates *= exclude_prev
        _, ind_cand = torch.topk(
            candidates,
            k=top_candidates,
        )
        if attn_mask is not None:
            out = self.ranking_model.forward(
                users, ind_cand, attn_mask.gather(-1, ind_cand)
            )
        else:
            out = self.ranking_model.forward(users, ind_cand)
        val, ind = torch.topk(
            out,
            k=top_ranking,
        )
        return val, ind_cand.gather(-1, ind)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        users, items, ratings, attn_mask = batch

        candidates = self.candidate_model.batch_predict(users, items)
        _, ind = torch.topk(
            candidates,
            k=128,
        )
        ratings = ratings.gather(-1, ind)
        preds = self.ranking_model.forward(
            users, items.gather(-1, ind), attn_mask.gather(-1, ind)
        )

        batch_size, item_size = ratings.size()
        train_loss = self.criterion(preds, ratings)
        for metric_name, metric in self.train_metrics.items():
            if metric_name.startswith("ndcg"):
                metric.update(
                    preds,
                    ratings.long(),
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
        batch: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        batch_idx: int,
    ) -> None:
        users, items, ratings, attn_mask, exclude_prev = batch
        candidates = self.candidate_model.batch_predict(users, items)
        _, ind = torch.topk(
            exclude_prev * candidates,
            k=128,
        )
        ratings = ratings.gather(-1, ind)
        preds = self.ranking_model.forward(
            users, items.gather(-1, ind), attn_mask.gather(-1, ind)
        )

        batch_size, item_size = ratings.size()
        val_loss = self.criterion(preds, items)
        for metric_name, metric in self.val_metrics.items():
            if metric_name.startswith("ndcg"):
                metric.update(
                    preds,
                    ratings.long(),
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
        users, items, ratings, attn_mask, exclude_prev = batch
        candidates = self.candidate_model.batch_predict(users, items)
        _, ind = torch.topk(
            exclude_prev * candidates,
            k=128,
        )
        ratings = ratings.gather(-1, ind)
        preds = self.ranking_model.forward(
            users, items.gather(-1, ind), attn_mask.gather(-1, ind)
        )

        batch_size, item_size = ratings.size()
        test_loss = self.criterion(preds, ratings)
        for metric_name, metric in self.test_metrics.items():
            if metric_name.startswith("ndcg"):
                metric.update(
                    preds,
                    ratings.long(),
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
                        lambda p: p.requires_grad, self.ranking_model.parameters()
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
