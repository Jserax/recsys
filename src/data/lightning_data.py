from typing import Optional, Tuple

import lightning as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class MFTrainDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        num_uniq_items: int,
        implicit_weights: Tuple[float, float] = (0.05, 50),
    ) -> None:
        data = torch.from_numpy(data.values)
        num_uniq_users = len(data[:, 0].unique())
        self.ratings = torch.zeros(num_uniq_users, num_uniq_items, dtype=torch.float32)
        self.ratings[data[:, 0], data[:, 1]] = data[:, 2].float()
        self.weights = torch.full_like(
            self.ratings, implicit_weights[0], dtype=torch.float32
        )

        self.weights[self.ratings > 0] = implicit_weights[1]
        super().__init__()

    def __len__(self) -> int:
        return self.ratings.shape[0]

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        return (
            index,
            self.ratings[index],
            self.weights[index],
        )


class MFTestDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        prev_data: pd.DataFrame,
        num_uniq_items: int,
        implicit_weights: Tuple[float, float] = (0.01, 10),
    ) -> None:
        data = torch.from_numpy(data.values)
        prev_data = torch.from_numpy(prev_data.values)
        num_uniq_users = len(data[:, 0].unique())
        self.ratings = torch.zeros(num_uniq_users, num_uniq_items, dtype=torch.float32)
        self.ratings[data[:, 0], data[:, 1]] = data[:, 2].float()
        self.weights = torch.full_like(
            self.ratings, implicit_weights[0], dtype=torch.float32
        )
        self.weights[self.ratings > 0] = implicit_weights[1]
        prev_ratings = torch.zeros(num_uniq_users, num_uniq_items, dtype=torch.float32)
        prev_ratings[prev_data[:, 0], prev_data[:, 1]] = prev_data[:, 2].float()
        self.prev_weights = torch.ones_like(self.ratings, dtype=torch.float32)
        self.prev_weights[prev_ratings > 0] = 0.0
        super().__init__()

    def __len__(self) -> int:
        return self.ratings.shape[0]

    def __getitem__(
        self, index: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            index,
            self.ratings[index],
            self.weights[index],
            self.prev_weights[index],
        )


class RankingDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        num_uniq_items: int,
        implicit_weights: Tuple[float, float] = (1.0, 1.0),
        max_len: int = 128,
        sorted_rating: bool = True,
    ) -> None:
        data = torch.from_numpy(data.values)
        num_uniq_users = len(data[:, 0].unique())
        self.ratings = torch.zeros(num_uniq_users, num_uniq_items, dtype=torch.float32)
        self.ratings[data[:, 0], data[:, 1]] = data[:, 2].float()
        self.ratings, self.items = torch.topk(
            self.ratings, k=max_len, dim=-1, sorted=sorted_rating
        )
        self.weights = torch.full_like(
            self.ratings, implicit_weights[0], dtype=torch.float32
        )
        self.weights[self.ratings > 0] = implicit_weights[1]

        super().__init__()

    def __len__(self) -> int:
        return self.ratings.shape[0]

    def __getitem__(
        self, index: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            index,
            self.items[index],
            self.ratings[index],
            self.weights[index],
        )


class RankingTrainDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        num_uniq_items: int,
        max_len: int = -1,
    ) -> None:
        data = torch.from_numpy(data.values)
        num_uniq_users = len(data[:, 0].unique())
        self.ratings = torch.zeros(num_uniq_users, num_uniq_items, dtype=torch.float32)
        self.ratings[data[:, 0], data[:, 1]] = data[:, 2].float()
        if max_len == -1:
            self.ratings, self.items = torch.topk(
                self.ratings, k=num_uniq_items, dim=-1, sorted=False
            )
        else:
            self.ratings, self.items = torch.topk(
                self.ratings, k=max_len, dim=-1, sorted=False
            )
        self.attn_mask = torch.zeros(
            num_uniq_users,
            num_uniq_items if max_len == -1 else max_len,
            dtype=torch.bool,
        )
        self.attn_mask[self.ratings > 0] = 1

        super().__init__()

    def __len__(self) -> int:
        return self.ratings.shape[0]

    def __getitem__(
        self, index: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            index,
            self.items[index],
            self.ratings[index],
            self.attn_mask[index],
        )


class RankingTestDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        prev_data: pd.DataFrame,
        num_uniq_items: int,
        max_len: int = -1,
    ) -> None:
        data = torch.from_numpy(data.values)
        prev_data = torch.from_numpy(prev_data.values)
        num_uniq_users = len(data[:, 0].unique())
        self.ratings = torch.zeros(num_uniq_users, num_uniq_items, dtype=torch.float32)
        self.ratings[data[:, 0], data[:, 1]] = data[:, 2].float()
        if max_len == -1:
            self.ratings, self.items = torch.topk(
                self.ratings, k=num_uniq_items, dim=-1, sorted=False
            )
        else:
            self.ratings, self.items = torch.topk(
                self.ratings, k=max_len, dim=-1, sorted=False
            )
        self.attn_mask = torch.zeros(num_uniq_users, num_uniq_items, dtype=torch.bool)
        self.attn_mask[self.ratings > 0] = 1

        prev_ratings = torch.zeros(num_uniq_users, num_uniq_items, dtype=torch.float32)
        prev_ratings[prev_data[:, 0], prev_data[:, 1]] = prev_data[:, 2].float()
        self.exclude_prev = torch.ones_like(self.ratings, dtype=torch.float32)
        self.exclude_prev[prev_ratings > 0] = 0.0
        super().__init__()

    def __len__(self) -> int:
        return self.ratings.shape[0]

    def __getitem__(
        self, index: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            index,
            self.items[index],
            self.ratings[index],
            self.attn_mask[index],
            self.exclude_prev[index],
        )


class MFDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        val_size: int = 5,
        test_size: int = 5,
        implicit_weights: Tuple[float, float] = (0.05, 5),
        num_workers: int = 2,
        pin_memory: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False)
        self.data_path = data_path
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.implicit_weights = implicit_weights
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size
        else:
            self.batch_size_per_device = self.batch_size
        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            data = pd.read_csv(
                self.data_path,
                sep="::",
                names=["UserId", "ProductId", "Rating", "Timestamp"],
            )

            user_to_index = {
                old: new for new, old in enumerate(data.UserId.unique(), start=0)
            }
            data.UserId = data.UserId.map(user_to_index)
            data.UserId = data.UserId.astype("int32")

            item_to_index = {
                old: new for new, old in enumerate(data.ProductId.unique(), start=0)
            }
            data.ProductId = data.ProductId.map(item_to_index)
            data.ProductId = data.ProductId.astype("int32")
            num_uniq_items = data.ProductId.nunique()
            data = data.sort_values(by="Timestamp", ascending=True)

            test = data.groupby("UserId").tail(self.test_size)
            train_val = data[~data.index.isin(test.index)]

            self.test_dataset = MFTestDataset(
                test,
                train_val,
                num_uniq_items,
                self.implicit_weights,
            )

            val = train_val.groupby("UserId").tail(self.val_size)
            train = train_val[~train_val.index.isin(val.index)]
            self.val_dataset = MFTestDataset(
                val,
                train,
                num_uniq_items,
                self.implicit_weights,
            )

            self.train_dataset = MFTrainDataset(
                train,
                num_uniq_items,
                self.implicit_weights,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


class RankingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        val_size: int = 5,
        test_size: int = 5,
        num_workers: int = 2,
        pin_memory: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False)
        self.data_path = data_path
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size
        else:
            self.batch_size_per_device = self.batch_size
        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            data = pd.read_csv(
                self.data_path,
                sep="::",
                names=["UserId", "ProductId", "Rating", "Timestamp"],
            )

            user_to_index = {
                old: new for new, old in enumerate(data.UserId.unique(), start=0)
            }
            data.UserId = data.UserId.map(user_to_index)
            data.UserId = data.UserId.astype("int32")

            item_to_index = {
                old: new for new, old in enumerate(data.ProductId.unique(), start=0)
            }
            data.ProductId = data.ProductId.map(item_to_index)
            data.ProductId = data.ProductId.astype("int32")
            num_uniq_items = data.ProductId.nunique()
            data = data.sort_values(by="Timestamp", ascending=True)

            test = data.groupby("UserId").tail(self.test_size)
            train_val = data[~data.index.isin(test.index)]

            self.test_dataset = RankingTestDataset(
                test,
                train_val,
                num_uniq_items,
                -1,
            )

            val = train_val.groupby("UserId").tail(self.val_size)
            train = train_val[~train_val.index.isin(val.index)]
            self.val_dataset = RankingTestDataset(
                val,
                train,
                num_uniq_items,
                -1,
            )

            self.train_dataset = RankingTrainDataset(
                train,
                num_uniq_items,
                -1,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
