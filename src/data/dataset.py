import torch
from torch.utils.data import Dataset


class DemandForecastDataset(Dataset):

    def __init__(
        self,
        X,
        y,
        task_ids
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )

        self.task_ids = torch.tensor(
            task_ids,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.y)

    def __getitem__(self, index):

        return {
            "sequence": self.X[index],
            "target": self.y[index],
            "task_id": self.task_ids[index]
        }