import torch


class TokenBatchLoader:
    """
    Samples random contiguous token sequences for
    autoregressive next-token prediction.
    """

    def __init__(
        self,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        context_length: int,
        batch_size: int,
        device: str
    ):
        self.train_data = train_data
        self.val_data = val_data

        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device


    def get_batch(self, split: str):
        """
        Returns:

            x: (batch_size, context_length)
            y: (batch_size, context_length)

        y is x shifted one token into the future.
        """

        if split == "train":
            data = self.train_data

        elif split == "val":
            data = self.val_data

        else:
            raise ValueError(
                "split must be 'train' or 'val'"
            )

        max_start = (
            len(data)
            - self.context_length
            - 1
        )

        if max_start <= 0:
            raise ValueError(
                "Dataset is too short for the "
                "requested context length."
            )

        # Pick random starting positions
        starts = torch.randint(
            0,
            max_start,
            (self.batch_size,)
        )

        # Input sequences
        x = torch.stack([
            data[
                i:i + self.context_length
            ]
            for i in starts
        ])

        # Same sequences shifted by 1 token
        y = torch.stack([
            data[
                i + 1:
                i + self.context_length + 1
            ]
            for i in starts
        ])

        return (
            x.to(self.device),
            y.to(self.device)
        )