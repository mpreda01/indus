import torch


class MaskedDepthRegressionLoss(torch.nn.Module):
    """
    L1 or L2 regression loss on metric depth, computed only over valid ground-truth pixels
    (finite and > 0; 0 marks out-of-range/sky pixels in the dataset).
    """

    def __init__(self, kind: str = 'l1'):
        super().__init__()
        if kind not in ('l1', 'l2'):
            raise ValueError(f'Unknown depth loss "{kind}", expected "l1" or "l2"')
        self.kind = kind

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = input.float()
        target = target.float()
        if input.ndim == 4:
            input = input.squeeze(1)
        if target.ndim == 4:
            target = target.squeeze(1)

        valid = torch.isfinite(target) & (target > 0)
        if not valid.any():
            # keep the graph connected so backward() does not fail on an all-invalid batch
            return input.sum() * 0.0

        diff = input[valid] - target[valid]
        return diff.abs().mean() if self.kind == 'l1' else (diff * diff).mean()
