from typing import Optional, Tuple

import torch


def ndcg(
    pred: torch.Tensor,
    target: torch.Tensor,
    k: int = 5,
    gain_type: str = "identity",
    reduction: str = "mean",
) -> torch.Tensor:
    def dcg(
        target: torch.Tensor, k: int = 5, gain_type: str = "identity"
    ) -> torch.Tensor:
        if not k == -1:
            target = target[..., :k]
        if gain_type == "exp2":
            gain = target.pow(2.0) - 1.0
        else:
            gain = target
        discount = 1 / torch.log2(torch.arange(gain.shape[-1], device=gain.device) + 2)
        return (gain * discount).sum(dim=-1)

    index = pred.argsort(dim=-1, descending=True)
    cur_dcg = dcg(target.gather(-1, index), k=k, gain_type=gain_type)
    index = target.argsort(dim=-1, descending=True)
    ideal_dcg = dcg(target.gather(-1, index), k=k, gain_type=gain_type) + 1e-9
    if reduction == "mean":
        return (cur_dcg / ideal_dcg).mean()
    elif reduction == "sum":
        return (cur_dcg / ideal_dcg).sum()
    else:
        return cur_dcg / ideal_dcg


def ideal_dcg(target: torch.Tensor, gain_type: str = "identity") -> torch.Tensor:
    target = target.gather(-1, target.argsort(dim=-1, descending=True))
    if gain_type == "exp2":
        gain = target.pow(2.0) - 1.0
    else:
        gain = target
    discount = 1 / torch.log2(torch.arange(gain.shape[-1], device=gain.device) + 2)
    return (gain * discount).sum(dim=-1) + 1e-9


class LambdaRankLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        pred: torch.Tensor,
        target: torch.Tensor,
        gain_type: str = "identity",
        top_k: int = 10,
        sigma: float = 1.0,
        grad_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return 1 - ndcg(
            pred=pred, target=target, k=top_k, gain_type=gain_type, reduction="mean"
        )

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        pred, target, gain_type, top_k, sigma, grad_weights = inputs
        if grad_weights is None:
            grad_weights = torch.ones_like(pred)
        ctx.save_for_backward(pred, target, grad_weights)
        ctx.gain_type = gain_type
        ctx.sigma = sigma

    @staticmethod
    def backward(ctx, grad_output) -> Tuple[torch.Tensor, None, None, None, None]:
        pred, target, grad_weights = ctx.saved_tensors
        sigma, gain_type = ctx.sigma, ctx.gain_type
        B = pred.size(0)
        pred = pred.view(B, -1, 1)
        target = target.view(B, -1, 1)
        pos_pairs_score_diff = 1.0 + torch.exp(sigma * (pred - pred.transpose(-2, -1)))
        rel_diff = target - target.transpose(-2, -1)
        pos_pairs = (rel_diff > 0).float()
        neg_pairs = (rel_diff < 0).float()
        Sij = pos_pairs - neg_pairs
        rank_net = sigma * (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff)

        N = 1 / ideal_dcg(target.view(B, -1), gain_type)
        gain_diff = target - target.transpose(-2, -1)
        rank_order = pred.view(B, -1).argsort(descending=True).argsort() + 1
        rank_order_tensor = rank_order.view(B, -1, 1)
        decay_diff = 1.0 / torch.log2(rank_order_tensor + 1.0) - 1.0 / torch.log2(
            rank_order_tensor.transpose(-2, -1) + 1.0
        )
        delta_ndcg = torch.abs(N.view(B, 1, 1) * gain_diff * decay_diff)

        lambda_update = rank_net * delta_ndcg
        lambda_update = torch.sum(lambda_update, -1, keepdim=False)
        return grad_output * grad_weights * lambda_update, None, None, None, None, None


def lambdarank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gain_type: str = "identity",
    top_k: int = 10,
    sigma: float = 1.0,
    grad_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss = LambdaRankLoss.apply(
        pred,
        target,
        gain_type,
        top_k,
        sigma,
        grad_weights,
    )
    return loss
