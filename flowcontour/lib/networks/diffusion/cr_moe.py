"""Contour-Routed Mixture-of-Experts（CR-MoE）。

整条闭合轮廓是路由单元。router 只看局部图像 token 的 mean/std，一次输出
[N, 3, 4] logits，分别给 DiT 第 2/4/6 层。同一轮廓上所有点共享 Top-2 专家，
没有点级路由。

被路由的块保留原来的 Dense SwiGLU 作为永远执行的共享专家，再加 4 个稀疏
残差专家。速度输出头保持 Dense。交叉注意力和时间/阶段 AdaLN 不进 router。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """把熵除以 log(E)，便于跨专家数比较。"""
    expert_count = int(probabilities.shape[-1])
    if expert_count <= 1:
        return probabilities.new_zeros(probabilities.shape[:-1])
    safe = probabilities.clamp_min(1e-8)
    return -(safe * safe.log()).sum(dim=-1) / math.log(float(expert_count))


class ContourRoutePath(nn.Module):
    """一次轮廓级决策，同时给出三层 routed block 的专家 logits。

    只用局部图像 token 的均值和标准差选专家。主干的全局/局部交叉注意力
    以及时间、阶段 AdaLN 仍照常进入 DiT，但不作为 router 输入。
    """

    def __init__(
        self,
        dim: int = 256,
        num_routed_blocks: int = 3,
        num_experts: int = 4,
        hidden_dim: int = 256,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if int(num_routed_blocks) < 1:
            raise ValueError("num_routed_blocks must be positive")
        if int(num_experts) < 2:
            raise ValueError("CR-MoE requires at least two experts")
        if float(temperature) <= 0.0:
            raise ValueError("router temperature must be positive")

        self.dim = int(dim)
        self.num_routed_blocks = int(num_routed_blocks)
        self.num_experts = int(num_experts)
        self.temperature = float(temperature)
        descriptor_dim = 2 * self.dim
        self.descriptor_dim = int(descriptor_dim)
        self.norm = nn.LayerNorm(descriptor_dim)
        self.trunk = nn.Sequential(
            nn.Linear(descriptor_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.route_heads = nn.Linear(
            int(hidden_dim), self.num_routed_blocks * self.num_experts
        )
        nn.init.normal_(self.route_heads.weight, std=1e-3)
        nn.init.zeros_(self.route_heads.bias)
        self._last_logits: torch.Tensor | None = None

    @staticmethod
    def _moments(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """沿点维汇聚：descriptor = concat(mean, std)。"""
        mean = tokens.float().mean(dim=1)
        variance = tokens.float().var(dim=1, unbiased=False)
        return mean, torch.sqrt(variance + 1e-6)

    def forward(
        self,
        local_image_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """输入 [N,L,D]，输出 [N, num_routed_blocks, num_experts]。"""
        if local_image_tokens.ndim != 3:
            raise ValueError("router token inputs must have shape [N,L,D]")
        if local_image_tokens.shape[-1] != self.dim:
            raise ValueError("router token channel mismatch")
        batch_size = int(local_image_tokens.shape[0])
        mean, scale = self._moments(local_image_tokens)
        descriptor = torch.cat((mean, scale), dim=-1)
        hidden = self.trunk(self.norm(descriptor))
        logits = self.route_heads(hidden).view(
            batch_size, self.num_routed_blocks, self.num_experts
        )
        self._last_logits = (logits / self.temperature).to(local_image_tokens.dtype)
        return self._last_logits

    def diagnostics(self) -> dict[str, torch.Tensor]:
        if self._last_logits is None:
            return {}
        probabilities = self._last_logits.float().softmax(dim=-1)
        top1 = probabilities.argmax(dim=-1)
        agreement = (
            (top1[:, 1:] == top1[:, :-1]).float().mean()
            if self.num_routed_blocks > 1
            else probabilities.new_ones(())
        )
        return {
            "route_entropy": _normalized_entropy(probabilities).mean().detach(),
            "adjacent_block_agreement": agreement.detach(),
        }


class ResidualSwiGLUExpert(nn.Module):
    """输出近零初始化的残差专家，避免破坏已有 Dense FFN。"""

    def __init__(
        self,
        dim: int = 256,
        hidden_dim: int = 256,
        output_init_std: float = 1e-4,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(int(dim), int(hidden_dim), bias=False)
        self.v = nn.Linear(int(dim), int(hidden_dim), bias=False)
        self.w2 = nn.Linear(int(hidden_dim), int(dim), bias=False)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.normal_(self.w2.weight, std=float(output_init_std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.v(x)) * self.w1(x))


class ContourSharedResidualExperts(nn.Module):
    """一条轮廓共用一组 Top-k 权重；专家输出加到共享 Dense FFN 上。"""

    def __init__(
        self,
        dim: int = 256,
        num_experts: int = 4,
        top_k: int = 2,
        expert_hidden_dim: int = 256,
        load_balance_weight: float = 1e-2,
        specialization_weight: float = 5e-3,
        expert_init_std: float = 1e-4,
    ) -> None:
        super().__init__()
        if not 1 <= int(top_k) <= int(num_experts):
            raise ValueError("top_k must lie in [1, num_experts]")
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.load_balance_weight = float(load_balance_weight)
        self.specialization_weight = float(specialization_weight)
        self.experts = nn.ModuleList(
            [
                ResidualSwiGLUExpert(
                    dim=int(dim),
                    hidden_dim=int(expert_hidden_dim),
                    output_init_std=float(expert_init_std),
                )
                for _ in range(self.num_experts)
            ]
        )
        self._last_regularization: torch.Tensor | None = None
        self._last_diagnostics: dict[str, torch.Tensor] = {}

    def _regularization(
        self,
        probabilities: torch.Tensor,
        top_indices: torch.Tensor,
    ) -> torch.Tensor:
        """负载均衡：软/硬使用率靠近均匀；特化：单次决策别太糊、边际别塌成单专家。"""
        uniform = probabilities.new_full(
            (self.num_experts,), 1.0 / float(self.num_experts)
        )
        soft_load = probabilities.float().mean(dim=0)
        hard_load = F.one_hot(
            top_indices, num_classes=self.num_experts
        ).float().mean(dim=(0, 1))
        load_penalty = (soft_load - uniform).square().mean()
        load_penalty = load_penalty + (hard_load - uniform).square().mean()

        conditional_entropy = _normalized_entropy(probabilities.float()).mean()
        marginal_entropy = _normalized_entropy(soft_load.unsqueeze(0)).mean()
        specialization_penalty = conditional_entropy + (1.0 - marginal_entropy)
        regularization = (
            self.load_balance_weight * load_penalty
            + self.specialization_weight * specialization_penalty
        )
        self._last_diagnostics = {
            "soft_load_min": soft_load.min().detach(),
            "hard_load_min": hard_load.min().detach(),
            "route_entropy": conditional_entropy.detach(),
            "dead_experts": (hard_load < 0.01).sum().detach(),
        }
        return regularization.to(probabilities.dtype)

    def forward(
        self,
        contour_tokens: torch.Tensor,
        route_logits: torch.Tensor,
    ) -> torch.Tensor:
        """route_logits 形状 [N, E]，没有点维。"""
        expected = (contour_tokens.shape[0], self.num_experts)
        if route_logits.shape != expected:
            raise ValueError(
                f"route logits must have shape {expected}, got {tuple(route_logits.shape)}"
            )
        logits = route_logits.float()
        probabilities = logits.softmax(dim=-1)
        top_values, top_indices = torch.topk(logits, k=self.top_k, dim=-1)
        top_weights = top_values.softmax(dim=-1).to(contour_tokens.dtype)
        output = torch.zeros_like(contour_tokens)

        for expert_id, expert in enumerate(self.experts):
            selected = torch.nonzero(top_indices == expert_id, as_tuple=False)
            if selected.numel() == 0:
                continue
            contour_ids = selected[:, 0]
            slots = selected[:, 1]
            expert_output = expert(contour_tokens[contour_ids])
            weights = top_weights[contour_ids, slots].view(-1, 1, 1)
            output.index_add_(0, contour_ids, expert_output * weights)

        self._last_regularization = self._regularization(
            probabilities, top_indices
        )
        return output

    def reg_loss(self) -> torch.Tensor:
        if self._last_regularization is None:
            return next(self.parameters()).new_zeros(())
        return self._last_regularization

    def diagnostics(self) -> dict[str, torch.Tensor]:
        return dict(self._last_diagnostics)


def collect_cr_moe_diagnostics(module: nn.Module) -> dict[str, torch.Tensor]:
    """前向后收集 router / 各 routed 层的负载与死专家诊断。"""

    diagnostics: dict[str, torch.Tensor] = {}
    router = getattr(module, "_contour_moe_router", None)
    if router is not None:
        for key, value in router.diagnostics().items():
            diagnostics[f"contour.{key}"] = value
    routed_indices = module._cr_moe_layer_indices
    layers = getattr(module, "dit_layers", ())
    for layer_index in routed_indices:
        experts = getattr(layers[layer_index], "routed_moe", None)
        if experts is None:
            continue
        for key, value in experts.diagnostics().items():
            diagnostics[f"block{layer_index + 1}.{key}"] = value
    return diagnostics


CRMoE = ContourSharedResidualExperts
ContourRoutedMoE = ContourSharedResidualExperts

__all__ = (
    "ContourRoutePath",
    "ContourSharedResidualExperts",
    "CRMoE",
    "ContourRoutedMoE",
    "collect_cr_moe_diagnostics",
)
