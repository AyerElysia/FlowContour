"""Keep the signed checkpoint's net.* namespace without another trainer."""

import torch
from torch import nn


class FlowContourNetworkWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, batch):
        output = self.net(batch["inp"], batch)
        loss = output["diff_loss"]
        if not torch.isfinite(loss).all():
            raise FloatingPointError("nonfinite Flow loss")
        return output, loss, {"loss": loss.detach(), "diff_loss": loss.detach()}, {}
