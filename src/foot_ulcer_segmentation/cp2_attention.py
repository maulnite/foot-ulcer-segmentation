"""Stage-local CP2 attention changes; the pinned source package is never edited.

CBAM channel/spatial definition adapted from Jongchan/attention-module,
459efad0e05ee7dde50c41ca10a3d0800bc3792a (MIT, licenses/cbam-MIT.txt).
"""

import types

import torch
from torch import nn
from torch.nn import functional as F

from .cp2_train import tensor_fingerprint
from .reference import build_model


VARIANTS = ("A1", "A2", "A3", "A5", "A6")


class LocalSpatial(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x)).sigmoid()


class ImageGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.hidden = nn.Linear(channels, max(8, channels // 16))
        self.output = nn.Linear(max(8, channels // 16), 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x.mean(dim=(2, 3)))))


class CapacityBlock(nn.Module):
    """An additive feature correction, not a sigmoid map multiplying x."""
    def __init__(self, channels):
        super().__init__()
        hidden = max(1, round(max(8, channels // 16) / 2))
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.expand = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, x):
        return self.expand(F.relu(self.reduce(self.depthwise(x))))


class CBAM(nn.Module):
    """Author avg+max channel MLP, then 7x7 spatial gate with author's BN."""
    def __init__(self, channels):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels, channels // 16), nn.ReLU(),
                                 nn.Linear(channels // 16, channels))
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False),
                                     nn.BatchNorm2d(1, eps=1e-5, momentum=.01))

    def forward(self, x):
        average = F.avg_pool2d(x, x.shape[2:]).flatten(1)
        maximum = F.max_pool2d(x, x.shape[2:]).flatten(1)
        x = x * (self.mlp(average) + self.mlp(maximum)).sigmoid()[:, :, None, None]
        maps = torch.cat((x.max(dim=1, keepdim=True).values, x.mean(dim=1, keepdim=True)), dim=1)
        return x * self.spatial(maps).sigmoid()


def fusion_weights(block, x):
    logits = block.fusion_log_prior[None].expand(x.shape[0], -1)
    if block.phase_a_variant == "A2":
        logits = block.fusion_logits[None].expand(x.shape[0], -1)
    elif block.phase_a_variant == "A3" and block.phase_a_intervention != "fixed_prior_gate":
        logits = logits + block.image_gate(x)
    weights = logits.softmax(dim=1)
    if block.phase_a_intervention == "drop_local":
        # Remove the branch and renormalize the remaining mass explicitly.
        mask = weights.new_tensor([1, 1, 0])[None]
        weights = weights * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return weights


def _forward(block, x, skip=None, strategy="add"):
    if skip is None or strategy != "add":
        raise ValueError("Only the two predeclared regular skip-connected stages may change")
    x = torch.cat((F.interpolate(x, scale_factor=2, mode="nearest"), skip), dim=1)
    variant = block.phase_a_variant
    weights = local = None
    if variant == "A6":
        fused = 3 * block.cbam(x)
    else:
        maximum, additive = block.attention1(x), block.attention2(x)
        if variant == "A5":
            fused = maximum + additive + block.capacity_block(x)
        else:
            local = block.local_attention(x)
            weights = fusion_weights(block, x)
            fused = 3 * (weights[:, 0, None, None, None] * (additive / 2)
                         + weights[:, 1, None, None, None] * maximum
                         + weights[:, 2, None, None, None] * (x * local))
    if block.phase_a_capture_enabled:
        entry = {"input_shape": list(x.shape),
                 "input_rms": x.detach().square().mean(dim=(1, 2, 3)).sqrt().cpu().tolist(),
                 "fused_rms": fused.detach().square().mean(dim=(1, 2, 3)).sqrt().cpu().tolist()}
        if weights is not None:
            entry["weights"] = weights.detach().cpu().tolist()
            entry["local_map_mean"] = local.detach().mean(dim=(1, 2, 3)).cpu().tolist()
            entry["local_map_std"] = local.detach().std(dim=(1, 2, 3), unbiased=False).cpu().tolist()
        block.phase_a_captures.append(entry)
    return block.conv1(fused)


def attach_attention(model, variant, config):
    if variant not in ("A0", *VARIANTS):
        raise ValueError("Unknown phase-A architecture")
    if config["phase_a"]["modified_stages"] != [2, 3] or len(model.decoder.blocks) != 5:
        raise ValueError("Stage-placement expansion is a separate experiment")
    if config["phase_a"]["fusion"]["scale"] != 3 or config["phase_a"]["cbam"]["output_scale"] != 3:
        raise ValueError("Fusion gain must match the registered scale convention")
    if variant == "A0":
        return model
    for stage in config["phase_a"]["modified_stages"]:
        block = model.decoder.blocks[stage]
        if block.attention_type != "pscse" or hasattr(block, "phase_a_variant"):
            raise ValueError("Expected untouched source P-scSE blocks")
        channels = block.conv1[0].in_channels
        if channels != {2: 176, 3: 128}[stage]:
            raise ValueError("Source decoder channel layout changed")
        block.phase_a_variant = variant
        block.phase_a_intervention = None
        block.phase_a_capture_enabled = False
        block.phase_a_captures = []
        # Construction must neither consume shared RNG nor overwrite source tensors.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config["seed"] + 20000 + stage)
            if variant in ("A1", "A2", "A3"):
                block.local_attention = LocalSpatial(channels)
                block.register_buffer("fusion_log_prior", torch.tensor(
                    config["phase_a"]["fusion"]["initial_weights"], dtype=torch.float32).log())
                if variant == "A2":
                    block.fusion_logits = nn.Parameter(block.fusion_log_prior.clone())
                if variant == "A3":
                    torch.manual_seed(config["seed"] + 20100 + stage)
                    block.image_gate = ImageGate(channels)
            elif variant == "A5":
                block.capacity_block = CapacityBlock(channels)
            else:
                torch.manual_seed(config["seed"] + 20200 + stage)
                block.cbam = CBAM(channels)
        block.forward = types.MethodType(_forward, block)
    return model


def build_attention_model(config, variant, initial):
    model = build_model(config, pretrained=False)
    model.load_state_dict(initial["state_dict"])
    before = tensor_fingerprint(model.state_dict())
    if before != config["phase_a"]["control_initial_tensor_sha256"]:
        raise ValueError("Shared source tensors differ from B1's actual initialization")
    attach_attention(model, variant, config)
    state = model.state_dict()
    for name, expected in initial["state_dict"].items():
        if not torch.equal(state[name], expected):
            raise ValueError(f"Attention changed a pairable original tensor: {name}")
    original_names = set(initial["state_dict"])
    additional = {name: value for name, value in state.items() if name not in original_names}
    local = {name: value for name, value in additional.items() if ".local_attention." in name}
    return model, {"source_tensor_sha256": before, "all_original_named_tensors_equal": True,
                   "additional_tensor_sha256": tensor_fingerprint(additional),
                   "local_tensor_sha256": tensor_fingerprint(local) if local else None,
                   "complete_initial_tensor_sha256": tensor_fingerprint(state),
                   "additional_tensors": {k: list(v.shape) for k, v in additional.items()},
                   "final_stage_forward_unchanged": True}


def model_profile(model, inputs):
    """Count actual called parameters and conv/linear MACs, not all operations."""
    active, handles, calls, macs = set(), [], {}, [0]

    def record(name):
        def hook(module, args, output):
            calls[name] = calls.get(name, 0) + 1
            active.update(id(p) for p in module.parameters(recurse=False))
            if isinstance(module, nn.Conv2d):
                macs[0] += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
            elif isinstance(module, nn.Linear):
                macs[0] += output.numel() * module.in_features
        return hook

    for name, module in model.named_modules():
        handles.append(module.register_forward_hook(record(name)))
    try:
        model.eval()
        with torch.no_grad():
            prediction = model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return {"registered_parameters": sum(p.numel() for p in model.parameters()),
            "active_forward_parameters": sum(p.numel() for p in model.parameters() if id(p) in active),
            "conv_linear_MACs_per_image": macs[0] // inputs.shape[0],
            "MAC_caveat": "multiply-accumulate counts for Conv2d/Linear only; pooling, normalization, interpolation and elementwise operations excluded",
            "input_shape": list(inputs.shape), "output_shape": list(prediction.shape),
            "decoder_calls": {name: calls.get(name, 0) for stage in range(5) for name in
                              [f"decoder.blocks.{stage}.{part}" for part in ("attention1", "attention2", "conv1", "conv2")]}}


def capture_attention(model, enabled=True, intervention=None):
    if intervention not in (None, "drop_local", "fixed_prior_gate"):
        raise ValueError("Unregistered intervention")
    for block in model.decoder.blocks:
        if hasattr(block, "phase_a_variant"):
            block.phase_a_capture_enabled = enabled
            block.phase_a_intervention = intervention
            block.phase_a_captures = []
