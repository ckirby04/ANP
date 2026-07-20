"""Device selection tests.

Every run before this existed trained on the 8 GB card while its config said
`cuda:0` and its provenance recorded `cuda:0`, because CUDA's default
FASTEST_FIRST ordering disagrees with nvidia-smi's. These tests pin the two
behaviours that prevent a repeat: selection happens by name, and a mismatch
fails loudly rather than running on the wrong hardware.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.network import (  # noqa: E402
    REQUIRED_DEVICE_ORDER,
    describe_cuda_devices,
    resolve_device,
    set_cuda_device_order,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                               reason="no CUDA device")


def test_device_order_is_pinned_to_pci_bus_id():
    set_cuda_device_order()
    assert os.environ["CUDA_DEVICE_ORDER"] == REQUIRED_DEVICE_ORDER


@cuda_only
def test_devices_are_described_with_name_and_vram():
    devices = describe_cuda_devices()
    assert devices
    for d in devices:
        assert d["name"] and d["total_vram_gb"] > 0
        assert set(d) == {"index", "name", "total_vram_gb", "capability"}


@cuda_only
def test_resolves_target_card_by_name_regardless_of_index():
    """The name decides, not the index in the config string."""
    devices = describe_cuda_devices()
    target = max(devices, key=lambda d: d["total_vram_gb"])
    # Ask for a deliberately wrong index; the name match must win.
    wrong = (target["index"] + 1) % max(1, len(devices))
    dev = resolve_device(f"cuda:{wrong}", target["name"], 0.0)
    assert dev.index == target["index"]
    assert torch.cuda.get_device_name(dev) == target["name"]


@cuda_only
def test_sets_current_device_to_the_selection():
    """autocast and GradScaler act on the current device, not the tensor's."""
    devices = describe_cuda_devices()
    target = max(devices, key=lambda d: d["total_vram_gb"])
    dev = resolve_device("cuda:0", target["name"], 0.0)
    assert torch.cuda.current_device() == dev.index


@cuda_only
def test_index_override_alone_is_ineffective_and_warns():
    """Overriding the index without the name does NOT move the job.

    The name match decides, so `--set device=cuda:N` on its own resolves back
    to the named card. It warns rather than failing, so a caller who wants a
    different physical card must change the name too. scripts/run_arm.ps1 -Gpu
    exists to set both together.
    """
    devices = describe_cuda_devices()
    if len(devices) < 2:
        pytest.skip("needs two CUDA devices")
    target = max(devices, key=lambda d: d["total_vram_gb"])
    other = min(devices, key=lambda d: d["total_vram_gb"])

    with pytest.warns(UserWarning, match="using the name match"):
        dev = resolve_device(f"cuda:{other['index']}", target["name"], 0.0)
    assert dev.index == target["index"], "index override should not win"


@cuda_only
def test_unknown_device_name_raises_with_inventory():
    with pytest.raises(RuntimeError, match="no CUDA device matching"):
        resolve_device("cuda:0", "RTX 9090 Ti Ultra", 0.0)


@cuda_only
def test_insufficient_vram_raises_naming_the_card():
    devices = describe_cuda_devices()
    target = max(devices, key=lambda d: d["total_vram_gb"])
    with pytest.raises(RuntimeError, match="below the required"):
        resolve_device("cuda:0", target["name"],
                       target["total_vram_gb"] + 100.0)


@cuda_only
def test_out_of_range_index_raises():
    n = len(describe_cuda_devices())
    with pytest.raises(RuntimeError, match="only .* device"):
        resolve_device(f"cuda:{n + 5}", "", 0.0)


def test_cpu_device_with_a_gpu_requirement_raises():
    with pytest.raises(RuntimeError, match="not CUDA"):
        resolve_device("cpu", "RTX 5060 Ti", 15.0)


def test_config_defaults_require_the_sixteen_gig_card():
    from config import Config
    cfg = Config()
    assert cfg.require_device_name == "RTX 5060 Ti"
    assert cfg.require_min_vram_gb == 15.0


def test_no_bare_cuda_device_strings_in_source():
    """No code path may target `cuda` or `cuda:0` outside the config default.

    Device TYPE strings, as in autocast("cuda") or GradScaler("cuda"), are
    fine: they follow torch.cuda.set_device, which resolve_device sets.
    """
    src = Path(__file__).resolve().parents[1] / "src"
    offenders = []
    for path in src.rglob("*.py"):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or "noqa: device" in line:
                continue
            for bad in ('torch.device("cuda")', "torch.device('cuda')",
                        '.to("cuda")', ".to('cuda')",
                        '.to("cuda:0")', ".to('cuda:0')", ".cuda()"):
                if bad in line:
                    offenders.append(f"{path.name}:{n}: {stripped}")
    assert not offenders, "bare device targets found:\n" + "\n".join(offenders)


def test_determinism_flags_are_not_silently_enabled():
    """Determinism is NOT enforced, and the docs must keep saying so.

    Enabling it mid-pilot would make the remaining arms non-comparable with
    dense_seed0 and rigl_seed0, which already ran without it.
    """
    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.deterministic
