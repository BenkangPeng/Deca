"""Preserve root causes instead of reporting every environment error as a bad tile."""
from io import StringIO
from types import SimpleNamespace

import pytest
import torch

import test_flash_attn as cli
from arch import H20
from kernels.flash_attn_tir import FAConfig


def test_cuda_preflight_preserves_driver_error(monkeypatch):
    cause = RuntimeError("NVIDIA driver is too old")

    def fail():
        raise cause

    monkeypatch.setattr(torch.cuda, "init", fail)
    with pytest.raises(RuntimeError, match="PyTorch CUDA initialization failed") as error:
        cli.require_torch_cuda()
    assert error.value.__cause__ is cause
    assert torch.__version__ in str(error.value)
    assert "driver is too old" in str(error.value)


@pytest.mark.parametrize("stage", ["build", "validation"])
def test_no_valid_config_preserves_cause(monkeypatch, stage):
    import Solver
    from cost_model import CostModel
    from RuntimeMod import RuntimeMod

    cfg = FAConfig()
    cause = RuntimeError("underlying " + stage + " failure")
    monkeypatch.setattr(cli, "require_torch_cuda", lambda: None)
    monkeypatch.setattr(Solver, "solve_flash_attn", lambda *a: [cfg])
    monkeypatch.setattr(CostModel, "rank_fa_configs", lambda *a, **kw: [cfg])

    def build(*args, **kwargs):
        if stage == "build":
            kwargs["failures"][cfg] = cause
            return {}
        return {cfg: "unused.so"}

    def load(*args):
        raise cause

    monkeypatch.setattr(cli, "build_all", build)
    monkeypatch.setattr(RuntimeMod, "load_flash_attn", load)
    with pytest.raises(RuntimeError, match="underlying " + stage) as error:
        cli.tune(SimpleNamespace(causal=False, top_n=1, jobs=1),
                 (1, 1, 128, 128, 64), H20(), "sm_90", StringIO())
    assert error.value.__cause__ is cause
    assert "1 solver candidates, 1 selected" in str(error.value)
    assert ("0 built" if stage == "build" else "1 built") in str(error.value)
