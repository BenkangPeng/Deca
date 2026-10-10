import math

from arch.Arch import Arch
from operators.Op import Operator


class CostModel:
    def __init__(self):
        pass

    def cost(self, kernel: Operator, arch: Arch, compute_pattern: list[tuple[int, int, int]]):
        pass


def fa_occupancy(cfg, arch: Arch, D: int, grid: int):
    """Returns (resident CTAs per SM, theoretical occupancy) of a FlashAttention config."""
    from kernels.flash_attn_tir import est_regs_per_thread, smem_bytes

    regs = min(255, est_regs_per_thread(cfg, D))
    regs = (regs + 7) // 8 * 8  # register allocation granularity
    regs_per_cta = regs * cfg.threads
    warps_per_cta = cfg.num_warps
    # The driver reserves 1 KB of shared memory per CTA.
    smem_per_cta = smem_bytes(cfg, D) + 1024
    smem_per_sm = max(arch.shared_mem_config_options) * 1024

    ctas = min(
        smem_per_sm // smem_per_cta,
        arch.max_registers_per_sm // regs_per_cta,
        arch.max_resident_warps_per_sm // warps_per_cta,
        arch.max_resident_blocks_per_sm,
    )
    ctas = max(int(ctas), 1)
    return ctas, ctas * warps_per_cta / arch.max_resident_warps_per_sm


def fa_score(cfg, arch: Arch, shape, causal: bool = False) -> float:
    """
    Heuristic score of a FlashAttention config, the larger the better.

    - roofline of one SM per FLOP: tensor-core time vs. shared-memory time
      (every warp reads the whole K/V tile, plus Q when `q_in_smem`, so the
      bytes per FLOP are 1/WM + q_in_smem/(2*Bc) with WM = Q rows per warp)
      vs. L2 time (every CTA streams K/V once: 1/Br bytes per FLOP); the
      non-bottleneck terms only partially overlap
    - O rescaling in the online softmax costs WM*D FP32 ops per KV tile
    - wave efficiency: the tail of the last wave is wasted
    - latency hiding: resident warps per SM, saturating at 8 (two per sub-core)
    - pipelining: a single K/V buffer already overlaps the K(j+1)/V(j) loads with
      compute; deeper rings mostly cost shared memory and slot arithmetic
    - padding (no swizzle) wastes shared memory and address bandwidth
    - register spills are penalized
    - causal: tiles crossing the diagonal are computed in full, so only the
      useful fraction of the computed score tiles counts
    """
    from kernels.flash_attn_tir import est_regs_per_thread

    B, H, S_q, S_kv, D = shape
    grid = (S_q // cfg.Br) * B * H
    ctas, _ = fa_occupancy(cfg, arch, D, grid)
    slots = arch.sm_nums * ctas
    waves = math.ceil(grid / slots)
    wave_eff = grid / (waves * slots)

    warps = ctas * cfg.num_warps
    hiding = min(warps, 8) / 8

    wm = cfg.Br // cfg.num_warps
    t_mma = arch.sm_nums / (arch.peak_mma_sync_flops * 1e9)
    t_smem = (1 / wm + int(cfg.q_in_smem) / (2 * cfg.Bc)) / arch.smem_bw_per_sm
    t_l2 = (1 / cfg.Br) * arch.sm_nums / (arch.l2_bw * 1e9)
    t_rescale = t_mma * 2.0 / cfg.Bc
    terms = sorted([t_mma, t_smem, t_l2])
    t_flop = terms[-1] + 0.3 * (terms[0] + terms[1]) + t_rescale

    pipe = 1.0 - 0.02 * (cfg.stages - 1)
    layout = 1.0 if cfg.swizzle else 0.96

    regs = est_regs_per_thread(cfg, D)
    spill = 1.0 if regs <= 255 else max(0.5, 1.0 - 0.002 * (regs - 255))

    useful = 1.0
    if causal:
        nq = S_q // cfg.Br
        computed = sum(-(-(i + 1) * cfg.Br // cfg.Bc) * cfg.Bc for i in range(nq)) * cfg.Br
        useful = S_q * (S_q + 1) / 2 / computed

    return wave_eff * hiding * pipe * layout * spill * useful / (t_flop / t_mma)


def rank_fa_configs(configs, arch: Arch, shape, causal: bool = False, top_n: int | None = None):
    """Sort FlashAttention configs by `fa_score` (best first)."""
    ranked = sorted(configs, key=lambda c: fa_score(c, arch, shape, causal), reverse=True)
    return ranked[:top_n] if top_n else ranked
