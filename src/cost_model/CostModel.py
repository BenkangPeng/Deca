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


def rowwise_occupancy(cfg, arch: Arch, op: str, N: int):
    """Resident CTAs per SM of a row-wise (Softmax / RMSNorm) config."""
    from kernels.rowwise_tir import est_regs_per_thread

    regs = (min(255, est_regs_per_thread(cfg, N, op)) + 7) // 8 * 8
    warps = cfg.threads // 32
    ctas = min(
        arch.max_registers_per_sm // (regs * cfg.threads),
        arch.max_resident_warps_per_sm // warps,
        arch.max_resident_blocks_per_sm,
    )
    return max(int(ctas), 1)


def rowwise_score(cfg, arch: Arch, op: str, M: int, N: int, type_size: int = 2) -> float:
    """
    Heuristic score of a memory-bound row-wise config, the larger the better.

    - memory parallelism: bytes of loads in flight per SM (every chunk of the
      row is requested up front in `reg` mode, `unroll` chunks in `twopass`)
      against Little's law, DRAM bandwidth per SM x ~800 ns of latency
    - phases: a CTA loads, reduces, then stores; with few resident CTAs per SM
      the reduction/store phase is not hidden by the loads of other CTAs
    - wave efficiency of the grid of M / rows_per_block CTAs
    - twopass re-reads the row; the re-read hits L2 when the rows in flight fit
    - cross-warp reductions cost a block barrier each
    - idle vector lanes of the last chunk (tail) and 8-byte accesses
    - cache hints keep twopass rows in L2 for the re-read (and stream y out)
    """
    from kernels.rowwise_tir import num_chunks

    ctas = rowwise_occupancy(cfg, arch, op, N)
    chunks = num_chunks(cfg, N)
    per_thread = (chunks if cfg.mode == "reg" else cfg.unroll) * cfg.vec * type_size
    in_flight = ctas * cfg.threads * per_thread
    need = arch.dram_bw * 1e9 / arch.sm_nums * 800e-9
    mem = min(1.0, in_flight / need)

    phase = 1.0 - 0.35 / ctas if cfg.mode == "reg" else 1.0 - 0.1 / ctas

    grid = M // cfg.rows_per_block
    slots = arch.sm_nums * ctas
    waves = math.ceil(grid / slots)
    wave_eff = grid / (waves * slots)

    passes = 1.0
    if cfg.mode == "twopass":
        footprint = arch.sm_nums * ctas * cfg.rows_per_block * N * type_size
        # Share of the rows in flight still in L2 at the re-read; evict-last
        # loads and streaming stores protect a larger part of the L2.
        hit = min(1.0, (0.8 if cfg.hint else 0.5) * arch.l2_bytes / footprint)
        # x is half of the traffic (y is written once)
        reread = 0.5 * (hit * arch.dram_bw / arch.l2_bw + (1.0 - hit))
        passes = 1.0 + reread

    nw = cfg.threads_per_row // 32
    reduce = 1.0 / (1.0 + 0.02 * math.log2(cfg.threads_per_row // 32 or 1) + (0.03 if nw > 1 else 0.0))

    tail = N / (chunks * cfg.threads_per_row * cfg.vec) if cfg.mode == "reg" else 1.0
    lanes = 1.0 if cfg.vec * type_size >= 16 else 0.97

    return mem * phase * wave_eff * reduce * (0.7 + 0.3 * tail) * lanes / passes


def gemm_tc_occupancy(cfg, arch: Arch):
    """Resident CTAs per SM of an FP16 Tensor Core GEMM config."""
    from kernels.gemm_tir import est_regs_per_thread, smem_bytes

    regs = (min(255, est_regs_per_thread(cfg)) + 7) // 8 * 8
    smem_per_sm = max(arch.shared_mem_config_options) * 1024
    ctas = min(
        smem_per_sm // (smem_bytes(cfg) + 1024),
        arch.max_registers_per_sm // (regs * cfg.threads),
        arch.max_resident_warps_per_sm // cfg.num_warps,
        arch.max_resident_blocks_per_sm,
    )
    return max(int(ctas), 1)


def gemm_wave_dram_bytes_per_flop(cfg, arch: Arch, M: int, N: int, K: int, slots: int) -> float:
    """DRAM bytes per FLOP of A / B (fp16) for one wave of `slots` concurrent CTAs.

    With grouped rasterization a wave covers `rows` CTA rows x `cols` CTA
    columns; every A row block and B column block of the wave is fetched once.
    When both operands fit comfortably in L2 they are fetched once overall.
    """
    if (M * K + K * N) * 2 <= 0.5 * arch.l2_bytes:
        return (M * K + K * N) * 2 / (2.0 * M * N * K)
    GM, GN = M // cfg.BM, N // cfg.BN
    g = cfg.raster
    cols = min(GN, math.ceil(slots / g))
    rows = min(GM, g * math.ceil(slots / (g * GN)))
    tiles = min(slots, GM * GN)
    return (rows * cfg.BM + cols * cfg.BN) / (tiles * cfg.BM * cfg.BN)


def gemm_mma_peak(cfg, arch: Arch) -> float:
    """mma.sync peak (GFLOPS) for the accumulation precision of `cfg`; GPUs
    without a faster fp16-accumulate rate fall back to the fp32 one."""
    if cfg.acc_f16:
        return getattr(arch, "peak_mma_sync_f16acc_flops", arch.peak_mma_sync_flops)
    return arch.peak_mma_sync_flops


def gemm_tc_score(cfg, arch: Arch, M: int, N: int, K: int) -> float:
    """
    Heuristic score of an FP16 Tensor Core GEMM config, the larger the better.

    Per-SM time per FLOP of the main loop, as a roofline whose non-bottleneck
    terms only partially overlap:
    - tensor cores (mma.sync, fp32 accumulation)
    - shared memory: every warp reads (WM + WN) * 16 * 2 bytes per k16 step for
      2 * WM * WN * 16 FLOPs, and the CTA writes (BM + BN) * BK * 2 bytes per stage
    - L2: every CTA streams its A rows and B columns, (BM + BN) / (BM * BN) B/FLOP
    - DRAM: the unique A / B blocks of one wave (grouped rasterization)
    Then:
    - wave efficiency of the CTA grid; latency hiding by resident warps
    - pipelining: bytes in flight (stages - 1 tiles per CTA) vs. Little's law
      on the L2 stream, ~1 us of latency
    - one block barrier per BK stage, hidden by other resident CTAs
    - ldmatrix issue overhead per MMA (small warp tiles)
    - epilogue: C is written once (fp16, 1 / K bytes per FLOP); direct
      fragment stores are 2-byte scattered writes, the staged epilogue writes
      16-byte coalesced chunks; it overlaps with other CTAs' main loops
    """
    ctas = gemm_tc_occupancy(cfg, arch)
    slots = arch.sm_nums * ctas
    grid = (M // cfg.BM) * (N // cfg.BN)
    waves = math.ceil(grid / slots)
    wave_eff = grid / (waves * slots)
    hiding = min(ctas * cfg.num_warps, 8) / 8

    sm = arch.sm_nums
    wm, wn = cfg.warp_m, cfg.warp_n
    peak = gemm_mma_peak(cfg, arch)
    t_mma = sm / (peak * 1e9)
    smem_bpf = (wm + wn) / (wm * wn) + (cfg.BM + cfg.BN) / (cfg.BM * cfg.BN)
    t_smem = smem_bpf / arch.smem_bw_per_sm
    l2_bpf = (cfg.BM + cfg.BN) / (cfg.BM * cfg.BN)
    t_l2 = l2_bpf * sm / (arch.l2_bw * 1e9)
    dram_bpf = gemm_wave_dram_bytes_per_flop(cfg, arch, M, N, K, slots)
    t_dram = dram_bpf * sm / (arch.dram_bw * 1e9)
    terms = sorted([t_mma, t_smem, t_l2, t_dram])
    t_flop = terms[-1] + 0.25 * sum(terms[:-1])

    # Bytes per SM that must be in flight to cover ~1 us of L2 latency.
    need = l2_bpf / t_mma * 1e-6
    in_flight = ctas * (cfg.stages - 1) * (cfg.BM + cfg.BN) * cfg.BK * 2
    pipe = min(1.0, in_flight / need) ** 0.5

    # SM cycles per stage at the tensor-core rate vs. ~120 cycles of barrier.
    flops_per_cycle = peak * 1e9 / sm / 2.52e9
    stage_cycles = 2 * cfg.BM * cfg.BN * cfg.BK / flops_per_cycle
    sync_eff = 1.0 / (1.0 + 120.0 / ctas / stage_cycles)

    ldm_per_mma = (wm // 16 + wn // 16) / ((wm // 16) * (wn // 8))
    issue = 1.0 / (1.0 + 0.15 * ldm_per_mma)

    epi_eff = 1.0 if cfg.epi_smem else 0.25
    t_epi = (1.0 / K) / epi_eff * sm / (arch.dram_bw * 1e9)
    t_epi *= 1.0 if ctas == 1 else 0.4

    return wave_eff * hiding * pipe * sync_eff * issue / ((t_flop + t_epi) / t_mma)


def rank_gemm_configs(configs, arch: Arch, M: int, N: int, K: int, top_n: int | None = None):
    """Sort FP16 Tensor Core GEMM configs by `gemm_tc_score` (best first)."""
    ranked = sorted(configs, key=lambda c: gemm_tc_score(c, arch, M, N, K), reverse=True)
    return ranked[:top_n] if top_n else ranked


def rank_rowwise_configs(configs, arch: Arch, op: str, M: int, N: int, top_n: int | None = None):
    """Sort row-wise configs by `rowwise_score` (best first)."""
    ranked = sorted(configs, key=lambda c: rowwise_score(c, arch, op, M, N), reverse=True)
    return ranked[:top_n] if top_n else ranked
