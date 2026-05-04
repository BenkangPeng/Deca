from typing import Tuple, Optional, List
from src.utils.helper import GPUDetector
from arch.Arch import Arch
from arch.H100 import H100
from arch.RTX4090 import RTX4090
from arch.V100 import V100

gpu_config = GPUDetector.create_gpu_instance(verbose=False)


def cal_theoretical_occupancy(
    config_size_kb: float,
    shared_mem_per_cta_kb: float,
    regs_per_cta: int,
    warps_per_cta: int,
    max_resident_blocks_per_sm: int,
    max_resident_warps_per_sm: int,
    max_registers_per_sm: int,
    grid_size: int,
    sm_nums: int,
) -> Tuple[int, float]:
    """Calculate theoretical occupancy for a given shared memory configuration."""
    # The maximum number of CTAs that can be executed on each SM.
    # Maybe some SMs are not fully occupied.
    max_ctas_per_sm_executed: int = (grid_size + sm_nums - 1) // sm_nums

    # Calculate the maximum number of CTAs limited by shared memory,
    # registers, warps, and blocks.
    ctas_by_smem = (
        int(config_size_kb // shared_mem_per_cta_kb)
        if shared_mem_per_cta_kb > 0
        else max_resident_blocks_per_sm
    )
    ctas_by_registers = max_registers_per_sm // regs_per_cta
    ctas_by_warps = max_resident_warps_per_sm // warps_per_cta
    ctas_by_blocks = min(max_resident_blocks_per_sm, max_ctas_per_sm_executed)

    # The maximum number of CTAs that can be executed on each SM.
    max_resident_ctas = int(
        min(ctas_by_smem, ctas_by_registers, ctas_by_warps, ctas_by_blocks)
    )
    max_resident_warps = warps_per_cta * max_resident_ctas

    # The theoretical occupancy is the ratio of the maximum number of
    # warps that can be executed on each SM to the maximum number of
    # warps that can be executed on each SM.
    return max_resident_ctas, max_resident_warps / max_resident_warps_per_sm

def byte_size(dtype: str) -> int:
    if dtype == "float32":
        return 4
    elif dtype == "float16":
        return 2
    else:
        raise ValueError(f"Unsupported data type: {dtype}")

def cal_theoretical_occupancy_tile(BM, BN, BK, RM, RN, RK, dtype: str) -> float:
    """Calculate theoretical occupancy for a given tile pattern."""
    # assert (
    #     False
    # ), "This function has some features not implemented yet (M, K, N, and arch)."

    config_size_kb = 64.0
    shared_mem_per_cta_kb: float = float((BM * BK + BN * BK) * byte_size(dtype)) / 1024.0
    regs_per_cta = RM * RN * byte_size(dtype) / 4
    assert BM % RM == 0
    assert BN % RN == 0
    block_size = (BM // RM) * (BN // RN)
    warp_size = 32
    warps_per_cta = block_size // warp_size

    if warps_per_cta <= 0:
        return 0

    max_resident_blocks_per_sm = gpu_config.max_resident_blocks_per_sm
    max_resident_warps_per_sm = gpu_config.max_resident_warps_per_sm
    max_registers_per_sm = gpu_config.max_registers_per_sm
    grid_size = ((4096 + BM - 1) // BM) * ((4096 + BN - 1) // BN)
    sm_nums = gpu_config.sm_nums

    max_resident_ctas, occ = cal_theoretical_occupancy(
        config_size_kb,
        shared_mem_per_cta_kb,
        regs_per_cta,
        warps_per_cta,
        max_resident_blocks_per_sm,
        max_resident_warps_per_sm,
        max_registers_per_sm,
        grid_size,
        sm_nums,
    )

    waves = float(grid_size) / max_resident_ctas

    return waves, occ


def simulate_shared_mem_config_size(
    gpu_config: Arch,
    grid_size: int,
    sm_nums: int,
    block_size: int,
    num_regs_per_thread: int,
    static_shared_mem_kb: float,
    dynamic_shared_mem_kb: Optional[float] = None,
    driver_shared_mem_kb: Optional[float] = None,
    warp_size: int = 32,
) -> dict[float, Tuple[int, float]]:
    """
    Simulate CUDA driver shared memory configuration size calculation logic.

    Core logic (based on user description):
    1. Calculate total shared memory per CTA X = static + dynamic + driver
    2. Choose configuration size from GPU available options
    3. Configuration must fit at least 1 CTA
    4. Choose configuration that maximizes occupancy
    5. Must satisfy register, shared memory, warp, block limitations

    Parameters:
        gpu_config: GPU configuration information
        grid_size: Grid size (number of CTAs)
        sm_nums: Number of SMs on the GPU
        block_size: Block size (number of threads)
        num_regs_per_thread: Number of registers used per thread
        static_shared_mem_kb: Static shared memory size (KB)
        dynamic_shared_mem_kb: Dynamic shared memory size (KB)
        driver_shared_mem_kb: Driver-allocated shared memory (KB)
        warp_size: Warp size (default 32)

    Returns:
        Dictionary mapping config size (KB) to (max_resident_ctas, theoretical_occupancy)
    """

    # Calculate the total shared memory per CTA.
    shared_mem_per_cta_kb = (
        static_shared_mem_kb
        + (driver_shared_mem_kb or 0.0)
        + (dynamic_shared_mem_kb or 0.0)
    )

    # Calculate the number of warps per CTA and the number of registers per CTA.
    warps_per_cta = block_size // warp_size
    regs_per_cta = num_regs_per_thread * block_size

    # Get the maximum number of resident blocks per SM, warps per SM, and registers per SM.
    max_resident_blocks_per_sm, max_resident_warps_per_sm, max_registers_per_sm = (
        gpu_config.max_resident_blocks_per_sm,
        gpu_config.max_resident_warps_per_sm,
        gpu_config.max_registers_per_sm,
    )

    smem_to_ctas_occupancy = dict[float, Tuple[int, float]]()

    # Calculate the theoretical occupancy for each shared memory configuration.
    for config_size_kb in sorted(gpu_config.shared_mem_config_options):
        max_resident_ctas, theoretical_occupancy = cal_theoretical_occupancy(
            config_size_kb,
            shared_mem_per_cta_kb,
            regs_per_cta,
            warps_per_cta,
            max_resident_blocks_per_sm,
            max_resident_warps_per_sm,
            max_registers_per_sm,
            grid_size,
            sm_nums,
        )
        smem_to_ctas_occupancy[float(config_size_kb)] = (
            max_resident_ctas,
            theoretical_occupancy,
        )

    return smem_to_ctas_occupancy


def optimal_smem_config_size(
    smem_to_ctas_occupancy: dict[float, Tuple[int, float]],
) -> float:
    """
    Find the optimal shared memory config size.

    Strategy: Among all config sizes that achieve the maximum occupancy,
    choose the smallest one. This balances performance (good occupancy)
    with L1 cache efficiency (not using more shared memory than necessary).

    Parameters:
        smem_to_ctas_occupancy: Dictionary mapping config size (KB) to
            (max_resident_ctas, theoretical_occupancy)

    Returns:
        Optimal config size in KB
    """
    if not smem_to_ctas_occupancy:
        raise ValueError("Empty occupancy data")

    max_occupancy = max(occ for _, (_, occ) in smem_to_ctas_occupancy.items())
    candidates = [
        size
        for size, (_, occ) in smem_to_ctas_occupancy.items()
        if occ == max_occupancy
    ]

    return min(candidates)
