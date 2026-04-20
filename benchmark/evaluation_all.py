import os
import subprocess

shapes = [ #(M, N, K)

    # Scale up the M dimension
    # (512, 4096, 1024),
    # (1024, 4096, 1024),
    (2048, 4096, 1024),
    (4096, 4096, 1024),
    (8192, 4096, 1024),
    (16384, 4096, 1024),

    # Scale up the N dimension
    (1024, 512, 1024),
    (1024, 1024, 1024),
    (1024, 2048, 1024),
    (1024, 4096, 1024),
    (1024, 8192, 1024),
]

GPU_id = 2

# Evaluation Ansor-AF-DS
def evaluation_ansor_af_ds():
    Ansor_AF_DS_path = f'/home/pengbenkang/github/HPCRL_tvm/benchmarks/test'

    for shape in shapes:
        M, N, K = shape
        cmd = f'''
        source ~/.bashrc && \
        export CUDA_VISIBLE_DEVICES={GPU_id} && \
        conda activate ics24 && \
        python dynamic_gradient_descent_search.py --shape {M} {N} {K}
        '''
        subprocess.run(cmd, shell=True, cwd=Ansor_AF_DS_path, executable='/bin/bash')

def evaluation_roller():
    Roller_path = f'/home/pengbenkang/github/NewRoller'
    for shape in shapes:
        M, N, K = shape
        cmd = f'''
        source ~/.bashrc && \
        export CUDA_VISIBLE_DEVICES={GPU_id} && \
        conda activate tvm-build-venv && \
        python test_op.py --shape {M} {N} {K}
        '''
        subprocess.run(cmd, shell=True, cwd=Roller_path, executable='/bin/bash')

def evaluation_deca():
    DECA_path = f'/home/pengbenkang/github/ConstructTuner'
    for shape in shapes:
        M, N, K = shape
        cmd = f'''
        source ~/.bashrc && \
        export CUDA_VISIBLE_DEVICES={GPU_id} && \
        conda activate tvm-build-venv && \
        export PYTHONPATH=$PYTHONPATH:/home/pengbenkang/github/ConstructTuner/src && \
        export PYTHONPATH=$PYTHONPATH:/home/pengbenkang/github/ConstructTuner/arch && \
        python test_op.py --shape {M} {N} {K}
        '''
        subprocess.run(cmd, shell=True, cwd=DECA_path, executable='/bin/bash')

def evaluation_tvm():
    tvm_tuning_script_path = f'/home/pengbenkang/github/ConstructTuner/benchmark/tvm'
    for shape in shapes:
        M, N, K = shape
        cmd = f'''
        source ~/.bashrc && \
        export CUDA_VISIBLE_DEVICES={GPU_id} && \
        conda activate tvm-build-venv && \
        python tvm_gemm_tuner.py --shape {M} {N} {K}
        '''
        subprocess.run(cmd, shell=True, cwd=tvm_tuning_script_path, executable='/bin/bash')

if __name__ == "__main__":
    # evaluation_roller()
    # evaluation_ansor_af_ds()
    # evaluation_deca()
    evaluation_tvm()
