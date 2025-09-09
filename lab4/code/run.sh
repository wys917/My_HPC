#!/bin/bash
#SBATCH --job-name=scaling_test
#SBATCH --partition=M7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:10:00
#SBATCH --output=reports_time/benchmark_%j.out
#SBATCH --error=reports_time/benchmark_%j.err

# 直接设置MPI路径，绕过spack环境问题
export PATH="/pxe/opt/spack/opt/spack/linux-debian12-haswell/gcc-12.2.0/intel-oneapi-mpi-2021.14.1-jdda552mqvxz4g6vuwkboc7biptbtgge/mpi/2021.14/bin:$PATH"
export LD_LIBRARY_PATH="/pxe/opt/spack/opt/spack/linux-debian12-haswell/gcc-12.2.0/intel-oneapi-mpi-2021.14.1-jdda552mqvxz4g6vuwkboc7biptbtgge/mpi/2021.14/lib:$LD_LIBRARY_PATH"
export I_MPI_PMI_LIBRARY=/slurm/libpmi2.so.0.0.0
mkdir -p reports_time



export OMP_NUM_THREADS=48  # 单进程使用更多线程

# Run BICGSTAB 直接运行（单进程）
./build/bicgstab data/case_2001.bin

