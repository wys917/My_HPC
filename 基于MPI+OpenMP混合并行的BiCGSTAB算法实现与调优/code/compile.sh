#!/bin/bash

# 先加载 spack 再加载编译环境
source /pxe/opt/spack/share/spack/setup-env.sh
export I_MPI_PMI_LIBRARY=/slurm/libpmi2.so.0.0.0
spack env activate hpc101-intel

# 确保MPI编译器在PATH中
export PATH="/pxe/opt/spack/opt/spack/linux-debian12-haswell/gcc-12.2.0/intel-oneapi-mpi-2021.14.1-jdda552mqvxz4g6vuwkboc7biptbtgge/mpi/2021.14/bin:$PATH"

# 运行你的命令
cmake -B build
cmake --build build
