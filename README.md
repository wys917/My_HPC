# 🚀 高性能计算实验项目集

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-blue.svg)](https://github.com/wys917/My_HPC)

本仓库包含了一系列高性能计算(HPC)相关的实验项目，涵盖了集群搭建、SIMD优化、混合并行编程等多个重要主题。每个项目都包含完整的源代码、实验报告和性能分析结果。

## 📋 项目概览

### 🏗️ [迷你集群搭建及HPL性能调优](./迷你集群搭建及HPL性能调优/)
- **技术栈**: VMware, Debian, OpenMPI, HPL, BLAS
- **核心成果**: 从零搭建4节点计算集群，HPL性能达到 **3.1339 GFLOPS**
- **关键技术**: 集群配置、网络组网、性能基准测试与参数调优

### ⚡ [基于AVX-512的整数量化矩阵乘法优化实验](./基于AVX-512的整数量化矩阵乘法优化实验/)
- **技术栈**:  AVX-512/AMX Intrinsics, CMake, Python
- **核心成果**: AVX-512实现 **48.67倍** 加速，AMX实现 **120.8倍** 加速
- **关键技术**: SIMD向量化、数据重排、内存访问优化

### 🔄 [基于MPI+OpenMP混合并行的BiCGSTAB算法实现与调优](./基于MPI+OpenMP混合并行的BiCGSTAB算法实现与调优/)
- **技术栈**:  MPI, OpenMP, Intel VTune, SLURM
- **核心成果**: OpenMP单节点实现 **108.8倍** 加速，混合并行达到 **30.4倍** 加速
- **关键技术**: 共享内存并行、分布式内存并行、性能剖析与瓶颈分析

## 🛠️ 技术栈总览

| 领域 | 技术 |
|------|------|
| **并行编程** | OpenMP, MPI, 混合并行模型 |
| **SIMD优化** | AVX-512, AMX, Intel Intrinsics |
| **系统配置** | Linux集群, SLURM, SSH免密, NFS |
| **性能分析** | Intel VTune Profiler, ITAC, HPL基准测试 |
| **开发工具** | CMake, GCC, Python, Gnuplot |

## 📊 性能成果汇总

| 项目 | 基准性能 | 优化后性能 | 加速比 | 关键优化技术 |
|------|----------|------------|--------|--------------|
| HPL集群测试 | - | 3.1339 GFLOPS | - | 参数调优 (N=2000, NB=224) |
| AVX-512优化 | 2.31s | 0.047s | **48.67x** | SIMD向量化 + 数据重排 |
| AMX优化 | 3.01s | 0.025s | **120.8x** | 二维瓦片矩阵计算 |
| OpenMP并行 | 93.88s | 3.49s | **108.8x** | 共享内存并行 (96线程) |
| MPI+OpenMP混合 | 379.38s | 12.5s | **30.4x** | 混合并行模型 |

## 🏗️ 项目结构

```
My_HPC/
├── 迷你集群搭建及HPL性能调优/
│   ├── assets/          # 图表和可视化结果
│   ├── code/           # 脚本和配置文件
│   ├── report/         # 实验报告 (Markdown + PDF)
│   └── result/         # HPL测试结果
├── 基于AVX-512的整数量化矩阵乘法优化实验/
│   ├── assets/         # 性能对比图表
│   ├── code/           # C++源码和CMake配置
│   └── report/         # 实验报告 (Markdown + PDF)
└── 基于MPI+OpenMP混合并行的BiCGSTAB算法实现与调优/
    ├── assets/         # VTune分析报告截图
    ├── code/           # 并行算法源码
    ├── report/         # 实验报告 (Markdown + PDF)
    └── result/         # 性能测试数据和VTune报告
```

## 🚀 快速开始

### 环境要求
- **操作系统**: Linux (推荐) 或 Windows with WSL
- **编译器**: GCC 9+ (支持AVX-512) 或 Intel ICC
- **并行库**: OpenMPI 4.0+, OpenMP 4.5+
- **工具链**: CMake 3.15+, Python 3.7+

### 编译运行示例
```bash
# 克隆仓库
git clone https://github.com/wys917/My_HPC.git
cd My_HPC

# AVX-512 矩阵乘法优化项目
cd "基于AVX-512的整数量化矩阵乘法优化实验/code/vector"
mkdir build && cd build
cmake .. && make
./lab2

# MPI+OpenMP 混合并行项目  
cd "../../基于MPI+OpenMP混合并行的BiCGSTAB算法实现与调优/code"
mkdir build && cd build
cmake .. && make
mpirun -np 2 ./bicgstab_solver
```

## 📈 学习价值

本项目集适合以下人群学习参考：
- **HPC入门者**: 从集群搭建到性能优化的完整实践流程
- **并行编程学习者**: OpenMP、MPI及混合并行模型的实际应用
- **性能优化工程师**: SIMD指令集优化和性能瓶颈分析方法
- **科学计算研究者**: 迭代求解器的并行化策略和性能调优

## 📝 实验报告

每个项目都包含详细的实验报告，内容涵盖：
- **问题背景与优化目标**
- **技术方案设计与实现**  
- **性能测试与结果分析**
- **关键挑战与解决方案**
- **量化的性能提升数据**

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！如果您发现了问题或有改进建议，请：
1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

## 👨‍💻 作者

**苏易文** - *浙江大学*
- 学号: 3240103466
- GitHub: [@wys917](https://github.com/wys917)

---

⭐ 如果这个项目对您有帮助，请给个 Star 支持一下！
