# 基于AVX-512的整数量化矩阵乘法优化

**实验者：** 苏易文 (学号: 3240103466)  
**日期：** 2025年7月6日  

## 📌 项目概述

本实验专注于优化整数量化矩阵乘法（`uint8_t * int8_t`），通过深入理解现代CPU的向量化指令集，实现了基于AVX-512和AMX的高性能SIMD优化。项目包含了从基础向量化到终极硬件加速的完整优化路径，展示了现代处理器架构在数值计算中的强大威力。

## 🎯 实验目标

- 掌握AVX-512向量指令集的使用方法和优化策略
- 理解数据重排（Data Reshaping）在SIMD优化中的关键作用
- 探索Intel AMX指令集在矩阵运算中的革命性优势
- 分析不同硬件加速技术的性能特征和适用场景
- 学习NumPy向量化编程的最佳实践

## ⚙️ 技术栈与环境

### 硬件要求
- **CPU**: 支持AVX-512和AMX指令集的Intel处理器
- **内存**: 至少8GB RAM
- **编译器**: 支持Intel intrinsics的现代C++编译器

### 软件环境
- **编程语言**: C++17, Python 3.x
- **编译工具**: CMake 3.10+, GCC/Clang
- **依赖库**: Intel intrinsics, NumPy
- **开发工具**: CMake, Make

## 🏗️ 项目结构

```
基于AVX-512的整数量化矩阵乘法优化/
├── README.md                           # 项目说明文档
├── report/                            # 实验报告
│   └── report.md                      # 详细实验报告
├── code/                              # 源代码
│   ├── vector/                        # C++ SIMD优化实现
│   │   ├── main.cpp                   # 主程序
│   │   ├── CMakeLists.txt             # CMake构建文件
│   │   ├── config.yaml                # 配置文件
│   │   ├── include/                   # 头文件
│   │   │   ├── buffer.h              # 缓冲区管理
│   │   │   ├── reshape.h             # 数据重排
│   │   │   └── tile.h                # AMX瓦片操作
│   │   ├── src/                      # 源文件
│   │   │   ├── buffer.cpp            # 缓冲区实现
│   │   │   └── reshape.cpp           # 数据重排实现
│   │   └── build/                    # 构建目录
│   └── numpy_example/                # Python示例
│       ├── main.py                   # 主程序
│       ├── utils/                    # 工具函数
│       │   └── timer.py              # 计时工具
│       └── bilinear_interp/          # 双线性插值示例
│           ├── baseline.py           # 基准实现
│           └── vectorized.py         # 向量化实现
└── assets/                           # 资源文件（图表等）
```

## 🚀 核心算法与实现

### 算法概述
本实验优化的核心算法是**整数量化矩阵乘法**：
```
C = A * B_transposed
```
其中：
- `A`: `uint8_t` 类型的 M×(K×4) 矩阵
- `B`: `int8_t` 类型的 N×(K×4) 矩阵  
- `C`: `uint32_t` 类型的 M×N 结果矩阵

### 1. AVX-512 优化实现

#### 核心策略
1. **数据重排（Data Reshaping）**: 通过`reshape`函数优化B矩阵的内存布局
2. **Tiling分块策略**: 将C矩阵分解为1×16的小块进行处理
3. **SIMD并行化**: 利用AVX-512指令实现高效的向量运算

#### 关键代码片段
```cpp
// 循环展开和Tiling: M维度每次处理1行，N维度每次处理16列
for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; j += 16) {
        __m512i c_vec_accumulator = _mm512_setzero_si512();
        
        for (int k = 0; k < K; ++k) {
            // 1. 从A矩阵加载并广播数据
            __m128i a_block_128 = _mm_cvtsi32_si128(*(const int*)(&A[i * K * 4 + k * 4]));
            __m512i a_vec_broadcasted = _mm512_broadcastd_epi32(a_block_128);
            
            // 2. 从重排后的B矩阵加载连续数据
            __m512i b_vec_packed = _mm512_loadu_si512((__m512i const*)&B_reshape[k * N * 4 + j * 4]);
            
            // 3. 执行核心的点积累加运算
            c_vec_accumulator = _mm512_dpbusd_epi32(c_vec_accumulator, a_vec_broadcasted, b_vec_packed);
        }
        
        // 4. 将结果写回C矩阵
        _mm512_storeu_si512((__m512i*)&C[i * N + j], c_vec_accumulator);
    }
}
```

### 2. AMX 终极优化（Bonus）

#### 核心特点
- **二维瓦片处理**: 从一维向量提升到二维矩阵瓦片
- **专用硬件加速**: 利用TMUL（Tile Multiplication Unit）
- **更高计算密度**: 单指令完成大规模矩阵运算

#### 关键代码片段
```cpp
// AMX配置和主循环
init_tile_config(M);

for (int j = 0; j < N; j += 16) {
    _tile_zero(0);  // 初始化结果瓦片
    
    for (int k = 0; k < K * 4; k += 64) {
        // 加载A瓦片（12x64）
        _tile_loadd(1, A + k, K * 4);
        
        // 加载B瓦片（16x64）
        void* b_addr = (uint8_t*)B_reshape + k_block_idx * 16 * 128 + j_block_idx * 64;
        _tile_loadd(2, b_addr, 128);
        
        // 执行瓦片矩阵乘法累加
        _tile_dpbusd(0, 1, 2);
    }
    
    // 存储结果瓦片
    _tile_stored(0, C + j, N * 4);
}
```

## 📊 性能测试结果

### AVX-512 优化结果

| 实现方法 | 运行时间 | 加速比 |
|---------|---------|-------|
| `naive_gemm` (基准) | 2.31009s | 1.0× |
| **AVX-512 优化版** | **0.0474591 s** | **48.68x** |

### AMX 优化结果（Bonus）

| 实现方法 | 运行时间 | 加速比 |
|---------|---------|-------|
| `naive_gemm` (基准) | 3.01185s | 1.0× |
| **AMX 优化版** | **0.0249321s** | **120.8×** |

### 性能分析
1. **AVX-512优化**: 通过向量化指令和数据重排，实现了67%的性能提升
2. **AMX优化**: 专用矩阵硬件带来了惊人的120倍加速比
3. **优化效果**: AMX相比AVX-512展现了专用硬件的绝对优势

## 🔧 使用方法

### 编译和运行C++版本

```bash
# 进入vector目录
cd code/vector

# 创建构建目录
mkdir -p build && cd build

# 配置和编译
cmake ..
make

# 运行程序
./lab2
```

### 运行Python示例

```bash
# 进入numpy_example目录
cd code/numpy_example

# 运行双线性插值示例
python main.py
```

## 💡 核心技术要点

### 1. 数据重排的重要性
- **问题**: 原始B矩阵布局不利于向量化访问
- **解决**: 通过块转置操作优化内存布局
- **效果**: 使相关数据在内存中连续存放，提高缓存效率

### 2. AVX-512关键指令解析
- `_mm512_broadcastd_epi32`: 广播A矩阵数据块
- `_mm512_loadu_si512`: 加载B矩阵连续数据
- `_mm512_dpbusd_epi32`: 执行16组4元素点积
- `_mm512_storeu_si512`: 写回计算结果

### 3. AMX架构优势
- **专用设计**: 为矩阵运算量身定制的硬件
- **二维处理**: 直接操作二维数据瓦片
- **高效流水**: TMUL单元提供极高计算吞吐量

## 📚 实验思考题解答

### 1. NumPy向量化中None的作用
在`bilinear_interp_vectorized.py`中，`None`用于增加维度，为广播机制做准备：
- `x_mul = (x - x_idx)[:,:,:,None]` 将形状从`(N,H2,W2)`变为`(N,H2,W2,1)`
- 支持与`(N,H2,W2,C)`形状的数组进行广播运算

### 2. 高级索引的广播机制
`a[n_idx, x_idx, y_idx]`通过广播实现批量索引：
- **最终形状**: `[N, H2, W2, C]`
- **作用机制**: 三个索引数组通过广播组合成坐标网格
- **效果**: 并行检索所有对应位置的像素数据

### 3. 矩阵转置算法中的Intel指令
详细的指令解析包括：
- `_mm256_loadu_epi64`: 非对齐加载256位数据
- `_mm512_castsi256_si512`: 类型转换（256位→512位）
- `_mm512_inserti64x4`: 插入256位数据到512位向量
- `_mm512_mask_permutexvar_epi64`: 掩码控制的可变置换

## 🏆 关键收获

1. **硬件理解**: 深入理解了现代CPU的向量化和矩阵计算单元
2. **优化策略**: 掌握了从数据布局到指令选择的系统性优化方法
3. **性能分析**: 学会了如何量化和分析不同优化技术的效果
4. **实践能力**: 提升了使用底层指令进行高性能计算的实践技能

## 📧 联系信息

如有问题或建议，请联系：
- **姓名**: 苏易文
- **学号**: 3240103466
- **日期**: 2025年7月6日

---

**注**: 本项目展示了从基础优化到硬件加速的完整技术路径，为高性能计算和深度学习加速提供了重要的技术参考。
