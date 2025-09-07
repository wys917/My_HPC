#include "buffer.h"
#include "tile.h"
#include "reshape.h"
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <chrono>

#define M 12
#define K 32
#define N 32

void naive_gemm(uint8_t* A, int8_t* B, uint32_t* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int kk = 0; kk < k * 4; kk++) {
                C[i * n + j] += A[i * k * 4 + kk] * B[j * k * 4 + kk];
            }
        }
    }
}

int main() {
    uint8_t* A = new uint8_t[M * K * 4];
    int8_t* B = new int8_t[N * K * 4];
    int8_t* B_reshape = new int8_t[N * K * 4];

    uint32_t* C = new uint32_t[M * N];
    uint32_t* C_ref = new uint32_t[M * N];

    init_buffer(A, M, K * 4);
    init_buffer(B, N, K * 4);

    // Naive Implementation (Baseline)
    std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < 100000; iter ++) {
        memset(C_ref, 0, sizeof(uint32_t) * M * N);
        naive_gemm(A, B, C_ref, M, N, K);
    }

    std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration1 = end_time - start_time;
    std::cout << "Time: " << duration1.count() << " s" << std::endl;

    // Optimized
    // Write your code here
    // You can only modify the code between 'start' and 'end'

    start_time = std::chrono::high_resolution_clock::now();
    //------------------------------set_tiledata_use();

    // Initialize AMX tile
    // start
//---------------------------init_tile_config(M);
    // end

    for (int iter = 0; iter < 100000; iter ++) {
        memset(C, 0, sizeof(uint32_t) * M * N);
        // reshape B
        reshape((uint8_t*)B, N, K * 4, (uint8_t*)B_reshape);

        // !!!Choose a matrix multiplication from the following to achieve!!!
        // do AVX GEMM 
        
// start
        // 循环展开和Tiling: M维度每次处理1行，N维度每次处理16列 (一个AVX512向量的宽度)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; j += 16) { // 每次计算C的一行中的16个元素

                // 准备一个向量累加器，它将同时保存 C[i][j] 到 C[i][j+15] 的16个结果
                __m512i c_vec_accumulator = _mm512_setzero_si512();

                // 沿 K 维度循环，按块(block)进行
                for (int k = 0; k < K; ++k) {
                    
                    // 1. 从A的第i行加载第k个1x4数据块 (32-bit)
                    //    然后将这4个uint8_t值广播(broadcast)到整个512-bit向量寄存器中
                    __m128i a_block_128 = _mm_cvtsi32_si128(*(const int*)(&A[i * K * 4 + k * 4]));
                    __m512i a_vec_broadcasted = _mm512_broadcastd_epi32(a_block_128);

                    // 2. 从B_reshape加载数据。
                    //    由于B_reshape的内存布局，B的第j到j+15列的第k个数据块现在是内存中连续的
                    //    所以我们只需要一次内存加载，就能把这16个块 (16 * 4 = 64字节) 全都读进来
                    __m512i b_vec_packed = _mm512_loadu_si512((__m512i const*)&B_reshape[k * N * 4 + j * 4]);

                    // 3. 执行核心计算
                    //    用_mm512_dpbusd_epi32指令，A的广播向量和B的打包向量进行点积累加
                    //    这一条指令就完成了16次 4-element dot-product
                    c_vec_accumulator = _mm512_dpbusd_epi32(c_vec_accumulator, a_vec_broadcasted, b_vec_packed);
                }

                // 4. 将累加器中得到的16个最终结果一次性写回C矩阵的对应位置
                _mm512_storeu_si512((__m512i*)&C[i * N + j], c_vec_accumulator);
            }
        }
        // end
    }
    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration2 = end_time - start_time;
    std::cout << "Time: " << duration2.count() << " s" << std::endl;

    // Verify the answer
    check_result(C, C_ref, M, N);
    double speedup = duration1.count() / duration2.count();
    std::cout << "Speedup: " << speedup << std::endl;

    return 0;
}

/*下面是AMX版本的代码
#include "buffer.h"
#include "tile.h"
#include "reshape.h"
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <chrono>

#define M 12
#define K 32
#define N 32

void naive_gemm(uint8_t* A, int8_t* B, uint32_t* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int kk = 0; kk < k * 4; kk++) {
                C[i * n + j] += A[i * k * 4 + kk] * B[j * k * 4 + kk];
            }
        }
    }
}

int main() {
    uint8_t* A = new uint8_t[M * K * 4];
    int8_t* B = new int8_t[N * K * 4];
    int8_t* B_reshape = new int8_t[N * K * 4];

    uint32_t* C = new uint32_t[M * N];
    uint32_t* C_ref = new uint32_t[M * N];

    init_buffer(A, M, K * 4);
    init_buffer(B, N, K * 4);

    // Naive Implementation (Baseline)
    std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < 100000; iter ++) {
        memset(C_ref, 0, sizeof(uint32_t) * M * N);
        naive_gemm(A, B, C_ref, M, N, K);
    }

    std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration1 = end_time - start_time;
    std::cout << "Time: " << duration1.count() << " s" << std::endl;

    // Optimized
    // Write your code here
    // You can only modify the code between 'start' and 'end'

    start_time = std::chrono::high_resolution_clock::now();
    set_tiledata_use();

    // Initialize AMX tile
    // start
    init_tile_config(M);
    // end

    for (int iter = 0; iter < 100000; iter ++) {
        memset(C, 0, sizeof(uint32_t) * M * N);
        // reshape B
        reshape((uint8_t*)B, N, K * 4, (uint8_t*)B_reshape);

        // !!!Choose a matrix multiplication from the following to achieve!!!
        //  AMX GEMM
        // start
        
        for (int j = 0; j < N; j += 16) {
            
            _tile_zero(0);

            for (int k = 0; k < K * 4; k += 64) {
                
                _tile_loadd(1, A + k, K * 4);

                int j_block_idx = j / 16;
                int k_block_idx = k / 64;
                void* b_addr = (uint8_t*)B_reshape + k_block_idx * 16 * 128 + j_block_idx * 64;
                
         
                _tile_loadd(2, b_addr, 128);

                _tile_dpbusd(0, 1, 2);
            }

            _tile_stored(0, C + j, N * 4);
        }
        // end
    }
    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration2 = end_time - start_time;
    std::cout << "Time: " << duration2.count() << " s" << std::endl;

    // Verify the answer
    check_result(C, C_ref, M, N);
    double speedup = duration1.count() / duration2.count();
    std::cout << "Speedup: " << speedup << std::endl;

    return 0;
}*/
