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
        for (int i = 0; i < M; i += 4) {
            for (int j = 0; j < N; j += 16) { // 每次计算C的一行中的16个元素

                // 在 j 循环内部...
                // 定义累加器...
                __m512i c_vec_0 = _mm512_setzero_si512();
                __m512i c_vec_1 = _mm512_setzero_si512();
                __m512i c_vec_2 = _mm512_setzero_si512();
                __m512i c_vec_3 = _mm512_setzero_si512();

                // K维度循环，步长为2
                for (int k = 0; k < K; k += 2) { 
                    // ====================== 加载阶段 (Load Phase) ======================
                    // 把未来两次迭代需要的所有数据，一次性全部发出加载指令
                    
                    // --- k iter 1 data ---
                    __m512i b_vec_k0 = _mm512_loadu_si512((__m512i const*)&B_reshape[(k+0) * N * 4 + j * 4]);
                    __m512i a_vec_0_k0 = _mm512_set1_epi32(*(const int*)(&A[(i+0) * K * 4 + (k+0) * 4]));
                    __m512i a_vec_1_k0 = _mm512_set1_epi32(*(const int*)(&A[(i+1) * K * 4 + (k+0) * 4]));
                    __m512i a_vec_2_k0 = _mm512_set1_epi32(*(const int*)(&A[(i+2) * K * 4 + (k+0) * 4]));
                    __m512i a_vec_3_k0 = _mm512_set1_epi32(*(const int*)(&A[(i+3) * K * 4 + (k+0) * 4]));
                    
                    // --- k iter 2 data ---
                    __m512i b_vec_k1 = _mm512_loadu_si512((__m512i const*)&B_reshape[(k+1) * N * 4 + j * 4]);
                    __m512i a_vec_0_k1 = _mm512_set1_epi32(*(const int*)(&A[(i+0) * K * 4 + (k+1) * 4]));
                    __m512i a_vec_1_k1 = _mm512_set1_epi32(*(const int*)(&A[(i+1) * K * 4 + (k+1) * 4]));
                    __m512i a_vec_2_k1 = _mm512_set1_epi32(*(const int*)(&A[(i+2) * K * 4 + (k+1) * 4]));
                    __m512i a_vec_3_k1 = _mm512_set1_epi32(*(const int*)(&A[(i+3) * K * 4 + (k+1) * 4]));

                    // ====================== 计算阶段 (Compute Phase) ======================
                    // 此刻，大部分加载延迟已经被隐藏，现在集中进行计算
                    
                    // --- k iter 1 computes ---
                    c_vec_0 = _mm512_dpbusd_epi32(c_vec_0, a_vec_0_k0, b_vec_k0);
                    c_vec_1 = _mm512_dpbusd_epi32(c_vec_1, a_vec_1_k0, b_vec_k0);
                    c_vec_2 = _mm512_dpbusd_epi32(c_vec_2, a_vec_2_k0, b_vec_k0);
                    c_vec_3 = _mm512_dpbusd_epi32(c_vec_3, a_vec_3_k0, b_vec_k0);

                    // --- k iter 2 computes ---
                    c_vec_0 = _mm512_dpbusd_epi32(c_vec_0, a_vec_0_k1, b_vec_k1);
                    c_vec_1 = _mm512_dpbusd_epi32(c_vec_1, a_vec_1_k1, b_vec_k1);
                    c_vec_2 = _mm512_dpbusd_epi32(c_vec_2, a_vec_2_k1, b_vec_k1);
                    c_vec_3 = _mm512_dpbusd_epi32(c_vec_3, a_vec_3_k1, b_vec_k1);
                }

                // ... 存储结果 ...

                // 循环结束后，将4个累加器的结果写回C矩阵
                _mm512_storeu_si512((__m512i*)&C[(i+0) * N + j], c_vec_0);
                _mm512_storeu_si512((__m512i*)&C[(i+1) * N + j], c_vec_1);
                _mm512_storeu_si512((__m512i*)&C[(i+2) * N + j], c_vec_2);
                _mm512_storeu_si512((__m512i*)&C[(i+3) * N + j], c_vec_3);
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
