/*
 * voxtral_cuda.cu - CUDA GPU acceleration for Voxtral inference
 *
 * Phase 1: cuBLAS matmul with BF16 weight caching.
 * Weights are uploaded to GPU once and reused. Activations are copied per call.
 */

#include "voxtral_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * Global State
 * ======================================================================== */

static int g_initialized = 0;
static cudaStream_t g_stream = NULL;
static cublasHandle_t g_cublas = NULL;
static size_t g_memory_used = 0;

/* ========================================================================
 * BF16 Weight Cache
 *
 * Maps CPU bf16 pointer -> GPU device bf16 pointer.
 * Linear-probe hash table. Weights are uploaded once and stay on GPU.
 * ======================================================================== */

#define WEIGHT_CACHE_SIZE 1024

typedef struct {
    const uint16_t *cpu_ptr;  /* key (NULL = empty slot) */
    void *gpu_ptr;            /* device pointer (bf16) */
    size_t num_elements;
} weight_cache_entry_t;

static weight_cache_entry_t g_weight_cache[WEIGHT_CACHE_SIZE];

static int weight_cache_find(const uint16_t *cpu_ptr) {
    unsigned long hash = ((unsigned long)cpu_ptr >> 4) % WEIGHT_CACHE_SIZE;
    for (int i = 0; i < WEIGHT_CACHE_SIZE; i++) {
        int idx = (hash + i) % WEIGHT_CACHE_SIZE;
        if (g_weight_cache[idx].cpu_ptr == cpu_ptr) return idx;
        if (g_weight_cache[idx].cpu_ptr == NULL) return -1;
    }
    return -1;
}

static void *weight_cache_get_or_upload(const uint16_t *cpu_ptr, size_t num_elements) {
    int idx = weight_cache_find(cpu_ptr);
    if (idx >= 0) return g_weight_cache[idx].gpu_ptr;

    /* Upload to GPU */
    void *gpu_ptr = NULL;
    size_t bytes = num_elements * sizeof(uint16_t);
    cudaError_t err = cudaMalloc(&gpu_ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA: weight upload failed (%zu bytes): %s\n",
                bytes, cudaGetErrorString(err));
        return NULL;
    }
    err = cudaMemcpyAsync(gpu_ptr, cpu_ptr, bytes, cudaMemcpyHostToDevice, g_stream);
    if (err != cudaSuccess) {
        cudaFree(gpu_ptr);
        return NULL;
    }
    cudaStreamSynchronize(g_stream);
    g_memory_used += bytes;

    /* Insert into cache */
    unsigned long hash = ((unsigned long)cpu_ptr >> 4) % WEIGHT_CACHE_SIZE;
    for (int i = 0; i < WEIGHT_CACHE_SIZE; i++) {
        idx = (hash + i) % WEIGHT_CACHE_SIZE;
        if (g_weight_cache[idx].cpu_ptr == NULL) {
            g_weight_cache[idx].cpu_ptr = cpu_ptr;
            g_weight_cache[idx].gpu_ptr = gpu_ptr;
            g_weight_cache[idx].num_elements = num_elements;
            return gpu_ptr;
        }
    }

    /* Cache full (shouldn't happen with 1024 slots and ~400 weight tensors) */
    fprintf(stderr, "CUDA: weight cache full\n");
    cudaFree(gpu_ptr);
    g_memory_used -= bytes;
    return NULL;
}

/* ========================================================================
 * Activation Buffer Pool
 *
 * Reusable GPU buffers for transient activation/output data.
 * Avoids cudaMalloc/cudaFree per matmul call.
 * ======================================================================== */

#define POOL_SIZE 4

typedef struct {
    void *ptr;
    size_t capacity;  /* bytes */
    int in_use;
} pool_entry_t;

static pool_entry_t g_pool[POOL_SIZE];

static void *pool_alloc(size_t bytes) {
    /* Find a free buffer large enough */
    for (int i = 0; i < POOL_SIZE; i++) {
        if (!g_pool[i].in_use && g_pool[i].capacity >= bytes) {
            g_pool[i].in_use = 1;
            return g_pool[i].ptr;
        }
    }

    /* Find a free slot (grow or create) */
    for (int i = 0; i < POOL_SIZE; i++) {
        if (!g_pool[i].in_use) {
            if (g_pool[i].ptr) {
                g_memory_used -= g_pool[i].capacity;
                cudaFree(g_pool[i].ptr);
            }
            /* Allocate with some headroom to reduce re-allocations */
            size_t alloc_size = bytes < 64 * 1024 * 1024 ? 64 * 1024 * 1024 : bytes;
            cudaError_t err = cudaMalloc(&g_pool[i].ptr, alloc_size);
            if (err != cudaSuccess) {
                /* Try exact size */
                alloc_size = bytes;
                err = cudaMalloc(&g_pool[i].ptr, alloc_size);
                if (err != cudaSuccess) {
                    g_pool[i].ptr = NULL;
                    g_pool[i].capacity = 0;
                    return NULL;
                }
            }
            g_pool[i].capacity = alloc_size;
            g_pool[i].in_use = 1;
            g_memory_used += alloc_size;
            return g_pool[i].ptr;
        }
    }

    /* All slots in use — fallback to direct allocation */
    void *ptr = NULL;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess) return NULL;
    g_memory_used += bytes;
    return ptr;
}

static void pool_free(void *ptr) {
    for (int i = 0; i < POOL_SIZE; i++) {
        if (g_pool[i].ptr == ptr) {
            g_pool[i].in_use = 0;
            return;
        }
    }
    /* Was a direct allocation, actually free it */
    cudaFree(ptr);
}

/* ========================================================================
 * Lifecycle
 * ======================================================================== */

extern "C" int vox_cuda_init(void) {
    if (g_initialized) return 1;

    /* Check for CUDA device */
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "CUDA: no devices found\n");
        return 0;
    }

    /* Check compute capability (need SM 8.0+ for native BF16) */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm = prop.major * 10 + prop.minor;
    if (sm < 80) {
        fprintf(stderr, "CUDA: device SM %d.%d < 8.0, BF16 not supported\n",
                prop.major, prop.minor);
        return 0;
    }

    fprintf(stderr, "CUDA: %s (SM %d.%d, %.0f MB)\n",
            prop.name, prop.major, prop.minor,
            prop.totalGlobalMem / (1024.0 * 1024.0));

    /* Create stream and cuBLAS handle */
    err = cudaStreamCreate(&g_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA: stream creation failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    cublasStatus_t cberr = cublasCreate(&g_cublas);
    if (cberr != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUDA: cuBLAS creation failed: %d\n", cberr);
        cudaStreamDestroy(g_stream);
        g_stream = NULL;
        return 0;
    }
    cublasSetStream(g_cublas, g_stream);

    /* Use TF32 for better performance on Ampere+ (still precise enough) */
    cublasSetMathMode(g_cublas, CUBLAS_TF32_TENSOR_OP_MATH);

    /* Clear caches */
    memset(g_weight_cache, 0, sizeof(g_weight_cache));
    memset(g_pool, 0, sizeof(g_pool));
    g_memory_used = 0;

    g_initialized = 1;
    return 1;
}

extern "C" int vox_cuda_available(void) {
    return g_initialized;
}

extern "C" void vox_cuda_shutdown(void) {
    if (!g_initialized) return;

    /* Free weight cache */
    for (int i = 0; i < WEIGHT_CACHE_SIZE; i++) {
        if (g_weight_cache[i].gpu_ptr) {
            cudaFree(g_weight_cache[i].gpu_ptr);
            g_weight_cache[i].gpu_ptr = NULL;
            g_weight_cache[i].cpu_ptr = NULL;
        }
    }

    /* Free pool buffers */
    for (int i = 0; i < POOL_SIZE; i++) {
        if (g_pool[i].ptr) {
            cudaFree(g_pool[i].ptr);
            g_pool[i].ptr = NULL;
            g_pool[i].capacity = 0;
        }
    }

    if (g_cublas) { cublasDestroy(g_cublas); g_cublas = NULL; }
    if (g_stream) { cudaStreamDestroy(g_stream); g_stream = NULL; }

    g_memory_used = 0;
    g_initialized = 0;
}

/* ========================================================================
 * F32 -> BF16 Conversion Kernel
 * ======================================================================== */

__global__ void kernel_f32_to_bf16(__nv_bfloat16 *out, const float *in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(in[idx]);
    }
}

/* ========================================================================
 * Matrix Multiplication
 * ======================================================================== */

/*
 * C[M,N] = A_f32[M,K] @ B_bf16[N,K]^T
 *
 * cuBLAS requires both A and B to be the same type for BF16 GEMM.
 * Strategy: convert A from F32 to BF16 on GPU, then run BF16 x BF16 -> F32 GEMM.
 *
 * Row-major layout. Uses cuBLAS column-major convention:
 *   C'[N,M] = B^T[N,K] * A_bf16'[K,M]
 */
extern "C" void vox_cuda_sgemm_bf16(int M, int N, int K,
                                      const float *A,
                                      const uint16_t *B_bf16,
                                      float *C) {
    if (!g_initialized) return;

    /* Look up or upload weights */
    void *d_B = weight_cache_get_or_upload(B_bf16, (size_t)N * K);
    if (!d_B) return;

    /* Allocate buffers: F32 input, BF16 converted input, F32 output */
    size_t a_f32_bytes = (size_t)M * K * sizeof(float);
    size_t a_bf16_bytes = (size_t)M * K * sizeof(__nv_bfloat16);
    size_t c_bytes = (size_t)M * N * sizeof(float);

    void *d_A_f32 = pool_alloc(a_f32_bytes);
    void *d_A_bf16 = pool_alloc(a_bf16_bytes);
    void *d_C = pool_alloc(c_bytes);
    if (!d_A_f32 || !d_A_bf16 || !d_C) {
        if (d_A_f32) pool_free(d_A_f32);
        if (d_A_bf16) pool_free(d_A_bf16);
        if (d_C) pool_free(d_C);
        return;
    }

    /* Upload F32 activations */
    cudaMemcpyAsync(d_A_f32, A, a_f32_bytes, cudaMemcpyHostToDevice, g_stream);

    /* Convert F32 -> BF16 on GPU */
    int total = M * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kernel_f32_to_bf16<<<blocks, threads, 0, g_stream>>>(
        (__nv_bfloat16 *)d_A_bf16, (const float *)d_A_f32, total);

    /* cuBLAS GEMM: C[M,N] = A_bf16[M,K] @ B_bf16[N,K]^T
     * Both inputs BF16, output F32, compute in F32.
     *
     * In column-major: C'[N,M] = B^T[N,K] * A_bf16'[K,M]
     * transa=T: stored B[K,N]col = B[N,K]row, transpose -> [N,K]
     * transb=N: stored A[K,M]col = A[M,K]row, no transpose -> [K,M] */
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(g_cublas,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 d_B, CUDA_R_16BF, K,          /* B_bf16[N,K] row-major */
                 d_A_bf16, CUDA_R_16BF, K,     /* A_bf16[M,K] row-major */
                 &beta,
                 d_C, CUDA_R_32F, N,           /* C_f32[M,N] row-major */
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    /* Download result */
    cudaMemcpyAsync(C, d_C, c_bytes, cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);

    pool_free(d_A_f32);
    pool_free(d_A_bf16);
    pool_free(d_C);
}

extern "C" void vox_cuda_sgemm(int M, int N, int K,
                                 const float *A,
                                 const float *B,
                                 float *C) {
    if (!g_initialized) return;

    size_t a_bytes = (size_t)M * K * sizeof(float);
    size_t b_bytes = (size_t)N * K * sizeof(float);
    size_t c_bytes = (size_t)M * N * sizeof(float);

    void *d_A = pool_alloc(a_bytes);
    void *d_B = pool_alloc(b_bytes);
    void *d_C = pool_alloc(c_bytes);
    if (!d_A || !d_B || !d_C) {
        if (d_A) pool_free(d_A);
        if (d_B) pool_free(d_B);
        if (d_C) pool_free(d_C);
        return;
    }

    cudaMemcpyAsync(d_A, A, a_bytes, cudaMemcpyHostToDevice, g_stream);
    cudaMemcpyAsync(d_B, B, b_bytes, cudaMemcpyHostToDevice, g_stream);

    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(g_cublas,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 d_B, CUDA_R_32F, K,
                 d_A, CUDA_R_32F, K,
                 &beta,
                 d_C, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaMemcpyAsync(C, d_C, c_bytes, cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);

    pool_free(d_A);
    pool_free(d_B);
    pool_free(d_C);
}

/* ========================================================================
 * Weight Warmup
 * ======================================================================== */

extern "C" void vox_cuda_warmup_bf16(const uint16_t *bf16_weights, size_t num_elements) {
    if (!g_initialized || !bf16_weights) return;
    weight_cache_get_or_upload(bf16_weights, num_elements);
}

/* ========================================================================
 * Shared Memory (Unified Memory)
 * ======================================================================== */

extern "C" void *vox_cuda_shared_alloc(size_t size) {
    if (!g_initialized) return calloc(1, size);

    void *ptr = NULL;
    cudaError_t err = cudaMallocManaged(&ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA: managed alloc failed (%zu bytes): %s\n",
                size, cudaGetErrorString(err));
        return calloc(1, size);
    }
    memset(ptr, 0, size);
    g_memory_used += size;
    return ptr;
}

extern "C" void vox_cuda_shared_free(void *ptr) {
    if (!ptr) return;
    if (!g_initialized) { free(ptr); return; }

    /* Try cudaFree — works for both managed and device memory.
     * If ptr was allocated with calloc (fallback), this will fail silently
     * and we'll free with free(). */
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        /* Not a CUDA allocation — use regular free */
        cudaGetLastError(); /* clear the error */
        free(ptr);
    }
}

extern "C" size_t vox_cuda_memory_used(void) {
    return g_memory_used;
}
