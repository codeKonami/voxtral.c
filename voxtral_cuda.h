/*
 * voxtral_cuda.h - CUDA GPU acceleration for Voxtral inference
 *
 * Provides cuBLAS-accelerated matrix multiplication with bf16 weight caching,
 * plus custom CUDA kernels for element-wise operations.
 */

#ifndef VOXTRAL_CUDA_H
#define VOXTRAL_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize CUDA acceleration. Returns 1 on success, 0 if unavailable.
 * Requires SM 8.0+ for native BF16 support. */
int vox_cuda_init(void);

/* Check if CUDA is initialized and available. */
int vox_cuda_available(void);

/* Cleanup all CUDA resources. */
void vox_cuda_shutdown(void);

/*
 * GPU-accelerated matrix multiplication with bf16 weights.
 * C[M,N] = A[M,K] @ B^T[N,K]
 *
 * A is f32 (activations), B is bf16 (weights, cached on GPU after first use),
 * C is f32 (output). B is always transposed (row-major weight layout).
 *
 * Weight buffers are cached on GPU after first use (uploaded once per unique
 * bf16 pointer). cuBLAS uses native BF16 (no conversion needed).
 */
void vox_cuda_sgemm_bf16(int M, int N, int K,
                          const float *A,
                          const uint16_t *B_bf16,
                          float *C);

/*
 * GPU-accelerated f32 matrix multiplication.
 * C[M,N] = A[M,K] @ B^T[N,K]
 */
void vox_cuda_sgemm(int M, int N, int K,
                     const float *A,
                     const float *B,
                     float *C);

/*
 * Pre-warm the bf16 weight cache by uploading to GPU.
 * Call during model loading to avoid first-use latency.
 */
void vox_cuda_warmup_bf16(const uint16_t *bf16_weights, size_t num_elements);

/*
 * GPU-shared memory allocation (unified memory, accessible from CPU and GPU).
 * Returns a pointer usable from both CPU and GPU.
 * Falls back to calloc if CUDA is not available.
 */
void *vox_cuda_shared_alloc(size_t size);
void  vox_cuda_shared_free(void *ptr);

/* GPU memory usage in bytes (for debugging). */
size_t vox_cuda_memory_used(void);

#ifdef __cplusplus
}
#endif

#endif /* VOXTRAL_CUDA_H */
