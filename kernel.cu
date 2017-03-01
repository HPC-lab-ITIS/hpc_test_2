#include <math.h>
#include "cuda_runtime.h"
#include "kernel.h"

__global__ void kernel_sum(const float* A, const float* B, float* C, int n_el);

void sum(const float* A, const float* B, float* C, int n_el) {

  int threadsPerBlock,blocksPerGrid;

  if (n_el<512){
    threadsPerBlock = n_el;
    blocksPerGrid   = 1;
  } else {
    threadsPerBlock = 512;
    blocksPerGrid   = ceil(double(n_el)/double(threadsPerBlock));
  }

  kernel_sum<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, n_el);
}

__global__ void kernel_sum(const float* A, const float* B, float* C, int n_el)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n_el) C[tid] = A[tid] + B[tid];
}