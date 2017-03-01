#include <math.h>
#include <time.h>
#include <iostream>
#include <stdexcept>
#include "cuda_runtime.h"
#include "kernel.h"


static const int n_el = 512;
static const size_t size = n_el * sizeof(float);

int main(){

  float* h_A = (float*)malloc(size);
  float* h_B = (float*)malloc(size);
  float* h_C = (float*)malloc(size);

  float *d_A,*d_B,*d_C;

  for (int i=0; i<n_el; i++){
    h_A[i]=sin(i);
    h_B[i]=cos(i);
  }

  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // call kernel function
  sum(d_A, d_B, d_C, n_el);

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);


  double err=0;
  for (int i=0; i<n_el; i++) {
    double diff=double((h_A[i]+h_B[i])-h_C[i]);
    err+=diff*diff;

    std::cout << "A+B: " << h_A[i]+h_B[i] << "\n";
    std::cout << "C: " << h_C[i] << "\n";
  }
  err=sqrt(err);
  std::cout << "err: " << err << "\n";

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return cudaDeviceSynchronize();
}
