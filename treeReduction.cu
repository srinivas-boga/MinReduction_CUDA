#include <iostream>
#include <cstdlib>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>



__global__ void treeReduction(int *arr, int* local_mins, int arr_size) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int sdata[blockDim.x];

    if(tid < arr_size) {
        sdata[threadIdx.x] = arr[tid];
    } else {
        sdata[threadIdx.x] = FLT_MAX;
    }

    // Wait for all threads to finish copying to shared memory
    __syncthreads();

    // Reduction, tree style, each iteration halves the number of comparisons
    for (int i = blockDim.x/2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            sdata[threadIdx.x] = min(sdata[threadIdx.x], sdata[threadIdx.x + 1]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        local_mins[blockIdx.x] = sdata[0];
    }
    
}


int main() {

    int arr_size = 1 << 20;
    int tsize = 1 << 10;

    int num_threads = arr_size/tsize;

    int *arr = (int*)malloc(arr_size*sizeof(int));
    int *local_mins = (int*)malloc(num_threads*sizeof(int));

    for (int i = 0; i < arr_size; i++) {
        arr[i] = rand() % 1000;
    }

    int *d_arr, *d_local_mins;
    cudaMalloc(&d_arr, arr_size*sizeof(int));
    cudaMalloc(&d_local_mins, num_threads*sizeof(int));

    cudaMemcpy(d_arr, arr, arr_size*sizeof(int), cudaMemcpyHostToDevice);

    naiveMin<<<num_threads, 1>>>(d_arr, d_local_mins, tsize, arr_size);

    cudaMemcpy(local_mins, d_local_mins, num_threads*sizeof(int), cudaMemcpyDeviceToHost);

    int min = local_mins[0];
    for (int i = 0; i < num_threads; i++) {
        if (local_mins[i] < min) {
            min = local_mins[i];
        }
    }

    std::cout << "Min: " << min << std::endl;

    free(arr);
    free(local_mins);
    cudaFree(d_arr);
    cudaFree(d_local_mins);

    return 0;

}