#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void naiveMin(int *arr, int* local_mins, int tsize, int arr_size) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int left = tid*tsize;
    int right = (tid+1)*tsize;

    if (right > arr_size) {
        right = arr_size;
    }

    int min = arr[left];
    for (int i = left; i < right; i++) {
        if (arr[i] < min) {
            min = arr[i];
        }
    }

    local_mins[tid] = min;
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