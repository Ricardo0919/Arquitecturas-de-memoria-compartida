// =================================================================
//
// File: example04.cu
// Author: Pedro Perez
// Description: This file contains the code that searches for the
// 				smallest value stored in an array using CUDA. To 
//              compile:
//		        nvcc -o app example04.cu
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <climits>
#include <cuda_runtime.h>
#include <algorithm>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 1000000000 //1e9
#define THREADS 512
#define BLOCKS	min(32, ((SIZE / THREADS) + 1))

__device__ int minimum(int a, int b) {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}

__global__ void minimum(int *array, int *results) {
    __shared__ int cache[THREADS];
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    int aux = array[0];
    while (index < SIZE) {
        if(array[index] < aux){
            aux = array[index];
        }
        
        index += (blockDim.x * gridDim.x);
    }

    cache[threadIdx.x] = aux;

    __syncthreads(); 

    int gap = blockDim.x / 2;

    while (gap > 0) {
        if (threadIdx.x < gap) {
            cache[threadIdx.x] = minimum(cache[threadIdx.x], cache[threadIdx.x+ gap]);
        }
        __syncthreads();
        gap /= 2;
    }

    if (threadIdx.x == 0) {
        results[blockIdx.x] = cache[threadIdx.x];
    }
}

int main(int argc, char* argv[]) {
    int *array, *results;
    int *device_array, *device_results;
    
    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    array =  new int[SIZE];
    random_array(array, SIZE);
    display_array("array", array);

    results = new int[BLOCKS];
    cudaMalloc((void**) &device_array, SIZE * sizeof(int) );
    cudaMalloc((void**) &device_results, BLOCKS * sizeof(int) );

    cudaMemcpy(device_array, array, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        minimum<<<BLOCKS, THREADS>>> (device_array, device_results);

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }

    cudaMemcpy(results, device_results, BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);

    int aux = array[0];
    for (int i = 0; i < BLOCKS; i++) {
        aux = min(aux, results[i]);
    }

    cout << "result = " << aux << "\n";
    cout << "avg time = " << fixed << setprecision(3) 
        << (timeElapsed / N) <<  " ms\n";

    cudaFree(device_results);
    cudaFree(device_array);

    delete [] array;
    delete [] results;

    return 0;
}