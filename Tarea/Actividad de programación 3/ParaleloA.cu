// =================================================================
// 
// File: ParaleloA.cu
// Author: Ricardo Sierra Roa - A01709887
// Description: Contar el número de pares que existen en un arreglo de 
// números enteros. El tamaño del arreglo debe ser 1e9 (1,000,000,000).
// 
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE  1000000000 // 1e9
#define THREADS 512
#define BLOCKS min(16, ((SIZE / THREADS) + 1))

__global__ void countEven(int *array, int *count){
    __shared__ int cache[THREADS];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int localCount = 0;

    while (index < SIZE){
        if (array[index] % 2 == 0){
            localCount++;
        }
        index += blockDim.x * gridDim.x;
    }

    cache[threadIdx.x] = localCount;

    __syncthreads(); 

    int gap = blockDim.x / 2;

    while (gap > 0){
        if (threadIdx.x < gap){
            cache[threadIdx.x] += cache[threadIdx.x+ gap];
        }
        __syncthreads();
        gap /= 2;
    }

    if (threadIdx.x == 0){
        count[blockIdx.x] = cache[threadIdx.x];
    }

}


int main(int argc, char* argv[]){
    int *array, *deviceArray;
    int *count, *deviceCount;

    high_resolution_clock::time_point start, end;
    double timeElapsed = 0;

    array = new int[SIZE];
    fill_array(array, SIZE);

    count = new int[BLOCKS];

    cudaMalloc((void**)&deviceArray, SIZE * sizeof(int));
    cudaMalloc((void**)&deviceCount, BLOCKS * sizeof(int));

    cudaMemcpy(deviceArray, array, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    cout << "Starting...\n";
    for (int j = 0; j < N; j++){
        start = high_resolution_clock::now();

        countEven<<<BLOCKS, THREADS>>>(deviceArray, deviceCount);

        end = high_resolution_clock::now();
        timeElapsed += duration<double, std::milli>(end - start).count();
    }

    cudaMemcpy(count, deviceCount, BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);

    int aux = 0;
    for (int i = 0; i < BLOCKS; i++){
        aux += count[i];
    }

    cout << "Count of even numbers: " << aux << "\n";
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    cudaFree(deviceArray);
    cudaFree(deviceCount);

    delete[] array;
    delete[] count;

    return 0;
}
