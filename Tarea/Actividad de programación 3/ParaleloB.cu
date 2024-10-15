// =================================================================
//
// File: ParaleloB.cpp
// Author: Ricardo Sierra Roa - A01709887
// Description: Sumar todos los números enteros primos que existen entre 
// 1 y 5,000,000 (5e6). El resultado esperado es  838,596,693,108. 
// NOTA: utiliza una variable double para la suma.
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

#define SIZE  5000000 // 5e6
#define THREADS 512
#define BLOCKS	min(16, ((SIZE / THREADS) + 1))

__device__ bool isPrime(int num){
    if (num <= 1){
        return false;
    }
    if (num == 2 || num == 3){
        return true;
    }
    if (num % 2 == 0 || num % 3 == 0){
        return false;
    }
    for (int i = 5; i * i <= num; i += 6){
        if (num % i == 0 || num % (i + 2) == 0){
            return false;
        }
    }
    return true;
}

__global__ void sumPrimes(long long *sum) {
    __shared__ long long cache[THREADS];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    long long localSum = 0;

    while (index < SIZE){
        if (isPrime(index)){
            localSum += index;
        }
        index += blockDim.x * gridDim.x;
    }

    cache[threadIdx.x] = localSum;

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
        sum[blockIdx.x] = cache[threadIdx.x];
    }
}

int main(int argc, char* argv[]){
    long long *sum;
    long long *deviceSum;

    high_resolution_clock::time_point start, end;
    double timeElapsed = 0;

    sum = new long long[BLOCKS];

    cudaMalloc((void**)&deviceSum, BLOCKS * sizeof(long long));  

    cout << "Starting...\n";
    for (int j = 0; j < N; j++){
        start = high_resolution_clock::now();

        sumPrimes<<<BLOCKS, THREADS>>>(deviceSum);

        end = high_resolution_clock::now();
        timeElapsed += duration<double, std::milli>(end - start).count();
    }

    cudaMemcpy(sum, deviceSum, BLOCKS * sizeof(long long), cudaMemcpyDeviceToHost);

    long long aux = 0;
    for (int i = 0; i < BLOCKS; i++){
        aux += sum[i];
    }

    cout << "Sum of prime numbers: " << aux << "\n";
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    cudaFree(deviceSum);

    delete[] sum; 

    return 0;
}
