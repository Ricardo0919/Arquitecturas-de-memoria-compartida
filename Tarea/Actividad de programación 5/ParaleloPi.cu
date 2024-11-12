// =================================================================
//
// File: ParaleloPi.cu
// Author: Ricardo Sierra Roa - A01709887
// Description: Aproximacion de Pi mediante la serie de Nilakantha 
// utilizando CUDA para procesamiento en la GPU.
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

#define SIZE 10000000 // 1e7
#define THREADS 512
#define BLOCKS min(16, ((SIZE / THREADS) + 1))

__global__ void calcularPi(double *pi_partial, int size) {
    __shared__ double cache[THREADS];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    double constantePi = 0.0;
    int signo = (index % 2 == 0) ? 1 : -1;

    while (index < size) {
        constantePi += signo * (4.0 / ((2.0 * index + 2) * (2.0 * index + 3) * (2.0 * index + 4)));
        signo *= -1;
        index += blockDim.x * gridDim.x;
    }
    cache[threadIdx.x] = constantePi;

    __syncthreads();

    int gap = blockDim.x / 2;
    while (gap > 0) {
        if (threadIdx.x < gap) {
            cache[threadIdx.x] += cache[threadIdx.x + gap];
        }
        __syncthreads();
        gap /= 2;
    }

    if (threadIdx.x == 0) {
        pi_partial[blockIdx.x] = cache[0];
    }
}

int main(int argc, char* argv[]) {
    double *pi_partial, *devicePiPartial;
    double pi = 3.0;

    high_resolution_clock::time_point start, end;
    double timeElapsed = 0;

    pi_partial = new double[BLOCKS];

    cudaMalloc((void**)&devicePiPartial, BLOCKS * sizeof(double));

    cout << "Starting...\n";
    for (int j = 0; j < N; j++) {
        pi = 3.0; 
        start = high_resolution_clock::now();

        calcularPi<<<BLOCKS, THREADS>>>(devicePiPartial, SIZE);

        cudaDeviceSynchronize();

        end = high_resolution_clock::now();
        timeElapsed += duration<double, std::milli>(end - start).count();

        cudaMemcpy(pi_partial, devicePiPartial, BLOCKS * sizeof(double), cudaMemcpyDeviceToHost);

        for (int i = 0; i < BLOCKS; i++) {
            pi += pi_partial[i];
        }
    }

    cout << "Aproximacion de pi: " << fixed << setprecision(15) << pi << "\n";
    cout << "avg time = " << fixed << setprecision(3) << (timeElapsed / N) << " ms\n";

    cudaFree(devicePiPartial);
    delete[] pi_partial;

    return 0;
}
