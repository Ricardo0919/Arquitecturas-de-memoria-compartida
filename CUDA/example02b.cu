// =================================================================
//
// File: example02.cpp
// Author: Pedro Perez
// Description: This file contains the code that looks for an element 
//				X within the array and replaces it with Y using CUDA. 
//              To compile:
//		        nvcc -o app example02.cu
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include "utils.h"
#include <cuda_runtime.h>
#include <algorithm>

using namespace std;
using namespace std::chrono;

#define SIZE 	1000000000 //1e9
#define THREADS 512
#define BLOCKS	min(16, ((SIZE / THREADS) + 1))

__global__ void replace(int *array, int *x, int *y) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    while (index < SIZE) {
        if(array[index] == *x){
            array[index] = *y;
        }

        index += (blockDim.x * gridDim.x);
    }
}

int main(int argc, char* argv[]) {
    //We will use pointers to handle large arrays.
    int *array, x, y;
    int *deviceArray, *device_x, *device_y;

    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    array = new int[SIZE];
    for (int i = 0; i < SIZE; i++) {
        array[i] = 1;
    }
    display_array("before", array);
    x = 1; 
    y = -1;
    
    cudaMalloc((void**) &deviceArray, SIZE * sizeof(int));
    cudaMalloc((void**) &device_x, sizeof(int));
    cudaMalloc((void**) &device_y, sizeof(int));

    cudaMemcpy(device_x, &x, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, &y, sizeof(int), cudaMemcpyHostToDevice);

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        cudaMemcpy(deviceArray, array, SIZE * sizeof(int), cudaMemcpyHostToDevice);
        start = high_resolution_clock::now();

        replace<<<BLOCKS, THREADS>>>(deviceArray, device_x, device_y);

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }
    
    cudaMemcpy(array, deviceArray, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    display_array("after", array);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    delete [] array;
    delete [] x;
    delete [] y;

    cudaFree(deviceArray);
    cudaFree(device_x);
    cudaFree(device_y);

    return 0;
}