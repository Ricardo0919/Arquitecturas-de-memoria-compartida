// =================================================================
//
// File: example01.cpp
// Author: Pedro Perez
// Description: This file implements the addition of two vectors. 
//				The time this implementation takes will be used as 
//				the basis to calculate the improvement obtained with 
//				parallel technologies.
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

const int SIZE = 1000000; // 1e6

__global__ void add_vector(int *c, int *a, int *b) {
    //Threads
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

int main(int argc, char* argv[]) {
    int *a, *b, *c;
    int *device_a, *device_b, *device_c;

    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    //We create the arrays.
    a = new int [SIZE];
    b = new int [SIZE];
    c = new int [SIZE];

    cudaMalloc((void**) &device_a, SIZE * sizeof(int));
    cudaMalloc((void**) &device_b, SIZE * sizeof(int));
    cudaMalloc((void**) &device_c, SIZE * sizeof(int));

    fill_array(a, SIZE);
    display_array("a:", a);
    fill_array(b, SIZE);
    display_array("b:", b);

    cudaMemcpy(device_a, a, SIZE*sizeof(int),
        cudaMemcpyHostToDevice);

    cudaMemcpy(device_b, b, SIZE*sizeof(int),
        cudaMemcpyHostToDevice);

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        add_vector<<<1, SIZE>>>(device_c, device_a, device_b);

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }

    cudaMemcpy(c, device_c, SIZE * sizeof(int),
        cudaMemcpyDeviceToHost);

    display_array("c:", c);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    delete [] a;
    delete [] b;
    delete [] c;

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return 0;
}