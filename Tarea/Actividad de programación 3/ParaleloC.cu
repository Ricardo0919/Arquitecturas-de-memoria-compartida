// =================================================================
//
// File: ParaleloC.cpp
// Author: Ricardo Sierra Roa - A01709887
// Description: Ordenar un arreglo de números enteros usando el algoritmo 
// "Ranking Sort"Links to an external site. El tamaño del arreglo 
// debe ser 1e4 (10,000).
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 	10000 // 1e4
#define THREADS 512
#define BLOCKS	min(16, ((SIZE / THREADS) + 1))

__global__ void rankingSort(int *array, int *sortArray) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    int position;
    while (index < SIZE) {
        position = 0;
        for (int j = 0; j < SIZE; j++){
            if ( (array[j] < array[index]) || (array[j] == array[index] && j < index) ) {
                position++; 
            }
        }
        sortArray[position] = array[index];
        index += (gridDim.x * blockDim.x);
    }
}

int main(){
    int *array, *deviceArray; 
    int *deviceSortArray, *sortArray;

    high_resolution_clock::time_point start, end;
    double timeElapsed;

    array = new int[SIZE];
    random_array(array, SIZE);

    sortArray = new int[SIZE];

    cudaMalloc( (void**) &deviceArray, SIZE * sizeof(int) );
    cudaMalloc( (void**) &deviceSortArray, SIZE * sizeof(int) );

    cudaMemcpy(deviceArray, array, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    cout << "Starting...\n";
    timeElapsed = 0;

    for (int j = 0; j < N; j++){
        start = high_resolution_clock::now();

        rankingSort<<<BLOCKS, THREADS>>> (deviceArray, deviceSortArray);
        cudaDeviceSynchronize();

        end = high_resolution_clock::now();
        timeElapsed += duration<double, std::milli>(end - start).count();
    }

    cudaMemcpy(sortArray, deviceSortArray, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    display_array("Sorted Array: ", sortArray);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    cudaFree(deviceArray);
    cudaFree(deviceSortArray);
    
    delete[] array;
    delete[] sortArray;

    return 0;
}