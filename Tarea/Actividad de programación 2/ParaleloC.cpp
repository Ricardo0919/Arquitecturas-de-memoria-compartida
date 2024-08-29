// =================================================================
//
// File: ParaleloC.cpp
// Author: Ricardo Sierra Roa - A01709887
// Description: Ordenar un arreglo de números enteros usando el algoritmo 
// "Ranking Sort"Links to an external site.. El tamaño del arreglo 
// debe ser 1x104 (10,000).
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include "utils.h"

using namespace std;
using namespace std::chrono;

const int SIZE = 10000; // 1e4
const int THREADS = std::thread::hardware_concurrency();

typedef struct {
    int *array;
    int start, end;
} Block;

void rankSort(Block &block) {
    int *rank = new int[SIZE];
    int *sortedArray = new int[SIZE];

    for (int i = block.start; i < block.end; i++) {
        rank[i] = 0;
        sortedArray[i] = -1;
    }

    for (int i = block.start; i < block.end; i++) {
        for (int j = block.start; j < block.end; j++) {
            if (block.array[i] > block.array[j]) {
                rank[i]++;
            }
        }
    }

    for (int i = block.start; i < block.end; i++) {
        int pos = rank[i];
        while (sortedArray[pos] != -1) {
            pos++;
            if (pos >= SIZE) {
                cerr << "Error: Array index out of bounds!" << endl;
                exit(EXIT_FAILURE);
            }
        }
        sortedArray[pos] = block.array[i];
    }

    for (int i = block.start; i < block.end; i++) {
        block.array[i] = sortedArray[i];
    }

    delete[] rank;
    delete[] sortedArray;
}

int main(int argc, char* argv[]) {
    int *array;

    high_resolution_clock::time_point start, end;
    double timeElapsed;

    array = new int [SIZE];

    random_array(array, SIZE);

    // Declarar bloques y threads
    Block blocks[THREADS];
    thread threads[THREADS];

    int blockSize = SIZE / THREADS;
    for (int i = 0; i < THREADS; i++) {
        blocks[i].start = i * blockSize;
        blocks[i].end = (i != (THREADS - 1) ? (i + 1) * blockSize : SIZE);
        blocks[i].array = array;
    }

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        for (int i = 0; i < THREADS; i++) {
            threads[i] = thread(rankSort, std::ref(blocks[i]));
        }
        
        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
        }
        
        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }

    display_array("Sorted Array: ", array);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    delete [] array;

    return 0;
}
