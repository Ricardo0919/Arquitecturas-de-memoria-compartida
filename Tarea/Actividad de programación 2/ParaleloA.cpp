// =================================================================
//
// File: ParaleloA.cpp
// Author: Ricardo Sierra Roa - A01709887
// Description: Contar el número de pares que existen en un arreglo de 
// números enteros. El tamaño del arreglo debe ser 1x109 (1,000,000,000).
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

const int SIZE = 1000000000; // 1e9
const int THREADS = std::thread::hardware_concurrency();

typedef struct {
    int *array, count;
    int start, end;
} Block;

void countEven(Block &block){
    block.count = 0;
    for(int i = block.start; i<block.end; i++){
        if( block.array[i]%2 == 0){
            block.count += 1;
        }
    }
}

int main(int argc, char* argv[]) {
    int *array, count=0;

    high_resolution_clock::time_point start, end;
    double timeElapsed;

    array = new int [SIZE];

    fill_array(array, SIZE);

    Block blocks[THREADS];
    thread threads[THREADS];

    int blockSize = SIZE / THREADS;
    for (int i = 0; i < THREADS; i++) {
        blocks[i].start = i * blockSize;
        blocks[i].end = (i != (THREADS - 1) ? (i + 1) * blockSize : SIZE);
        blocks[i].array = array;
        blocks[i].count = 0;
    }

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        count = 0;
        start = high_resolution_clock::now();
        
        for (int i = 0; i < THREADS; i++) {
            threads[i] = thread(countEven, std::ref(blocks[i]));
        }
        
        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
        }

        for (int i = 0; i < THREADS; i++) {
            count += blocks[i].count;
        }

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }

    cout << "Count of even numbers: " << count << "\n";
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    delete [] array;

    return 0;
}