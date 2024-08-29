// =================================================================
//
// File: example04.cpp
// Author: Pedro Perez
// Description: This file implements the algorithm to find the 
//				minimum value in an array. The time this 
//				implementation takes will be used as the basis to 
//				calculate the improvement obtained with parallel 
//				technologies.
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <climits>
#include <thread>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 1000000000 // 1e9
#define THREADS std::thread::hardware_concurrency()

typedef struct {
    int start, end, result, *array;
} Block;

//Opcion rapida, pero lenta
void min(Block &block) {
    block.result = block.array[block.start];
    for (int i = block.start+1; i < block.end; i++) {
        if (block.array[i] < block.result) {
            block.result = block.array[i];
        }
    }
}

//Opcion mucho mas rapida
void minimum(Block &block) {
    int aux = block.array[block.start];
    for (int i = block.start+1; i < block.end; i++) {
        if (block.array[i] < aux) {
            aux = block.array[i];
        }
    }

    block.result = aux;
}

int main(int argc, char* argv[]) {
    int *array, result;

    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    array = new int [SIZE];
    
    random_array(array, SIZE);
    display_array("array:", array);

    // Declaración correcta de threads y blocks
    int blockSize = SIZE / THREADS;

    thread threads[THREADS];
    Block blocks[THREADS];

    for (int i = 0; i < THREADS; i++) {
        blocks[i].start = i * blockSize;
        blocks[i].end = (i != (THREADS - 1) ? (i + 1) * blockSize : SIZE);
        blocks[i].array = array;
        blocks[i].result = 0;
    }

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        // Lanzar hilos
        for (int i = 0; i < THREADS; i++) {
            threads[i] = thread(minimum, std::ref(blocks[i]));
        }


        result = array[9999];

        // Unir hilos
        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
            if(blocks[i].result < result){
                result = blocks[i].result;
            }
        }

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }
    cout << "result = " << result << "\n";
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    delete [] array;

    return 0;
}
