// =================================================================
//
// File: example02.cpp
// Author: Pedro Perez
// Description: This file contains the code that looks for an element 
//				X within the array and replaces it with Y. The time 
//				it takes to implement this will be used as the basis 
//				for calculating the improvement obtained with parallel 
//				technologies. The time this implementation takes.
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <thread>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 1000000000  // 1e9
#define THREADS std::thread::hardware_concurrency()

typedef struct {
    int start, end, x, y, *array;
} Block;

void replace(Block &block) {
    for (int i = block.start; i < block.end; i++) {
        if (block.array[i] == block.x) {
            block.array[i] = block.y;  // Corrección: asignación en lugar de comparación
        }
    }
}

int main(int argc, char* argv[]) {
    int *array, *aux;

    // Variables para medir el tiempo de ejecución.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    array = new int[SIZE];
    for (int i = 0; i < SIZE; i++) {
        array[i] = 1;
    }
    display_array("before", array);
    
    aux = new int[SIZE];

    // Declaración correcta de threads y blocks
    thread threads[THREADS];
    Block blocks[THREADS];

    int blockSize = SIZE / THREADS;
    for (int i = 0; i < THREADS; i++) {
        blocks[i].start = i * blockSize;
        blocks[i].end = (i != (THREADS - 1) ? (i + 1) * blockSize : SIZE);
        blocks[i].x = 1;
        blocks[i].y = 2;  // Valor a reemplazar, ejemplo: 1 por 2
        blocks[i].array = aux;
    }

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        memcpy(aux, array, sizeof(int) * SIZE);
        
        start = high_resolution_clock::now();

        // Lanzar hilos
        for (int i = 0; i < THREADS; i++) {
            threads[i] = thread(replace, std::ref(blocks[i]));
        }

        // Unir hilos
        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
        }

        end = high_resolution_clock::now();
        timeElapsed += duration<double, std::milli>(end - start).count();
    }
    
    display_array("after", aux);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    delete[] array;
    delete[] aux;

    return 0;
}
