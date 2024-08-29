// =================================================================
//
// File: example05.cpp
// Author: Pedro Perez
// Description: This file contains the approximation of Pi using the 
//				Monte-Carlo method. The time this implementation 
//				takes will be used as the basis to calculate the 
//				improvement obtained with parallel technologies.
//
// Reference:
//	https://www.geogebra.org/m/cF7RwK3H
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <thread>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define INTERVAL 		 10000 // 1e4
#define NUMBER_OF_POINTS (INTERVAL * INTERVAL) // 1e8
#define THREADS std::thread::hardware_concurrency()

typedef struct {
    int start, end, result;
} Block;


void aprox_pi(Block &block) {
    double x, y, dist;
    int count;

    count = 0;
    for (int i = block.start; i < block.end; i++) {
        x = double(rand() % (INTERVAL + 1)) / ((double) INTERVAL);
        y = double(rand() % (INTERVAL + 1)) / ((double) INTERVAL);
        dist = (x * x) + (y * y);
        if (dist <= 1) {
            count++;
        }
    }

    block.result = count;
}

int main(int argc, char* argv[]) {
    double result;
    int count;
    
    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    // Declaración correcta de threads y blocks
    int blockSize = NUMBER_OF_POINTS / THREADS;

    thread threads[THREADS];
    Block blocks[THREADS];

    for (int i = 0; i < THREADS; i++) {
        blocks[i].start = i * blockSize;
        blocks[i].end = (i != (THREADS - 1) ? (i + 1) * blockSize : NUMBER_OF_POINTS);
        blocks[i].result = 0;
    }
    
    srand(time(0));

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        // Lanzar hilos
        for (int i = 0; i < THREADS; i++) {
            threads[i] = thread(aprox_pi, std::ref(blocks[i]));
        }

        count = 0;

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
    
    result = ((double)(4.0 * result)) / ((double) NUMBER_OF_POINTS);
    cout << "result = " << fixed << setprecision(20)  << result << "\n";
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    return 0;
}
