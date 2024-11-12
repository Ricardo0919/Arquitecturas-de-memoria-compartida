// =================================================================
//
// File: ParaleloPi.cpp
// Author: Ricardo Sierra Roa - A01709887
// Description: Aproximacion de Pi, mediante la serie de Nilakantha
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

const int SIZE = 10000000; // 1e7
const int THREADS = std::thread::hardware_concurrency();

typedef struct {
    double pi;
    int start, end;
} Block;

void calcularPi(Block &block) {
    double constantePi = 0.0;
    int signo = 1;

    for (int i = block.start; i<block.end; i++) {
        constantePi += signo *(4.0 / ((2.0 * i) * (2.0 * i + 1) * (2.0 * i + 2)));
        signo *= -1;
    }

    block.pi = constantePi;
}

int main(int argc, char* argv[]) {
    double pi = 3.0;

    high_resolution_clock::time_point start, end;
    double timeElapsed;

    Block blocks[THREADS];
    thread threads[THREADS];

    int blockSize = SIZE / THREADS;
    for (int i = 0; i < THREADS; i++) {
        blocks[i].start = i * blockSize+1;
        blocks[i].end = (i != (THREADS - 1) ? (i + 1) * blockSize : SIZE+1);
        blocks[i].pi = 0.0;
    }

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        pi = 3.0;
        start = high_resolution_clock::now();
        
        for (int i = 0; i < THREADS; i++) {
            threads[i] = thread(calcularPi, std::ref(blocks[i]));
        }
        
        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
        }

        for (int i = 0; i < THREADS; i++) {
            pi += blocks[i].pi;
        }

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }

    cout << "Aproximacion de pi: " << fixed << setprecision(15) << pi << "\n";
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    return 0;
}