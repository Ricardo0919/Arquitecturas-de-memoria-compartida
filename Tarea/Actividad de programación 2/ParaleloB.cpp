// =================================================================
//
// File: ParaleloB.cpp
// Author: Ricardo Sierra Roa - A01709887
// Description: Sumar todos los números enteros primos que existen entre 
// 1 y 5,000,000 (5x106). El resultado esperado es  838,596,693,108. 
// NOTA: utiliza una variable double para la suma.
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

const int SIZE = 5000000; // 5e6
const int THREADS = std::thread::hardware_concurrency();

typedef struct {
    long long sum;
    int start, end;
} Block;

bool isPrime(int num) {
    if (num <= 1) {
        return false;
    }
    if (num == 2 || num == 3) {
        return true;
    }
    if (num % 2 == 0 || num % 3 == 0) {
        return false;
    }
    for (int i = 5; i * i <= num; i += 6) {
        if (num % i == 0 || num % (i + 2) == 0) {
            return false;
        }
    }
    return true;
}

void sumPrimes(Block &block) {
    double aux = 0:
    for (int i = block.start; i <= block.end; i++) {
        if (isPrime(i)) {
            aux += i;
        }
    }
    block.sum = aux; 
}

int main(int argc, char* argv[]) {
    long long sum;

    high_resolution_clock::time_point start, end;
    double timeElapsed;

    Block blocks[THREADS];
    thread threads[THREADS];

    int blockSize = SIZE / THREADS;
    for (int i = 0; i < THREADS; i++) {
        blocks[i].start = i * blockSize;
        blocks[i].end = (i != (THREADS - 1) ? (i + 1) * blockSize : SIZE);
        blocks[i].sum = 0;
    }

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        sum = 0;
        start = high_resolution_clock::now();
        
        for (int i = 0; i < THREADS; i++) {
            threads[i] = thread(sumPrimes, std::ref(blocks[i]));
        }
        
        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
        }

        for (int i = 0; i < THREADS; i++) {
            sum += blocks[i].sum;
        }

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }

    cout << "Sum of prime numbers: " << sum << "\n";
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    return 0;
}
