// =================================================================
//
// File: SecuencialB.cpp
// Author: Ricardo Sierra Roa - A01709887
// Description: Sumar todos los números enteros primos que existen entre 
// 1 y 5,000,000 (5e6). El resultado esperado es  838,596,693,108. 
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
#include "utils.h"

using namespace std;
using namespace std::chrono;

const int SIZE = 5000000; // 5e6

bool isPrime(int num){
    if (num <= 1){
        return false;
    }
    if (num == 2 || num == 3){
        return true;
    }
    if (num % 2 == 0 || num % 3 == 0){
        return false;
    }
    for (int i = 5; i * i <= num; i += 6){
        if (num % i == 0 || num % (i + 2) == 0){
            return false;
        }
    }
    return true;
}

void sumPrimes(long long &sum, int size){
    sum = 0; 
    for (int i = 1; i <= size; i++){
        if (isPrime(i)){
            sum += i;
        }
    }
}

int main(int argc, char* argv[]){
    long long sum;

    high_resolution_clock::time_point start, end;
    double timeElapsed;

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++){
        start = high_resolution_clock::now();
        
        sumPrimes(sum, SIZE);

        end = high_resolution_clock::now();
        timeElapsed += duration<double, std::milli>(end - start).count();
    }

    cout << "Sum of prime numbers: " << sum << "\n";
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    return 0;
}
