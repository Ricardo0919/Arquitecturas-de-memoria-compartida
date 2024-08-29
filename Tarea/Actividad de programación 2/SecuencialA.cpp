// =================================================================
//
// File: SecuencialA.cpp
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
#include "utils.h"

using namespace std;
using namespace std::chrono;

const int SIZE = 1000000000; // 1e9

void countEven(int *array, int &count, int size){
    count = 0;
    for(int i = 0; i<size; i++){
        if( array[i]%2 == 0){
            count += 1;
        }
    }
}

int main(int argc, char* argv[]) {
    int *array, count=0;

    high_resolution_clock::time_point start, end;
    double timeElapsed;

    array = new int [SIZE];

    fill_array(array, SIZE);

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();
        
        countEven(array, count, SIZE);

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