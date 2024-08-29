// =================================================================
//
// File: SecuencialC.cpp
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
#include "utils.h"

using namespace std;
using namespace std::chrono;

const int SIZE = 10000; // 1e4

void rankSort(int *array, int size) {
    int *rank = new int[size];
    int *sortedArray = new int[size];

    for (int i = 0; i < size; i++) {
        rank[i] = 0;
        sortedArray[i] = -1;
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (array[i] > array[j]) {
                rank[i]++;
            }
        }
    }

    for (int i = 0; i < size; i++) {
        int pos = rank[i];
        while (sortedArray[pos] != -1) {
            pos++;
            if (pos >= size) {
                cerr << "Error: Array index out of bounds!" << endl;
                exit(EXIT_FAILURE);
            }
        }
        sortedArray[pos] = array[i];
    }

    for (int i = 0; i < size; i++) {
        array[i] = sortedArray[i];
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

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        rankSort(array, SIZE);

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
