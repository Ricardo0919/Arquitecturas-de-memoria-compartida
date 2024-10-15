// =================================================================
//
// File: SecuencialC.cpp
// Author: Ricardo Sierra Roa - A01709887
// Description: Ordenar un arreglo de números enteros usando el algoritmo 
// "Ranking Sort"Links to an external site. El tamaño del arreglo 
// debe ser 1e4 (10,000).
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <chrono>
#include <iomanip>
#include "utils.h"

using namespace std;
using namespace std::chrono;

const int SIZE = 10000; // 1e4

void rankingSort(int *array, int *sortArray, int size){
    int position;
    for (int i = 0; i < size; i++){
        position = 0;
        for (int j = 0; j < size; j++){ 
            if ( (array[j] < array[i]) || (array[j] == array[i] && j < i) ){
                position++;
            } 
        }
        sortArray[position] = array[i];
    }
}

int main() {
    int *array = new int[SIZE];

    high_resolution_clock::time_point start, end;
    double timeElapsed;

    random_array(array, SIZE); 

    int newArray[SIZE];

    cout << "Starting...\n";
    timeElapsed = 0;

    for (int j = 0; j < N; j++){
        start = high_resolution_clock::now();

        rankingSort(array, newArray, SIZE);

        end = high_resolution_clock::now();
        timeElapsed += duration<double, milli>(end - start).count();
    }
    
    display_array("Sorted Array: ", newArray);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    delete[] array; 
    
    return 0;
}