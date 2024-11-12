// =================================================================
//
// File: SecuencialPi.cpp
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
#include "utils.h"

using namespace std;
using namespace std::chrono;

const int SIZE = 10000000; // 1e7

void calcularPi(double &pi, int size) {
    double constantePi = 3.0;
    int signo = 1;

    for (int i = 1; i <= size; i++) {
        constantePi += signo *(4.0 / ((2.0 * i) * (2.0 * i + 1) * (2.0 * i + 2)));
        signo *= -1;
    }

    pi = constantePi;
}

int main(int argc, char* argv[]) {
    double pi = 0.0;

    high_resolution_clock::time_point start, end;
    double timeElapsed;

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();
        
        calcularPi(pi, SIZE);

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }

    cout << "Aproximacion de pi: " << fixed << setprecision(15) << pi << "\n";
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    return 0;
}