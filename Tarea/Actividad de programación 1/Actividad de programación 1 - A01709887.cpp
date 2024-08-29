// =================================================================
//
// File: main.cpp
// Author: Ricardo Sierra Roa - A01709887
// Date: 14/08/2024
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <thread>
#include <cmath>

using namespace std;

void taskRoot(int limit){
    for(int i = 1; i <= limit; i++){
        cout << "Number = " << i << " val = " << sqrt(i) << "\n";
    }
}

void taskSquare(int limit){
    for(int i = 1; i <= limit; i++){
        cout << "Number = " << i << " val = " << i*i << "\n";
    }
}

int main(int argc, char* argv[]){
    thread t;

    cout << "Square Root" << "\n";
    t = thread(taskRoot, 10);
    
    t.join();

    cout << "Power" << "\n";
    t = thread(taskSquare, 10);

    t.join();

    return 0;
}