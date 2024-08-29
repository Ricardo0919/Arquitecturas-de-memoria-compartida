#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include "utils.h"

using namespace std;
using namespace std::chrono;

const int SIZE = 1000000000; // 1e9
const int THREADS = std::thread::hardware_concurrency();

typedef struct {
    int *a, *b, *c;
    int start, end;
} Block;

void add_vector(Block &block) {
    for (int i = block.start; i < block.end; i++) {
        block.c[i] = block.a[i] + block.b[i];
    }
}

int main(int argc, char* argv[]) {
    int *a, *b, *c;

    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    a = new int [SIZE];
    b = new int [SIZE];
    c = new int [SIZE];

    fill_array(a, SIZE);
    display_array("a:", a);
    fill_array(b, SIZE);
    display_array("b:", b);

    // Declarar bloques y threads
    Block blocks[THREADS];
    thread threads[THREADS];

    int blockSize = SIZE / THREADS;
    for (int i = 0; i < THREADS; i++) {
        blocks[i].start = i * blockSize;
        blocks[i].end = (i != (THREADS - 1) ? (i + 1) * blockSize : SIZE);
        blocks[i].a = a;
        blocks[i].b = b;
        blocks[i].c = c;
    }

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        // Crear y lanzar hilos
        for (int i = 0; i < THREADS; i++) {
            threads[i] = thread(add_vector, std::ref(blocks[i]));
        }

        // Esperar a que todos los hilos terminen
        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
        }

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }
    display_array("c:", c);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) << " ms\n";

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
