#include <iostream>
#include <iomanip>
#include <thread>

#define THREADS 4

using namespace std;

void task(int id, int limit){
    for(int i = 1; i <= limit; i++){
        cout << "id = " << id << " val = " << i << "\n";
    }
}

int main(int argc, char* argv[]){
    thread t[THREADS];

    for(int i = 0; i < THREADS; i++){
        t[i] = thread(task, i, 20);
    }
    
    for(int i = 0; i < THREADS; i++){
        t[i].join();
    }

    return 0;
}