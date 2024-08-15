#include <iostream>
#include <iomanip>
#include <thread>

using namespace std;

#define THREADS 4

typedef struct {
    int id, start, end;
} Block;

void task(Block &b){
    for(int i = b.start; i < b.end; i++){
        cout << "id = " << b.id << " val = " << i << "\n";
    }
}

int main(int argc, char* argv[]){
    Block blocks[THREADS];
    thread t[THREADS];

    for(int i = 0; i < THREADS; i++){
        blocks[i].id = i;
        blocks[i].start = i * 10;
        blocks[i].end = (i+1) * 10;
    }
    
    for(int i = 0; i < THREADS; i++){
        t[i] = thread(task, std::ref(blocks[i]));
    }

    for(int i = 0; i < THREADS; i++){
        t[i].join();
    }

    return 0;
}