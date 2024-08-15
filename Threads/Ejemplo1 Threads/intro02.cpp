#include <iostream>
#include <iomanip>
#include <thread> 

using namespace std;

void task(int limit){
    for(int i = 1; i <= limit; i++){
        cout << i << " ";
    }
    cout << "\n";
}

int main(int argc, char* argv[]){
    thread t;

    t = thread(task, 20);
    
    t.join();

    return 0;
}