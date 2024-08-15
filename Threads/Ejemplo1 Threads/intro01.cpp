#include <iostream>
#include <iomanip>
#include <thread> 

using namespace std;

void task(){
    for(int i = 1; i <= 10; i++){
        cout << i << " ";
    }
    cout << "\n";
}

int main(int argc, char* argv[]){
    thread t;

    t = thread(task);
    
    t.join();

    return 0;
}