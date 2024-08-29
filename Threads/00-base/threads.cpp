#include <iostream>
#include <thread>

int main() {
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        std::cout << "No se pudo determinar el número de hilos soportados." << std::endl;
    } else {
        std::cout << "Número de hilos soportados: " << num_threads << std::endl;
    }
    return 0;
}
