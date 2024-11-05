// =================================================================
// 
// File: barberProblem.cpp
// Author: Ricardo Sierra Roa - A01709887
// Description: Problema del barbero (mutex y condition_variable)
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <queue>
#include <set>
#include <chrono>
#include <cstdlib>
#include <map>
#include <atomic>

using namespace std;

// Variables y sincronizacion
mutex mtx;
condition_variable customerReady;
queue<int> waitingCustomers; // Cola de clientes esperando
set<int> customersInBarberShop; // Conjunto de clientes que ya estan en la barberia
set<int> customersBeingServed; // Conjunto de clientes que estan siendo atendidos
const int maxSeats = 4;      // Numero maximo de sillas disponibles
int availableSeats = maxSeats; // Numero de sillas disponibles

// Contadores para cada cliente
map<int, int> attendCount; // Contador de veces que el cliente fue atendido
map<int, int> fullShopCount; // Contador de veces que el cliente encontro la barberia llena
atomic<bool> stopProgram(false); // Variable para detener el programa

void barber() {
    while (!stopProgram) {
        unique_lock<mutex> lock(mtx);
        // Espera a que haya al menos un cliente en la barberia
        customerReady.wait(lock, []() { return !waitingCustomers.empty() || stopProgram; });

        if (stopProgram) break;

        int currentCustomer = waitingCustomers.front();
        waitingCustomers.pop();
        customersInBarberShop.erase(currentCustomer); // Elimina al cliente de la lista de espera
        customersBeingServed.insert(currentCustomer); // Marca al cliente como atendido
        cout << "El barbero se despierta y empieza a cortar el cabello del cliente " << currentCustomer << ".\n";
        lock.unlock();

        // Simula el tiempo de corte de cabello
        this_thread::sleep_for(chrono::seconds(3));

        lock.lock();
        availableSeats++; // Libera la silla despues de cortar el cabello
        customersBeingServed.erase(currentCustomer); // Elimina al cliente de la lista de atendidos

        // Incrementa el contador de veces atendido y muestra el mensaje
        attendCount[currentCustomer]++;
        cout << "El barbero termino de cortar el cabello del cliente " << currentCustomer << ".\n";
        cout << "Sillas disponibles: " << availableSeats << "\n";

        // Verifica si se debe detener el programa
        bool allReachedLimit = true;
        for (int i = 1; i <= 7; ++i) {
            int totalAttempts = attendCount[i] + fullShopCount[i];
            if (totalAttempts < 5) {
                allReachedLimit = false;
                break;
            }
        }
        if (allReachedLimit) {
            stopProgram = true;
            customerReady.notify_all(); // Notifica a los clientes para detenerlos
        }

        lock.unlock();
    }
}

void customer(int id) {
    while (!stopProgram) {
        unique_lock<mutex> lock(mtx);
        // Verifica si el cliente no esta ya en la barberia o siendo atendido
        if (customersInBarberShop.find(id) == customersInBarberShop.end() && customersBeingServed.find(id) == customersBeingServed.end()) {
            if (attendCount[id] + fullShopCount[id] >= 5) {
                lock.unlock();
                break; // El cliente deja de intentar si ya ha alcanzado 5 intentos
            }

            if (availableSeats > 0) {
                // El cliente se sienta y ocupa una silla
                availableSeats--; // Ocupa una silla al sentarse
                waitingCustomers.push(id);
                customersInBarberShop.insert(id); // Marca al cliente como dentro de la barberia
                cout << "Cliente " << id << " se sienta y espera. Sillas disponibles: " << availableSeats << "\n";
                customerReady.notify_one(); // Notifica al barbero que hay un cliente
            } else {
                // Incrementa el contador de veces que el cliente encontro la barberia llena y muestra el mensaje
                fullShopCount[id]++;
                cout << "Cliente " << id << " encontro la barberia llena. Volvera mas tarde.\n";
            }
        }
        
        lock.unlock();

        // Simula el tiempo antes de que el cliente vuelva a intentar
        this_thread::sleep_for(chrono::seconds(rand() % 5 + 1));
    }
}

int main() {
    srand(time(NULL)); // Inicializa la semilla aleatoria para tiempos de espera

    // Inicializa los contadores para cada cliente
    for (int i = 1; i <= 7; ++i) {
        attendCount[i] = 0;
        fullShopCount[i] = 0;
    }

    // Crea el hilo del barbero
    thread barberThread(barber);

    // Crea hilos de clientes
    vector<thread> customerThreads;
    for (int i = 1; i <= 7; ++i) {
        customerThreads.emplace_back(customer, i);
    }

    // Espera a que los hilos de los clientes terminen
    for (auto& th : customerThreads) {
        th.join();
    }

    // Detenemos el hilo del barbero (en este ejemplo, es infinito)
    if (barberThread.joinable()) {
        barberThread.join();
    }

    // Muestra el resumen de las veces que cada cliente fue atendido o encontro la barberia llena
    cout << "\nResumen de clientes:\n";
    for (int i = 1; i <= 7; ++i) {
        int totalAttempts = attendCount[i] + fullShopCount[i];
        cout << "Cliente " << i << " fue atendido " << attendCount[i] << " veces, encontro la barberia llena " << fullShopCount[i] << " veces, total de intentos: " << totalAttempts << ".\n";
    }

    return 0;
}