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

using namespace std;

// Variables y sincronizacion
mutex mtx;
condition_variable customerReady;
queue<int> waitingCustomers; // Cola de clientes esperando
set<int> customersInBarberShop; // Conjunto de clientes que ya estan en la barberia
set<int> customersBeingServed; // Conjunto de clientes que estan siendo atendidos
const int maxSeats = 4;      // Numero maximo de sillas disponibles
int availableSeats = maxSeats; // Numero de sillas disponibles

void barber() {
    while (true) {
        unique_lock<mutex> lock(mtx);
        // Espera a que haya al menos un cliente en la barberia
        customerReady.wait(lock, []() { return !waitingCustomers.empty(); });

        int currentCustomer = waitingCustomers.front();
        waitingCustomers.pop();
        customersInBarberShop.erase(currentCustomer); // Elimina al cliente de la lista de espera
        customersBeingServed.insert(currentCustomer); // Marca al cliente como atendido
        cout << "El barbero se despierta y empieza a cortar el cabello del cliente " << currentCustomer << ".\n";
        lock.unlock();

        // Simula el tiempo de corte de cabello
        this_thread::sleep_for(chrono::seconds(3));

        lock.lock();
        availableSeats++; // Libera la silla después de cortar el cabello
        customersBeingServed.erase(currentCustomer); // Elimina al cliente de la lista de atendidos
        cout << "El barbero termino de cortar el cabello del cliente " << currentCustomer << ".\n";
        cout << "Sillas disponibles: " << availableSeats << "\n";
        lock.unlock();
    }
}

void customer(int id) {
    for (int i = 0; i < 5; ++i) {
        unique_lock<mutex> lock(mtx);
        // Verifica si el cliente no está ya en la barbería o siendo atendido
        if (customersInBarberShop.find(id) == customersInBarberShop.end() && customersBeingServed.find(id) == customersBeingServed.end()) {
            if (availableSeats > 0) {
                // El cliente se sienta y ocupa una silla
                availableSeats--; // Ocupa una silla al sentarse
                waitingCustomers.push(id);
                customersInBarberShop.insert(id); // Marca al cliente como dentro de la barbería
                cout << "Cliente " << id << " se sienta y espera. Sillas disponibles: " << availableSeats << "\n";
                customerReady.notify_one(); // Notifica al barbero que hay un cliente
            } else {
                // Si no hay sillas disponibles, el cliente se va
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
    barberThread.detach();

    return 0;
}
