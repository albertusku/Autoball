#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cmath>
#include "PIDController.h"
#include "StepperMotor.h"

const std::string SOCKET_PATH = "/tmp/pid_socket";
const double DT = 0.05;                 // 50 ms
const double DEAD_ZONE_PX = 10.0;       // Zona muerta en píxeles
const double MAX_SPEED = 600.0;         // Máx pasos/s
const int STEPS_PER_UPDATE = 4;         // Pasos a emitir por ciclo si se supera la zona muerta

int main() {
    StepperMotor motor("gpiochip0", 17, 27);
    PIDController pid(0.4, 0.01, 0.05);

    std::cout << "Control PID en ejecución. Esperando datos por socket...\n";

    while (true) {
        double error_px = pid.receive_error_from_socket(SOCKET_PATH);

        // Aplica zona muerta
        if (std::abs(error_px) < DEAD_ZONE_PX) {
            motor.stop();
            std::this_thread::sleep_for(std::chrono::milliseconds(int(DT * 1000)));
            continue;
        }

        // PID
        double speed = pid.compute(error_px, DT);  // pasos/s
        speed = std::clamp(speed, -MAX_SPEED, MAX_SPEED);

        // Calcula delay entre pasos
        double steps_per_sec = std::abs(speed);
        int delay_us = int(1e6 / (2 * steps_per_sec));  // cada pulso (1 ciclo = 2 pulsos)

        motor.rotate_steps(speed > 0, STEPS_PER_UPDATE, delay_us);
        std::this_thread::sleep_for(std::chrono::milliseconds(int(DT * 1000)));
    }

    return 0;
}
