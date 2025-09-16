#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cmath>
#include "PIDController.h"
#include "StepperMotor.h"
#include "auto_utils/logger.h"

const std::string SOCKET_PATH = "/tmp/pid_socket";
const double DT = 0.05;                 // 50 ms
const double DEAD_ZONE_PX = 10.0;       // Zona muerta en píxeles
const double MAX_SPEED = 600.0;         // Máx pasos/s
const int STEPS_PER_UPDATE = 4;         // Pasos por ciclo si se supera la zona muerta

int main() {
    StepperMotor motor("gpiochip0", 17, 27);
    PIDController pid(0.4, 0.01, 0.05);
    Logger logger("Main");

    logger.log("INFO", "Starting control loop...");

    // ---- Warm-up: espera hasta recibir el primer datagrama ----
    {
        logger.log("INFO", "Waiting for first datagram on " + SOCKET_PATH + " ...");
        const int max_attempts = 100;           // ~5 s con DT=50 ms
        bool got_first = false;
        for (int i = 0; i < max_attempts; ++i) {
            double v = pid.receive_error_from_socket(SOCKET_PATH);
            // Nota: la función devuelve el último valor válido o 0.0 si aún no hay nada.
            // Consideramos "recibido" si no es NaN (y permitimos 0.0 como valor real posible).
            if (std::isfinite(v)) {
                got_first = true;
                logger.log("INFO", "First datagram received: " + std::to_string(v));
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(int(DT * 1000)));
        }
        if (!got_first) {
            logger.log("WARN", "No datagrams received yet; the loop will keep trying.");
        }
    }
    // -----------------------------------------------------------

    while (true) {
        // Intenta leer un datagrama (la función tiene timeout interno ~100 ms)
        double error_px = pid.receive_error_from_socket(SOCKET_PATH);
        logger.log("DEBUG", "Received error: " + std::to_string(error_px));

        if (!std::isfinite(error_px)) {
            // Si por cualquier motivo llega algo no parseable (NaN),
            // evitamos actuar en este ciclo.
            logger.log("WARN", "Invalid error received (NaN/Inf). Skipping cycle.");
            std::this_thread::sleep_for(std::chrono::milliseconds(int(DT * 1000)));
            continue;
        }

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
        double steps_per_sec = std::max(std::abs(speed), 1.0); // evitar división por 0
        int delay_us = int(1e6 / (2 * steps_per_sec));         // 1 ciclo = 2 pulsos

        logger.log("INFO", "Computed speed: " + std::to_string(speed) + " steps/s");
        motor.rotate_steps(speed > 0, STEPS_PER_UPDATE, delay_us);

        std::this_thread::sleep_for(std::chrono::milliseconds(int(DT * 1000)));
    }

    return 0;
}
