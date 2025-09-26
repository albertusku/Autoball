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

    // ---- Warm-up: wait till first datagram ----
    {
        logger.log("INFO", "Waiting for first datagram on " + SOCKET_PATH + " ...");
        const int max_attempts = 100;           // ~5 s with DT=50 ms
        bool got_first = false;
        for (int i = 0; i < max_attempts; ++i) {
            double v = pid.receive_error_from_socket(SOCKET_PATH);
            // Note: the function returns the last valid value or 0.0 if there is none yet.
            // We consider it received if it is not NaN (and allow 0.0 as a valid possible value).
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
        // Attempts to read a datagram (the function has an internal timeout of ~100 ms)
        double error_px = pid.receive_error_from_socket(SOCKET_PATH);
        logger.log("DEBUG", "Received error: " + std::to_string(error_px));

        if (!std::isfinite(error_px)) {
            // If for any reason something non-parsable (NaN) arrives,
            // we avoid acting in this cycle.
            logger.log("WARN", "Invalid error received (NaN/Inf). Skipping cycle.");
            std::this_thread::sleep_for(std::chrono::milliseconds(int(DT * 1000)));
            continue;
        }

        // Applies dead zone
        if (std::abs(error_px) < DEAD_ZONE_PX) {
            motor.stop();
            std::this_thread::sleep_for(std::chrono::milliseconds(int(DT * 1000)));
            continue;
        }

        // PID
        double speed = pid.compute(error_px, DT);  // steps/s
        speed = std::clamp(speed, -MAX_SPEED, MAX_SPEED);

        // Calculates delay between steps
        double steps_per_sec = std::max(std::abs(speed), 1.0); // avoid division by 0
        int delay_us = int(1e6 / (2 * steps_per_sec));         // 1 cycle = 2 pulses

        logger.log("INFO", "Computed speed: " + std::to_string(speed) + " steps/s");
        motor.rotate_steps(speed > 0, STEPS_PER_UPDATE, delay_us);

        std::this_thread::sleep_for(std::chrono::milliseconds(int(DT * 1000)));
    }

    return 0;
}
