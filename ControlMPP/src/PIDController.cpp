#include "PIDController.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <auto_utils/logger.h>

PIDController::PIDController(double kp, double ki, double kd)
    : kp(kp), ki(ki), kd(kd), integral(0.0), prev_error(0.0) {}

static Logger logger("PIDController");

double PIDController::compute(double error, double dt) {
    integral += error * dt;
    double derivative = (error - prev_error) / dt;
    prev_error = error;
    return kp * error + ki * integral + kd * derivative;
}

void PIDController::reset() {
    integral = 0.0;
    prev_error = 0.0;
}

double PIDController::receive_error_from_socket(const std::string& socket_path) {
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        logger.log("ERROR", "Socket creation failed");
        return 0.0;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
        logger.log("ERROR", "Failed to connect to socket: " + socket_path);
        close(sock);
        return 0.0;
    }

    char buffer[64];
    int len = read(sock, buffer, sizeof(buffer) - 1);
    close(sock);

    if (len <= 0) {
        logger.log("ERROR", "Failed to read from socket or no data received");
        return 0.0;
    }

    buffer[len] = '\0';
    return std::stod(buffer);
}