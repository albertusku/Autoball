// PIDController.cpp
#include "PIDController.h"

// POSIX
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <unistd.h>

// STL
#include <algorithm>
#include <locale>
#include <cstring>
#include <cstdio>
#include <cctype>
#include <iostream>

// Tu logger
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
    static int fd = -1;
    static double last_valid = 0.0;
    static bool timeout_set = false;

    auto log_errno = [](const std::string& where){
        char msg[256];
        std::snprintf(msg, sizeof(msg), "%s: %s", where.c_str(), std::strerror(errno));
        logger.log("ERROR", msg);
    };

    auto trim = [](std::string& s){
        auto notspace = [](unsigned char ch){ return !std::isspace(ch); };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), notspace));
        s.erase(std::find_if(s.rbegin(), s.rend(), notspace).base(), s.end());
    };

    // Inicializar socket y hacer bind la primera vez
    if (fd == -1) {
        if (socket_path.size() >= sizeof(sockaddr_un::sun_path)) {
            logger.log("ERROR", "Socket path too long: " + socket_path);
            return last_valid;
        }

        fd = ::socket(AF_UNIX, SOCK_DGRAM, 0);
        if (fd < 0) {
            log_errno("socket()");
            return last_valid;
        }

        // Limpiar path previo y bind
        ::unlink(socket_path.c_str());
        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

        if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == -1) {
            log_errno("bind()");
            ::close(fd);
            fd = -1;
            return last_valid;
        }

        // Permisos (opcional): permitir escritura a otros procesos/usuarios
        if (::chmod(socket_path.c_str(), 0666) == -1) {
            log_errno("chmod()");
            // no es fatal
        }

        // Timeout de recepción (opcional): 100 ms
        timeval tv{};
        tv.tv_sec = 0;
        tv.tv_usec = 100000; // 100ms
        if (::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == -1) {
            log_errno("setsockopt(SO_RCVTIMEO)");
        } else {
            timeout_set = true;
        }

        logger.log("INFO", "UNIX DGRAM receiver bound at " + socket_path);
    }

    // Recibir un datagrama
    char buf[128];
    ssize_t n = ::recv(fd, buf, sizeof(buf) - 1, 0);
    if (n < 0) {
        if (timeout_set && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            // Timeout: devolvemos el último valor
            logger.log("DEBUG", "recv() timeout; returning last value");
            return last_valid;
        }
        log_errno("recv()");
        return last_valid;
    }
    if (n == 0) {
        // datagrama vacío
        logger.log("WARN", "Empty datagram received");
        return last_valid;
    }

    buf[n] = '\0';
    std::string s(buf);
    trim(s);

    // Asegurar punto decimal por si viniera coma
    std::replace(s.begin(), s.end(), ',', '.');

    try {
        // Forzar locale clásico para std::stod si el global estuviera cambiado
        std::locale::global(std::locale::classic());
        double val = std::stod(s);
        last_valid = val;
        return val;
    } catch (const std::exception& e) {
        logger.log("ERROR", std::string("Parse error on '") + s + "': " + e.what());
        return last_valid;
    }
}
