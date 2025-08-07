#ifndef PIDCONTROLLER_H
#define PIDCONTROLLER_H

#include <string>

class PIDController {
public:
    PIDController(double kp, double ki, double kd);
    double compute(double error, double dt);
    void reset();
    double receive_error_from_socket(const std::string& socket_path);

private:
    double kp, ki, kd;
    double integral;
    double prev_error;
};

#endif // PIDCONTROLLER_H