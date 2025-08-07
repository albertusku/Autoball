#include "StepperMotor.h"
#include <unistd.h>
#include <iostream>

StepperMotor::StepperMotor(const std::string& chip_name, int step_pin, int dir_pin) {
    chip = gpiod_chip_open_by_name(chip_name.c_str());
    if (!chip) {
        std::cerr << "Error: No se pudo abrir el chip " << chip_name << "\n";
        exit(1);
    }

    step_line = gpiod_chip_get_line(chip, step_pin);
    dir_line  = gpiod_chip_get_line(chip, dir_pin);

    if (!step_line || !dir_line) {
        std::cerr << "Error: No se pudieron obtener las líneas GPIO\n";
        gpiod_chip_close(chip);
        exit(1);
    }

    if (gpiod_line_request_output(step_line, "stepper", 0) < 0 ||
        gpiod_line_request_output(dir_line,  "stepper", 0) < 0) {
        std::cerr << "Error: No se pudieron configurar los pines como salida\n";
        gpiod_chip_close(chip);
        exit(1);
    }
}

StepperMotor::~StepperMotor() {
    gpiod_line_release(step_line);
    gpiod_line_release(dir_line);
    gpiod_chip_close(chip);
}

void StepperMotor::step(bool clockwise) {
    gpiod_line_set_value(dir_line, clockwise ? 1 : 0);
    gpiod_line_set_value(step_line, 1);
    usleep(500);  // pulso alto
    gpiod_line_set_value(step_line, 0);
    usleep(500);  // pulso bajo
}

void StepperMotor::rotate_steps(bool clockwise, int steps, int delay_us) {
    gpiod_line_set_value(dir_line, clockwise ? 1 : 0);
    for (int i = 0; i < steps; ++i) {
        gpiod_line_set_value(step_line, 1);
        usleep(delay_us);
        gpiod_line_set_value(step_line, 0);
        usleep(delay_us);
    }
}

void StepperMotor::stop() {
    gpiod_line_set_value(step_line, 0);
}
