#include "StepperMotor.h"
#include <unistd.h>
#include <iostream>
#include <auto_utils/logger.h>

static Logger logger("StepperMotor");
StepperMotor::StepperMotor(const std::string& chip_name, int step_pin, int dir_pin) {
    chip = gpiod_chip_open_by_name(chip_name.c_str());
    if (!chip) {
        logger.log("ERROR", "Failed to open GPIO chip: " + chip_name);
        exit(1);
    }

    step_line = gpiod_chip_get_line(chip, step_pin);
    dir_line  = gpiod_chip_get_line(chip, dir_pin);

    if (!step_line || !dir_line) {
        logger.log("ERROR", "Failed to get GPIO lines for step or direction pins");
        gpiod_chip_close(chip);
        exit(1);
    }

    if (gpiod_line_request_output(step_line, "stepper", 0) < 0 ||
        gpiod_line_request_output(dir_line,  "stepper", 0) < 0) {
        logger.log("ERROR", "Failed to request GPIO lines as output");
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
    usleep(500);  
    gpiod_line_set_value(step_line, 0);
    usleep(500);  
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
