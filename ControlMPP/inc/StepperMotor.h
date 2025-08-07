#ifndef STEPPERMOTOR_H
#define STEPPERMOTOR_H

#include <string>
#include <gpiod.h>

class StepperMotor {
public:
    StepperMotor(const std::string& chip_name, int step_pin, int dir_pin);
    ~StepperMotor();

    void step(bool clockwise);
    void rotate_steps(bool clockwise, int steps, int delay_us);
    void stop();

private:
    gpiod_chip* chip;
    gpiod_line* step_line;
    gpiod_line* dir_line;
};

#endif // STEPPERMOTOR_H
