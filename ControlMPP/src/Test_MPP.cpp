#include <gpiod.h>
#include <unistd.h>
#include <iostream>

#define CHIP_NAME "gpiochip0"
#define STEP_PIN 17  // GPIO17 físico: pin 11
#define DIR_PIN  27  // GPIO27 físico: pin 13

void pulse_step(gpiod_line* step, int delay_us) {
    gpiod_line_set_value(step, 1);
    usleep(delay_us);
    gpiod_line_set_value(step, 0);
    usleep(delay_us);
}

void rotate(gpiod_line* step, gpiod_line* dir, bool clockwise, int steps, int delay_us) {
    gpiod_line_set_value(dir, clockwise ? 1 : 0);
    std::cout << "→ Rotando en sentido " << (clockwise ? "horario" : "antihorario") << "..." << std::endl;
    for (int i = 0; i < steps; ++i) {
        pulse_step(step, delay_us);
    }
}

int main() {
    std::cout << "Iniciando prueba del motor paso a paso...\n";

    gpiod_chip* chip = gpiod_chip_open_by_name(CHIP_NAME);
    if (!chip) {
        std::cerr << "No se pudo abrir gpiochip0.\n";
        return 1;
    }

    gpiod_line* step = gpiod_chip_get_line(chip, STEP_PIN);
    gpiod_line* dir = gpiod_chip_get_line(chip, DIR_PIN);

    if (!step || !dir) {
        std::cerr << "Error al obtener líneas GPIO.\n";
        return 1;
    }

    gpiod_line_request_output(step, "stepper_test", 0);
    gpiod_line_request_output(dir, "stepper_test", 0);

    rotate(step, dir, true, 200, 1000);   // 200 pasos CW, 1ms por flanco
    sleep(1);
    rotate(step, dir, false, 200, 1000);  // 200 pasos CCW

    gpiod_line_release(step);
    gpiod_line_release(dir);
    gpiod_chip_close(chip);

    std::cout << "Prueba finalizada.\n";
    return 0;
}
