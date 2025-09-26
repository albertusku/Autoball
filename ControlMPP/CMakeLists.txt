cmake_minimum_required(VERSION 3.10)
project(ControlMPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_DIR "/home/ruiz17/Autoball")

# path to code
include_directories(
    inc
    ${CMAKE_DIR}
)

add_executable(TestMPP src/Test_MPP.cpp)
add_executable(ControlMPP
    src/main.cpp
    src/PIDController.cpp
    src/StepperMotor.cpp
    ${CMAKE_DIR}/auto_utils/logger.cpp
    
)

# link libgpiod
find_package(PkgConfig REQUIRED)
pkg_check_modules(GPIOD REQUIRED IMPORTED_TARGET libgpiod)

target_link_libraries(ControlMPP PRIVATE PkgConfig::GPIOD)
target_link_libraries(TestMPP PRIVATE PkgConfig::GPIOD)

