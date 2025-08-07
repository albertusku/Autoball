cmake_minimum_required(VERSION 3.10)
project(ControlMPP)

set(CMAKE_CXX_STANDARD 17)

# Ruta del código fuente
add_executable(TestMPP src/Test_MPP.cpp)

# Vincula libgpiod
find_package(PkgConfig REQUIRED)
pkg_check_modules(GPIOD REQUIRED IMPORTED_TARGET libgpiod)

target_link_libraries(TestMPP PRIVATE PkgConfig::GPIOD)
