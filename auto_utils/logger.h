#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <sstream>

class Logger {
public:
    Logger(const std::string& process_name);
    void log(const std::string& level, const std::string& message);

private:
    std::string process;
    std::ofstream logfile;
    std::string current_date();
    std::string timestamp();
};
