#include "logger.h"

Logger::Logger(const std::string& process_name)
    : process(process_name) {
    std::string log_dir = "/home/ruiz17/Autoball/logs";
    std::system(("mkdir -p " + log_dir).c_str());

    std::string filename = log_dir + "/autoball_" + current_date() + ".log";
    logfile.open(filename, std::ios::app);
}

void Logger::log(const std::string& level, const std::string& message) {
    std::ostringstream full;
    full << timestamp() << " [" << process << "] " << level << ": " << message << "\n";
    logfile << full.str();
    logfile.flush(); // Opcional
    std::cout << full.str(); // También a terminal
}

std::string Logger::current_date() {
    std::time_t t = std::time(nullptr);
    char buf[11];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", std::localtime(&t));
    return std::string(buf);
}

std::string Logger::timestamp() {
    std::time_t t = std::time(nullptr);
    char buf[20];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return std::string(buf);
}
