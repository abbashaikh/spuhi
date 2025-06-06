#pragma once

#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <iostream>

class Logger {
public:
    Logger(const std::string& filename, bool enable = true);
    ~Logger();

    void log(const std::string& message);
    void enable(bool on);

private:
    bool logToFile;
    bool enabled;
    std::ofstream logfile;

    std::string current_time() const;
};