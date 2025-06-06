#include "Logger.h"

Logger::Logger(const std::string& filename, bool enable)
    : logToFile(enable), enabled(enable)
{
    if (logToFile) {
        logfile.open(filename, std::ios::out | std::ios::app);
        if (!logfile.is_open()) {
            std::cerr << "Could not open log file: " << filename << std::endl;
            logToFile = false;
        }
    }
}

Logger::~Logger() {
    if (logfile.is_open()) {
        logfile.close();
    }
}

void Logger::enable(bool on) {
    enabled = on;
}

void Logger::log(const std::string& message) {
    if (!enabled) return;

    std::string timestamp = current_time();
    std::string fullMsg = "[" + timestamp + "] " + message;

    if (logToFile && logfile.is_open()) {
        logfile << fullMsg << std::endl;
    } else {
        std::cout << fullMsg << std::endl;
    }
}

std::string Logger::current_time() const {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm buf;
#ifdef _WIN32
    localtime_s(&buf, &in_time_t);
#else
    localtime_r(&in_time_t, &buf);
#endif
    std::stringstream ss;
    ss << std::put_time(&buf, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}
