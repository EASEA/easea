


#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

//#define CANCEL() {raise(SIGINT);}

/* Simple logger for printing different kinds of messages to console */

#define LOG_MSG(type, ...)  Output(type, formatString(__VA_ARGS__), __func__, __FILE__, __LINE__);

#define LOG_ERROR(code, ...)                                           \
  throw Exception(code, formatString(__VA_ARGS__), __func__, __FILE__, __LINE__);

#define LOG_CHECK(condition, code, msg, ...)                    \
  if (!(condition)) {                                             \
    EASEA_ERROR(code, std::string("Failed `" #condition "`: ") + msg,  \
               #__VA_ARGS__);                                    \
  }

/* Type of message */
enum class msgType {
    DEBUG,
    INFO,
    WARNING
};
/* Type of error */
enum class errorCode {
    undefined,
    value,
    type,
    memory,
    io,
    os,
    target_specific
};

class Exception : public std::exception {

protected:
    errorCode code_;  // code of error
    std::string full_msg_; // message to be shown
    std::string msg_;      // error message
    std::string func_;     // function name
    std::string file_;     // file name
    int line_;        // number of line

public:
  Exception(errorCode code, const std::string &msg, const std::string &func, const std::string &file, int line);
  errorCode getCode() const {return code_;};
  virtual ~Exception() throw();
  virtual const char *what() const throw();
};


class Output {

protected :
    msgType type_;    // type of messgae
    std::string full_msg_; // message to be shown
    std::string msg_;      // message
    std::string func_;     // function name
    std::string file_;     // file name
    int line_;        // number of line


public :
    Output(msgType type, const std::string  &msg, const std::string &func,
            const std::string &file, int line);

    virtual ~Output();
};

template <typename T, typename... Args>
std::string formatString(const std::string &format, T first, Args... rest) {
    int size = std::snprintf(NULL, 0, format.c_str(), first, rest...);
    std::vector<char> buffer(size + 1);

    snprintf(buffer.data(), size + 1, format.c_str(), first, rest...);

    return std::string(buffer.data(), buffer.data() + size);
}

inline std::string formatString(const std::string &format) {
    for ( std::string::const_iterator itr = format.begin(); itr != format.end(); itr++) {
        if (*itr == '%') {
            if (*(itr + 1) == '%')
                itr++;
        /*    else {
                LOG_ERROR(errorCode::value, "Invalid format std::string %s",
                   format.c_str());
            }*/
        }
    }
    return format;

}

#define STOP() {LOG_MSG(msgType::INFO, "FORCE STOP from %s",  __builtin_FUNCTION()); raise(SIGINT);}
