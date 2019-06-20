


#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using std::string;
using std::snprintf;
using std::vector;

/* Simple logger for printing different kinds of messages to console */

#define LOG_MSG(type, msg, ...)  Output(type, formatString(msg, ##__VA_ARGS__), __func__, __FILE__, __LINE__);

#define LOG_ERROR(code, msg, ...)                                           \
  throw Exception(code, formatString(msg, ##__VA_ARGS__), __func__, __FILE__, __LINE__);

#define LOG_CHECK(condition, code, msg, ...)                    \
  if (!(condition)) {                                             \
    EASEA_ERROR(code, string("Failed `" #condition "`: ") + msg,  \
               ##__VA_ARGS__);                                    \
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
    string full_msg_; // message to be shown
    string msg_;      // error message
    string func_;     // function name
    string file_;     // file name
    int line_;        // number of line

public:
  Exception(errorCode code, const string &msg, const string &func, const string &file, int line);
  errorCode getCode() const {return code_;};
  virtual ~Exception() throw();
  virtual const char *what() const throw();
};


class Output {

protected :
    msgType type_;    // type of messgae
    string full_msg_; // message to be shown
    string msg_;      // message
    string func_;     // function name
    string file_;     // file name
    int line_;        // number of line


public :
    Output(msgType type, const string  &msg, const string &func,
            const string &file, int line);

    virtual ~Output();
};

template <typename T, typename... Args>
string formatString(const string &format, T first, Args... rest) {
    int size = snprintf(NULL, 0, format.c_str(), first, rest...);
    vector<char> buffer(size + 1);

    snprintf(buffer.data(), size + 1, format.c_str(), first, rest...);

    return string(buffer.data(), buffer.data() + size);
}

inline string formatString(const string &format) {
    for ( std::string::const_iterator itr = format.begin(); itr != format.end(); itr++) {
        if (*itr == '%') {
            if (*(itr + 1) == '%')
                itr++;
        /*    else {
                LOG_ERROR(errorCode::value, "Invalid format string %s",
                   format.c_str());
            }*/
        }
    }
    return format;

}

