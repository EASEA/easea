#include <CLogger.h>

#include <sstream>
#include <iostream>

using std::string;

/* Types of Error */
string getErrorMsg(errorCode code) {
    switch (code) {
#define CASE_ERROR_MSG(codeName)  \
   case errorCode::codeName:          \
   	return #codeName;
    CASE_ERROR_MSG(undefined);
    CASE_ERROR_MSG(value);
    CASE_ERROR_MSG(type);
    CASE_ERROR_MSG(memory);
    CASE_ERROR_MSG(io);
    CASE_ERROR_MSG(os);
    CASE_ERROR_MSG(target_specific);
#undef CASE_ERROR_MSG
  }
  return std::string();
}
/* Types of message */
string getMsgType(msgType type) {
    switch (type) {
#define CASE_MSG_TYPE(typeName)  \
   case msgType::typeName:          \
   	return #typeName;
    CASE_MSG_TYPE(DEBUG);
    CASE_MSG_TYPE(INFO);
    CASE_MSG_TYPE(WARNING);
#undef CASE_MSG_TYPE
  }
  return std::string();
}
    Output::Output(msgType type, const string &msg, const string &func,
            const string &file, int line) : type_(type), msg_(msg), func_(func), file_(file), line_(line) {


        std::ostringstream ss;
        if (type == msgType::INFO)
		ss << "EASEA LOG [" << getMsgType(type) << "]: "<< msg_ << std::endl;
	else{

        	ss << "EASEA LOG [" << getMsgType(type) << "]: from func " << func_
        	<< ":" << std::endl
        	<< msg_ << std::endl;
	}
        full_msg_ = ss.str();
        std::cout << full_msg_.c_str();
    }

    Output::~Output(){}


    Exception::Exception(errorCode code, const string &msg, const string &func,
            const string &file, int line) : code_(code), msg_(msg), func_(func), file_(file), line_(line) {

        std::ostringstream ss;
        ss << "EASEA LOG [ERROR]: "<< getErrorMsg(code_) << " error in func " << func_  <<  " ( "
        << file_ << "," << " line " << line_ << ") :" << std::endl
        << msg_ << std::endl;

        full_msg_ = ss.str();
    }

    Exception::~Exception() throw(){}

    const char *Exception::what() const throw(){ return full_msg_.c_str(); }

