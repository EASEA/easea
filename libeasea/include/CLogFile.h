#pragma once

#include <iostream>
#include <mutex>
#include <string>
#include <fstream>
#include <iomanip>

namespace easea {
    extern std::ofstream log_file;

    inline bool print() {

        return !!(log_file << std::endl);
    }

    template<typename T>
    bool print( T&& value)
    {

        return !!(log_file << std::forward<T>(value) << std::endl);
    }

    template<typename First, typename ... Rest>
    bool print( First&& first, Rest&& ... rest)
    {
        return !!(log_file << std::forward<First>(first)) && print( std::forward<Rest>(rest)...);
    }




    class log_stream {
    public:
	
        log_stream()
        {

        }

        template <typename... Args>
        bool operator ()(Args&&... args) {
            bool OK = print(std::forward<Args>(args)...);
            return OK;
        }
	

    private:
        std::string name = "[EASEA RAPPORT]";
    };
    


}
