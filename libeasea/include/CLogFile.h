#pragma once

#include <iostream>
#include <mutex>
#include <string>
#include <fstream>
#include <iomanip>

namespace easena {

    inline bool print(std::ostream& out) {
        return !!(out << std::endl);
    }

    template<typename T>
    bool print(std::ostream& out, T&& value)
    {
        return !!(out << std::forward<T>(value) << std::endl);
    }

    template<typename First, typename ... Rest>
    bool print(std::ostream& out, First&& first, Rest&& ... rest)
    {
        return !!(out << std::forward<First>(first)) && print(out, std::forward<Rest>(rest)...);
    }

    std::mutex logger_mtx;

    class log_stream {
    public:
        log_stream(std::string str, std::ostream& ifile)
            : name(str)
            , file(ifile)
        {
            std::string s{ "[" };
            name = s + name + "] ";
        }

        template <typename... Args>
        bool operator() (Args&&... args) {
            bool OK = print(file, std::forward<Args>(args)...);
            {
                std::lock_guard<std::mutex> lck(logger_mtx);
                print(std::cout, name, std::forward<Args>(args)...);
                if (!OK) {
                    print(std::cout, name, "-- Error writing to log file. --");
                }
            }
            return OK;
        }

    private:
        std::string name;
        std::ostream& file;
    };


}
