#pragma once

#include <chrono>
#include <iostream>

namespace easena {

class CProgressBar {

private:
    unsigned int counter = 0;

    const unsigned int nbTotal;
    const unsigned int width;
    const char sbComplited = '=';
    const char sbIncomplited = ' ';
    const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

public:
    CProgressBar(unsigned int total, unsigned int width, char complete, char incomplete) :
            nbTotal{total}, width{width}, sbComplited{complete}, sbIncomplited{incomplete} {}

    CProgressBar(unsigned int total, unsigned int width) : nbTotal{total}, width{width} {}

    unsigned int operator++() { return ++counter; }

    void display() const {
        float progress = (float) counter / nbTotal;
        int pos = (int) (width * progress);

        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

        std::cout << "[";

        for (int i = 0; i < width; ++i) {
            if (i < pos) std::cout << sbComplited;
            else if (i == pos) std::cout << ">";
            else std::cout << sbIncomplited;
        }
        std::cout << "] " << int(progress * 100.0) << "% "
                  << float(time_elapsed) / 1000.0 << "s\r";

        std::cout.flush();
    }
    
    void complited() const {
        display();
        std::cout << std::endl;
    }
};
}
