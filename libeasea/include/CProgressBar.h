#pragma once

#include <chrono>
#include <iostream>

namespace easena {

class CProgressBar {

private:
    unsigned int counter = 0;

    const unsigned int nbTotal;
    const unsigned int width = 50;
    const char sbComplited = '=';
    const char sbIncomplited = ' ';
    const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

public:
    CProgressBar(unsigned int total,  char complete, char incomplete) :
            nbTotal{total}, sbComplited{complete}, sbIncomplited{incomplete} {


    }

    CProgressBar(unsigned int total) : nbTotal{total} {}

    unsigned int operator++() { return ++counter; }

    void init() {
	std::cout << "0%   10   20   30   40   50   60   70   80   90   100%\n";
        std::cout << "[----|----|----|----|----|----|----|----|----|----|]\n";

    }
    void display() const {
        float progress =  static_cast<float>(counter) / static_cast<float>(nbTotal);
        int pos = static_cast<int>(static_cast<float>(width) * progress);

        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        std::cout << "[";

        for (int i = 0; i < static_cast<int>(width); ++i) {
            if (i < pos) std::cout << sbComplited;
            else if (i == pos) std::cout << ">";
            else std::cout << sbIncomplited;
        }
        std::cout << "] " << static_cast<int>(progress * 100.f) << "% "
                  << "Eval. time: " << float(time_elapsed) / 1000.f << "s\r";

        std::cout.flush();
    }
    
    void complited() const {
        display();
        std::cout << std::endl;
    }
};
}
