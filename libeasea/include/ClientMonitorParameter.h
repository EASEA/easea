/**
 * @file ClientMonitorParameter.h
 * @author Pallamidessi Joseph
 * @version 1.0
 *
 * @section LICENSE
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details at
 * http://www.gnu.org/copyleft/gpl.html
**/  

#ifndef CLIENTMONITORPARAMETER_H__
#define CLIENTMONITORPARAMETER_H__

#include <iostream>
#include <sstream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include "CEvolutionaryAlgorithm.h"
#include "CMonitorUtils.h"

class ClientMonitorParameter:public MonitorParameter {
  public:
    ClientMonitorParameter (CEvolutionaryAlgorithm* parent);
    virtual ~ClientMonitorParameter();
    virtual void fill();
    virtual void sending();
    virtual void reception();
    virtual size_t size();
    char* serialize();
    int serialSize();
    void deserialize(char* buf);
    void processBuffer(std::string line); 
    int getTime();

    float best;
    float worst;
    float stdev;
    float average;
    CEvolutionaryAlgorithm* source;
};


#endif /* end of include guard: CLIENTMONITORPARAMETER_H__ */
