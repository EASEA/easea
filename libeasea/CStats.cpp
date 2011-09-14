#include "include/CStats.h"

CStats::CStats(){
    this->totalNumberOfImmigrants=0;
    this->currentNumberOfImmigrants=0;

    this->totalNumberOfImmigrantReproductions=0;
    this->currentNumberOfImmigrantReproductions=0;

    this->currentAverageFitness=0.0;
    this->currentStdDev=0.0;
}

CStats::~CStats(){
}

void CStats::resetCurrentStats(){
    this->totalNumberOfImmigrants += this->currentNumberOfImmigrants;
    this->totalNumberOfImmigrantReproductions += this->currentNumberOfImmigrantReproductions;

    this->currentNumberOfImmigrants=0;

    this->currentNumberOfImmigrantReproductions=0;

    this->currentAverageFitness=0.0;
    this->currentStdDev=0.0;
}
