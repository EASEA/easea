#ifndef CSTATS_H_
#define CSTATS_H_

class CStats {

  public:
    int totalNumberOfImmigrants;
    int currentNumberOfImmigrants;

    int totalNumberOfImmigrantReproductions;
    int currentNumberOfImmigrantReproductions;

    double currentAverageFitness;
    double currentStdDev;

  public:
    CStats();
    ~CStats();
    void resetCurrentStats();
};

#endif /* CSTATS_H_ */
