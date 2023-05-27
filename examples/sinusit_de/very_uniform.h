#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <float.h>

using namespace std;

//Generate ranks to sample in nus acquisition in a realistic manner
//The first element is mandatory in this version
//input : iacqu_size is the size of the acquisition after NUS
//input : nnus_mult is the NUS multiplier we have to consider for nsampling
//output : nranks is the output ranks/coordinates classified in growing order.
void very_uniform_nus(int nacq_size, int nnus_mult, int* nranks){

    vector<int> vgaps_1stpart; // First half of the ranks
    vector<int> vgaps_2ndpart; // Second half to be considered as "symetric"

    vgaps_1stpart.push_back(1); // First element mandatory

    for(int i=1; i<nacq_size/2; i++){ // First part to be picked up randomly within [0,2*nnus_mult]
        vgaps_1stpart.push_back((int)(((float)rand()/((float)RAND_MAX+1.0))*(nnus_mult*2-1)+1));
    }

    for(int i=0; i<nacq_size/2; i++){ // Second part computed in a symmetric manner to have the right mean in the end
        vgaps_2ndpart.push_back(2*nnus_mult-vgaps_1stpart.at(i));
    }

    vgaps_1stpart.insert(vgaps_1stpart.end(), vgaps_2ndpart.begin(), vgaps_2ndpart.end()); // Concatenate both vectors

    nranks[0]=1; //First element
    vgaps_1stpart.erase(vgaps_1stpart.begin()); // Make sure not to count it twice

    //Mix up the ranks to have something more random
    for(int i=1; i<nacq_size; i++){
        int nindex = (int)((float)rand()/((float)RAND_MAX+1.0)*(nacq_size-i));
        nranks[i]=nranks[i-1]+vgaps_1stpart.at(nindex);
        vgaps_1stpart.erase(vgaps_1stpart.begin()+nindex);
    }

    //Clear memory
    vgaps_1stpart.clear();
    vgaps_1stpart.shrink_to_fit();
    vgaps_2ndpart.clear();
    vgaps_2ndpart.shrink_to_fit();

    return;
}

//Generate ranks to sample in grs acquisition in a realistic manner
//The first element is not mandatory in this version
//input : iacqu_size is the size of the acquisition after NUS
//input : nnus_mult is the NUS multiplier we have to consider for nsampling
//output : nranks is the output ranks/coordinates classified in growing order.
void very_uniform_grs(int nacq_size, int nnus_mult, int* nranks){

    vector<int> vgaps_1stpart; // First half of the ranks
    vector<int> vgaps_2ndpart; // Second half to be considered as "symetric"

    for(int i=0; i<nacq_size/2; i++){ // First part to be picked up randomly within [0,2*nnus_mult]
        vgaps_1stpart.push_back((int)(((float)rand()/((float)RAND_MAX+1.0))*(nnus_mult*2-1)+1));
    }

    for(int i=0; i<nacq_size/2; i++){ // Second part computed in a symmetric manner to have the right mean in the end
        vgaps_2ndpart.push_back(2*nnus_mult-vgaps_1stpart.at(i));
    }

    // Concatenate both vectors
    vgaps_1stpart.insert(vgaps_1stpart.end(), vgaps_2ndpart.begin(), vgaps_2ndpart.end());

    //Mix up the ranks to have something more random
    for(int i=0; i<nacq_size; i++){
        int nindex = (int)((float)rand()/((float)RAND_MAX+1.0)*(nacq_size-i));
        nranks[i]=(i==0 ? 0 : nranks[i-1])+vgaps_1stpart.at(nindex);
        vgaps_1stpart.erase(vgaps_1stpart.begin()+nindex);
    }

    //Clear memory
    vgaps_1stpart.clear();
    vgaps_1stpart.shrink_to_fit();
    vgaps_2ndpart.clear();
    vgaps_2ndpart.shrink_to_fit();

    return;
}
