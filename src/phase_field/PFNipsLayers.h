
# ifndef PFNIPSLAYERS_H
# define PFNIPSLAYERS_H

# include <vector>
# include "../base/CuesoBase.h"
# include "../utils/rand.h"
# include "../utils/GetPot"
# include <curand.h>
# include <curand_kernel.h>


using std::vector;

class PFNipsLayers : public CuesoBase {

    private:
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // TODO possibly change all doubles to floats
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        int current_step;
        int nx,ny,nz;
        int nxyz;
        int numSteps;
        int numOutputs;
        int outInterval;
        int holdSteps;
        float co;        // polymer concentration fields
        float co1;
        float r1;
        float M;
        float M1;
        float mobReSize;
        float kap;
        float dt;
        float dx,dy,dz;
        float water_CB;
        int bathHeight;
        int zone1;
        int zone2;
        int blendZone;
        float blendWeight;
        float NS_in_dope;
        float phiCutoff;
        float chiPS;
        float chiPN;
        float chiPP;
        float W_S;
        float W_P1;
        float W_P2;
        float N;
        float A;
        float Tbath;
        float Tinit;
        float Tcast;
        float initNoise;
        float noiseStr;
        float D0;
        float D01;
        float Dw;
        float gamma;
        float nu;
        float gammaDw;
        float nuDw;
        float Mweight;
        float Mvolume;
        bool bx,by,bz;
        Rand rng;
        vector<float> c;
        vector<float> c1;
        vector<float> water;
        vector<float> Mob;
    
        // cuda members
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // TODO change arrays to floats
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        int size;
        float * c_d;        // concentration array
        float * c1_d;
        float * df_d;       // chemical potential array
        float * df1_d;
        float * w_d;        // non-solvent array
        float * muNS_d;     // laplacian array for fickian diffusion
        float * cpyBuff_d; 			// Copy buffer for ansynchronous data transfer
        float * Mob_d;     			// mobility
        float * Mob1_d;
        float * nonUniformLap_d;	    // laplacian of mobility and df
        curandState * devState;         // state for cuRAND
        float seed;             // seed for cuRAND !!!! trying float !!!! remove unsigned
        dim3 blocks;
        dim3 blockSize;
        

    public:

        PFNipsLayers(const GetPot&);
        ~PFNipsLayers();
        void initSystem();
        void computeInterval(int interval);
        void writeOutput(int step);
        void runUnitTests();

    private:
        // unit tests
        bool lapKernUnitTest();

};

# endif  // PFNIPSLAYERS_H
