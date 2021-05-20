/*
 * PFNipsLayersKernels.h
 * 
 *
 * Distributed under terms of the MIT license.
 */

#ifndef PFNIPSLAYERSKERNELS_H
#define PFNIPSLAYERSKERNELS_H
# include <curand.h>
# include <curand_kernel.h>

// kernel for evolving c-field using finite difference to solve
// the Cahn-Hilliard equation

__global__ void calculateLapBoundaries_NIPS(float* c,float* c1, float* df, float* df1, int nx, int ny, int nz, 
								       float h, bool bX, bool bY, bool bZ);


__global__ void calculateChemPotFH_NIPS(float* c,float* c1, float* w, float* df,float* df1,float chiPP, float kap, float A, float chiPS, float chiPN, float N, int nx, int ny, int nz, int current_step, float dt);


__global__ void calculateMobility_NIPS(float* c,float* c1,float* Mob,float* Mob1, float M,float M1,float mobReSize,int nx,int ny,int nz,
float phiCutoff, float N,float gamma,float nu,float D0,float D01,float Mweight, float Mvolume,float Tcast);


__global__ void lapChemPotAndUpdateBoundaries_NIPS(float* c,float* c1,float* df,float* df1,float* Mob,float* Mob1, float M, float M1, float dt, int nx, int ny, int nz, float h, bool bX, bool bY, bool bZ);

__global__ void vitrify_NIPS(float* c,float* c1,float* Mob,float* Mob1, float phiCutoff, int nx, int ny, int nz);
 
// kernel for evolving water field using Fick's 2nd law...

__global__ void calculate_muNS_NIPS(float*w, float*c,float*c1, float* muNS, /*float* Mob,*/ float Dw, float water_CB, int nx, int ny, int nz);

__global__ void calculate_water_diffusion(int zone1,int zone2,int bathHeight,float*c,float*c1,float*Mob,float Dw,float W_N, float W_P1, float W_P2,float gammaDw,float nuDw,float Mweight,float Mvolume,int nx, int ny, int nz);

__global__ void calculateLapBoundaries_muNS_NIPS(float* df, float* muNS, int nx, int ny, int nz, float h, bool bX, bool bY, bool bZ);

__global__ void calculateNonUniformLapBoundaries_muNS_NIPS(float* muNS, float* Mob,float* nonUniformLap, int nx, int ny, int nz, float h, bool bX, bool bY, bool bZ);

__global__ void update_water_NIPS(float* w,float* df, float* Mob, float* nonUniformLap,float Dw, float dt, int nx, int ny, int nz, float h, bool bX, bool bY, bool bZ);

__global__ void init_cuRAND_NIPS(unsigned long seed, curandState* state,int nx, int ny, int nz);


__global__ void addNoise_NIPS(float* c,float* c1, int nx, int ny, int nz, float dt, int current_step,
                         float water_CB,float phiCutoff, curandState * state,float noiseStr);

__global__ void populateCopyBuffer_NIPS(float* c, float* cpyBuff, int nx, int ny, int nz);


// kernel for testing the laplacian function
// changed doubles to floats...
__global__ void testLap_NIPS(float* f, int nx, int ny, int nz, float h, 
				            bool bX, bool bY, bool bZ);


__global__ void testNonUniformMob_NIPS(float *f, float* b,int gid, int nx, int ny, int nz, float h,
								  bool bX, bool bY, bool bZ);

#endif /* !PFNIPSLAYERSKERNELS_H */
