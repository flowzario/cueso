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

__global__ void calculateLapBoundaries_NIPS(double* c, double* df,int nx, int ny, int nz, 
								       double h, bool bX, bool bY, bool bZ);


__global__ void calculateChemPotFH_NIPS(double* c,double* c1, double* w,double* df,/*double* df1,*/ double kap, double A, double chiPS, double chiPN, double chiPP, double N, int nx, int ny, int nz, int current_step, double dt);


__global__ void calculateMobility_NIPS(double* c,double* Mob,double M,double mobReSize,int nx,int ny,int nz,
double phiCutoff, double N, double gamma, double nu, double D0, double Mweight, double Mvolume, double Tcast);


__global__ void lapChemPotAndUpdateBoundaries_NIPS(double* c,/*double* c1,*/double* df,/*double* df1,*/ double* Mob,/*double* nonUniformLap,*/ double M, double M1, double dt, int nx, int ny, int nz, double h, bool bX, bool bY, bool bZ);
 
// kernel for evolving water field using Fick's 2nd law...

__global__ void calculate_muNS_NIPS(double*w, double*c, double* muNS, double* Mob, double Dw, double water_CB,double gamma, double nu, double Mweight, double Mvolume, int nx, int ny, int nz);

__global__ void calculate_water_diffusion(double*w,double*c,double*c1,double*Mob,double Dw,double Dw1,double water_CB,int nx, int ny, int nz);

__global__ void calculateLapBoundaries_muNS_NIPS(double* df, double* muNS, int nx, int ny, int nz, double h, bool bX, bool bY, bool bZ);

/*__global__ void calculateNonUniformLapBoundaries_muNS_NIPS(double* muNS, double* Mob,double* nonUniformLap, int nx, int ny, int nz, double h, bool bX, bool bY, bool bZ);*/

__global__ void update_water_NIPS(double* w,double* df, double* Mob, /*double* nonUniformLap,*/double Dw, double dt, int nx, int ny, int nz, double h, bool bX, bool bY, bool bZ);

__global__ void init_cuRAND_NIPS(unsigned long seed, curandState* state,int nx, int ny, int nz);


__global__ void addNoise_NIPS(double* c, int nx, int ny, int nz, double dt, int current_step,
                         double water_CB,double phiCutoff, curandState * state);

__global__ void populateCopyBuffer_NIPS(double* c, double* cpyBuff, int nx, int ny, int nz);


// kernel for testing the laplacian function
__global__ void testLap_NIPS(double* f, int nx, int ny, int nz, double h, 
				            bool bX, bool bY, bool bZ);


__global__ void testNonUniformMob_NIPS(double *f, double* b,int gid, int nx, int ny, int nz, double h,
								  bool bX, bool bY, bool bZ);

#endif /* !PFNIPSLAYERSKERNELS_H */
