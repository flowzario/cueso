 /*
 * PFNipsLayersKernels.cpp
 * Copyright (C) 2020 M. Rosario Cervellere <rosario.cervellere@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "PFNipsLayersKernels.h"
#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>



// -------------------------------------------------------
// Device Functions
// -------------------------------------------------------


/**********************************************************
   * Laplacian of non-uniform mobility field 
   * for Cahn-Hilliard Euler update
   ********************************************************/

__device__ float laplacianNonUniformMob_NIPS(float *f, float *Mob,int gid, int x, int y, int z,
                                         int nx, int ny, int nz, float h, bool bX, bool bY, bool bZ)
{
	// get id of neighbors for no-flux and PBCs
   int xlid,xrid,ylid,yrid,zlid,zrid;
   // -----------------------------------
   // X-Direction Boundaries
   // -----------------------------------
	if (bX) {
		// PBCs (x-dir.)
		if(x == 0) xlid = nx*ny*z + nx*y + nx-1;
		else xlid = nx*ny*z + nx*y + x-1;
		if(x == nx-1) xrid = nx*ny*z + nx*y + 0;
		else xrid = nx*ny*z + nx*y + x+1;
	}
	else {
	 	// no-flux BC (x-dir.)
		if (x == 0) xlid = nx*ny*z + nx*y + x;
		else xlid = nx*ny*z + nx*y + x-1;
		if (x == nx-1) xrid = nx*ny*z + nx*y + x;
		else xrid = nx*ny*z + nx*y + x+1;
   }
   // -----------------------------------
   // Y-Direction Boundaries
   // -----------------------------------
	if (bY) {
		// PBC Apply
	   if(y == 0) ylid = nx*ny*z + nx*(ny-1) + x;
    	else ylid = nx*ny*z + nx*(y-1) + x;
    	if(y == ny-1) yrid = nx*ny*z + nx*0 + x;
    	else yrid = nx*ny*z + nx*(y+1) + x;
   }
   else {
   	// no-flux BC (y-dir.)
      if(y == 0) ylid = nx*ny*z + nx*y + x;
    	else ylid = nx*ny*z + nx*(y-1) + x;
    	if(y == ny-1) yrid = nx*ny*z + nx*y + x;
    	else yrid = nx*ny*z + nx*(y+1) + x;
	}
   // -----------------------------------
   // Z-Direction Boundaries
   // -----------------------------------
	if (bZ) {
		// PBC Apply (z-dir.)
   	if(z == 0) zlid = nx*ny*(nz-1) + nx*y + x;
    	else zlid = nx*ny*(z-1) + nx*y + x;
    	if(z == nz-1) zrid = nx*ny*0 + nx*y + x;
    	else zrid = nx*ny*(z+1) + nx*y + x;
   }
	else {
		// no-flux BC (z-dir.)
		if(z == 0) zlid = nx*ny*z + nx*y + x;
    	else zlid = nx*ny*(z-1) + nx*y + x;
    	if(z == nz-1) zrid = nx*ny*z + nx*y + x;
    	else zrid = nx*ny*(z+1) + nx*y + x;
	}

    // ------------------------------------------
    // begin laplacian
	// ------------------------------------------
	
    // get values of neighbors for mobility
    float mobXl = Mob[xlid];
    float mobXr = Mob[xrid];
    float mobYl = Mob[ylid];
    float mobYr = Mob[yrid];
    float mobZl = Mob[zlid];
    float mobZr = Mob[zrid];
    // get values of neighbors for mu
    float xl = f[xlid];
    float xr = f[xrid];
    float yl = f[ylid];
    float yr = f[yrid];
    float zl = f[zlid];
    float zr = f[zrid];
    // get value of current points
    float bo = Mob[gid];
    float fo = f[gid];
    // begin laplacian
    float bx1 = 2.0/(1.0/mobXl + 1.0/bo);
    float bx2 = 2.0/(1.0/mobXr + 1.0/bo);
    float by1 = 2.0/(1.0/mobYl + 1.0/bo);
    float by2 = 2.0/(1.0/mobYr + 1.0/bo);
    float bz1 = 2.0/(1.0/mobZl + 1.0/bo);
    float bz2 = 2.0/(1.0/mobZr + 1.0/bo);
    float lapx = (xl*bx1 + xr*bx2 - (bx1+bx2)*fo)/(h*h); 
    float lapy = (yl*by1 + yr*by2 - (by1+by2)*fo)/(h*h);
    float lapz = (zl*bz1 + zr*bz2 - (bz1+bz2)*fo)/(h*h);
    float lapNonUniform = lapx + lapy + lapz;
    return lapNonUniform;
}   
   

/*********************************************************
   * Compute Laplacian with user specified 
   * boundary conditions (UpdateBoundaries)
   ******************************************************/
	
__device__ float laplacianUpdateBoundaries_NIPS(float* f,int gid, int x, int y, int z, 
								            int nx, int ny, int nz, float h, 
								            bool bX, bool bY, bool bZ)
{
    // get id of neighbors with periodic boundary conditions
    // and no-flux conditions
    int xlid,xrid,ylid,yrid,zlid,zrid;
    // -----------------------------------
    // X-Direction Boundaries
    // -----------------------------------
    if (bX) {
        // PBCs (x-dir.)
        if(x == 0) xlid = nx*ny*z + nx*y + nx-1;
        else xlid = nx*ny*z + nx*y + x-1;
        if(x == nx-1) xrid = nx*ny*z + nx*y + 0;
        else xrid = nx*ny*z + nx*y + x+1;
    }
    else {
        // no-flux BC (x-dir.)
		if (x == 0) xlid = nx*ny*z + nx*y + x;
		else xlid = nx*ny*z + nx*y + x-1;
		if (x == nx-1) xrid = nx*ny*z + nx*y + x;
		else xrid = nx*ny*z + nx*y + x+1;
    }
    // -----------------------------------
    // Y-Direction Boundaries
    // -----------------------------------
	if (bY) {
        // PBC Apply
        if(y == 0) ylid = nx*ny*z + nx*(ny-1) + x;
    	else ylid = nx*ny*z + nx*(y-1) + x;
    	if(y == ny-1) yrid = nx*ny*z + nx*0 + x;
    	else yrid = nx*ny*z + nx*(y+1) + x;
    }
    else {
   	// no-flux BC (y-dir.)
        if(y == 0) ylid = nx*ny*z + nx*y + x;
    	else ylid = nx*ny*z + nx*(y-1) + x;
    	if(y == ny-1) yrid = nx*ny*z + nx*y + x;
    	else yrid = nx*ny*z + nx*(y+1) + x;
    }
    // -----------------------------------
    // Z-Direction Boundaries
    // -----------------------------------
	if (bZ) {
		// PBC Apply (z-dir.)
   	if(z == 0) zlid = nx*ny*(nz-1) + nx*y + x;
    	else zlid = nx*ny*(z-1) + nx*y + x;
    	if(z == nz-1) zrid = nx*ny*0 + nx*y + x;
    	else zrid = nx*ny*(z+1) + nx*y + x;
    }
	else {
		// no-flux BC (z-dir.)
		if(z == 0) zlid = nx*ny*z + nx*y + x;
    	else zlid = nx*ny*(z-1) + nx*y + x;
    	if(z == nz-1) zrid = nx*ny*z + nx*y + x;
    	else zrid = nx*ny*(z+1) + nx*y + x;
	}
    // get values of neighbors
    float xl = f[xlid];
    float xr = f[xrid];
    float yl = f[ylid];
    float yr = f[yrid];
    float zl = f[zlid];
    float zr = f[zrid];
    float lap = (xl+xr+yl+yr+zl+zr-6.0*f[gid])/(h*h);
    return lap;
}


/*********************************************************
   * Compute nabla with user specified 
   * boundary conditions (UpdateBoundaries)
   * TODO IS THIS NEEDED?
   ******************************************************/
	
/*__device__ float nablaUpdateBoundaries_NIPS(float* f,int gid, int x, int y, int z, 
								            int nx, int ny, int nz, float h, 
								            bool bX, bool bY, bool bZ)
{
    // get id of neighbors with periodic boundary conditions
    // and no-flux conditions
    int xlid,xrid,ylid,yrid,zlid,zrid;
    // -----------------------------------
    // X-Direction Boundaries
    // -----------------------------------
    if (bX) {
        // PBCs (x-dir.)
        if(x == 0) xlid = nx*ny*z + nx*y + nx-1;
        else xlid = nx*ny*z + nx*y + x-1;
        if(x == nx-1) xrid = nx*ny*z + nx*y + 0;
        else xrid = nx*ny*z + nx*y + x+1;
    }
    else {
        // no-flux BC (x-dir.)
		if (x == 0) xlid = nx*ny*z + nx*y + x;
		else xlid = nx*ny*z + nx*y + x-1;
		if (x == nx-1) xrid = nx*ny*z + nx*y + x;
		else xrid = nx*ny*z + nx*y + x+1;
    }
    // -----------------------------------
    // Y-Direction Boundaries
    // -----------------------------------
	if (bY) {
        // PBC Apply
        if(y == 0) ylid = nx*ny*z + nx*(ny-1) + x;
    	else ylid = nx*ny*z + nx*(y-1) + x;
    	if(y == ny-1) yrid = nx*ny*z + nx*0 + x;
    	else yrid = nx*ny*z + nx*(y+1) + x;
    }
    else {
   	// no-flux BC (y-dir.)
        if(y == 0) ylid = nx*ny*z + nx*y + x;
    	else ylid = nx*ny*z + nx*(y-1) + x;
    	if(y == ny-1) yrid = nx*ny*z + nx*y + x;
    	else yrid = nx*ny*z + nx*(y+1) + x;
    }
    // -----------------------------------
    // Z-Direction Boundaries
    // -----------------------------------
	if (bZ) {
		// PBC Apply (z-dir.)
   	if(z == 0) zlid = nx*ny*(nz-1) + nx*y + x;
    	else zlid = nx*ny*(z-1) + nx*y + x;
    	if(z == nz-1) zrid = nx*ny*0 + nx*y + x;
    	else zrid = nx*ny*(z+1) + nx*y + x;
    }
	else {
		// no-flux BC (z-dir.)
		if(z == 0) zlid = nx*ny*z + nx*y + x;
    	else zlid = nx*ny*(z-1) + nx*y + x;
    	if(z == nz-1) zrid = nx*ny*z + nx*y + x;
    	else zrid = nx*ny*(z+1) + nx*y + x;
	}
    // get values of neighbors
    float xl = f[xlid];
    float xr = f[xrid];
    float yl = f[ylid];
    float yr = f[yrid];
    float zl = f[zlid];
    float zr = f[zrid];
    float nab = (xr+yr+zr-xl-yl-zl)/(2*h);
    return nab;
}
*/


/*************************************************************
  * compute chi with linear weighted average
  ***********************************************************/

__device__ float chiDiffuse_NIPS(float locWater, float chiPS, float chiPN)
{
    float chi = chiPN*locWater + chiPS*(1.0-locWater);
	return chi;
}


/*************************************************************
	* Compute the chemical potential using the 1st derivative
	* of the  binary Flory-Huggins free energy of mixing with
	* respect to c
	*
	* F = c*log(c)/N + (1-c)*log(1-c) + chi*c*(1-c)
	*
	*
	* dF/dc = (log(c) + 1)/N - log(1 - c) - 1.0 
	*         + chi*(1 - 2*c)
	*
	***********************************************************/

/*__device__ float freeEnergyBiFH_NIPS(float cc, float chi, float N, float lap_c, float kap, float A)
{
   float c_fh = 0.0;
   if (cc < 0.0) c_fh = 0.0001;
   else if (cc > 1.0) c_fh = 0.999;
   else c_fh = cc;
   float FH = (log(c_fh) + 1.0)/N - log(1.0-c_fh) - 1.0 + chi*(1.0-2.0*c_fh) - kap*lap_c;
   if (cc <= 0.0) FH = -1.5*A*sqrt(-cc) - kap*lap_c;   
   return FH;
}*/


__device__ float freeEnergyTernaryFH_NIPS(float cc, float cc1, float chi, float chiPP, float N, float lap_c, float kap, float A)
{
    // make sure everythin is in the range of 0-1
    float cc_fh = 0.0;
    float cc1_fh = 0.0;
    float n_fh = 0.0;
    if (cc <= 0.0) cc_fh = 1e-6;
    else if (cc >= 1.0) cc_fh = 1.0;
    else cc_fh = cc;
    if (cc1 <= 0.0) cc1_fh = 0.0;
    else if (cc1 >= 1.0) cc1_fh = 1.0;
    else cc1_fh = cc1;
    n_fh = 1.0 - cc_fh - cc1_fh;
    if (n_fh <= 0.0) n_fh = 1e-6;
    else if (n_fh >= 1.0) n_fh = 1.0;
    else n_fh = 1.0 - cc_fh - cc1_fh;
    // 1st derivative from FH from Tree et. al 2019
    // with our substitution (1.0-c-c1) for c_N
    // remove the 0.5... not sure why doug tree had that in his work...
    // here we use chiP1-S = chiP2-S
    //float FH = (chiPP*cc1_fh - 2.0*chi*cc_fh - 2.0*chi*cc1_fh + chi) + (log(cc_fh)+1.0)/N - log(n_fh) -1;
    float FH = (log(cc_fh)+1.0)/N - log(n_fh) - 1.0 + chiPP*cc1_fh - chi*(2.0*cc_fh + 2.0*cc1_fh - 1.0); 
    
    // without the substitution (1-c-c1) for c_n
    //float FH = 0.5*chiPP*cc1 + 0.5*chi*n_fh + log(cc)/N + 1.0/N;
    // subtract kap*lap_c for CH
    FH -= kap*lap_c;
    // if our values go over 1 or less than 0, push back toward [0,1]
    if (cc < 0.0) FH = -1.5*A*sqrt(-cc) - kap*lap_c;  
    if (cc > 1.0) FH = 1.5*A*sqrt(cc - 1.0) - kap*lap_c;
    return FH;
}

/*************************************************************
  * Compute second derivative of FH with respect to phi
  * using the ternary FH expression
  ***********************************************************/
  
__device__ float d2dc2_FH_NIPS(float cc, float cc1, float N)
{
   
    float cc_fh = 0.0;
    float cc1_fh = 0.0;
    float ss_fh = 0.0;
    // check first concentration in bounds
    if (cc < 0.0) cc_fh = 1e-6;
    else if (cc > 1.0) cc_fh = 0.999999;
    else cc_fh = cc;
    // check second concentration in bounds
    if (cc1 < 0.0) cc1_fh = 1e-6;
    else if (cc1 > 1.0) cc1_fh = 0.999999;
    else cc1_fh = cc1;
    //check to see 1-cc_fh-cc1_fh isn't going past zero
    ss_fh = 1.0 - cc_fh - cc1_fh;
    if (ss_fh >= 1.0) ss_fh = 1.0;
    else if (ss_fh <= 0.0) ss_fh = 1e-6;
    float FH_2 = 0.5 * (1.0/(N*cc_fh) + 1.0/(ss_fh));
    return FH_2;	
}

/*************************************************************
  * Compute diffusion coefficient via phillies eq.
  ***********************************************************/

__device__ float philliesDiffusion_NIPS(float cc, float gamma, float nu, 
								    float D0, float Mweight, float Mvolume)
{
	float cc_d = 1.0;
	float rho = Mweight/Mvolume;
	if (cc >= 1.0) cc_d = 1.0 * rho; // convert phi to g/L	
	else if (cc < 0.0) cc_d = 1e-6 * rho; // convert phi to g/L 
	else cc_d = cc * rho; // convert phi to g/L
	float Dp = D0 * exp(-gamma * pow(cc_d,nu));
	return Dp;
}


// -------------------------------------------------------
// Device Kernels for Testing
// -------------------------------------------------------


/****************************************************************
  * Kernels for unit testing the laplacian devices 
  ***************************************************************/

__global__ void testLap_NIPS(float* f, int nx, int ny, int nz, float h, bool bX, bool bY, bool bZ)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        f[gid] = laplacianUpdateBoundaries_NIPS(f,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
    }
}

__global__ void testLapNonUniformMob_NIPS(float* f, float *Mob, int nx, int ny, int nz, float h, bool bX, bool bY, bool bZ)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        f[gid] = laplacianNonUniformMob_NIPS(f,Mob,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
    }
}




// -------------------------------------------------------
// Device Kernels for Simulation
// -------------------------------------------------------


/*********************************************************
  * Compute the laplacian of the concentration array c and w
  * and store it in the device array df and wdf
  *******************************************************/

__global__ void calculateLapBoundaries_NIPS(float* c, float* c1, float* df, float* df1, int nx, int ny, int nz, 
													float h, bool bX, bool bY, bool bZ)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        df[gid] = laplacianUpdateBoundaries_NIPS(c,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
        df1[gid] = laplacianUpdateBoundaries_NIPS(c1,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
    }
}




/*********************************************************
  * Computes the chemical potential of a concentration
  * order parameter and stores it in the df_d array.
  *******************************************************/


__global__ void calculateChemPotFH_NIPS(float* c,float* c1,float* w,float* df,float*df1,float chiPP, float kap, float A, float chiPS, float chiPN, float N, int nx, int ny, int nz, int current_step, float dt)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        float cc = c[gid];
        float cc1 = c1[gid];
        float ww = w[gid];
        float lap_c = df[gid];
        float lap_c1 = df1[gid];
        // compute interaction parameter
        float chi = chiDiffuse_NIPS(ww,chiPS,chiPN);
        // compute chemical potential
        df[gid] = freeEnergyTernaryFH_NIPS(cc,cc1,chi,chiPP,N,lap_c,kap,A);
        df1[gid] = freeEnergyTernaryFH_NIPS(cc1,cc,chi,chiPP,N,lap_c1,kap,A);
    }
}


/*********************************************************
  * Computes the mobility of a concentration order
  * parameter and stores it in the Mob_d array.
  *******************************************************/
  
__global__ void calculateMobility_NIPS(float* c,float* c1,float* Mob,float* Mob1, float M,float M1,float mobReSize, int nx, int ny, int nz, float phiCutoff, float N,float gamma, float nu, float D0,float D01, float Mweight, float Mvolume, float Tcast)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        float cc = c[gid];
        float cc1 = c1[gid];
        
        /*if (cc < 0.0) cc = 1e-6;
        else if (cc > 1.0) cc = 1.0;
        if (cc1 < 0.0) cc = 1e-6;
        else if (cc1 > 1.0) cc1 = 1.0;*/
        
        float d2FH = d2dc2_FH_NIPS(cc,cc1,N);
        float d2FH_1 = d2dc2_FH_NIPS(cc1,cc,N);
        float D = philliesDiffusion_NIPS(cc,gamma,nu,D0,Mweight,Mvolume);
        float D1 = philliesDiffusion_NIPS(cc1,gamma,nu,D01,Mweight,Mvolume);
        float mobility = D/d2FH;
        float mobility1 = D1/d2FH_1;
        //float mobility = M*cc*(1.0-cc);
        //if (cc < 0.0) cc = 0.0;
        //else if (cc > 1.0) cc = 1.0;
        
        if (mobility > 1.0) mobility = 1.0;     // making mobility max = 1
        else if (mobility <= 0.0) mobility = 1e-6; // mobility min = 0.001
        if (mobility1 > 1.0) mobility = 1.0;
        else if (mobility1 <= 0) mobility1 = 1e-6;
        // use constant mobility for debugging 
        // TODO - find issues with weird numbers...
        Mob[gid] = mobility;
        Mob1[gid] = mobility1;
    }
}


__global__ void vitrify_NIPS(float* c, float* c1, float* Mob,float* Mob1, float phiCutoff,int nx, int ny, int nz)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        float cc = c[gid];
        float cc1 = c1[gid];
        if (cc + cc1 >= phiCutoff) {Mob[gid] *= 1e-6; Mob1[gid] *= 1e-6;}
    }
}


/************************************************************************************
  * Computes the non-uniform mobility and chemical potential laplacian, multiplies 
  * it by the time step to get the RHS of the CH equation, then uses this RHS value 
  * to perform an Euler update of the concentration in time.
  ***********************************************************************************/

__global__ void lapChemPotAndUpdateBoundaries_NIPS(float* c,float* c1, float* df,float* df1, float* Mob, float*Mob1,/*float* nonUniformLap,*/ float M, float M1, float dt, int nx, int ny, int nz, float h,bool bX, bool bY, bool bZ)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        // compute chemical potential laplacain with non-uniform mobility
        // and user defined boundaries (no-flux or PBCs)
        // not using this anymore... causing issues
        //nonUniformLap[gid] = laplacianNonUniformMob_NIPS(df,Mob,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
        //c[gid] += nonUniformLap[gid]*dt;
        
        // calculate non-uniform laplacian without nonUniform array/field (save memory)
        // do euler update
        // commenting this out for debugging 
        // TODO
        float nonUniformLap_c = laplacianNonUniformMob_NIPS(df,Mob,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
        float nonUniformLap_c1 = laplacianNonUniformMob_NIPS(df1,Mob1,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
        c[gid] += nonUniformLap_c*dt;
        c1[gid] += nonUniformLap_c1*dt;
        
        // compute laplacian of chemical potential and update with constant mobility
        // compute laplacian and do euler update
        //float lap_c = laplacianUpdateBoundaries_NIPS(df,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
        //float lap_c1 = laplacianUpdateBoundaries_NIPS(df1,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
        //c[gid] += 1.0*lap_c*dt;
        //c1[gid] += 1.0*lap_c1*dt;
    } 
}



__global__ void calculate_muNS_NIPS(float*w, float*c,float*c1, float* muNS, /*float* Mob,*/ float Dw, float water_CB, int nx, int ny, int nz)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        
        // calculate mu for NonSolvent NS diffusion
        // make x = 0 coagulation bath composition
        if (idx == 0) w[gid] = water_CB;
        float ww = w[gid];
        // assign muNS for calculating laplacian
        muNS[gid] =  ww;
        
        // now assign diffusion to mobility array
        // check that polymer < 1.0 and greater than 0.0
        /*float cc = c[gid];
        if (cc < 0.0) cc = 0.0;
        else if (cc > 1.0) cc = 1.0;
        float cc1 = c1[gid];
        if (cc1 < 0.0) cc1 = 0.0;
        else if (cc1 > 1.0) cc1 = 1.0;
        float cN = 1.0 - cc - cc1;
        if (cN < 0.0) cN = 0.0;
        else if (cN > 1.0) cN = 1.0;
        float D_N = 1.0*cN - 0.5*cc - 0.5*cc1;
        // assign mobility to D_NS
        Mob[gid] = D_N;
        if (D_N < 0.0) Mob[gid] = 0.0;
        if (D_N > 1.0) Mob[gid] = 1.0;*/
    }
    
}



/********************************************
* Use this for constant NS diffusion
*
/********************************************/
__global__ void calculateLapBoundaries_muNS_NIPS(float* df, float* muNS, int nx, int ny, int nz, float h, bool bX, bool bY, bool bZ)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        df[gid] = laplacianUpdateBoundaries_NIPS(muNS,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
    }
}



__global__ void calculateNonUniformLapBoundaries_muNS_NIPS(float* muNS, float* Mob,float* nonUniformLap, int nx, int ny, int nz, float h, bool bX, bool bY, bool bZ)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        nonUniformLap[gid] = laplacianNonUniformMob_NIPS(muNS,Mob,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
    }
}


// TODO add depth checks for water layer, 1st layer, and second layer
__global__ void calculate_water_diffusion(int zone1, int zone2, int bathHeight,float*c,float*c1,float*Mob,float Dw,float W_S,float W_P1,float W_P2,float gammaDw,float nuDw,float Mweight,float Mvolume,int nx, int ny, int nz)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        // currently don't need polymer concentration
        
        int gid = nx*ny*idz + nx*idy + idx;
    
        float cc = c[gid];
        float cc1 = c1[gid];
        // check top layer 
        // TODO these may not be necessary...
        // float check if they still are
        if (cc > 1.0) cc = 0.99999;
        if (cc <= 0.0) cc = 1e-5;
        // check bottom layer
        if (cc1 > 1.0) cc1 = 0.99999;
        if (cc1 <= 0.0) cc1 = 1e-5;
        
        float Dw = 1.0;
        if (idx < bathHeight) Dw = W_S;
        else if (idx >= bathHeight && idx < (bathHeight+zone1)) Dw = (1.0-cc)*W_P1;
        else if (idx >= (bathHeight+zone1) && idx < (bathHeight+zone1+zone2)) Dw = (1.0-cc1)*W_P2;

        
        // ------------------------------
        // phillies method (scrapped)
        // here we're getting incorrect results due to the additive nature
        // scrap phillies method
        // ------------------------------
        //float Dphil_c = philliesDiffusion_NIPS(cc, gammaDw, nuDw, W_P1, Mweight, Mvolume);
        //float Dphil_c2 = philliesDiffusion_NIPS(cc1,gammaDw, nuDw, W_P2, Mweight, Mvolume);
        //if (Dphil_c <= 0.0) Dphil = 0.001;
        //float D_combined = Dphil_c + Dphil_c2;
        Mob[gid] = Dw;
    }
}

__global__ void update_water_NIPS(float* w,float* df, float* Mob, float* nonUniformLap,float Dw, float dt, int nx, int ny, int nz, float h, bool bX, bool bY, bool bZ)
{
    // here we're re-using the Mob array for Dw_nonUniform
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        
        // adding back in nonUniformLaplacian
        // TODO do we need the nonUniformLaplacian Kernel? aparantly so...
        float nonUniformLapNS = nonUniformLap[gid]; //laplacianNonUniformMob_NIPS(df,Mob,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
        //float nonUniformLapNS = laplacianNonUniformMob_NIPS(df,Mob,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
        w[gid] += nonUniformLapNS*dt;
        //w[gid] += 20*df[gid]*dt;
    }
}


/**********************************************************************
  * initialize cuRAND for thermal fluctuations of polymerconcentration
  *********************************************************************/
__global__ void init_cuRAND_NIPS(unsigned long seed,curandState *state,int nx,int ny,int nz)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        curand_init(seed,gid,0,&state[gid]);
    }
}


/************************************************************
  * Add random fluctuations for non-trivial solution (cuRand)
  * TODO - fix droplet issues
  ***********************************************************/
__global__ void addNoise_NIPS(float *c,float* c1,int nx, int ny, int nz, float dt, int current_step, 
                         float water_CB,float phiCutoff,curandState *state,float noiseStr)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        // TODO - fix droplet issues
        int gid = nx*ny*idz + nx*idy + idx;
        float noise = curand_uniform(&state[gid]);
        float cc = c[gid];
        float cc1 = c1[gid];
        // add random fluctuations with euler update
        
        // define noise limit 
        float noiseLimit = noiseStr*0.5*dt;
        float scaledNoise = noiseStr*(noise-0.5)*dt;
        // no fluctuations for phi >= cutoff
        if (cc + cc1 >= phiCutoff) scaledNoise = 0.0; 
        // no fluctuations for phi <= 0
        // TODO - only look at single poly concentration
        else if (cc /*+ cc1*/ <= noiseLimit) scaledNoise = 0.0;  
        // if noise makes negative concetration -> add concentration and don't subtract
        //else if (cc <= 0.0 && scaledNoise < 0.0) scaledNoise = 0;
        c[gid] += scaledNoise;
    }
}


/*********************************************************
  * Copies the contents of c into cpyBuffer so the c data
  * can be asynchronously transfered from the device to
  * the host.
  *******************************************************/

__global__ void populateCopyBuffer_NIPS(float* c,float* cpyBuff, int nx, int ny, int nz)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        // copy the contents of c to cpyBuff
        cpyBuff[gid] = c[gid];
    }
}
