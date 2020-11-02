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

__device__ double laplacianNonUniformMob_NIPS(double *f, double *Mob,int gid, int x, int y, int z,
                                         int nx, int ny, int nz, double h, bool bX, bool bY, bool bZ)
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
    double mobXl = Mob[xlid];
    double mobXr = Mob[xrid];
    double mobYl = Mob[ylid];
    double mobYr = Mob[yrid];
    double mobZl = Mob[zlid];
    double mobZr = Mob[zrid];
    // get values of neighbors for mu
    double xl = f[xlid];
    double xr = f[xrid];
    double yl = f[ylid];
    double yr = f[yrid];
    double zl = f[zlid];
    double zr = f[zrid];
    // get value of current points
    double bo = Mob[gid];
    double fo = f[gid];
    // begin laplacian
    double bx1 = 2.0/(1.0/mobXl + 1.0/bo);
    double bx2 = 2.0/(1.0/mobXr + 1.0/bo);
    double by1 = 2.0/(1.0/mobYl + 1.0/bo);
    double by2 = 2.0/(1.0/mobYr + 1.0/bo);
    double bz1 = 2.0/(1.0/mobZl + 1.0/bo);
    double bz2 = 2.0/(1.0/mobZr + 1.0/bo);
    double lapx = (xl*bx1 + xr*bx2 - (bx1+bx2)*fo)/(h*h); 
    double lapy = (yl*by1 + yr*by2 - (by1+by2)*fo)/(h*h);
    double lapz = (zl*bz1 + zr*bz2 - (bz1+bz2)*fo)/(h*h);
    double lapNonUniform = lapx + lapy + lapz;
    return lapNonUniform;
}   
   

/*********************************************************
   * Compute Laplacian with user specified 
   * boundary conditions (UpdateBoundaries)
   ******************************************************/
	
__device__ double laplacianUpdateBoundaries_NIPS(double* f,int gid, int x, int y, int z, 
								            int nx, int ny, int nz, double h, 
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
    double xl = f[xlid];
    double xr = f[xrid];
    double yl = f[ylid];
    double yr = f[yrid];
    double zl = f[zlid];
    double zr = f[zrid];
    double lap = (xl+xr+yl+yr+zl+zr-6.0*f[gid])/(h*h);
    return lap;
}


/*************************************************************
  * compute chi with linear weighted average
  ***********************************************************/

__device__ double chiDiffuse_NIPS(double locWater, double chiPS, double chiPN)
{
    double chi = chiPN*locWater + chiPS*(1.0-locWater);
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

/*__device__ double freeEnergyBiFH_NIPS(double cc, double chi, double N, double lap_c, double kap, double A)
{
   double c_fh = 0.0;
   if (cc < 0.0) c_fh = 0.0001;
   else if (cc > 1.0) c_fh = 0.999;
   else c_fh = cc;
   double FH = (log(c_fh) + 1.0)/N - log(1.0-c_fh) - 1.0 + chi*(1.0-2.0*c_fh) - kap*lap_c;
   if (cc <= 0.0) FH = -1.5*A*sqrt(-cc) - kap*lap_c;   
   return FH;
}*/


__device__ double freeEnergyTernaryFH_NIPS(double cc, double cc1, double chi, double chiPP, double N, double lap_c, double kap, double A)
{
    // make sure everythin is in the range of 0-1
    double cc_fh = 0.0;
    double cc1_fh = 0.0;
    double n_fh = 0.0;
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
    //double FH = (chiPP*cc1_fh - 2.0*chi*cc_fh - 2.0*chi*cc1_fh + chi) + (log(cc_fh)+1.0)/N - log(n_fh) -1;
    double FH = (log(cc_fh)+1.0)/N - log(n_fh) - 1.0 + chiPP*cc1_fh - chi*(2.0*cc_fh + 2.0*cc1_fh - 1.0); 
    
    // without the substitution (1-c-c1) for c_n
    //double FH = 0.5*chiPP*cc1 + 0.5*chi*n_fh + log(cc)/N + 1.0/N;
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
  
__device__ double d2dc2_FH_NIPS(double cc, double cc1, double N)
{
   
    double cc_fh = 0.0;
    double cc1_fh = 0.0;
    double ss_fh = 0.0;
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
    double FH_2 = 0.5 * (1.0/(N*cc_fh) + 1.0/(ss_fh));
    return FH_2;	
}

/*************************************************************
  * Compute diffusion coefficient via phillies eq.
  ***********************************************************/

__device__ double philliesDiffusion_NIPS(double cc, double gamma, double nu, 
								    double D0, double Mweight, double Mvolume)
{
	double cc_d = 1.0;
	double rho = Mweight/Mvolume;
	if (cc >= 1.0) cc_d = 1.0 * rho; // convert phi to g/L	
	else if (cc < 0.0) cc_d = 1e-6 * rho; // convert phi to g/L 
	else cc_d = cc * rho; // convert phi to g/L
	double Dp = D0 * exp(-gamma * pow(cc_d,nu));
	return Dp;
}


// -------------------------------------------------------
// Device Kernels for Testing
// -------------------------------------------------------


/****************************************************************
  * Kernels for unit testing the laplacian devices 
  ***************************************************************/

__global__ void testLap_NIPS(double* f, int nx, int ny, int nz, double h, bool bX, bool bY, bool bZ)
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

__global__ void testLapNonUniformMob_NIPS(double* f, double *Mob, int nx, int ny, int nz, double h, bool bX, bool bY, bool bZ)
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

__global__ void calculateLapBoundaries_NIPS(double* c, double* c1, double* df, double* df1, int nx, int ny, int nz, 
													double h, bool bX, bool bY, bool bZ)
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


__global__ void calculateChemPotFH_NIPS(double* c,double* c1,double* w,double* df,double*df1,double chiPP, double kap, double A, double chiPS, double chiPN, double N, int nx, int ny, int nz, int current_step, double dt)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        double cc = c[gid];
        double cc1 = c1[gid];
        double ww = w[gid];
        double lap_c = df[gid];
        double lap_c1 = df1[gid];
        // compute interaction parameter
        double chi = chiDiffuse_NIPS(ww,chiPS,chiPN);
        // compute chemical potential
        df[gid] = freeEnergyTernaryFH_NIPS(cc,cc1,chi,chiPP,N,lap_c,kap,A);
        df1[gid] = freeEnergyTernaryFH_NIPS(cc1,cc,chi,chiPP,N,lap_c1,kap,A);
    }
}


/*********************************************************
  * Computes the mobility of a concentration order
  * parameter and stores it in the Mob_d array.
  *******************************************************/
  
__global__ void calculateMobility_NIPS(double* c,double* c1,double* Mob,double* Mob1, double M,double M1,double mobReSize, int nx, int ny, int nz, double phiCutoff, double N,double gamma, double nu, double D0,double D01, double Mweight, double Mvolume, double Tcast)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        double cc = c[gid];
        double cc1 = c1[gid];
        
        /*if (cc < 0.0) cc = 1e-6;
        else if (cc > 1.0) cc = 1.0;
        if (cc1 < 0.0) cc = 1e-6;
        else if (cc1 > 1.0) cc1 = 1.0;*/
        
        double d2FH = d2dc2_FH_NIPS(cc,cc1,N);
        double d2FH_1 = d2dc2_FH_NIPS(cc1,cc,N);
        double D = philliesDiffusion_NIPS(cc,gamma,nu,D0,Mweight,Mvolume);
        double D1 = philliesDiffusion_NIPS(cc1,gamma,nu,D01,Mweight,Mvolume);
        double mobility = D/d2FH;
        double mobility1 = D1/d2FH_1;
        //double mobility = M*cc*(1.0-cc);
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


__global__ void vitrify_NIPS(double* c, double* c1, double* Mob,double* Mob1, double phiCutoff,int nx, int ny, int nz)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        double cc = c[gid];
        double cc1 = c1[gid];
        if (cc + cc1 >= phiCutoff) {Mob[gid] *= 1e-6; Mob1[gid] *= 1e-6;}
    }
}

/************************************************************************************
  * Computes the non-uniform mobility and chemical potential laplacian, multiplies 
  * it by the time step to get the RHS of the CH equation, then uses this RHS value 
  * to perform an Euler update of the concentration in time.
  ***********************************************************************************/

__global__ void lapChemPotAndUpdateBoundaries_NIPS(double* c,double* c1, double* df,double* df1, double* Mob, double*Mob1,/*double* nonUniformLap,*/ double M, double M1, double dt, int nx, int ny, int nz, double h,bool bX, bool bY, bool bZ)
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
        double nonUniformLap_c = laplacianNonUniformMob_NIPS(df,Mob,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
        double nonUniformLap_c1 = laplacianNonUniformMob_NIPS(df1,Mob1,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
        c[gid] += nonUniformLap_c*dt;
        c1[gid] += nonUniformLap_c1*dt;
        
        // compute laplacian of chemical potential and update with constant mobility
        // compute laplacian and do euler update
        //double lap_c = laplacianUpdateBoundaries_NIPS(df,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
        //double lap_c1 = laplacianUpdateBoundaries_NIPS(df1,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
        //c[gid] += 1.0*lap_c*dt;
        //c1[gid] += 1.0*lap_c1*dt;
    } 
}



__global__ void calculate_muNS_NIPS(double*w, double*c,double*c1, double* muNS, /*double* Mob,*/ double Dw, double water_CB, int nx, int ny, int nz)
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
        double ww = w[gid];
        // assign muNS for calculating laplacian
        muNS[gid] =  ww;
        
        // now assign diffusion to mobility array
        // check that polymer < 1.0 and greater than 0.0
        /*double cc = c[gid];
        if (cc < 0.0) cc = 0.0;
        else if (cc > 1.0) cc = 1.0;
        double cc1 = c1[gid];
        if (cc1 < 0.0) cc1 = 0.0;
        else if (cc1 > 1.0) cc1 = 1.0;
        double cN = 1.0 - cc - cc1;
        if (cN < 0.0) cN = 0.0;
        else if (cN > 1.0) cN = 1.0;
        double D_N = 1.0*cN - 0.5*cc - 0.5*cc1;
        // assign mobility to D_NS
        Mob[gid] = D_N;
        if (D_N < 0.0) Mob[gid] = 0.0;
        if (D_N > 1.0) Mob[gid] = 1.0;*/
    }
    
}



__global__ void calculateLapBoundaries_muNS_NIPS(double* df, double* muNS, int nx, int ny, int nz, double h, bool bX, bool bY, bool bZ)
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

// TODO is this function necessary...
// check later... first run some simulations
__global__ void calculateNonUniformLapBoundaries_muNS_NIPS(double* muNS, double* Mob,double* nonUniformLap, int nx, int ny, int nz, double h, bool bX, bool bY, bool bZ)
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

__global__ void calculate_water_diffusion(double*c,double*c1,double*Mob,double Dw,double W_S,double W_P1,double W_P2,int nx, int ny, int nz)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        double cc = c[gid];
        double cc1 = c1[gid];
        // check top layer 
        if (cc > 1.0) cc = 1.0;
        if (cc < 0.0) cc = 0.0;
        // check bottom layer
        if (cc1 > 1.0) cc1 = 1.0;
        if (cc1 < 0.0) cc1 = 0.0;
        // calculate solvent concentratio
        double cS = 1.0 - cc -cc1;
        if (cS > 1.0) cS = 1.0;
        if (cS < 0.0) cS = 0.0;
        //double checkZero = cS + cc + cc1;
        // TODO does this remove the instability...
        // what value should we use here...
        // TODO remove this if its unnecessary
        //if (checkZero <= 0.0) checkZero = 1.0;
        // calculate diffusion
        double Dweight = (W_S*cS + cc*W_P1 + cc1*W_P2)/*/(W_S + W_P1 + W_P2)*/;
        if (Dweight <= 0) Dweight = 0.001;
        // TODO why does this work.... and the others dont
        //double dw = Dw*Dweight;
        // TODO is this rational?
        // the below isn't necessary
        // see if instability improves
        //if (Dweight > Dw) Dweight = Dw;
        //if (Dweight < 0.0) Dweight = 0.1;
        Mob[gid] = Dweight;
    }
}

__global__ void update_water_NIPS(double* w,double* df, double* Mob, double* nonUniformLap,double Dw, double dt, int nx, int ny, int nz, double h, bool bX, bool bY, bool bZ)
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
        double nonUniformLapNS = nonUniformLap[gid]; //laplacianNonUniformMob_NIPS(df,Mob,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
        //double nonUniformLapNS = laplacianNonUniformMob_NIPS(df,Mob,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
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
  ***********************************************************/
__global__ void addNoise_NIPS(double *c,double* c1,int nx, int ny, int nz, double dt, int current_step, 
                         double water_CB,double phiCutoff,curandState *state,double noiseStr)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        // TODO add noiseScale variable
        int gid = nx*ny*idz + nx*idy + idx;
        double noise = curand_uniform_double(&state[gid]);
        double cc = c[gid];
        double cc1 = c1[gid];
        //double noiseScale = 1.0;
        // add random fluctuations with euler update
        if (cc + cc1 >= phiCutoff) noise = 0.5; // no fluctuations for phi < 0
        else if (cc + cc1 <= 0.0) noise = 0.5;  // no fluctuations for phi > phiCutoff
        c[gid] += noiseStr*(noise-0.5)*dt;
    }
}


/*********************************************************
  * Copies the contents of c into cpyBuffer so the c data
  * can be asynchronously transfered from the device to
  * the host.
  *******************************************************/

__global__ void populateCopyBuffer_NIPS(double* c,double* cpyBuff, int nx, int ny, int nz)
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
