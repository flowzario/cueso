# include <iostream>    // endl
# include <fstream>     // for ofstream
# include <string>      // for string
# include <sstream>     // for stringstream
# include <math.h>
# include "PFNipsLayers.h"
# include "PFNipsLayersKernels.h"
# include "../utils/cudaErrorMacros.h" // for cudaCheckErrors & cudaCheckAsyncErrors


using std::string;
using std::stringstream;
using std::cout;
using std::endl;
using std::ofstream;

// -------------------------------------------------------------------------
// Constructor:
// -------------------------------------------------------------------------

PFNipsLayers::PFNipsLayers(const GetPot& input_params)
    : rng(1234)
{

    // ---------------------------------------
    // Assign variables from 'input_params':
    // ---------------------------------------

    nx = input_params("Domain/nx",1);
    ny = input_params("Domain/ny",1);
    nz = input_params("Domain/nz",1);
    nxyz = nx*ny*nz;
    dx = input_params("Domain/dx",1.0);
    dy = input_params("Domain/dy",1.0);
    dz = input_params("Domain/dz",1.0);
    dt = input_params("Time/dt",1.0);
    bx = input_params("PFNipsLayers/bx",0);
    by = input_params("PFNipsLayers/by",1);
    bz = input_params("PFNipsLayers/bz",1);
    numSteps = input_params("Time/nstep",1);
    holdSteps = input_params("PFNipsLayers/holdSteps",0);
    co = input_params("PFNipsLayers/co",0.20);
    co1 = input_params("PFNipsLayers/co1",0.20);
    r1 = input_params("PFNipsLayers/r1",0.5);
    M = input_params("PFNipsLayers/M",1.0);
    M1 = input_params("PFNipsLayers/M1",1.0);
    mobReSize = input_params("PFNipsLayers/mobReSize",0.35);
    kap = input_params("PFNipsLayers/kap",1.0);
    water_CB = input_params("PFNipsLayers/water_CB",1.0);
    bathHeight = input_params("PFNipsLayers/bathHeight",0);
    zone1 = r1*(nx-bathHeight);
    zone2 = nx - zone1 - bathHeight;
    blendZone = input_params("PFNipsLayers/blendZone",0);
    blendWeight = input_params("PFNipsLayers/blendWeight",1.0);
    NS_in_dope = input_params("PFNipsLayers/NS_in_dope",0.0);
    mobReSize = input_params("PFNipsLayers/mobReSize",1.0);
    chiPS = input_params("PFNipsLayers/chiPS",0.034);
    chiPN = input_params("PFNipsLayers/chiPN",1.5);
    chiPP = input_params("PFNipsLayers/chiPP",1.0);
    W_S = input_params("PFNipsLayers/W_S",1.0);
    W_P1 = input_params("PFNipsLayers/W_P1",1.0);
    W_P2 = input_params("PFNipsLayers/W_P2",1.0);
    phiCutoff = input_params("PFNipsLayers/phiCutoff",0.75);
    N = input_params("PFNipsLayers/N",100.0);
    A = input_params("PFNipsLayers/A",1.0);
    Tinit = input_params("PFNipsLayers/Tinit",298.0);
    Tcast = input_params("PFNipsLayers/Tcast",298.0);
    noiseStr = input_params("PFNipsLayers/noiseStr",0.1);
    initNoise = input_params("PFNipsLayers/initNoise",0.1);
        // ----------------------------------------------------------------
        // TODO 
        // ADD noiseStr to all areas where used (initializing domain)
        // -----------------------------------------------------------------
    D0 = input_params("PFNipsLayers/D0",1.0);
    D01 = input_params("PFNipsLayers/D01",1.0);
    Dw = input_params("PFNipsLayers/Dw",10.0);
    numOutputs = input_params("Output/numOutputs",1);
    outInterval = numSteps/numOutputs;
    Mweight = input_params("PFNipsLayers/Mweight",1.0);
    Mvolume = input_params("PFNipsLayers/Mvolume",1.0);
    gamma = input_params("PFNipsLayers/gamma",1.0);
    nu = input_params("PFNipsLayers/nu",1.0);
    gammaDw = input_params("PFNipsLayers/gammaDw",1.0);
    nuDw = input_params("PFNipsLayers/nuDw",1.0);
    // ---------------------------------------
    // Set up cuda kernel launch variables:
    // ---------------------------------------

    blockSize.x = input_params("GPU/blockSize.x",0);
    blockSize.y = input_params("GPU/blockSize.y",0);
    blockSize.z = input_params("GPU/blockSize.z",0);

    // set default kernel launch parameters
    if(blockSize.x == 0) blockSize.x = 32;
    if(blockSize.y == 0) blockSize.y = 32;
    if(blockSize.z == 0) blockSize.z = 1;

    // calculate the number of blocks to be used (3-D block grid)
    int totalBlockSize = blockSize.x*blockSize.y*blockSize.z;
    blocks.x = (nx + blockSize.x - 1)/blockSize.x;
    blocks.y = (ny + blockSize.y - 1)/blockSize.y;
    blocks.z = (nz + blockSize.z - 1)/blockSize.z;

    // perform some assumption checking
    int numBlocks = blocks.x*blocks.y*blocks.z;
    int totalNumThreads = numBlocks*totalBlockSize;
    if(totalNumThreads < nxyz)
        throw "GPU Kernel Launch setup lacks sufficient threads!\n";
    if(totalBlockSize > 1024)
        throw "Total number of threads per block exceeds 1024";

}



// -------------------------------------------------------------------------
// Destructor:
// -------------------------------------------------------------------------

PFNipsLayers::~PFNipsLayers()
{

    // ----------------------------------------
    // free up device memory:
    // ----------------------------------------

    cudaFree(c_d);
    cudaFree(c1_d);
    cudaFree(df_d);
    cudaFree(df1_d);
    cudaFree(Mob_d);
    cudaFree(Mob1_d);
    cudaFree(w_d);
    cudaFree(muNS_d);
    cudaFree(nonUniformLap_d);
    cudaFree(cpyBuff_d);
    cudaFree(devState);
}



// -------------------------------------------------------------------------
// Initialize system:
// -------------------------------------------------------------------------

void PFNipsLayers::initSystem()
{
		
    // ----------------------------------------
    // Initialize concentration fields:
    // ----------------------------------------
    srand(time(NULL));      // setting the seed  
    float r = 0.0;
    int xHolder = 0;
    int zoneLoc = 0;
    float blendLoc = 0.0;
    float blendScale = 0.0;
    //int zone1 = r1*(nx-bathHeight);
    //int zone2 = nx - zone1 - bathHeight;
    if (initNoise > noiseStr*0.5*dt) initNoise = noiseStr*0.5*dt;
    for(size_t i=0;i<nxyz;i++) {
        while (xHolder < bathHeight){
            //r = (float)rand()/RAND_MAX;
            water.push_back(water_CB);
            c.push_back(0.0);
            c1.push_back(0.0);
            Mob.push_back(1.0);
            xHolder++;
        }
        xHolder = 0;
        while (xHolder < zone1) 
        {  
            if (xHolder > (zone1 - blendZone/2)){
                // blend scale is a function that is approx 1 at x=0
                // and approx 0 at x=2
                blendLoc = 2.0*zoneLoc/(blendZone);
                blendScale =  1.0 - (tanh(blendWeight*(blendLoc-1)/2)+1)/2;
                r = (float)rand()/RAND_MAX;
                c.push_back((blendScale)*(co + 0.1*(r-0.5)));
                // add small concentration to opposite layer
                // trying to take it back out to 0.0
                // minimum noise threshold
                // check for negative issues        
                c1.push_back((1.0 - blendScale)*(co + 0.1*(r-0.5)));
                water.push_back(NS_in_dope);
                Mob.push_back(1.0);
                xHolder++;
                zoneLoc++;
            }
            else 
            {
                r = (float)rand()/RAND_MAX;
                c.push_back(co + 0.1*(r-0.5));
                // add small concentration to opposite layer
                // trying to take it back out to 0.0
                // minimum noise threshold
                c1.push_back(initNoise);
                water.push_back(NS_in_dope);
                Mob.push_back(1.0);
                xHolder++;
            }
        }
        xHolder = 0;
        while (xHolder < zone2)
        {
            if (xHolder < blendZone/2) {
                blendLoc = 2.0*zoneLoc/(blendZone);
                blendScale =  1.0 - (tanh(blendWeight*(blendLoc-1)/2)+1)/2;
                r = (float)rand()/RAND_MAX;
                // should we check for negative numbers
                c.push_back((blendScale)*(co + 0.1*(r-0.5)));
                // add small concentration to opposite layer
                // trying to take it back out to 0.0
                // minimum noise threshold
                c1.push_back((1.0 - blendScale)*(co + 0.1*(r-0.5)));
                water.push_back(NS_in_dope);
                Mob.push_back(1.0);
                xHolder++;
                zoneLoc++;
            }
            else {
                r = (float)rand()/RAND_MAX;
                // add small concentration to opposite layer
                // trying to take it back out to 0.0
                // minimum noise threshold
                c.push_back(initNoise);
                c1.push_back(co1 + 0.1*(r-0.5));
                water.push_back(NS_in_dope);
                Mob.push_back(1.0);
                xHolder++;    
            }
            
        }
        zoneLoc = 0;
        xHolder = 0;
    }
    // ----------------------------------------
    // Allocate memory on device and copy data
    // and copy data from host to device
    // ----------------------------------------

    // allocate memory on device
    size = nxyz*sizeof(float);
    // allocate polymer species (top layer)
    cudaMalloc((void**) &c_d,size);
    cudaCheckErrors("cudaMalloc fail");
    // allocate space for laplacian
    cudaMalloc((void**) &df_d,size);
    cudaCheckErrors("cudaMalloc fail");
    // allocate polymer species (bottom layer)
    cudaMalloc((void**) &c1_d,size);
    cudaCheckErrors("cudaMalloc fail");
    // allocate space for laplacian
    cudaMalloc((void**) &df1_d,size);
    cudaCheckErrors("cudaMalloc fail");
    // allocate water concentration
    cudaMalloc((void**) &w_d,size);
    cudaCheckErrors("cudaMalloc fail");
    // allocate space for laplacian
    cudaMalloc((void**) &muNS_d,size);
    cudaCheckErrors("cudaMalloc fail");
    // copy buffer
    cudaMalloc((void**) &cpyBuff_d,size);
    cudaCheckErrors("cudaMalloc fail");
    // allocate mobility
    cudaMalloc((void**) &Mob_d,size);
    cudaCheckErrors("cudaMalloc fail");
    // allocate mobility
    cudaMalloc((void**) &Mob1_d,size);
    cudaCheckErrors("CudaMalloc fail");
    // allocate nonuniform laplacian for NS update
    cudaMalloc((void**) &nonUniformLap_d,size);
    cudaCheckErrors("cudaMalloc fail");
    // allocate memory for cuRAND state
    cudaMalloc((void**) &devState,nxyz*sizeof(curandState));
    cudaCheckErrors("cudaMalloc fail");
    
    // copy concentration and water array to device
    cudaMemcpy(c_d,&c[0],size,cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D fail");
    cudaMemcpy(c1_d,&c1[0],size,cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D fail");
    cudaMemcpy(w_d,&water[0],size,cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D fail");
    /*cudaMemcpy(Mob_d,&Mob[0],size,cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D fail");*/
    
    // ----------------------------------------
    // Initialize thermal fluctuations of
    // polymer concentration
    // ----------------------------------------
    
    init_cuRAND_NIPS<<<blocks,blockSize>>>(time(NULL),devState,nx,ny,nz);
    
}



// -------------------------------------------------------------------------
// Take one step forward in time:
// -------------------------------------------------------------------------

void PFNipsLayers::computeInterval(int interval)
{

    // ----------------------------------------
    //	Set the time step:
    // ----------------------------------------

    current_step = interval*outInterval;

    // ----------------------------------------
    //	Evolve system by solving CH equation:
    // ----------------------------------------

    for(size_t i=0;i<outInterval;i++)
    {
        // ------------------------
        // compute CH for c and c1
        // ------------------------
        
        // calculate the laplacian of c_d, c1_d, and store in df_d, df1_d
        calculateLapBoundaries_NIPS<<<blocks,blockSize>>>(c_d,c1_d,df_d,df1_d,nx,ny,nz,dx,bx,by,bz); 
        cudaCheckAsyncErrors("calculateLap polymer kernel fail");
        cudaDeviceSynchronize();
        // calculate the laplacian of c1_d and store in df_d
        //calculateLapBoundaries_NIPS<<<blocks,blockSize>>>(c1_d,df1_d,nx,ny,nz,dx,bx,by,bz); 
        //cudaCheckAsyncErrors("calculateLap polymer kernel fail");
        //cudaDeviceSynchronize();
        
        // calculate the chemical potential for c_d, c1_d, and store in df_d, df1_d
        calculateChemPotFH_NIPS<<<blocks,blockSize>>>(c_d,c1_d,w_d,df_d,df1_d,chiPP,kap,A,chiPS,chiPN,N,nx,ny,nz,current_step,dt);
        cudaCheckAsyncErrors("calculateChemPotFH kernel fail");
        cudaDeviceSynchronize();
        // calculate the chemical potential for c1 and store in df1_d
        //calculateChemPotFH_NIPS<<<blocks,blockSize>>>(c1_d,c_d,w_d,df1_d,chiPP,kap,A,chiPS,chiPN,N,nx,ny,nz,current_step,dt);
        //cudaCheckAsyncErrors("calculateChemPotFH kernel fail");
        //cudaDeviceSynchronize();
        
        // calculate mobility for first polymer species and store it in Mob_d
        calculateMobility_NIPS<<<blocks,blockSize>>>(c_d,c1_d,Mob_d,Mob1_d,M,M1,mobReSize,nx,ny,nz,phiCutoff,N,gamma,nu,D0,D01,Mweight,Mvolume,Tcast);
        cudaCheckAsyncErrors("calculateMobility kernel fail");
        cudaDeviceSynchronize();
        // calculate mobility for second polymer species and store it in Mob1_d                                                               
        //calculateMobility_NIPS<<<blocks,blockSize>>>(c1_d,c_d,Mob1_d,M1,mobReSize,nx,ny,nz,phiCutoff,N,gamma,nu,D0,Mweight,Mvolume,Tcast);    
        //cudaCheckAsyncErrors("calculateMobility kernel fail");                                                                               
        //cudaDeviceSynchronize();     
       
        // vitrify mobility for tob and bottom layer Mob and Mob1
        vitrify_NIPS<<<blocks,blockSize>>>(c_d,c1_d,Mob_d,Mob1_d,phiCutoff,nx,ny,nz);
        cudaCheckAsyncErrors("vitrify NIPS kernel fail");
        cudaDeviceSynchronize();
        
        // calculate the laplacian of the chemical potential, then update c_d and c1_d
        // using an Euler update
        lapChemPotAndUpdateBoundaries_NIPS<<<blocks,blockSize>>>(c_d,c1_d,df_d,df1_d,Mob_d,Mob1_d,M,M1,dt,nx,ny,nz,dx,bx,by,bz);
        cudaCheckAsyncErrors("lapChemPotAndUpdateBoundaries kernel fail");
        cudaDeviceSynchronize();
        
        
        // ------------------------------------------------
        // compute CH for c1.......is this needed??????
        // ------------------------------------------------
        
        // ---------------------------
        // compute water diffusion 
        // TODO fix non-uniform
        // ---------------------------
        if (current_step >= holdSteps)
        {
            // 1 calculate mu for Nonsolvent diffusion
            // removed water diffusivity scaling and added to 
            // calculate_water_diffusion
            // removing NS diffusion for debugging TODO
            calculate_muNS_NIPS<<<blocks,blockSize>>>(w_d,c_d,c1_d,muNS_d,/*Mob_d,*/Dw,water_CB,/*gammaDw,nuDw,Mweight,Mvolume,*/nx,ny,nz);
            cudaCheckAsyncErrors('calculate muNS kernel fail');
            cudaDeviceSynchronize();
        
            // 2 calculate laplacian for muNS - constant diffusion
            /*calculateLapBoundaries_muNS_NIPS<<<blocks,blockSize>>>(df_d,muNS_d,nx,ny,nz,dx,bx,by,bz);
            cudaCheckAsyncErrors('calculateLap water kernel fail');    
            cudaDeviceSynchronize();*/
        
            // calcualte diffusion of water based on which layer you're in
            // added zone1, zone2, and bathHeight
            calculate_water_diffusion<<<blocks,blockSize>>>(zone1,zone2,bathHeight,c_d,c1_d,Mob_d,Dw,W_S,W_P1,W_P2,gammaDw,nuDw,Mweight,Mvolume,nx,ny,nz);
            cudaCheckAsyncErrors('calculate water diffusivity fail');
            cudaDeviceSynchronize();
        
            // 3 calculate non-uniform laplacian for diffusion and concentration 
            calculateNonUniformLapBoundaries_muNS_NIPS<<<blocks,blockSize>>>(muNS_d,Mob_d,nonUniformLap_d,nx,ny,nz,dx,bx,by,bz);
            cudaCheckAsyncErrors('calculateNonUniformLap muNS kernel fail');
            cudaDeviceSynchronize();
        
            // 4 euler update water diffusing
            update_water_NIPS<<<blocks,blockSize>>>(w_d,df_d,Mob_d,nonUniformLap_d,Dw,dt,nx,ny,nz,dx,bx,by,bz);
            cudaCheckAsyncErrors("updateWater kernel fail");
            cudaDeviceSynchronize();
        }

        
        // ----------------------------
        // add thermal fluctuations
        // ----------------------------
        
        // add thermal fluctuations of polymer concentration c
        addNoise_NIPS<<<blocks,blockSize>>>(c_d, c1_d, nx, ny, nz, dt, current_step, water_CB, phiCutoff, devState,noiseStr);
        cudaCheckAsyncErrors("addNoise kernel fail");
        cudaDeviceSynchronize();
        
        // add thermal fluctuations of polymer concentration c1
        addNoise_NIPS<<<blocks,blockSize>>>(c1_d, c_d, nx, ny, nz, dt, current_step, water_CB, phiCutoff, devState,noiseStr);
        cudaCheckAsyncErrors("addNoise kernel fail");
        cudaDeviceSynchronize();
        
    }

    // ----------------------------------------
    //	Copy data back to host for writing:
    // ----------------------------------------
    
    // polymer concentration
    populateCopyBuffer_NIPS<<<blocks,blockSize>>>(c_d,cpyBuff_d,nx,ny,nz);
    cudaMemcpyAsync(&c[0],c_d,size,cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpyAsync D2H fail");
    cudaDeviceSynchronize();
    // second polymer concentration
    populateCopyBuffer_NIPS<<<blocks,blockSize>>>(c1_d,cpyBuff_d,nx,ny,nz);
    cudaMemcpyAsync(&c1[0],c1_d,size,cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpyAsync D2H fail");
    cudaDeviceSynchronize();
    // nonsolvent concentration
    populateCopyBuffer_NIPS<<<blocks,blockSize>>>(w_d,cpyBuff_d,nx,ny,nz);
    cudaMemcpyAsync(&water[0],w_d,size,cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpyAsync D2H fail");
    cudaDeviceSynchronize();
    // water diffusivity
    /*populateCopyBuffer_NIPS<<<blocks,blockSize>>>(Mob_d,cpyBuff_d,nx,ny,nz);
    cudaMemcpyAsync(&Mob[0],Mob_d,size,cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpyAsync D2H fail");
    cudaDeviceSynchronize();*/
}



// -------------------------------------------------------------------------
// Write output:
// -------------------------------------------------------------------------

void PFNipsLayers::writeOutput(int step)
{

    // -----------------------------------
    // Define the file location and name:
    // -----------------------------------

    ofstream outfile;
    ofstream outfile2;
    ofstream outfile3;
    ofstream outfile4;
    stringstream filenamecombine;
    stringstream filenamecombine2;
    stringstream filenamecombine3;
    stringstream filenamecombine4;
    float point = 0.0;
    float point2 = 0.0;
    
    filenamecombine << "vtkoutput/c_t_" << step << ".vtk";
    string filename = filenamecombine.str();
    outfile.open(filename.c_str(), std::ios::out);

    // -----------------------------------
    //	Write the 'vtk' file header:
    // -----------------------------------

    string d = "   ";
    outfile << "# vtk DataFile Version 3.1" << endl;
    outfile << "VTK file containing grid data" << endl;
    outfile << "ASCII" << endl;
    outfile << " " << endl;
    outfile << "DATASET STRUCTURED_POINTS" << endl;
    outfile << "DIMENSIONS" << d << nx << d << ny << d << nz << endl;
    outfile << "ORIGIN " << d << 0 << d << 0 << d << 0 << endl;
    outfile << "SPACING" << d << 1.0 << d << 1.0 << d << 1.0 << endl;
    outfile << " " << endl;
    outfile << "POINT_DATA " << nxyz << endl;
    outfile << "SCALARS c float" << endl;
    outfile << "LOOKUP_TABLE default" << endl;

    // -----------------------------------
    //	Write the data:
    // NOTE: x-data increases fastest,
    //       then y-data, then z-data
    // -----------------------------------

    for(size_t k=0;k<nz;k++)
        for(size_t j=0;j<ny;j++)
            for(size_t i=0;i<nx;i++)
            {
                int id = nx*ny*k + nx*j + i;
                point = c[id];
                if (abs(point) < 1e-30) point = 0.0; // making really small numbers == 0 
                outfile << point << endl;
            }

    // -----------------------------------
    //	Close the file:
    // -----------------------------------

    outfile.close();
    
    // vtkoutput for second polymer
    filenamecombine3 << "vtkoutput/c_b_" << step << ".vtk";
    string filename3 = filenamecombine3.str();
    outfile3.open(filename3.c_str(), std::ios::out);

    // -----------------------------------
    //	Write the 'vtk' file header:
    // -----------------------------------

    outfile3 << "# vtk DataFile Version 3.1" << endl;
    outfile3 << "VTK file containing grid data" << endl;
    outfile3 << "ASCII" << endl;
    outfile3 << " " << endl;
    outfile3 << "DATASET STRUCTURED_POINTS" << endl;
    outfile3 << "DIMENSIONS" << d << nx << d << ny << d << nz << endl;
    outfile3 << "ORIGIN " << d << 0 << d << 0 << d << 0 << endl;
    outfile3 << "SPACING" << d << 1.0 << d << 1.0 << d << 1.0 << endl;
    outfile3 << " " << endl;
    outfile3 << "POINT_DATA " << nxyz << endl;
    outfile3 << "SCALARS c float" << endl;
    outfile3 << "LOOKUP_TABLE default" << endl;

    // -----------------------------------
    //	Write the data:
    // NOTE: x-data increases fastest,
    //       then y-data, then z-data
    // -----------------------------------

    for(size_t k=0;k<nz;k++)
        for(size_t j=0;j<ny;j++)
            for(size_t i=0;i<nx;i++)
            {
                int id = nx*ny*k + nx*j + i;
                point = c1[id]; // or declare float here
                if (abs(point) < 1e-30) point = 0.0; // making really small numbers == 0 
                outfile3 << point << endl; // TODO convert float to float
            }

    // -----------------------------------
    //	Close the file:
    // -----------------------------------

    outfile3.close();
    // vtkoutput for water
    // -----------------------------------
    // Define the file location and name:
    // -----------------------------------


    filenamecombine2 << "vtkoutput/w_" << step << ".vtk";
    string filename2 = filenamecombine2.str();
    outfile2.open(filename2.c_str(), std::ios::out);

    // -----------------------------------
    //	Write the 'vtk' file header:
    // -----------------------------------

    outfile2 << "# vtk DataFile Version 3.1" << endl;
    outfile2 << "VTK file containing grid data" << endl;
    outfile2 << "ASCII" << endl;
    outfile2 << " " << endl;
    outfile2 << "DATASET STRUCTURED_POINTS" << endl;
    outfile2 << "DIMENSIONS" << d << nx << d << ny << d << nz << endl;
    outfile2 << "ORIGIN " << d << 0 << d << 0 << d << 0 << endl;
    outfile2 << "SPACING" << d << 1.0 << d << 1.0 << d << 1.0 << endl;
    outfile2 << " " << endl;
    outfile2 << "POINT_DATA " << nxyz << endl;
    outfile2 << "SCALARS w float" << endl;
    outfile2 << "LOOKUP_TABLE default" << endl;

    // -----------------------------------
    //	Write the data:
    // NOTE: x-data increases fastest,
    //       then y-data, then z-data
    // -----------------------------------

    for(size_t k=0;k<nz;k++)
        for(size_t j=0;j<ny;j++)
            for(size_t i=0;i<nx;i++)
            {
                int id = nx*ny*k + nx*j + i;
                point = water[id];
                // for paraview
                if (abs(point) < 1e-30) point = 0.0; // making really small numbers == 0 
                outfile2 << point << endl;
            }

    // -----------------------------------
    //	Close the file:
    // -----------------------------------

    outfile2.close();
    
    filenamecombine4 << "vtkoutput/c_comb_" << step << ".vtk";
    string filename4 = filenamecombine4.str();
    outfile4.open(filename4.c_str(), std::ios::out);

    // -----------------------------------
    //	Write the 'vtk' file header:
    // -----------------------------------

    outfile4 << "# vtk DataFile Version 3.1" << endl;
    outfile4 << "VTK file containing grid data" << endl;
    outfile4 << "ASCII" << endl;
    outfile4 << " " << endl;
    outfile4 << "DATASET STRUCTURED_POINTS" << endl;
    outfile4 << "DIMENSIONS" << d << nx << d << ny << d << nz << endl;
    outfile4 << "ORIGIN " << d << 0 << d << 0 << d << 0 << endl;
    outfile4 << "SPACING" << d << 1.0 << d << 1.0 << d << 1.0 << endl;
    outfile4 << " " << endl;
    outfile4 << "POINT_DATA " << nxyz << endl;
    outfile4 << "SCALARS c float" << endl;
    outfile4 << "LOOKUP_TABLE default" << endl;

    // -----------------------------------
    //	Write the data:
    // NOTE: x-data increases fastest,
    //       then y-data, then z-data
    // -----------------------------------

    for(size_t k=0;k<nz;k++)
        for(size_t j=0;j<ny;j++)
            for(size_t i=0;i<nx;i++)
            {
                int id = nx*ny*k + nx*j + i;
                point = c[id];
                point2 = c1[id];
                point += point2;
                // for paraview
                if (abs(point) < 1e-30) point = 0.0; // making really small numbers == 0 
                outfile4 << point << endl;
            }

    // -----------------------------------
    //	Close the file:
    // -----------------------------------

    outfile4.close();
    
    
}



// -------------------------------------------------------------------------
// Run unit tests for this App:
// -------------------------------------------------------------------------

void PFNipsLayers::runUnitTests()
{
    bool pass;
    pass = lapKernUnitTest();
    if(pass)
        cout << "\t- lapKernUnitTest -------------- PASSED\n";
    else
        cout << "\t- lapKernUnitTest -------------- FAILED\n";
}



// -------------------------------------------------------------------------
// Unit tests for this App:
// -------------------------------------------------------------------------

bool PFNipsLayers::lapKernUnitTest()
{
    // 3X3X3 scalar field with ones except the central node
    float sf[27] = {1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1};
    float solution[27] = {0,0,0,0,-1,0,0,0,0,0,-1,0,-1,6,-1,0,-1,0,0,0,0,0,-1,0,0,0,0};
    // allocate space on device
    float* sf_d;
    cudaMalloc((void**) &sf_d,27*sizeof(float));
    cudaCheckErrors("cudaMalloc fail");
    // copy sf to device
    cudaMemcpy(sf_d,sf,27*sizeof(float),cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D fail");
    // launch kernel
    dim3 grid(1,1,3);
    dim3 TpB(32,32,1);
    testLap_NIPS<<<grid,TpB>>>(sf_d,3,3,3,1.0,bx,by,bz);
    // copy data back to host
    cudaMemcpy(sf,sf_d,27*sizeof(float),cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H fail");
    // print out results
    for(size_t i=0;i<27;i++)
        /* cout << "i=" << i << " sf=" << sf[i] << " sol=" << solution[i] << endl; */
        if( sf[i] != solution[i]) 
        {
            cout << "i=" << i << " sf=" << sf[i] << " sol=" << solution[i] << endl;
            return false;
        }
    return true;
}
