#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define totaldegrees 180
#define binsperdegree 4
#define threadsperblock 512

// data for the real galaxies will be read into these arrays
float *ra_real, *decl_real;
// number of real galaxies
int NoofReal;

// data for the simulated random galaxies will be read into these arrays
float *ra_sim, *decl_sim;
// number of simulated random galaxies
int NoofSim;

unsigned int *histogramDR, *histogramDD, *histogramRR;
unsigned int *d_histogram;

double *omega;



__device__ float angular_distance(float ra_1, float decl_1, float ra_2, float decl_2){
    float ra1_rad, decl1_rad, ra2_rad, decl2_rad, pi, theta;
    
    pi = acos(-1.0);
    ra1_rad = ra_1/60.0 * pi/180.0;
    decl1_rad = decl_1/60.0*pi/180.0;
    ra2_rad = ra_2/60.0 * pi/180.0;
    decl2_rad = decl_2/60.0*pi/180.0;
    
    float argument = sin(decl1_rad) * sin(decl2_rad) + cos(decl1_rad) * cos(decl2_rad) * cos(ra1_rad - ra2_rad);

    // Avoid OOR due to floating point arithmetic
    if (argument > 1.0) argument = 1.0;
    if (argument < -1.0) argument = -1.0;
    
    theta = acos(argument);
    return theta;
}

__global__ void computeHistogramDD(unsigned int *d_histDD, float *ra_real, float *decl_real, int *d_NoofReal) 
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long NoofReal = *d_NoofReal;

    if (idx < NoofReal*NoofReal) {
        long i = idx / NoofReal;
        long j = idx % NoofReal;
        float theta_rad = angular_distance(ra_real[i], decl_real[i], ra_real[j], decl_real[j]);
        float theta = theta_rad * 180.0 / acos(-1.0);
        int bin = (int) (theta * binsperdegree);
 
        atomicAdd(&d_histDD[bin], 1);
        
    }
    
}

__global__ void computeHistogramRR(unsigned int *d_histRR, float *ra_sim, float *decl_sim, int *d_NoofSim) 
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long NoofSim = *d_NoofSim;

    if (idx < NoofSim*NoofSim) {
        long i = idx / NoofSim;
        long j = idx % NoofSim;
        float theta_rad = angular_distance(ra_sim[i], decl_sim[i], ra_sim[j], decl_sim[j]);
        float theta = theta_rad * 180.0 / acos(-1.0);
        int bin = (int) (theta * binsperdegree);
        atomicAdd(&d_histRR[bin], 1);
    }
}

__global__ void computeHistogramDR(unsigned int *d_histDR, float *ra_sim, float *decl_sim, float *ra_real, float *decl_real,
                                    int *d_NoofSim, int *d_NoofReal) 
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long NoofReal = *d_NoofReal;
    long NoofSim = *d_NoofSim;

    if (idx < NoofSim*NoofReal) {
        long i = idx / NoofReal;
        long j = idx % NoofSim;
        float theta_rad = angular_distance(ra_real[i], decl_real[i], ra_sim[j], decl_sim[j]);
        float theta = theta_rad * 180.0 / acos(-1.0);
        int bin = (int) (theta * binsperdegree);
        atomicAdd(&d_histDR[bin], 1);
    }
}


int main(int argc, char *argv[])
{
   int    i;
   int    readdata(char *argv1, char *argv2);
   int    getDevice(int deviceno);
   unsigned long long histogramDRsum, histogramDDsum, histogramRRsum;
   double w;
   double start, end, kerneltime;
   struct timeval _ttime;
   struct timezone _tzone;
   cudaError_t myError;
   void compute_omega();

   FILE *outfil;

   if ( argc != 4 ) {printf("Usage: a.out real_data random_data output_data\n");return(-1);}

   if ( getDevice(0) != 0 ) return(-1);

   if ( readdata(argv[1], argv[2]) != 0 ) return(-1);

   
    histogramDR = (unsigned int *)calloc(totaldegrees*binsperdegree, sizeof(unsigned int));
    histogramDD = (unsigned int *)calloc(totaldegrees*binsperdegree, sizeof(unsigned int));
    histogramRR = (unsigned int *)calloc(totaldegrees*binsperdegree, sizeof(unsigned int));
    
    omega = (double *)calloc(totaldegrees*binsperdegree, sizeof(double));


   // allocate memory on the GPU
    unsigned int *d_histDR, *d_histDD, *d_histRR;
    float *d_ra_real, *d_decl_real, *d_ra_sim, *d_decl_sim; 
    int *d_NoofReal, *d_NoofSim;

    cudaMalloc((void **) &d_histDR, totaldegrees*binsperdegree*sizeof(unsigned int));
    cudaMalloc((void **) &d_histDD, totaldegrees*binsperdegree*sizeof(unsigned int));
    cudaMalloc((void **) &d_histRR, totaldegrees*binsperdegree*sizeof(unsigned int));

    cudaMalloc((void **) &d_ra_real, NoofReal*sizeof(float));
    cudaMalloc((void **) &d_decl_real, NoofReal*sizeof(float));
    cudaMalloc((void **) &d_ra_sim, NoofSim*sizeof(float));
    cudaMalloc((void **) &d_decl_sim, NoofSim*sizeof(float));
    cudaMalloc((void **) &d_NoofReal, sizeof(int));
    cudaMalloc((void **) &d_NoofSim, sizeof(int));


   // copy data to the GPU
    cudaMemcpy(d_ra_real, ra_real, NoofReal*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_decl_real, decl_real, NoofReal*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ra_sim, ra_sim, NoofSim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_decl_sim, decl_sim, NoofSim*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_NoofReal, &NoofReal, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_NoofSim, &NoofSim, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(d_histDR, 0, totaldegrees*binsperdegree*sizeof(unsigned int));
    cudaMemset(d_histDD, 0, totaldegrees*binsperdegree*sizeof(unsigned int));
    cudaMemset(d_histRR, 0, totaldegrees*binsperdegree*sizeof(unsigned int));

    long total_threads = (long)NoofReal * NoofReal;
    int blocks = (total_threads + threadsperblock - 1) / threadsperblock;

    kerneltime = 0.0;
    gettimeofday(&_ttime, &_tzone);
    start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

// run your kernel here

    computeHistogramDD<<<blocks, threadsperblock>>>(d_histDD, d_ra_real, d_decl_real, d_NoofReal);
    myError = cudaGetLastError();
    if ( myError != cudaSuccess ) {
        printf("CUDA error DD: %s\n", cudaGetErrorString(myError));
        return(-1);
    }
    cudaDeviceSynchronize();
    
    computeHistogramRR<<<blocks, threadsperblock>>>(d_histRR, d_ra_sim, d_decl_sim, d_NoofSim);
    myError = cudaGetLastError();
    if ( myError != cudaSuccess ) {
        printf("CUDA error RR: %s\n", cudaGetErrorString(myError));
        return(-1);
    }
    cudaDeviceSynchronize();

    computeHistogramDR<<<blocks, threadsperblock>>>(d_histDR, d_ra_sim, d_decl_sim, d_ra_real, d_decl_real, d_NoofSim, d_NoofReal);
    myError = cudaGetLastError();
    if ( myError != cudaSuccess ) {
        printf("CUDA error DR: %s\n", cudaGetErrorString(myError));
        return(-1);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(histogramDD, d_histDD, totaldegrees*binsperdegree*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(histogramRR, d_histRR, totaldegrees*binsperdegree*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(histogramDR, d_histDR, totaldegrees*binsperdegree*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    myError = cudaGetLastError();
    if ( myError != cudaSuccess ) {
        printf("CUDA error: %s\n", cudaGetErrorString(myError));
        return(-1);
    }

    for(int x=0; x<totaldegrees*binsperdegree; x++){
        histogramDRsum += (unsigned long long)histogramDR[x];
        histogramDDsum += (unsigned long long)histogramDD[x];
        histogramRRsum += (unsigned long long)histogramRR[x];
    }

    printf("   histogramDRsum = %llu\n", histogramDRsum);
    printf("   histogramDDsum = %llu\n", histogramDDsum);
    printf("   histogramRRsum = %llu\n", histogramRRsum);

    compute_omega();
    gettimeofday(&_ttime, &_tzone);
    end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
    kerneltime += end-start;
    printf("   Run time = %.lf secs\n",kerneltime);


    printf("   Writing output to %s\n",argv[3]);
    outfil = fopen(argv[3],"w");
    if ( outfil == NULL ) {printf("Cannot open output file %s\n",argv[3]);return(-1);}
    // write omega to the output file
    for (i=0; i<totaldegrees*binsperdegree; i++) fprintf(outfil,"%u %u %u %lf\n", histogramDD[i], histogramRR[i], histogramDR[i], omega[i]);

    // Free memory
    cudaFree(d_histRR);
    cudaFree(d_histDR);
    cudaFree(d_histDD);
    cudaFree(d_ra_real);
    cudaFree(d_decl_real);
    cudaFree(d_ra_sim);
    cudaFree(d_decl_sim);


    free(histogramRR);
    free(histogramDR);
    free(histogramDD);

    return(0);
}


int readdata(char *argv1, char *argv2)
{
  int i,linecount;
  char inbuf[180];
  double ra, dec, phi, theta, dpi;
  FILE *infil;
                                         
  printf("   Assuming input data is given in arc minutes!\n");
                          // spherical coordinates phi and theta in radians:
                          // phi   = ra/60.0 * dpi/180.0;
                          // theta = (90.0-dec/60.0)*dpi/180.0;

  dpi = acos(-1.0);
  infil = fopen(argv1,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv1);return(-1);}

  // read the number of galaxies in the input file
  int announcednumber;
  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv1);return(-1);}
  linecount =0;
  while ( fgets(inbuf,180,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv1, linecount);
  else 
      {
      printf("   %s does not contain %d galaxies but %d\n",argv1, announcednumber,linecount);
      return(-1);
      }

  NoofReal = linecount;
  ra_real   = (float *)calloc(NoofReal,sizeof(float));
  decl_real = (float *)calloc(NoofReal,sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i = 0;
  while ( fgets(inbuf,80,infil) != NULL )
      {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) 
         {
         printf("   Cannot read line %d in %s\n",i+1,argv1);
         fclose(infil);
         return(-1);
         }
      ra_real[i]   = (float)ra;
      decl_real[i] = (float)dec;
      ++i;
      }

  fclose(infil);

  if ( i != NoofReal ) 
      {
      printf("   Cannot read %s correctly\n",argv1);
      return(-1);
      }

  infil = fopen(argv2,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv2);return(-1);}

  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv2);return(-1);}
  linecount =0;
  while ( fgets(inbuf,80,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv2, linecount);
  else
      {
      printf("   %s does not contain %d galaxies but %d\n",argv2, announcednumber,linecount);
      return(-1);
      }

  NoofSim = linecount;
  ra_sim   = (float *)calloc(NoofSim,sizeof(float));
  decl_sim = (float *)calloc(NoofSim,sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i =0;
  while ( fgets(inbuf,80,infil) != NULL )
      {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) 
         {
         printf("   Cannot read line %d in %s\n",i+1,argv2);
         fclose(infil);
         return(-1);
         }
      ra_sim[i]   = (float)ra;
      decl_sim[i] = (float)dec;
      ++i;
      }

  fclose(infil);

  if ( i != NoofSim ) 
      {
      printf("   Cannot read %s correctly\n",argv2);
      return(-1);
      }

  return(0);
}

int getDevice(int deviceNo)
{

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("   Found %d CUDA devices\n",deviceCount);
  if ( deviceCount < 0 || deviceCount > 128 ) return(-1);
  int device;
  for (device = 0; device < deviceCount; ++device) {
       cudaDeviceProp deviceProp;
       cudaGetDeviceProperties(&deviceProp, device);
       printf("      Device %s                  device %d\n", deviceProp.name,device);
       printf("         compute capability            =        %d.%d\n", deviceProp.major, deviceProp.minor);
       printf("         totalGlobalMemory             =       %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
       printf("         l2CacheSize                   =   %8d B\n", deviceProp.l2CacheSize);
       printf("         regsPerBlock                  =   %8d\n", deviceProp.regsPerBlock);
       printf("         multiProcessorCount           =   %8d\n", deviceProp.multiProcessorCount);
       printf("         maxThreadsPerMultiprocessor   =   %8d\n", deviceProp.maxThreadsPerMultiProcessor);
       printf("         sharedMemPerBlock             =   %8d B\n", (int)deviceProp.sharedMemPerBlock);
       printf("         warpSize                      =   %8d\n", deviceProp.warpSize);
       printf("         clockRate                     =   %8.2lf MHz\n", deviceProp.clockRate/1000.0);
       printf("         maxThreadsPerBlock            =   %8d\n", deviceProp.maxThreadsPerBlock);
       printf("         asyncEngineCount              =   %8d\n", deviceProp.asyncEngineCount);
       printf("         f to lf performance ratio     =   %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
       printf("         maxGridSize                   =   %d x %d x %d\n",
                          deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
       printf("         maxThreadsDim in thread block =   %d x %d x %d\n",
                          deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
       printf("         concurrentKernels             =   ");
       if(deviceProp.concurrentKernels==1) printf("     yes\n"); else printf("    no\n");
       printf("         deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);
       if(deviceProp.deviceOverlap == 1)
       printf("            Concurrently copy memory/execute kernel\n");
       }

    cudaSetDevice(deviceNo);
    cudaGetDevice(&device);
    if ( device != deviceNo ) printf("   Unable to set device %d, using device %d instead",deviceNo, device);
    else printf("   Using CUDA device %d\n\n", device);

return(0);
}

void compute_omega(){
    int i;
    int bins = totaldegrees * binsperdegree;

    for (i=0; i<bins; i++){
        if (histogramRR[i] == 0){
            omega[i] = 0;
        }
        else { 
            omega[i] = (double)((double)histogramDD[i] - 2*(double)histogramDR[i] + (double)histogramRR[i]) / (double)histogramRR[i];
        }
    }
}