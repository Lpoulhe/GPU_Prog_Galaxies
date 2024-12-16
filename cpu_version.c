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
int    NoofReal;

// data for the simulated random galaxies will be read into these arrays
float *ra_sim, *decl_sim;
// number of simulated random galaxies
int    NoofSim;

unsigned int *histogramDR, *histogramDD, *histogramRR;
unsigned int *d_histogram;
float *omega;

int main(){
    int readdata(char *argv1, char *argv2);
    void histograms();
    double start, end, kerneltime;
    struct timeval _ttime;
    struct timezone _tzone;

    if (readdata("data_100k_arcmin.dat", "rand_100k_arcmin.dat") != 0) return -1;

    histogramDR = (unsigned int *)calloc(totaldegrees*binsperdegree,sizeof(unsigned int));
    histogramDD = (unsigned int *)calloc(totaldegrees*binsperdegree,sizeof(unsigned int));
    histogramRR = (unsigned int *)calloc(totaldegrees*binsperdegree,sizeof(unsigned int));
    omega = (float *)calloc(totaldegrees*binsperdegree,sizeof(float));

    kerneltime = 0.0;
    gettimeofday(&_ttime, &_tzone);
    start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

    histograms();
    printf("histograms done\n");
    
    int x;
    unsigned long long check = 0;
    for (x=0; x<totaldegrees * binsperdegree; x++){
        // printf("histogramDD[%d]: %d\n", x, histogramDD[x]);
        check += (unsigned long long)(histogramDD[x]);
    }
    printf("check: %llu\n", check);

    compute_omega();

    free(histogramDD);
    free(histogramDR);
    free(histogramRR);
    free(omega);


    gettimeofday(&_ttime, &_tzone);
    end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
    kerneltime += end-start;
    printf("   Run time = %.lf secs\n",kerneltime);

    return 0;
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


void  histograms()
{
  int i,j;
  float pi, theta, theta_rad;
  int   index;
  int   bins = totaldegrees * binsperdegree;
  float angular_distance(float ra_1, float decl_1, float ra_2, float decl_2);

  pi = acos(-1.0);
  for ( i=0; i<NoofReal; i++){
    for ( j=i; j<NoofReal; j++){
        theta_rad = angular_distance(ra_real[i], decl_real[i], ra_real[j], decl_real[j]);
        theta = theta_rad * 180.0/pi;
        int index = (int)(theta*binsperdegree);

        (i==j) ? (histogramDD[index]++) : (histogramDD[index]+=2);
    } 
  }   
  for ( i=0; i<NoofSim; i++){
    for ( j=i; j<NoofSim; j++){
        theta_rad = angular_distance(ra_sim[i], decl_sim[i], ra_sim[j], decl_sim[j]);
        theta = theta_rad * 180.0/pi;
        int index = (int)(theta*binsperdegree);

        (i==j) ? (histogramRR[index]++) : (histogramRR[index]+=2);
    } 
  }   
  for ( i=0; i<NoofReal; i++){
    for ( j=i; j<NoofSim; j++){
        theta_rad = angular_distance(ra_real[i], decl_real[i], ra_sim[j], decl_sim[j]);
        theta = theta_rad * 180.0/pi;
        int index = (int)(theta*binsperdegree);

        (i==j) ? (histogramDR[index]++) : (histogramDR[index]+=2);
    } 
  }
}

float angular_distance(float ra_1, float decl_1, float ra_2, float decl_2){
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

void compute_omega(){
    int i;
    int bins = totaldegrees * binsperdegree;

    for (i=0; i<bins; i++){
        omega[i] = (histogramDD[i] - 2*histogramDR[i] + histogramRR[i]) / histogramRR[i];
    }
}