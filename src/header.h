// Include libraries
#include <complex.h>
#include <stdlib.h>

// Define macro (your parameters)
#define dt 0.05          // time grid spacing
#define num_t_real 1000  // number of total time grids
#define nbf 2         // number of basis function
#define num_t_image 256 // number of imagary time grids (Chebyshev grids)
#define num_so 800     // number of stochastic orbitals
#define nthreads 4


typedef struct rtprop{
  int numProc;
  int myid;
  int num_t_image_proc;
  int rTime;
  int omp_thread_num;
//  double *girdImageTime;   // imagiary time grid (Chebyshev grids); size num_t_image
  double complex *stoR;   // stochastic orbitals R; size num_so*nbf*nbf
  double complex *stoRprime; // stochastic orbitals R prime; size num_so*nbf*nbf
  double complex *fockMat; // Fock matrix (real, imaginary part set to 0, read in); size nbf*nbf
  double complex *D;       // interpolation coefficient; size num_t_image_proc*num_t_image*num_t_image
  double complex *Z;       // D*Gm; size: num_t_image_proc*num_t_image*nbf*nbf
  double complex *Gm;      // imaginary time green's function (read in); size num_t_image*nbf*nbf
  double complex *UdtMat;  // Propgathion of Fock matrix; size: nbf*nbf
  double complex *VdtMat;  // F^(-1)*(1-U); size: nbf*nbf
  double complex *Gceil;  // Ceil green's function; size: num_t_image_proc*num_t_real*nbf*nbf
  double complex *sigmaceil; // Ceil self-energy; size: num_t_image_proc*nbf*nbf
  double complex *sigmar; // retarded self-energy; size: num_t_real*nbf*nbf
  double complex *integralOne; // The first integral; size num_t_image_proc*nbf*nbf
  double complex *integralTwo; // The second integral; size num_t_image_proc*nbf*nbf
}RTPROP;

void readInput(RTPROP* );
void realTimePropagate(RTPROP *);

