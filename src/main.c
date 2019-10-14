#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "header.h"


int main(int argc, char *argv[]){
  
  RTPROP *realTimeProp=NULL;
  int numProc;
  int myid;
  int omp_thread_num;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numProc);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  realTimeProp = (RTPROP*)malloc(sizeof(RTPROP));
  realTimeProp->numProc = numProc;
  realTimeProp->myid = myid;

//omp total thread number
  realTimeProp->omp_thread_num = nthreads;

  readInput(realTimeProp);
  printf("what is wrong\n");
  printf("what is wrong again\n");

  realTimePropagate(realTimeProp);
  
  //outputSigmaR(realTimeProp);

  MPI_Finalize();
  return 1;
}

