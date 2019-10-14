#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include "header.h"

/***************************************************/
void readInput(RTPROP *realTimeProp){
  int i,j,k;
  int iProc;
  int numProc = realTimeProp->numProc;
  int myid = realTimeProp->myid;
  int num_t_image_proc;

  FILE *fockMatFile,*DFile;
  MPI_Status status;
  double *temp,*temp2;
  double complex *temp3;

  // Calculate num_t_image_proc
  if(num_t_image%(numProc*2)!=0){
    printf("We are not ready for uneven distribution tau!\n");
    fflush(stdout);
    exit(0);
  }
  else{
    realTimeProp->num_t_image_proc = num_t_image/numProc;
    num_t_image_proc = num_t_image/numProc;
  }  


  realTimeProp->fockMat = (double complex*)calloc(nbf*nbf,sizeof(double complex));
  realTimeProp->UdtMat = (double complex*)calloc(nbf*nbf,sizeof(double complex));
  realTimeProp->VdtMat = (double complex*)calloc(nbf*nbf,sizeof(double complex));
  realTimeProp->stoR = (double complex*)calloc(num_so*nbf*nbf,sizeof(double complex));
  realTimeProp->stoRprime = (double complex*)calloc(num_so*nbf*nbf,sizeof(double complex));
  realTimeProp->Gm = (double complex*)calloc(num_t_image*nbf*nbf,sizeof(double complex));
  realTimeProp->D = (double complex*)calloc(num_t_image_proc*num_t_image*num_t_image,sizeof(double complex));
  realTimeProp->Z = (double complex*)calloc(num_t_image_proc*num_t_image*nbf*nbf,sizeof(double complex));
  realTimeProp->Gceil = (double complex*)calloc(num_t_image_proc*num_t_real*nbf*nbf,sizeof(double complex));
  realTimeProp->sigmaceil = (double complex*)calloc(num_t_image_proc*nbf*nbf,sizeof(double complex));
  realTimeProp->sigmar = (double complex*)calloc(num_t_real*nbf*nbf,sizeof(double complex));
  realTimeProp->integralOne = (double complex*)calloc(num_t_image_proc*nbf*nbf,sizeof(double complex));
  realTimeProp->integralTwo = (double complex*)calloc(num_t_image_proc*nbf*nbf,sizeof(double complex));
  
  // Read in fock matrix
  temp = (double*)calloc(nbf*nbf,sizeof(double));
  if(myid==0){
    fockMatFile = fopen("F.npy","rb");
    fread(temp,nbf*nbf*sizeof(double),1,fockMatFile);
    fclose(fockMatFile);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(temp,nbf*nbf,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  for(i=0;i<nbf*nbf;i++)realTimeProp->fockMat[i]=temp[i];
  free(temp);

  // Read in Udt
  temp3 = (double complex*)calloc(nbf*nbf,sizeof(double complex));
  if(myid==0){
    fockMatFile = fopen("Udt.npy","rb");
    fread(temp3,nbf*nbf*sizeof(double complex),1,fockMatFile);
    fclose(fockMatFile);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(temp3,nbf*nbf,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  for(i=0;i<nbf*nbf;i++)realTimeProp->UdtMat[i]=temp3[i];
  free(temp3);

  // Read in Vdt
  temp3 = (double complex*)calloc(nbf*nbf,sizeof(double complex));
  if(myid==0){
    fockMatFile = fopen("Vdt.npy","rb");
    fread(temp3,nbf*nbf*sizeof(double complex),1,fockMatFile);
    fclose(fockMatFile);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(temp3,nbf*nbf,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  for(i=0;i<nbf*nbf;i++)realTimeProp->VdtMat[i]=temp3[i];
  free(temp3);

  // Read in stoR
  temp = (double*)calloc(num_so*nbf*nbf,sizeof(double));
  if(myid==0){
    fockMatFile = fopen("R.npy","rb");
    fread(temp,num_so*nbf*nbf*sizeof(double),1,fockMatFile);
    fclose(fockMatFile);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(temp,num_so*nbf*nbf,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  for(i=0;i<num_so*nbf*nbf;i++)realTimeProp->stoR[i]=temp[i]+0*I;
  free(temp);

  // Read in stoRprime
  temp = (double*)calloc(num_so*nbf*nbf,sizeof(double));
  if(myid==0){
    fockMatFile = fopen("Rprime.npy","rb");
    fread(temp,num_so*nbf*nbf*sizeof(double),1,fockMatFile);
    fclose(fockMatFile);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(temp,num_so*nbf*nbf,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  for(i=0;i<num_so*nbf*nbf;i++)realTimeProp->stoRprime[i]=temp[i]+0*I;
  free(temp);
 
  // Read in Gm
  temp = (double*)calloc(num_t_image*nbf*nbf,sizeof(double));
  if(myid==0){
    fockMatFile = fopen("Gm.npy","rb");
    fread(temp,num_t_image*nbf*nbf*sizeof(double),1,fockMatFile);
    fclose(fockMatFile);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(temp,num_t_image*nbf*nbf,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  for(i=0;i<num_t_image*nbf*nbf;i++)realTimeProp->Gm[i]=temp[i]+0*I;
  free(temp);

  // Read in D matrix
  int num_t_total = num_t_image*num_t_image*num_t_image;
  int num_t_total_proc = num_t_image_proc*num_t_image*num_t_image;
  if(myid==0){
    temp = (double*)calloc(num_t_image*num_t_image*num_t_image,sizeof(double));
    DFile = fopen("D.npy","rb");
    fread(temp,num_t_image*num_t_image*num_t_image*sizeof(double),1,DFile);
    fclose(DFile);
  }
  temp2 = (double*)calloc(num_t_image_proc*num_t_image*num_t_image,sizeof(double));
  if(myid==0){
    memcpy(&temp2[0],&temp[0],num_t_total_proc/2*sizeof(double));
    memcpy(&temp2[num_t_total_proc/2],&temp[num_t_total-num_t_total_proc/2],num_t_total_proc/2*sizeof(double));
  }
  for(iProc=1;iProc<numProc;iProc++){
    if(myid==0){
      MPI_Send(&temp[iProc*num_t_image_proc*num_t_image*num_t_image/2],
               num_t_image_proc*num_t_image*num_t_image/2,
               MPI_DOUBLE,iProc,iProc,MPI_COMM_WORLD);
    }
    if(myid==iProc){
      MPI_Recv(&temp2[0],num_t_image_proc*num_t_image*num_t_image/2,
               MPI_DOUBLE,0,iProc,MPI_COMM_WORLD,&status);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(myid==0){
      MPI_Send(&temp[(2*numProc-iProc-1)*num_t_image_proc*num_t_image*num_t_image/2],
               num_t_image_proc*num_t_image*num_t_image/2,
               MPI_DOUBLE,iProc,iProc,MPI_COMM_WORLD);
    }
    if(myid==iProc){
      MPI_Recv(&temp2[num_t_image_proc*num_t_image*num_t_image/2],
               num_t_image_proc*num_t_image*num_t_image/2,
               MPI_DOUBLE,0,iProc,MPI_COMM_WORLD,&status);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  for(i=0;i<num_t_image_proc*num_t_image*num_t_image;i++)realTimeProp->D[i] = temp2[i];
  if(myid==0)free(temp);
  free(temp2);
}
/***************************************************/



