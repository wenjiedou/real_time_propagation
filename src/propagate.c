#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <mpi.h>
#include <mkl.h>
#include <omp.h>
#include <string.h>
#include "header.h"

void calcZ(RTPROP *);
void initGceil(RTPROP *);
void calcIntegralOne(RTPROP *);
void calcIntegralTwo(RTPROP *);
void updateG(RTPROP *);
void updateSigma(RTPROP *);
void communicateSigma(RTPROP *,double complex *);
int omp_get_thread_num();

void  realTimePropagate(RTPROP *realTimeProp){
  int j;
  int rTime;
  FILE *file;
  

  // Generate Z = D*Gm
  calcZ(realTimeProp);
  
  // 0 time step 
  rTime = 0;
  realTimeProp->rTime = rTime;
  initGceil(realTimeProp);
  updateSigma(realTimeProp);

  for(rTime=1;rTime<num_t_real;rTime++){
    realTimeProp->rTime = rTime;

    calcIntegralOne(realTimeProp);
    calcIntegralTwo(realTimeProp);
    updateG(realTimeProp);
    updateSigma(realTimeProp);
  }

//  for(j=0;j<nbf*nbf;j++) printf("sigmar %f %f \n", creal(realTimeProp->sigmar[j]),cimag(realTimeProp->sigmar[j]));
//  for(j=0;j<nbf*nbf;j++) printf("sigmar %f %f \n", creal(realTimeProp->sigmar[nbf*nbf+j]),cimag(realTimeProp->sigmar[nbf*nbf+j]));
//  for(j=0;j<nbf*nbf;j++) printf("sigmar %f %f \n", creal(realTimeProp->sigmar[2*nbf*nbf+j]),cimag(realTimeProp->sigmar[2*nbf*nbf+j]));
  for(j=0;j<nbf*nbf;j++) printf("sigmar %f %f \n", creal(realTimeProp->sigmar[10*nbf*nbf+j]),cimag(realTimeProp->sigmar[10*nbf*nbf+j]));
  printf("I am here! haha\n");

  if(realTimeProp->myid==0){
     file = fopen("sigma.npy","wb");
     fwrite(realTimeProp->sigmar,2*num_t_real*nbf*nbf*sizeof(double),1,file);
     fclose(file);
   }


}

void calcZ(RTPROP *realTimeProp){
  int i;
  int num_t_image_proc = realTimeProp->num_t_image_proc;
  double complex alpha = 1.0;
  double complex beta = 0.0;
  double complex *Z = realTimeProp->Z; 
  double complex *Gm = realTimeProp->Gm;
  double complex *D = realTimeProp->D;
  
  cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
              num_t_image_proc*num_t_image,nbf*nbf,num_t_image,&alpha,D,
              num_t_image,Gm,nbf*nbf,&beta,
              Z,nbf*nbf);

}

void initGceil(RTPROP *realTimeProp){
  int iProc,i;
  int num_t_image_proc = realTimeProp->num_t_image_proc;
  int numProc = realTimeProp->numProc;
  int myid = realTimeProp->myid;
  double complex alpha = -1.0*I; 
  double complex *Gm = realTimeProp->Gm;
  double complex *Gceil = realTimeProp->Gceil;
  double complex *flipGm = (double complex*)calloc(num_t_image*nbf*nbf,sizeof(double complex));

  for(i=0;i<num_t_image;i++){
           memcpy(&flipGm[i*nbf*nbf],&Gm[(num_t_image-i-1)*nbf*nbf],
                  nbf*nbf*sizeof(double complex));                
  }

  cblas_zscal(num_t_image*nbf*nbf,&alpha,flipGm,1);
  
  for(iProc=0;iProc<numProc;iProc++){
      if(myid==iProc){
           for(i=0;i<num_t_image_proc/2;i++){
               memcpy(&Gceil[i*num_t_real*nbf*nbf],
                      &flipGm[iProc*num_t_image_proc*nbf*nbf+i*nbf*nbf],
                      nbf*nbf*sizeof(double complex));                

               memcpy(&Gceil[(num_t_image_proc/2+i)*num_t_real*nbf*nbf],
                      &flipGm[(2*numProc-iProc-1)*num_t_image_proc*nbf*nbf/2+i*nbf*nbf],
                      nbf*nbf*sizeof(double complex));                
               }
          }
   }

   free(flipGm);
}

void calcIntegralOne(RTPROP *realTimeProp){
  int rTime = realTimeProp->rTime;
  int omp_thread_num = realTimeProp->omp_thread_num;
  int iThread;
  int num_t_image_proc = realTimeProp->num_t_image_proc;
  int i,j,k;
  double complex alpha_dt = dt;
  double complex alpha = 1.0;
  double complex beta = 0.0;
  double complex *sigmar = realTimeProp->sigmar;
  double complex *Gceil = realTimeProp->Gceil;
  double complex *integralOne = realTimeProp->integralOne;
  double complex *SigGceil_threads = (double complex*)calloc(omp_thread_num*nbf*nbf,sizeof(double complex));
  double complex *SigGceil = (double complex*)calloc(nbf*nbf,sizeof(double complex));

//  omp_set_num_threads(omp_thread_num);
  omp_set_num_threads(omp_thread_num);

  for(i=0;i<num_t_image_proc;i++){
      cblas_zscal(omp_thread_num*nbf*nbf,&beta,SigGceil_threads,1);
       #pragma omp parallel private(j,iThread)
       {
         iThread = omp_get_thread_num();
         #pragma omp for
         for(j=0;j<rTime;j++){
         // SigGceil = sigmar * gceil
           cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,nbf,nbf,nbf,&alpha_dt,&sigmar[j*nbf*nbf],
                       nbf,&Gceil[i*num_t_real*nbf*nbf+(rTime-j-1)*nbf*nbf],nbf,&alpha,&SigGceil_threads[iThread*nbf*nbf],nbf);
          }
        } //end omp

      // omp reduction
      cblas_zscal(nbf*nbf,&beta,SigGceil,1);
      for(iThread=0;iThread<omp_thread_num;iThread++){
           cblas_zaxpy(nbf*nbf,&alpha,&SigGceil_threads[iThread*nbf*nbf],1,SigGceil,1);
          }

      memcpy(&integralOne[i*nbf*nbf],SigGceil,nbf*nbf*sizeof(double complex));                
      } // end for i
      
  free(SigGceil_threads);
  free(SigGceil);
}


void calcIntegralTwo(RTPROP *realTimeProp){
  int i,j,k;
  int num_t_image_proc = realTimeProp->num_t_image_proc;
  int omp_thread_num = realTimeProp->omp_thread_num;
  int iThread;
  int iTau;
  double complex alpha = 1.0;
  double complex beta = 0.0;
  double complex *integralTwo = realTimeProp->integralTwo;
  double complex *Z = realTimeProp->Z;
  double complex *sigmaceilTotal = (double complex*)calloc(num_t_image*nbf*nbf,sizeof(double complex));
  double complex *Zsigma_threads = (double complex*)calloc(omp_thread_num*nbf*nbf,sizeof(double complex));
  double complex *Zsigma = (double complex*)calloc(nbf*nbf,sizeof(double complex));

  // get sigmaceil for all threads
  communicateSigma(realTimeProp,sigmaceilTotal);

  omp_set_num_threads(omp_thread_num);

  // Integral 2
  for(i=0;i<num_t_image_proc;i++){

       cblas_zscal(omp_thread_num*nbf*nbf,&beta,Zsigma_threads,1);
       #pragma omp parallel private(iTau,iThread)
       {
         iThread = omp_get_thread_num();
         #pragma omp for
         for(iTau=0;iTau<num_t_image;iTau++){
         // Zsigma =  SigmaceilTotal*Z 
           cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,nbf,nbf,nbf,&alpha,&sigmaceilTotal[iTau*nbf*nbf],
                       nbf,&Z[i*num_t_image*nbf*nbf+iTau*nbf*nbf],nbf,&alpha,&Zsigma_threads[iThread*nbf*nbf],nbf);
          }
        } //end omp

      // omp reduction
      cblas_zscal(nbf*nbf,&beta,Zsigma,1);
      for(iThread=0;iThread<omp_thread_num;iThread++){
           cblas_zaxpy(nbf*nbf,&alpha,&Zsigma_threads[iThread*nbf*nbf],1,Zsigma,1);
          }

      memcpy(&integralTwo[i*nbf*nbf],Zsigma,nbf*nbf*sizeof(double complex));                

    } // end for i


    free(sigmaceilTotal);
    free(Zsigma_threads);
    free(Zsigma);
}

void updateG(RTPROP *realTimeProp){
  int i,j,k;
  double complex alpha = 1.0;
  double complex alpha_min = -1.0;
  double complex beta = 0.0;
  int num_t_image_proc = realTimeProp->num_t_image_proc;
  int rTime = realTimeProp->rTime;
  double complex *Gceil = realTimeProp->Gceil;
  double complex *integralOne = realTimeProp->integralOne;
  double complex *integralTwo = realTimeProp->integralTwo;
  double complex *UdtMat = realTimeProp->UdtMat;
  double complex *VdtMat = realTimeProp->VdtMat;
  double complex *integral = (double complex*)calloc(nbf*nbf,sizeof(double complex));

   for(i=0;i<num_t_image_proc;i++){
       // Gceil_dt = U*Gceil - V*(integralOne + integralTwo) 
       // integral = integralOne + integralTwo 
       cblas_zscal(nbf*nbf,&beta,integral,1);
       cblas_zaxpy(nbf*nbf,&alpha,&integralOne[i*nbf*nbf],1,integral,1);
       cblas_zaxpy(nbf*nbf,&alpha,&integralTwo[i*nbf*nbf],1,integral,1);
       // Gceil_dt = - V*intergral
       cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,nbf,nbf,nbf,&alpha_min,VdtMat,
                       nbf,integral,nbf,&beta,&Gceil[(i*num_t_real+rTime)*nbf*nbf],nbf);
       // Gceil_dt = Gceil_dt + U*Gceil
       cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,nbf,nbf,nbf,&alpha,UdtMat,
                       nbf,&Gceil[(i*num_t_real+rTime-1)*nbf*nbf],nbf,&alpha,&Gceil[(i*num_t_real+rTime)*nbf*nbf],nbf);
    }      

   free(integral);

   
}


void communicateSigma(RTPROP *realTimeProp,double complex *sigmaceilTotal){
  int i,j,k;
  int iProc;
  int numProc = realTimeProp->numProc;
  int myid = realTimeProp->myid;
  int num_t_image_proc = realTimeProp->num_t_image_proc;
  int num_se_total = num_t_image*nbf*nbf;
  int num_se_total_proc = num_t_image_proc*nbf*nbf;
  double complex *sigmaceil = realTimeProp->sigmaceil;
  MPI_Status status;

  // 3. MPI Communication
  if(myid==0){
    memcpy(&sigmaceilTotal[0],&sigmaceil[0],num_se_total_proc/2*sizeof(double complex));
    memcpy(&sigmaceilTotal[num_se_total-num_se_total_proc/2],&sigmaceil[num_se_total_proc/2],num_se_total_proc/2*sizeof(double complex));
  }
  for(iProc=1;iProc<numProc;iProc++){
     if(myid==iProc){  
       MPI_Send(&sigmaceil[0],num_se_total_proc,
                MPI_DOUBLE,0,iProc,MPI_COMM_WORLD);
     }
     if(myid==0){
       MPI_Recv(&sigmaceilTotal[iProc*num_se_total_proc/2],num_se_total_proc,
                MPI_DOUBLE,iProc,iProc,MPI_COMM_WORLD,&status);
     }
     MPI_Barrier(MPI_COMM_WORLD);
     if(myid==iProc){
       MPI_Send(&sigmaceil[num_se_total_proc/2],num_se_total_proc,
                MPI_DOUBLE,0,iProc,MPI_COMM_WORLD);
     }
     if(myid==0){
       MPI_Recv(&sigmaceilTotal[(2*numProc-iProc-1)*num_se_total_proc/2],
                num_se_total_proc,
                MPI_DOUBLE,iProc,iProc,MPI_COMM_WORLD,&status);
     }
     MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&sigmaceilTotal[0],num_t_image*nbf*nbf*2,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
}


