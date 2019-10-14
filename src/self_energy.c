#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <mpi.h>
#include <mkl.h>
#include <omp.h>
#include <string.h>
#include "header.h"


void updateSigma(RTPROP *realTimeProp){
  int i,j,k;
  int isto;
  int myid = realTimeProp->myid;
  int rTime = realTimeProp->rTime;
  int num_t_image_proc = realTimeProp->num_t_image_proc;
  int omp_thread_num = realTimeProp->omp_thread_num;
  int iThread;
  double complex alpha = 1.0;
  double complex alpha_min = -1.0;
  double complex beta = 0.0;
  double complex E;
  double complex *sigmaceil = realTimeProp->sigmaceil;
  double complex *sigmar = realTimeProp->sigmar;
  double complex *gceil = realTimeProp->Gceil;
  double complex *gceil_tau = (double complex*)calloc(nbf*nbf,sizeof(double complex)); // tau
  double complex *gceil_beta_m_tau = (double complex*)calloc(nbf*nbf,sizeof(double complex)); // beta-tau
  double complex *stoR = realTimeProp->stoR;
  double complex *stoRprime = realTimeProp->stoRprime;
  double complex *sigma_dir = (double complex*)calloc(nbf*nbf,sizeof(double complex));
  double complex *sigma_exch = (double complex*)calloc(nbf*nbf,sizeof(double complex));
  double complex *sigma_dir_threads = (double complex*)calloc(omp_thread_num*nbf*nbf,sizeof(double complex));
  double complex *sigma_exch_threads = (double complex*)calloc(omp_thread_num*nbf*nbf,sizeof(double complex));

  MPI_Status status;


  omp_set_num_threads(omp_thread_num);

  for(i=0;i<num_t_image_proc;i++){ 
    memcpy(&gceil_tau[0],&gceil[i*num_t_real*nbf*nbf+rTime*nbf*nbf],nbf*nbf*sizeof(double complex));
    // conjugate of gtb
    for (j=0;j<nbf*nbf;j++) {
       gceil_beta_m_tau[j] = conj(gceil[(num_t_image_proc-i-1)*num_t_real*nbf*nbf+rTime*nbf*nbf+j]);
    }
    cblas_zscal(omp_thread_num*nbf*nbf,&beta,sigma_exch_threads,1);
    cblas_zscal(omp_thread_num*nbf*nbf,&beta,sigma_dir_threads,1);
    #pragma omp parallel private(isto,j,E,iThread)
    {
      iThread = omp_get_thread_num();
      //cblas_zscal(omp_thread_num*nbf*nbf,&beta,sigma_exch_threads,1);
      //cblas_zscal(omp_thread_num*nbf*nbf,&beta,sigma_dir_threads,1);
      double complex *Xqn = (double complex*)calloc(nbf*nbf,sizeof(double complex));
      double complex *Dmn = (double complex*)calloc(nbf*nbf,sizeof(double complex));
      double complex *Emn = (double complex*)calloc(nbf*nbf,sizeof(double complex));
      double complex *Fkj = (double complex*)calloc(nbf*nbf,sizeof(double complex));
      double complex *Lij = (double complex*)calloc(nbf*nbf,sizeof(double complex));
      double complex *Xql = (double complex*)calloc(nbf*nbf,sizeof(double complex));
      double complex *Dml = (double complex*)calloc(nbf*nbf,sizeof(double complex));
      double complex *Fmj = (double complex*)calloc(nbf*nbf,sizeof(double complex));
      double complex *Ljl = (double complex*)calloc(nbf*nbf,sizeof(double complex));
      double complex *Eil = (double complex*)calloc(nbf*nbf,sizeof(double complex));
      double complex *exch = (double complex*)calloc(nbf*nbf,sizeof(double complex));
     #pragma omp for

      for(isto=0;isto<num_so;isto++){
        // 1. Direct terms
        cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,nbf,nbf,nbf,&alpha,gceil_beta_m_tau,
                    nbf,&stoRprime[isto*nbf*nbf],nbf,&beta,Xqn,nbf);

                    
        cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,nbf,nbf,nbf,&alpha,&stoR[isto*nbf*nbf],
                    nbf,Xqn,nbf,&beta,Dmn,nbf);
                    
        cblas_zgemm(CblasRowMajor,CblasTrans,CblasNoTrans,nbf,nbf,nbf,&alpha,&gceil_tau[0],
                    nbf,Dmn,nbf,&beta,Emn,nbf);
        E = 0.0;
        for(j=0;j<nbf;j++) E += Emn[j*nbf+j];

        cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,nbf,nbf,nbf,&alpha,&gceil_tau[0],
                    nbf,&stoRprime[isto*nbf*nbf],nbf,&beta,Fkj,nbf);

        cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,nbf,nbf,nbf,&alpha,&stoR[isto*nbf*nbf],
                    nbf,Fkj,nbf,&beta,Lij,nbf);

        cblas_zaxpy(nbf*nbf,&E,Lij,1,&sigma_dir_threads[iThread*nbf*nbf],1);


        // 2. Exchange terms
        cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasTrans,nbf,nbf,nbf,&alpha,gceil_beta_m_tau,
                    nbf,&stoRprime[isto*nbf*nbf],nbf,&beta,Xql,nbf);

        cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,nbf,nbf,nbf,&alpha,&stoR[isto*nbf*nbf],
                    nbf,Xql,nbf,&beta,Dml,nbf);

        cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,nbf,nbf,nbf,&alpha,&gceil_tau[0],
                    nbf,&stoRprime[isto*nbf*nbf],nbf,&beta,Fmj,nbf);

        cblas_zgemm(CblasRowMajor,CblasTrans,CblasNoTrans,nbf,nbf,nbf,&alpha,Fmj,
                    nbf,Dml,nbf,&beta,Ljl,nbf);

        cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,nbf,nbf,nbf,&alpha,&stoR[isto*nbf*nbf],
                    nbf,&gceil_tau[0],nbf,&beta,Eil,nbf);

        cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasTrans,nbf,nbf,nbf,&alpha,Eil,
                    nbf,Ljl,nbf,&beta,exch,nbf);

        cblas_zaxpy(nbf*nbf,&alpha,exch,1,&sigma_exch_threads[iThread*nbf*nbf],1);

      }//endfor isto

      free(Xqn);
      free(Dmn);
      free(Emn);
      free(Fkj);
      free(Lij);
      free(Xql);
      free(Dml);
      free(Fmj);
      free(Ljl);
      free(Eil);
      free(exch);

    }//end omp
    double complex pre1 = 2.0/num_so;
    double complex pre2 = 1.0/num_so;
    cblas_zscal(nbf*nbf,&beta,sigma_dir,1);
    cblas_zscal(nbf*nbf,&beta,sigma_exch,1);
    for(iThread=0;iThread<omp_thread_num;iThread++){
      cblas_zaxpy(nbf*nbf,&pre1,&sigma_dir_threads[iThread*nbf*nbf],1,sigma_dir,1);
      cblas_zaxpy(nbf*nbf,&pre2,&sigma_exch_threads[iThread*nbf*nbf],1,sigma_exch,1);
    }


    //sigma_dir = sigma_dir - sigma_exch
    cblas_zaxpy(nbf*nbf,&alpha_min,sigma_exch,1,sigma_dir,1);
    //sigmaceil = sigma_dir
    memcpy(&sigmaceil[i*nbf*nbf],sigma_dir,nbf*nbf*sizeof(double complex));                

  } // end for i

  // 3. Generate sigmar and broadcast Sigmar
  if(myid==0){
    memcpy(&sigmar[rTime*nbf*nbf],&sigmaceil[0],nbf*nbf*sizeof(double complex));
    cblas_zaxpy(nbf*nbf,&alpha,&sigmaceil[(num_t_image_proc-1)*nbf*nbf],1,&sigmar[rTime*nbf*nbf],1);
    cblas_zscal(nbf*nbf,&alpha_min,&sigmar[rTime*nbf*nbf],1);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&sigmar[rTime*nbf*nbf],nbf*nbf*2,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD); 
 
  // free these 
  free(sigma_dir);
  free(sigma_exch);
  free(sigma_dir_threads);
  free(sigma_exch_threads);
  free(gceil_tau);
  free(gceil_beta_m_tau);

}

