#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <mkl.h>
#include <string.h>

int main(int argc, char *argv[]){
  
  int i,j;
  int m=2;
  int n=4;
  int k=3;
  double complex alpha = 1.0; 
  double complex beta = 0.0; 
  double complex *A = (double complex*)calloc(m*k,sizeof(double complex));
  double complex *B = (double complex*)calloc(k*n,sizeof(double complex));
  double complex *C = (double complex*)calloc(m*n,sizeof(double complex));


 
  for (i=0;i<m;i++){
     for (j=0;j<k;j++){
       A[i*k+j] = 0.1*i + j;
     }
  }   

  for (i=0;i<k;i++){
     for (j=0;j<n;j++){
       B[i*n+j] = 0.2*i + 0.3*j;
     }
  }
  cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
              m,n,k,&alpha,A,
              k,B,n,&beta,
              C,n);

  printf("Matrix A \n");
  for (i=0;i<m;i++){
     for (j=0;j<k;j++){
       printf("%d %d %f\n",i,j,A[i*k+j]);
      }
  }    

  printf("Matrix B \n");
  for (i=0;i<k;i++){
     for (j=0;j<n;j++){
       printf("%d %d %f\n",i,j,B[i*n+j]);
      }
  }
  printf("Matrix C \n");
  for (i=0;i<m;i++){
     for (j=0;j<n;j++){
       printf("%d %d %f\n",i,j,C[i*n+j]);
      }
  }

  printf("I am here! haha\n");

  return 1;
}
