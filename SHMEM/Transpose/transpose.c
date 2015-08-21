/*
Copyright (c) 2013, Intel Corporation

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions 
are met:

* Redistributions of source code must retain the above copyright 
      notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above 
      copyright notice, this list of conditions and the following 
      disclaimer in the documentation and/or other materials provided 
      with the distribution.
* Neither the name of Intel Corporation nor the names of its 
      contributors may be used to endorse or promote products 
      derived from this software without specific prior written 
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
*/

/*******************************************************************

NAME:    transpose

PURPOSE: This program tests the efficiency with which a square matrix
         can be transposed and stored in another matrix. The matrices
         are distributed identically.
  
USAGE:   Program inputs are the matrix order, the number of times to 
         repeat the operation, and the communication mode

         transpose <# iterations> <matrix order> [tile size]

         An optional parameter specifies the tile size used to divide the 
         individual matrix blocks for improved cache and TLB performance. 
  
         The output consists of diagnostics to make sure the 
         transpose worked and timing statistics.

FUNCTIONS CALLED:

         Other than SHMEM or standard C functions, the following 
         functions are used in this program:

          wtime()           Portable wall-timer interface.
          bail_out()        Determine global error and exit if nonzero.

HISTORY: Written by Tom St. John, July 2015.  
         
  
*******************************************************************/

/******************************************************************
                     Layout nomenclature                         
                     -------------------

o Each rank owns one block of columns (Colblock) of the overall
  matrix to be transposed, as well as of the transposed matrix.
o Colblock is stored contiguously in the memory of the rank. 
  The stored format is column major, which means that matrix
  elements (i,j) and (i+1,j) are adjacent, and (i,j) and (i,j+1)
  are "order" words apart
o Colblock is logically composed of #ranks Blocks, but a Block is
  not stored contiguously in memory. Conceptually, the Block is 
  the unit of data that gets communicated between ranks. Block i of 
  rank j is locally transposed and gathered into a buffer called Work, 
  which is sent to rank i, where it is scattered into Block j of the 
  transposed matrix.
o When tiling is applied to reduce TLB misses, each block gets 
  accessed by tiles. 
o The original and transposed matrices are called A and B

 -----------------------------------------------------------------
|           |           |           |                             |
| Colblock  |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|        -------------------------------                          |
|           |           |           |                             |
|           |  Block    |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|        -------------------------------                          |
|           |Tile|      |           |                             |
|           |    |      |           |   Overall Matrix            |
|           |----       |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|        -------------------------------                          |
|           |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
|           |           |           |                             |
 -----------------------------------------------------------------*/

#include <par-res-kern_general.h>
#include <par-res-kern_shmem.h>

#define A(i,j)        A_p[(i+istart)+order*(j)]
#define B(i,j)        B_p[(i+istart)+order*(j)]
#define Work_in(phase, i,j)  Work_in_p[phase-1][i+Block_order*(j)]
#define Work_out(i,j) Work_out_p[i+Block_order*(j)]

double local_trans_time, trans_time;
double abserr, abserr_tot;
long pSync[_SHMEM_BCAST_SYNC_SIZE];
double pWrk[_SHMEM_BCAST_SYNC_SIZE];

int main(int argc, char ** argv)
{
  int Block_order;         /* number of columns owned by rank       */
  int Block_size;          /* size of a single block                */
  int Colblock_size;       /* size of column block                  */
  int Tile_order=32;       /* default Tile order                    */
  int tiling;              /* boolean: true if tiling is used       */
  int Num_procs;           /* number of ranks                       */
  int order;               /* order of overall matrix               */
  int send_to, recv_from;  /* ranks with which to communicate       */
  long bytes;              /* combined size of matrices             */
  int my_ID;               /* rank                                  */
  int root=0;              /* rank of root                          */
  int iterations;          /* number of times to do the transpose   */
  int i, j, it, jt, istart;/* dummies                               */
  int iter;                /* index of iteration                    */
  int phase;               /* phase inside staged communication     */
  int colstart;            /* starting column for owning rank       */
  int error;               /* error flag                            */
  double *A_p;             /* original matrix column block          */
  double *B_p;             /* transposed matrix column block        */
  double **Work_in_p;      /* workspace for the transpose function  */
  double *Work_out_p;      /* workspace for the transpose function  */
  double epsilon = 1.e-8;  /* error tolerance                       */
  double avgtime;          /* timing parameters                     */
  int *recv_flag;
  int *arguments;

/*********************************************************************
** Initialize the SHMEM environment
*********************************************************************/

  start_pes(0);
  my_ID=shmem_my_pe();
  Num_procs=shmem_n_pes();

  for(i=0;i<_SHMEM_BCAST_SYNC_SIZE;i++)
    pSync[i]=_SHMEM_SYNC_VALUE;

  arguments=(int*)shmalloc(3*sizeof(int));

/*********************************************************************
** process, test and broadcast input parameters
*********************************************************************/
  error = 0;
  if (my_ID == root) {
    if (argc != 3 && argc != 4){
      printf("Usage: %s <# iterations> <matrix order> [Tile size]\n",
                                                               *argv);
      error = 1; goto ENDOFTESTS;
    }

    iterations  = atoi(*++argv);
    arguments[0]=iterations;
    if(iterations < 1){
      printf("ERROR: iterations must be >= 1 : %d \n",iterations);
      error = 1; goto ENDOFTESTS;
    }

    order = atoi(*++argv);
    arguments[1]=order;
    if (order < Num_procs) {
      printf("ERROR: matrix order %d should at least # procs %d\n", 
             order, Num_procs);
      error = 1; goto ENDOFTESTS;
    }
    if (order%Num_procs) {
      printf("ERROR: matrix order %d should be divisible by # procs %d\n",
             order, Num_procs);
      error = 1; goto ENDOFTESTS;
    }

    if (argc == 4) Tile_order = atoi(*++argv);
    arguments[2]=Tile_order;

    ENDOFTESTS:;
  }
  bail_out(error, pSync);

  if (my_ID == root) {
    printf("SHMEM matrix transpose: B = A^T\n");
    printf("Number of ranks      = %d\n", Num_procs);
    printf("Matrix order         = %d\n", order);
    printf("Number of iterations = %d\n", iterations);
    if ((Tile_order > 0) && (Tile_order < order))
          printf("Tile size            = %d\n", Tile_order);
    else  printf("Untiled\n");
  }
  
  shmem_barrier_all();

  /*  Broadcast input data to all ranks */
  shmem_broadcast32(&arguments[0], &arguments[0], 3, root, 0, 0, Num_procs, pSync);

  iterations=arguments[0];
  order=arguments[1];
  Tile_order=arguments[2];

  shmem_barrier_all();
  shfree(arguments);

  for(i=0;i<_SHMEM_REDUCE_SYNC_SIZE;i++)
    pSync[i]=_SHMEM_SYNC_VALUE;

  /* a non-positive tile size means no tiling of the local transpose */
  tiling = (Tile_order > 0) && (Tile_order < order);
  bytes = 2 * sizeof(double) * order * order;

/*********************************************************************
** The matrix is broken up into column blocks that are mapped one to a 
** rank.  Each column block is made up of Num_procs smaller square 
** blocks of order block_order.
*********************************************************************/

  Block_order    = order/Num_procs;
  colstart       = Block_order * my_ID;
  Colblock_size  = order * Block_order;
  Block_size     = Block_order * Block_order;

/*********************************************************************
** Create the column block of the test matrix, the row block of the 
** transposed matrix, and workspace (workspace only if #procs>1)
*********************************************************************/
  A_p = (double *)malloc(Colblock_size*sizeof(double));
  if (A_p == NULL){
    printf(" Error allocating space for original matrix on node %d\n",my_ID);
    error = 1;
  }
  bail_out(error, pSync);

  B_p = (double *)malloc(Colblock_size*sizeof(double));
  if (B_p == NULL){
    printf(" Error allocating space for transpose matrix on node %d\n",my_ID);
    error = 1;
  }
  bail_out(error, pSync);

  if (Num_procs>1) {
    Work_in_p   = (double**)malloc((Num_procs-1)*sizeof(double));

    Work_out_p = (double*)shmalloc(Block_size*sizeof(double));
    recv_flag=(int*)shmalloc((Num_procs-1)*sizeof(int));
    if ((Work_in_p == NULL)||(Work_out_p==NULL) || (recv_flag == NULL)){
      printf(" Error allocating space for work or flags on node %d\n",my_ID);
      error = 1;
    }
    bail_out(error, pSync);
    for(i=0;i<(Num_procs-1);i++) {
      Work_in_p[i]=(double*)shmalloc(Block_size*sizeof(double));
      if (Work_in_p[i] == NULL) {
        printf(" Error allocating space for work on node %d\n",my_ID);
        error = 1;
      }
      bail_out(error, pSync);
    }

    for(i=0;i<Num_procs-1;i++)
      recv_flag[i]=0;
  }
  
  /* Fill the original column matrix                                                */
  istart = 0;  
  for (j=0;j<Block_order;j++) 
    for (i=0;i<order; i++)  {
      A(i,j) = (double) (order*(j+colstart) + i);
      B(i,j) = -1.0;
  }

  for (iter = 0; iter<=iterations; iter++){

    /* start timer after a warmup iteration                                        */
    if (iter == 1) { 
      shmem_barrier_all();
      local_trans_time = wtime();
    }

    /* do the local transpose                                                     */
    istart = colstart; 
    if (!tiling) {
      for (i=0; i<Block_order; i++) 
        for (j=0; j<Block_order; j++) {
          B(j,i) = A(i,j);
	}
    }
    else {
      for (i=0; i<Block_order; i+=Tile_order) 
        for (j=0; j<Block_order; j+=Tile_order) 
          for (it=i; it<MIN(Block_order,i+Tile_order); it++)
            for (jt=j; jt<MIN(Block_order,j+Tile_order);jt++)
              B(jt,it) = A(it,jt); 
    }

    for (phase=1; phase<Num_procs; phase++){
      recv_from = (my_ID + phase            )%Num_procs;
      send_to   = (my_ID - phase + Num_procs)%Num_procs;

      istart = send_to*Block_order; 
      if (!tiling) {
        for (i=0; i<Block_order; i++) 
          for (j=0; j<Block_order; j++){
	    Work_out(j,i) = A(i,j);
	  }
      }
      else {
        for (i=0; i<Block_order; i+=Tile_order) 
          for (j=0; j<Block_order; j+=Tile_order) 
            for (it=i; it<MIN(Block_order,i+Tile_order); it++)
              for (jt=j; jt<MIN(Block_order,j+Tile_order);jt++) {
                Work_out(jt,it) = A(it,jt); 
	      }
      }

      shmem_double_put(&Work_in_p[phase-1][0], &Work_out_p[0], Block_size, send_to);
      shmem_fence();
      shmem_int_inc(&recv_flag[phase-1], send_to);

      shmem_int_wait_until(&recv_flag[phase-1], SHMEM_CMP_EQ, iter+1);

      istart = recv_from*Block_order; 
      /* scatter received block to transposed matrix; no need to tile */
      for (j=0; j<Block_order; j++)
        for (i=0; i<Block_order; i++) 
          B(i,j) = Work_in(phase, i,j);
    }  /* end of phase loop  */
  } /* end of iterations */

  local_trans_time = wtime() - local_trans_time;

  for(i=0;i<_SHMEM_BCAST_SYNC_SIZE;i++)
    pSync[i]=_SHMEM_SYNC_VALUE;

  shmem_barrier_all();

  shmem_double_max_to_all(&trans_time, &local_trans_time, 1, 0, 0, Num_procs, pWrk, pSync);

  abserr = 0.0;
  istart = 0;
  for (j=0;j<Block_order;j++) for (i=0;i<order; i++) {
      abserr += ABS(B(i,j) - (double)(order*i + j+colstart));
  }

  for(i=0;i<_SHMEM_BCAST_SYNC_SIZE;i++)
    pSync[i]=_SHMEM_SYNC_VALUE;

  shmem_barrier_all();

  shmem_double_sum_to_all(&abserr_tot, &abserr, 1, 0, 0, Num_procs, pWrk, pSync);

  if (my_ID == root) {
    if (abserr_tot < epsilon) {
      printf("Solution validates\n");
      avgtime = trans_time/(double)iterations;
      printf("Rate (MB/s): %lf Avg time (s): %lf\n",1.0E-06*bytes/avgtime, avgtime);
#ifdef VERBOSE
      printf("Summed errors: %f \n", abserr);
#endif
    }
    else {
      printf("ERROR: Aggregate squared error %lf exceeds threshold %e\n", abserr, epsilon);
      error = 1;
    }
  }

  bail_out(error, pSync);

  if (Num_procs>1) shfree(recv_flag);
  for(i=0;i<Num_procs-1;i++)
    shfree(Work_in_p[i]);

  free(Work_in_p);

  //shmem_finalize();
  exit(EXIT_SUCCESS);

}  /* end of main */
