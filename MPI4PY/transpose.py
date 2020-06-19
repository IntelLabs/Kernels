#!/usr/bin/env python3
#
# Copyright (c) 2015, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
# * Neither the name of Intel Corporation nor the names of its
#      contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

#*******************************************************************
#
# NAME:    transpose
#
# PURPOSE: This program measures the time for the transpose of a
#          column-major stored matrix into a row-major stored matrix.
#
# USAGE:   Program input is the matrix order and the number of times to
#          repeat the operation:
#
#          transpose <# iterations> <matrix_size>
#
#          The output consists of diagnostics to make sure the
#          transpose worked and timing statistics.
#
# HISTORY: Written by  Rob Van der Wijngaart, February 2009.
#          Converted to Python by Jeff Hammond, February 2016.
#          Converted to MPI4PY by Babu Pillai, June 2020.
# *******************************************************************

import sys
if sys.version_info >= (3, 3):
    from time import perf_counter as timer
else:
    from timeit import default_timer as timer
from mpi4py import MPI

def main():

    comm = MPI.COMM_WORLD
    num_procs = comm.Get_size()
    my_id = comm.Get_rank()

    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    order=0
    iterations=0

    if my_id==0:
        print('Python version = ', str(sys.version_info.major)+'.'+str(sys.version_info.minor))
        print('Parallel Research Kernels version ') #, PRKVERSION
        print('Python Matrix transpose: B = A^T')

        if len(sys.argv) != 3:
            print('argument count = ', len(sys.argv))
            sys.exit("Usage: python -m mpi4py transpose.py <# iterations> <matrix order>")

        iterations = int(sys.argv[1])
        if iterations < 1:
            sys.exit("ERROR: iterations must be >= 1")

        order = int(sys.argv[2])
        if order < 1 or order%num_procs>0:
            sys.exit("ERROR: order must be a multiple of # procs")

        print('Number of ranks:     = ', num_procs)
        print('Number of iterations = ', iterations)
        print('Matrix order         = ', order)

    order = comm.bcast(order, root=0)
    iterations = comm.bcast(iterations, root=0)

    # ********************************************************************
    # ** Allocate space for the input and transpose matrix
    # ********************************************************************

    block_order = order//num_procs
    colstart = block_order * my_id

    # 0.0 is a float, which is 64b (53b of precision)
    A = [[(i+colstart)*order+j+0.0 for j in range(order)] for i in range(block_order)]
    B = [[0.0 for j in range(order)] for i in range(block_order)]
    
    for k in range(0,iterations+1):

        if k==1:
           comm.Barrier()
           t0 = timer()

        # do local transpose
        row = my_id*block_order
        for i in range(block_order):
            for j in range(block_order):
                B[i][j+row] += A[j][i+row]

        for phase in range(1,num_procs):
            send_to = (my_id+phase)%num_procs
            recv_from = (my_id-phase+num_procs)%num_procs
            row = send_to*block_order
            sblk = [ [ A[i][j] for j in range(row, row+block_order) ] for i in range(block_order) ]
            rblk = comm.sendrecv( sblk, dest=send_to, source=recv_from )
            row = recv_from*block_order
            for i in range(block_order):
                for j in range(block_order):
                    B[i][j+row] += rblk[j][i]

        for i in range(block_order):
            for j in range(order):
                A[i][j] += 1.0

    t1 = timer()
    trans_time = t1 - t0
    trans_time = comm.reduce(trans_time, op=MPI.MAX, root=0)

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    addit = (iterations * (iterations+1))/2
    abserr = 0.0;
    for i in range(block_order):
        for j in range(order):
            temp    = (order*j+colstart+i) * (iterations+1)
            abserr += abs(B[i][j] - float(temp+addit))
    abserr = comm.reduce(abserr, op=MPI.SUM, root=0)

    if my_id==0:
        epsilon=1.e-8
        nbytes = 2 * order**2 * 8 # 8 is not sizeof(double) in bytes, but allows for comparison to C etc.
        if abserr < epsilon:
            print('Solution validates')
            avgtime = trans_time/iterations
            print('Rate (MB/s): ',1.e-6*nbytes/avgtime, ' Avg time (s): ', avgtime)
        else:
            print('error ',abserr, ' exceeds threshold ',epsilon)
            sys.exit("ERROR: solution did not validate")


if __name__ == '__main__':
    main()

