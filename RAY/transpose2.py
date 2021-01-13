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
#          Parallelized on Ray by Padmanabhan Pillai, March 2020.
#          - v2 - implement in Ray dataflow style 
# *******************************************************************

# /******************************************************************
#                      Layout nomenclature                         
#                      -------------------
# 
# o Each rank owns one block of columns (Colblock) of the overall
#   matrix to be transposed, as well as of the transposed matrix.
# o Colblock is stored contiguously in the memory of the rank. 
#   The stored format is column major, which means that matrix
#   elements (i,j) and (i+1,j) are adjacent, and (i,j) and (i,j+1)
#   are "order" words apart
# o Colblock is logically composed of #ranks Blocks, but a Block is
#   not stored contiguously in memory. Conceptually, the Block is 
#   the unit of data that gets communicated between ranks. Block i of 
#   rank j is locally transposed and gathered into a buffer called Work, 
#   which is sent to rank i, where it is scattered into Block j of the 
#   transposed matrix.
# o When tiling is applied to reduce TLB misses, each block gets 
#   accessed by tiles. 
# o The original and transposed matrices are called A and B
# 
#  -----------------------------------------------------------------
# |           |           |           |                             |
# | Colblock  |           |           |                             |
# |           |           |           |                             |
# |           |           |           |                             |
# |           |           |           |                             |
# |        -------------------------------                          |
# |           |           |           |                             |
# |           |  Block    |           |                             |
# |           |           |           |                             |
# |           |           |           |                             |
# |           |           |           |                             |
# |        -------------------------------                          |
# |           |Tile|      |           |                             |
# |           |    |      |           |   Overall Matrix            |
# |           |----       |           |                             |
# |           |           |           |                             |
# |           |           |           |                             |
# |        -------------------------------                          |
# |           |           |           |                             |
# |           |           |           |                             |
# |           |           |           |                             |
# |           |           |           |                             |
# |           |           |           |                             |
#  -----------------------------------------------------------------*/

import sys
print('Python version = ', str(sys.version_info.major)+'.'+str(sys.version_info.minor))
if sys.version_info >= (3, 3):
    from time import process_time as timer
else:
    from timeit import default_timer as timer

import time
import ray
import os

@ray.remote(num_cpus=1)
class transpose():
    def __init__(self, my_id):
        self.my_id = my_id

    def init(self, n, order):
        self.order = order
        self.n = n
        self.block_order = int(self.order/self.n)
        self.colstart = self.block_order*self.my_id
        self.A = [[(i+self.colstart)*self.order+j+0.0 for j in range(self.order)] for i in range(self.block_order)]
        self.B = [[0.0 for j in range(self.order)] for i in range(self.block_order)]
        return self.my_id

    # Note:  Each iteration actually does B+=A^T, A+=1
    # For worker k, this takes as input the list of kth blocks (futures) from all workers
    def run_step(self, blocks):
        for s, blk in enumerate(blocks):
            row = s*self.block_order
            block = ray.get(blk)
            for i in range(self.block_order):
                for j in range(self.block_order):
                    self.B[i][j+row] += block[j][i]
        for i in range(self.block_order):
            for j in range(self.order):
                self.A[i][j]+=1

    def get_block(self, block_id):
        row = block_id*self.block_order
        return [C[row:row+self.block_order] for C in self.A]

    def start_time(self):
        self.start_time = timer()

    def wait_end(self):
        t = timer()
        return t - self.start_time

    def validate(self, iterations):
        addit = (iterations * (iterations+1))/2
        abserr = 0.0;
        for i in range(self.block_order):
            for j in range(self.order):
                temp    = (self.order*j+self.colstart+i) * (iterations+1)
                abserr += abs(self.B[i][j] - float(temp+addit))
        return abserr


def main():

    # ********************************************************************
    # initialize Ray
    # ********************************************************************

    ray_address = os.getenv("ray_address")
    ray_redis_pass = os.getenv("ray_redis_password")
    if ray_address==None:
    	ray_address = "auto"
    if ray_redis_pass==None:
    	ray_redis_pass = ""
    try:
    	ray.init(address=ray_address, _redis_password=ray_redis_pass)
    except:
    	ray.init()
    assert ray.is_initialized() == True
    cores = ray.available_resources()['CPU']
    print ("Ray initialized;  available cores:", cores)    

    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    print('Parallel Research Kernels version ') #, PRKVERSION
    print('Python Matrix transpose: B = A^T')

    if len(sys.argv) not in [3,4]:
        print('argument count = ', len(sys.argv))
        sys.exit("Usage: ./transpose <# iterations> <matrix order> [<num workers>]")

    iterations = int(sys.argv[1])
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    order = int(sys.argv[2])

    nworkers = int(cores)
    if len(sys.argv)==4:
        nworkers = int(sys.argv[3])
    if nworkers<1:
        sys.exit("ERROR: Num workers must be >=1")
    if nworkers>cores:
        sys.exit("ERROR: Too many workers requested")
    if order < nworkers:
        sys.exit("ERROR: order must be at least # of workers")
    if order%nworkers != 0:
        sys.exit("ERROR: order must be divisible by # of workers")

    print('Number of iterations = ', iterations)
    print('Matrix order         = ', order)
    print('Number of workers    = ', nworkers)


    # ********************************************************************
    # ** Initialize workers, matrices.
    # ********************************************************************

    robjs = [transpose.remote(x) for x in range(0,nworkers)]
    ray.get([x.init.remote(nworkers, order) for x in robjs])  
    # getting the results to make sure objs ready before timing

    # ********************************************************************
    # ** Run iterations.
    # ********************************************************************

    for i in range(iterations+1):
        if (i==1):
            ray.get([x.start_time.remote() for x in robjs])
            t0 = time.time()  # wall-clock time
        blocks = [[y.get_block.remote(i) for y in robjs] for i in range(nworkers)]
        [x.run_step.remote( blocks[i] ) for i,x in enumerate(robjs)]
    # get results (times measured by run() on each worker)
    fut_times = [x.wait_end.remote() for x in robjs]
    t2 = time.time()
    trans_time = max(ray.get(fut_times))
    t1 = time.time()
    trans_time2 = t1-t0
    print (trans_time, trans_time2, t2-t0)


    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    abserr = sum(ray.get( [x.validate.remote(iterations) for x in robjs] ))

    epsilon=1.e-8
    nbytes = 2 * order**2 * 8 # 8 is not sizeof(double) in bytes, but allows for comparison to C etc.
    if abserr < epsilon:
        print('Solution validates')
        avgtime = trans_time/iterations
        avgtime2 = trans_time2/iterations
        print('Rate (MB/s): ',1.e-6*nbytes/avgtime, ' Avg time (s): ', avgtime, '(max worker)')
        print('Rate (MB/s): ',1.e-6*nbytes/avgtime2, ' Avg time (s): ', avgtime2, '(wall-clock)')
    else:
        print('error ',abserr, ' exceeds threshold ',epsilon)
        sys.exit("ERROR: solution did not validate")




if __name__ == '__main__':
    main()

