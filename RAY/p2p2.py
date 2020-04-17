#!/usr/bin/env python3
#
# Copyright (c) 2015, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
# * Neither the name of Intel Corporation nor the names of its
#       contributors may be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
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
# NAME:    Pipeline
#
# PURPOSE: This program tests the efficiency with which point-to-point
#          synchronization can be carried out. It does so by executing
#          a pipelined algorithm on an m*n grid. The first array dimension
#          is distributed among the threads (stripwise decomposition).
#
# USAGE:   The program takes as input the
#          dimensions of the grid, and the number of iterations on the grid
#
#                <progname> <iterations> <m> <n> [<batch size> <# workers>]
#
#          The output consists of diagnostics to make sure the
#          algorithm worked, and of timing statistics.
#
# HISTORY: - Written by Rob Van der Wijngaart, February 2009.
#          - Converted to Python by Jeff Hammond, February 2016.
#          - Parallelized on Ray by Padmanabhan Pillai, March 2020.
#          - v2 - implement in Ray dataflow style 
#
# *******************************************************************

import sys
print('Python version = ', str(sys.version_info.major)+'.'+str(sys.version_info.minor))
if sys.version_info >= (3, 3):
    from time import process_time as timer
else:
    from timeit import default_timer as timer

import time
import ray
import os

@ray.remote
class pipeline():
    def __init__(self, my_id):
        self.my_id = my_id

    def init(self, nw, m, n, batch):
        self.m = m
        self.n = n
        self.nw = nw
        self.batch = batch if batch!=0 else n
        width = int((self.m-1)/self.nw)
        leftover = (self.m-1)%self.nw
        if self.my_id < leftover:
            istart = (width+1)*self.my_id+1
            self.width = width+1
        else:
            istart = width*self.my_id+leftover+1
            self.width = width
        self.A = [[0.0 for j in range(n)] for i in range(self.width+1)]
        for i in range(self.width+1):
            self.A[i][0]=float(i+istart-1)
        if self.my_id==0:
            for j in range(n):
                self.A[0][j]=float(j)

    def run_step(self, b, data):
        # For convenience
        my_id = self.my_id
        A = self.A
        w = self.width
        n = self.n
        nw = self.nw
        bs = self.batch

        js = b*bs+1
        je = min(n,js+bs)
        if my_id>0:
            self.set_data(js, data)
        for i in range(1,w+1):
            for j in range(js,je):
                A[i][j] = A[i][j-1] + A[i-1][j] - A[i-1][j-1]


    def get_data(self, i, b):
        return self.A[self.width][i:i+b]

    def set_data(self, i, d):
        self.A[0][i:i+len(d)] = d

    def set_home(self, d):
        self.A[0][0] = -d[0]

    def start_time(self):
        self.start_time = timer()

    def wait_end(self):
        t = timer()
        return t - self.start_time

    def validate(self):
        return -self.A[0][0]


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
    	ray.init(address=ray_address, redis_password=ray_redis_pass)
    except:
    	ray.init()
    assert ray.is_initialized() == True
    cores = ray.available_resources()['CPU']
    print ("Ray initialized;  available cores:", cores)    

    # ********************************************************************
    # read and test input parameters
    # ********************************************************************

    print('Parallel Research Kernels version ')#, PRKVERSION
    print('Python pipeline execution on 2D grid')

    if len(sys.argv) not in [4,5,6]:
        print('argument count = ', len(sys.argv))
        sys.exit("Usage: ./synch_p2p <# iterations> <first array dimension> <second array dimension> [<batch size> <number of workers>]")

    iterations = int(sys.argv[1])
    if iterations < 1:
        sys.exit("ERROR: iterations must be >= 1")

    m = int(sys.argv[2])
    if m < 1:
        sys.exit("ERROR: array dimension must be >= 1")

    n = int(sys.argv[3])
    if n < 1:
        sys.exit("ERROR: array dimension must be >= 1")

    batch = 0
    if len(sys.argv) > 4:
        batch = int(sys.argv[4])
    if batch<0:
        sys.exit("ERROR: Batch size must be positive, or zero to indicate all at once")
    if batch==0:
        batch = n
    numbatches = int((n+batch-1)/batch)

    nworkers = cores
    if len(sys.argv) > 5:
        nworkers = int(sys.argv[5])
    if nworkers<1:
        sys.exit("ERROR: Num workers must be >=1")
    if nworkers>cores:
        sys.exit("ERROR: Too many workers requested")

    print('Grid sizes               = ', m, n)
    print('Number of workers        = ', nworkers)
    print('Batch size               = ', batch)
    print('Number of iterations     = ', iterations)


    # ********************************************************************
    # ** Initialize workers, matrices.
    # ********************************************************************

    robjs = [pipeline.remote(x) for x in range(0,nworkers)]
    ray.get([x.init.remote(nworkers, m, n, batch) for x in robjs])  
    # getting the results to make sure objs ready before timing

    # ********************************************************************
    # ** Run iterations.
    # ********************************************************************

    t0 = time.time()  # wall-clock time
    for i in range(iterations+1):
        if (i==1):
            [x.start_time.remote() for x in robjs]
        for b in range(numbatches):
            js = batch*b+1
            for k in range(nworkers):
                d=0 if k==0 else robjs[k-1].get_data.remote(js,batch)
                robjs[k].run_step.remote(b, d)
        robjs[0].set_home.remote(robjs[-1].get_data.remote(n-1,1))
    # get results (times measured by run() on each worker)
    fut_times = [x.wait_end.remote() for x in robjs]
    t2 = time.time()
    pipeline_time = max(ray.get(fut_times))
    t1 = time.time()
    pipeline_time2 = t1-t0
    print (pipeline_time, pipeline_time2, t2-t0)

    # ********************************************************************
    # ** Analyze and output results.
    # ********************************************************************

    epsilon=1.e-8
    grid_val = ray.get(robjs[0].validate.remote())

    # verify correctness, using top right value
    corner_val = float((iterations+1)*(n+m-2))
    if (abs(grid_val - corner_val)/corner_val) < epsilon:
        print('Solution validates')
        avgtime = pipeline_time/iterations
        avgtime2 = pipeline_time2/(iterations+1)
        print('Rate (MFlops/s): ',1.e-6*2*(m-1)*(n-1)/avgtime,' Avg time (s): ',avgtime, "(max worker)")
        print('Rate (MFlops/s): ',1.e-6*2*(m-1)*(n-1)/avgtime2,' Avg time (s): ',avgtime2, "(wall-clock)")
    else:
        print('ERROR: checksum ',grid_val,' does not match verification value', corner_val)
        sys.exit()


if __name__ == '__main__':
    main()

