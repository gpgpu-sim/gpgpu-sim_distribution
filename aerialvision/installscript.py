#!/usr/bin/env python

# Copyright (C) 2009 by Aaron Ariel, Tor M. Aamodt and the University of British 
# Columbia, Vancouver, BC V6T 1Z4, All Rights Reserved.
# 
# THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
# TERMS AND CONDITIONS.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# 
# NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
# are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
# (property of NVIDIA).  The files benchmarks/BlackScholes/ and 
# benchmarks/template/ are derived from the CUDA SDK available from 
# http://www.nvidia.com/cuda (also property of NVIDIA).  The files from 
# src/intersim/ are derived from Booksim (a simulator provided with the 
# textbook "Principles and Practices of Interconnection Networks" available 
# from http://cva.stanford.edu/books/ppin/). As such, those files are bound by 
# the corresponding legal terms and conditions set forth separately (original 
# copyright notices are left in files from these sources and where we have 
# modified a file our copyright notice appears before the original copyright 
# notice).  
# 
# Using this version of GPGPU-Sim requires a complete installation of CUDA 
# which is distributed seperately by NVIDIA under separate terms and 
# conditions.  To use this version of GPGPU-Sim with OpenCL requires a
# recent version of NVIDIA's drivers which support OpenCL.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 
# 3. Neither the name of the University of British Columbia nor the names of
# its contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
# 
# 4. This version of GPGPU-SIM is distributed freely for non-commercial use only.  
#  
# 5. No nonprofit user may place any restrictions on the use of this software,
# including as modified by the user, by any other authorized user.
# 
# 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung, 
# Ali Bakhoda, George L. Yuan, at the University of British Columbia, 
# Vancouver, BC V6T 1Z4


import os
#
# make sure you have the following installed before running this script
# (a) python-dev|python-devel
# (b) Tkinter 
# (c) tcl-devel
# (d) tk-devel

os.system('mkdir $HOME/local')
os.system('mkdir $HOME/local/test')
os.system('mkdir $HOME/.gpgpu_sim')
os.system('mkdir $HOME/.gpgpu_sim/aerialvision')

###Installing Pmw
os.system('echo "XYZ 2"; curl -L -o "$HOME/local/test/Pmw.1.3.2.tar.gz" "http://sourceforge.net/projects/pmw/files/Pmw/Pmw.1.3.2/Pmw.1.3.2.tar.gz/download"')
os.system('echo "XYZ 3"; tar xvfz $HOME/local/test/Pmw.1.3.2.tar.gz -C $HOME/local/test')
os.system('echo "XYZ 4"; cd $HOME/local/test/Pmw.1.3.2/src/; ./setup.py build')
os.system('echo "XYZ 5"; cd $HOME/local/test/Pmw.1.3.2/src/; ./setup.py install --prefix=$HOME/local/test')
os.system('echo "XYZ 6"; rm $HOME/local/test/Pmw.1.3.2.tar.gz')


####Installing PLY
os.system('echo "XYZ 7"; curl -L -o "$HOME/local/test/ply-3.2.tar.gz" "http://www.dabeaz.com/ply/ply-3.2.tar.gz"')
os.system('echo "XYZ 8"; tar xvfz $HOME/local/test/ply-3.2.tar.gz -C $HOME/local/test')
os.system('echo "XYZ 9"; cd $HOME/local/test/ply-3.2/; python setup.py build')
os.system('echo "XYZ 10"; export PYTHONPATH=$PYTHONPATH:$HOME/local/test/lib/python2.6/site-packages:$HOME/local/test/lib64/python2.6/site-packages; cd $HOME/local/test/ply-3.2/; python setup.py install --prefix=$HOME/local/test')
os.system('echo "XYZ 11"; rm $HOME/local/test/ply-3.2.tar.gz')

#####Installing Numpy
os.system('echo "XYZ 12"; curl -L -o "$HOME/local/test/numpy-1.3.0.tar.gz" "http://sourceforge.net/projects/numpy/files/NumPy/1.3.0/numpy-1.3.0.tar.gz/download"')
os.system('echo "XYZ 13"; tar xvfz $HOME/local/test/numpy-1.3.0.tar.gz -C $HOME/local/test')
os.system('echo "XYZ 14"; rm $HOME/local/test/numpy-1.3.0.tar.gz')
os.system('echo "XYZ 15"; python $HOME/local/test/numpy-1.3.0/setup.py build')
os.system('echo "XYZ 16"; python $HOME/local/test/numpy-1.3.0/setup.py install --prefix=$HOME/local/test')

#####Installing LibPng
os.system('echo "XYZ 17"; curl -L -o "$HOME/local/test/libpng-1.2.37.tar.gz" "http://sourceforge.net/projects/libpng/files/libpng-stable/1.2.37/libpng-1.2.37.tar.gz/download"')
os.system('echo "XYZ 18"; tar xvfz $HOME/local/test/libpng-1.2.37.tar.gz -C $HOME/local/test')
os.system('echo "XYZ 19"; rm $HOME/local/test/libpng-1.2.37.tar.gz')
os.system('echo "XYZ 20"; cd $HOME/local/test/libpng-1.2.37')
os.system('echo "XYZ 21"; cd $HOME/local/test/libpng-1.2.37; ./configure --prefix=$HOME/local/test')
os.system('echo "XYZ 22"; cd $HOME/local/test/libpng-1.2.37; make')
os.system('echo "XYZ 23"; cd $HOME/local/test/libpng-1.2.37; make install')

####Installing matplotlib
os.system('echo "XYZ 24"; curl -L -o "$HOME/local/test/matplotlib-0.98.5.3.tar.gz" "http://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-0.98.5/matplotlib-0.98.5.3.tar.gz/download"')
os.system('echo "XYZ 25"; tar xvfz $HOME/local/test/matplotlib-0.98.5.3.tar.gz -C $HOME/local/test')
os.system('echo "XYZ 26"; rm $HOME/local/test/matplotlib-0.98.5.3.tar.gz')
os.system('echo "XYZ 27"; export PYTHONPATH=$PYTHONPATH:$HOME/local/test/lib/python2.6/site-packages:$HOME/local/test/lib64/python2.6/site-packages; export CPLUS_INCLUDE_PATH=$HOME/local/test/include; export PKG_CONFIG_PATH=$HOME/local/test/lib/pkgconfig; cd $HOME/local/test/matplotlib-0.98.5.3;python setup.py build')
os.system('echo "XYZ 28"; export PYTHONPATH=$PYTHONPATH:$HOME/local/test/lib/python2.6/site-packages:$HOME/local/test/lib64/python2.6/site-packages; export CPLUS_INCLUDE_PATH=$HOME/local/test/include; export PKG_CONFIG_PATH=$HOME/local/test/lib/pkgconfig; cd $HOME/local/test/matplotlib-0.98.5.3/; python setup.py install --prefix=$HOME/local/test')
os.system('echo "XYZ 29"; echo "Please add the following to your environment (e.g., via your .bashrc file)\n    export PYTHONPATH=\$PYTHONPATH:$HOME/local/test/lib/python2.6/site-packages:$HOME/local/test/lib64/python2.6/site-packages\n    export CPLUS_INCLUDE_PATH=$HOME/local/test/include\n    export PKG_CONFIG_PATH=$HOME/local/test/lib/pkgconfig\n"')
