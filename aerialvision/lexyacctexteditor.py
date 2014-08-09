#!/usr/bin/env python

# Copyright (C) 2009 by Aaron Ariel, Wilson W. L. Fung
# and the University of British Columbia, Vancouver, 
# BC V6T 1Z4, All Rights Reserved.
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



import sys
import re
sys.path.insert(0,"Lib/site-packages/ply-3.2/ply-3.2")
import ply.lex as lex
import ply.yacc as yacc
import variableclasses as vc

def textEditorParseMe(filename):
    
    tokens = ['FILENAME', 'NUMBERSEQUENCE']
    
    def t_FILENAME(t):
        r'[a-zA-Z_/.][a-zA-Z0-9_/.]*\.ptx'
        return t

    def t_NUMBERSEQUENCE(t):
        r'[0-9 :]+'
        return t
        
    t_ignore = '\t: '
    
    def t_newline(t):
        r'\n+'
        t.lexer.lineno += t.value.count("\n")
        
    def t_error(t):
        print "Illegal character '%s'" % t.value[0]
        t.lexer.skip(1)
        
    lex.lex()
    
    count = []
    latency = []
    organized = {}

    def p_sentence(p):
      '''sentence : FILENAME NUMBERSEQUENCE'''
      tmp1 = []
      tmp = p[2].split(':')
      for x in tmp:
        x = x.strip()
        tmp1.append(x)
      organized[int(tmp1[0])] = tmp1[1].split(' ')
          

    def p_error(p):
      if p:
          print("Syntax error at '%s'" % p.value)
          print p
      else:
          print("Syntax error at EOF")

        
    yacc.yacc()
    
    file = open(filename, 'r')
    while file:
        line = file.readline()
        if not line : break
        if (line.startswith('kernel line :')) :
            line = line.strip()
            ptxLineStatName = line.split(' ')
            ptxLineStatName = ptxLineStatName[3:]
        else: 
            yacc.parse(line[0:-1])
        
        
    return organized
  
  
def ptxToCudaMapping(filename):
  map = {}
  file = open(filename, 'r')
  bool = 0
  count = 0
  loc = 0
  while file:
    line = file.readline()
    if not line: break
    try:
      map[loc].append(count)
    except:
      map[loc] = []
      map[loc].append(count)

    m = re.search('\.loc\s+(\d+)\s+(\d+)\s+(\d+)', line)
    if (m != None):
      loc = int(m.group(2))

    count += 1
  x = map.keys()
  return map
    

#Unit test / playground
def main():
    data = textEditorParseMe(sys.argv[1])
    print data[100]
   
if __name__ == "__main__":
    main()
  
  
  
