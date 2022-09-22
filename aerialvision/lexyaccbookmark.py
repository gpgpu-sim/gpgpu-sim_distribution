#!/usr/bin/env python

# Copyright (C) 2009 by Aaron Ariel, Tor M. Aamodt and the University of British 
# Columbia, Vancouver, 
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


import os
import sys
import ply.lex as lex
import ply.yacc as yacc
import variableclasses as vc

def parseMe():
  

  
    #The lexer
        
    # List of token names.   This is always required
    tokens = ['WORD',
        'EQUALS',
        'VALUE',
        'NUMBER',
        'NOTHING'
    ]
    
    # Regular expression rules for tokens
    
    def t_VALUE(t):
        r'["][a-zA-Z()0-9\._ ]+["]'
        return t
    
    def t_EQUALS(t):
        r'[=][ ]'
        return t
    
    def t_WORD(t):
        r'[a-zA-Z_]+[ ]'
        return t

    def t_NUMBER(t):
        r'["][\d]+["]'
        return t
    
    def t_NOTHING(t):
        r'["]["]'
        return t
        
    t_ignore = '[\n]+'
    

        
    def t_error(t):
        print "Illegal character '%s'" % t.value[0]
        t.lexer.skip(1) 
    
    lex.lex()    
    
    listBookmarks = []

        
    def p_sentence(p):

        '''sentence : WORD EQUALS VALUE 
                    | WORD EQUALS NUMBER
                    | WORD EQUALS NOTHING'''
        p[1] = p[1][0:-1]
        p[3] = p[3][1:-1]

        if p[1] == 'title':
            listBookmarks[-1].title = p[3]

        elif p[1] == 'description':
            listBookmarks[-1].description = p[3]

        elif p[1] == 'dataChosenX':
            listBookmarks[-1].dataChosenX.append(p[3])
            
        elif p[1] == 'dataChosenY':
            listBookmarks[-1].dataChosenY.append(p[3])

        elif p[1] == 'graphChosen':
            listBookmarks[-1].graphChosen.append(p[3])

            
        elif p[1] == 'dydx':
            listBookmarks[-1].dydx.append(p[3])
        
        elif p[1] == 'START':
            listBookmarks.append(vc.bookmark())
        
        elif p[1] == 'ReasonForFile':
            pass

        else:
            print 'An Parsing Error has occurred'
            

    



    def p_error(p):
        if p:
            print("Syntax error at '%s'" % p.value)
        else:
            print("Syntax error at EOF")
    
    yacc.yacc()
   
    try:
        file = open(os.environ['HOME'] + '/.gpgpu_sim/aerialvision/bookmarks.txt', 'r')
        inputData = file.readlines()
    except IOError,e:
        if e.errno == 2:
            inputData = ''
        else:
            raise e

    for x in inputData:
        yacc.parse(x[0:-1]) # ,debug=True)
        
    return listBookmarks
