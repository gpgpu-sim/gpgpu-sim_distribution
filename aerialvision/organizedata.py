#!/usr/bin/env python

# Copyright (C) 2009 by Aaron Ariel, Tor M. Aamodt, Wilson W. L. Fung
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

import os
import array
#from numpy import array
import numpy
import lexyacctexteditor
import variableclasses as vc

global convertCFLog2CUDAsrc
global skipCFLog

convertCFLog2CUDAsrc = 0
skipCFLog = 1 

CFLOGInsnInfoFile = ''
CFLOGptxFile = ''
# Obtain the files required to parse CFLOG files from the source code view tab input
def setCFLOGInfoFiles(sourceViewFileList):

    global CFLOGInsnInfoFile 
    global CFLOGptxFile

    if CFLOGInsnInfoFile == '' and len(sourceViewFileList[2]) > 0:
        CFLOGInsnInfoFile = sourceViewFileList[2][0]
    if CFLOGptxFile == '' and len(sourceViewFileList[1]) > 0:
        CFLOGptxFile = sourceViewFileList[1][0]

def organizedata(fileVars):

    organizeFunction = {
        'scalar':OrganizeScalar,        # Scalar data
        'impVec':nullOrganizedShader,   # Implicit vector data for multiple units (used by Shader Core stats)
        'stackbar':nullOrganizedStackedBar, # Stacked bars 
        'idxVec':nullOrganizedDram,     # Vector data with index  (used by DRAM stats)
        'idx2DVec':nullOrganizedDramV2, # Vector data with 2D index  (used by DRAM access stats)
        'sparse':OrganizeSparse,        # Vector data with 2D index  (used by DRAM access stats)
        'custom':0
    }
    data_type_char = {int:'I', float:'f'}

    print "Organizing data into internal format..."

    # Organize globalCycle in advance because it is used as a reference
    if ('globalCycle' in fileVars):
        statData = fileVars['globalCycle']
        fileVars['globalCycle'].data = organizeFunction[statData.organize](statData.data, data_type_char[statData.datatype])

    # Organize other stat data into internal format
    for statName, statData in fileVars.iteritems():
        if (statName != 'CFLOG' and statName != 'globalCycle' and statData.organize != 'custom'):
            fileVars[statName].data = organizeFunction[statData.organize](statData.data, data_type_char[statData.datatype])
  
    # Custom routines to organize stat data into internal format
    if fileVars.has_key('averagemflatency'):
        zeros = []
        for count in range(len(fileVars['averagemflatency'].data),len(fileVars['globalCycle'].data)):
            zeros.append(0)
        fileVars['averagemflatency'].data = zeros + fileVars['averagemflatency'].data

    if (skipCFLog == 0) and fileVars.has_key('CFLOG'):
        ptxFile = CFLOGptxFile
        statFile = CFLOGInsnInfoFile
        
        print "PC Histogram to CUDA Src = %d" % convertCFLog2CUDAsrc
        parseCFLOGCUDA = convertCFLog2CUDAsrc

        if parseCFLOGCUDA == 1:
            print "Obtaining PTX-to-CUDA Mapping from %s..." % ptxFile
            map = lexyacctexteditor.ptxToCudaMapping(ptxFile.rstrip())
            print "Obtaining Program Range from %s..." % statFile
            maxStats = max(lexyacctexteditor.textEditorParseMe(statFile.rstrip()).keys())

        if parseCFLOGCUDA == 1:
            newMap = {}
            for lines in map:
                for ptxLines in map[lines]:
                    newMap[ptxLines] = lines
            print "    Total number of CUDA src lines = %s..." % len(newMap)
            
            markForDel = []
            for ptxLines in newMap:
                if ptxLines > maxStats:
                    markForDel.append(ptxLines)
            for lines in markForDel:
                del newMap[lines]
            print "    Number of touched CUDA src lines = %s..." % len(newMap)
    
        fileVars['CFLOGglobalPTX'] = vc.variable('',2,0)
        fileVars['CFLOGglobalCUDA'] = vc.variable('',2,0)
        
        count = 0
        for iter in fileVars['CFLOG']:

            print "Organizing data for %s" % iter

            fileVars[iter + 'PTX'] = fileVars['CFLOG'][iter]
            fileVars[iter + 'PTX'].data = CFLOGOrganizePTX(fileVars['CFLOG'][iter].data, fileVars['CFLOG'][iter].maxPC)
            if parseCFLOGCUDA == 1:
                fileVars[iter + 'CUDA'] = vc.variable('',2,0)
                fileVars[iter + 'CUDA'].data = CFLOGOrganizeCuda(fileVars[iter + 'PTX'].data, newMap)

            try:
                if count == 0:
                    fileVars['CFLOGglobalPTX'] = fileVars[iter + 'PTX']
                    if parseCFLOGCUDA == 1:
                        fileVars['CFLOGglobalCUDA'] = fileVars[iter + 'CUDA']
                else:
                    for rows in range(0, len(fileVars[iter + 'PTX'].data)):
                        for columns in range(0, len(fileVars[iter + 'PTX'].data[rows])):
                            fileVars['CFLOGglobalPTX'].data[rows][columns] += fileVars[iter + 'PTX'].data[rows][columns]
                    if parseCFLOGCUDA == 1:
                        for rows in range(0, len(fileVars[iter + 'CUDA'].data)):
                            for columns in range(0, len(fileVars[iter + 'CUDA'].data[rows])): 
                                fileVars['CFLOGglobalCUDA'].data[rows][columns] += fileVars[iter + 'CUDA'].data[rows][columns]
            except:
                print "Error in generating globalCFLog data"

            count += 1
        del fileVars['CFLOG']


    return fileVars

def OrganizeScalar(data, datatype_c):
    organized = [0] + data;
    organized = array.array(datatype_c, organized)
    return organized;

def nullOrganizedShader(nullVar, datatype_c):
    #need to organize this array into usable information
    count = 0
    organized = []
    
    #determining how many shader cores are present
    for x in reversed(nullVar):
        if x != 'NULL':
            count += 1
        elif count != 0:
            break
    numPlots = count
    count = 0
    
    #initializing 2D list
    for x in range(0, numPlots):
        organized.append(array.array(datatype_c, [0]))
    
    #filling up list appropriately
    for x in range(0,(len(nullVar))):
        if nullVar[x] == 'NULL':
            while count < numPlots:
                organized[count].append(0)
                count += 1
            count=0
        else:
            organized[count].append(nullVar[x])
            count += 1

    #for x in range(0,len(organized)):
    #    organized[x] = [0] + organized[x]
    
    return organized

def nullOrganizedStackedBar(nullVar, datatype_c):
    organized = nullOrganizedShader(nullVar, datatype_c)

    # group data points to improve display speed
    if len(organized[0]) > 512:
        n_data = len(organized[0]) // 512 + 1 
        newLen = 512
        for row in range (0,len(organized)):
            newy = array.array(datatype_c, [0 for col in range(newLen)])
            for col in range(0, len(organized[row])):
                newcol = col / n_data
                newy[newcol] += organized[row][col]
            for col in range(0, len(newy)):
                newy[col] /= n_data 
            organized[row] = newy

    return organized
    
def nullOrganizedDram(nullVar, datatype_c):
    organized = [array.array(datatype_c, [0])]
    mem = 1
    for iter in nullVar:
        if iter == 'NULL':
            mem = 1
            continue
        elif mem == 1:
            memNum = iter
            mem = 0
            continue
        else:
            try:
                organized[memNum].append(iter)
            except:
                organized.append(array.array(datatype_c, [0]))
                organized[memNum].append(iter)
    return organized

def nullOrganizedDramV2(nullVar, datatype_c):
    organized = {}
    mem = 1
    for iter in nullVar:
        if iter == 'NULL':
            mem = 1
            continue
        elif mem == 1:
            ChipNum = iter
            mem += 1
            continue
        elif mem == 2:
            BankNum = iter
            mem = 0
            continue
        else:
            try:
                key = str(ChipNum) + '.' + str(BankNum)
                organized[key].append(iter)
            except:
                organized[key] = array.array(datatype_c, [0])
                organized[key].append(iter)

    return organized

def OrganizeSparse(variable, datatype_c):
    data = numpy.array(variable[0], dtype=numpy.int32)
    row = numpy.array(variable[1], dtype=numpy.int32)
    col = numpy.array(variable[2], dtype=numpy.int32)
    del variable[0:]
    #organized = sparse.coo_matrix((data, (row, col)))
    organized = [data, row, col]

    return organized

def CFLOGOrganizePTX(list, maxPC):
    count = 0
    
    organizedThreadCount = list[1]
    organizedPC = list[0]

    nCycles = len(organizedPC)
    final_template = [0 for cycle in range(nCycles)]
    final = [array.array('I', final_template) for pc in range(maxPC + 1)] # fill the 2D array with zeros

    for cycle in range(0, nCycles):
        pcList = organizedPC[cycle]
        threadCountList = organizedThreadCount[cycle] 
        for n in range(0, len(pcList)):
            final[pcList[n]][cycle] = threadCountList[n]
    
    return final

def CFLOGOrganizeCuda(list, ptx2cudamap):
    #We need to aggregate lines of PTX together
    cudaMaxLineNo = max(ptx2cudamap.keys())
    tmp = {}
    #need to fill up the final matrix appropriately

    nSamples = len(list[0])

    # create a dictionary of empty data array (one array per cuda source line)
    for ptxline, cudaline in ptx2cudamap.iteritems():
        if tmp.has_key(cudaline):
            pass
        else:
            tmp[cudaline] = [0 for lengthData in range(nSamples)]


    for cudaline in tmp:
        for ptxLines, mapped_cudaline in ptx2cudamap.iteritems():
            if mapped_cudaline == cudaline:
                for lengthData in range(nSamples):
                    tmp[cudaline][lengthData] += list[ptxLines][lengthData]

    
    final = []           
    for iter in range(min(tmp.keys()),max(tmp.keys())):
        if tmp.has_key(iter):
            final.append(tmp[iter])            
        else:
            final.append([0 for lengthData in range(nSamples)])

    return final
                

#def stackedBar(nullVar):
#    #Need to initialize organize ar
#    organized = [[]]
#    for iter in nullVar:
#        if iter != 'NULL':
#            organized[-1].append(iter)
#        else:
#            organized.append([])
#    organized.remove([])
#    return organized


