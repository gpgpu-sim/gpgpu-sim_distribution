#!/usr/bin/env python

# Copyright (C) 2009 by Aaron Ariel, Tor M. Aamodt, Andrew Turner, Wilson W. L.
# Fung, Zev Weiss and the University of British Columbia, Vancouver, 
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
import Tkinter as Tk
import Pmw
import lexyacc
import guiclasses
import tkFileDialog as Fd
import organizedata
import os
import os.path


global TabsForGraphs
global Filenames
global TabsForText
global SourceCode

Filenames = []
TabsForGraphs = []
vars = {}
TabsForText = []

userSettingPath = os.path.join(os.environ['HOME'], '.gpgpu_sim', 'aerialvision')

def checkEmpty(list):
    bool = 0
    try:
        if type(list[0]).__name__ == 'list':
            for iter in list:
                for x in iter:
                    if ((int(x) != 0) and (x != 'NULL')):
                        bool = 1
            return bool
        else:
            for x in list:
                if ((x != 'NULL') and (int(x) != 0)):
                    bool = 1
            return bool
    except:
        for x in list:
            if ((int(x) != 0) and (x != 'NULL')):
                bool = 1
        return bool

def fileInput(cl_files=None):
    # The Main Window Stuff
    
    # Instantiate the window
    instance = Tk.Tk();
    instance.title("File Input")
    #set the window size
    root = Tk.Frame(instance, width = 1100, height = 550, bg = 'white');
    root.pack_propagate(0);
    root.pack();

    #Title at top of Page
    rootTitle = Tk.Label(root, text='AerialVision 1.0', font = ("Gill Sans MT", 20, "bold"), bg = 'white');
    rootTitle.pack(side = Tk.TOP);
    fileInputTitle = Tk.Label(root, text='Please Fill Out Specifications \n to Get Started', font = ("Gill Sans MT", 15, "bold", "underline"), bg = 'white', width = 400)
    fileInputTitle.pack(side = Tk.TOP)
    
    inputTabs = Pmw.NoteBook(root)
    inputTabs.pack(fill = 'both', expand = 'True')
    
    fileInputOuter = inputTabs.add('File Inputs for Visualizer')
    fileInputTextEditor = inputTabs.add('File Inputs for Source Code Viewer')

    
    
    #################### The visualizer side #############################3
    
    
    fileInput = Tk.Frame(fileInputOuter, bg = 'white', borderwidth = 5, relief = Tk.GROOVE)
    fileInput.pack()
    
    specChoices = Tk.Frame(fileInput, bg= 'white')
    specChoices.pack(side = Tk.TOP, anchor = Tk.W, pady = 30)
    addFile = Tk.Frame(specChoices, bg = 'white')
    addFile.pack(side = Tk.TOP, anchor = Tk.W)
    lAddFile = Tk.Label(addFile, text= "Add Input File: ", bg= 'white')
    lAddFile.pack(side = Tk.LEFT)
    eAddFile = Tk.Entry(addFile, width= 30, bg = 'white')
    eAddFile.pack(side = Tk.LEFT)
    bClearEntry = Tk.Button(addFile, text = "Clear", command = (lambda: clearField(eAddFile)))
    bClearEntry.pack(side = Tk.LEFT)

    bAddFileSubmit = Tk.Button(addFile, text = "Add File", command = (lambda: addToListbox(cFilesAdded, eAddFile.get(),eAddFile)))
    bAddFileSubmit.pack(side = Tk.LEFT)
    
    
    
    #Loading the most recent directory visited as the first directory
    try:
        loadfile = open(os.path.join(userSettingPath, 'recentfiles.txt'), 'r')
        tmprecentfile = loadfile.readlines()
        recentfile = ''  
        tmprecentfile = tmprecentfile[len(tmprecentfile) -1]
        tmprecentfile = tmprecentfile.split('/')
        for iter in range(1,len(tmprecentfile) - 1):
            recentfile = recentfile + '/' + tmprecentfile[iter]
    except IOError,e:
        if e.errno == 2:
            # recentfiles.txt does not exist, ignore and use CWD
            recentfile = '.'
        else:
            raise e

    bAddFileBrowse = Tk.Button(addFile, text = "Browse", command = (lambda: eAddFile.insert(0,Fd.askopenfilename(initialdir=recentfile ))))
    bAddFileBrowse.pack(side = Tk.LEFT)
    bAddFileRecentFiles = Tk.Button(addFile, text = "Recent Files", command = (lambda: loadRecentFile(eAddFile)))
    bAddFileRecentFiles.pack(side = Tk.LEFT)
    filesAdded = Tk.Frame(specChoices, bg = 'white')
    filesAdded.pack(side = Tk.TOP, anchor = Tk.W)
    lFilesAdded = Tk.Label(filesAdded, text = "Files Added: ", bg = 'white')
    lFilesAdded.pack(side = Tk.LEFT)
    cFilesAdded = Tk.Listbox(filesAdded, width = 100, height = 5, bg = 'white')
    cFilesAdded.pack(side = Tk.LEFT)
    bFilesAddedRem = Tk.Button(filesAdded, text = "Remove", command = (lambda: delFileFromListbox(cFilesAdded)))
    bFilesAddedRem.pack(side = Tk.LEFT)
    screenRes = Tk.Frame(specChoices, bg = 'white')
    screenRes.pack(side = Tk.TOP, anchor= Tk.W)
    lScreenResolution = Tk.Label(screenRes, text = "Choose Closest Screen Resolution: ", bg = 'white')
    lScreenResolution.pack(side = Tk.LEFT)
    Modes = [("1024 x 768", "1"), ("1600 x 1200", "3") ]
    num = Tk.StringVar()
    num.set("3")
    for text, mode in Modes:
        bRes = Tk.Radiobutton(specChoices, text = text, variable = num, value = mode, bg = 'white')
        bRes.pack(anchor = Tk.W)

    # add files specified on command line to listbox
    if cl_files:
        #print "adding", cl_files
        addListToListbox(cFilesAdded,cl_files)
        recentfile = cl_files[len(cl_files) -1]

    # check box to skip parsing of CFLog 
    skipCFLogVar = Tk.IntVar()
    skipCFLogVar.set(1)
    cbSkipCFLog = Tk.Checkbutton(specChoices, text = "Skip CFLog parsing", bg = 'white', variable = skipCFLogVar)
    cbSkipCFLog.pack(side = Tk.LEFT)
    # check box to activate converting CFLog to CUDA source line
    cflog2cuda = Tk.IntVar()
    cflog2cuda.set(0)
    cbCFLog2CUDAsrc = Tk.Checkbutton(specChoices, text = "Convert CFLog to CUDA source line", bg = 'white', variable = cflog2cuda)
    cbCFLog2CUDAsrc.pack(side = Tk.LEFT)
    
    ############### The text editor side ##################################
    
    fileInputTE = Tk.Frame(fileInputTextEditor, bg = 'white', borderwidth = 5, relief = Tk.GROOVE)
    fileInputTE.pack()
    
    ###### INPUT a Text File ###############
    specChoicesTE = Tk.Frame(fileInputTE, bg= 'white')
    specChoicesTE.pack(side = Tk.TOP, anchor = Tk.W, pady = 30)
    addFileTE = Tk.Frame(specChoicesTE, bg = 'white')
    addFileTE.pack(side = Tk.TOP, anchor = Tk.W)
    lAddFileTE = Tk.Label(addFileTE, text= "Add CUDA Source Code File: ", bg= 'white')
    lAddFileTE.pack(side = Tk.LEFT)
    eAddFileTE = Tk.Entry(addFileTE, width= 30, bg = 'white')
    eAddFileTE.pack(side = Tk.LEFT)
    bClearEntryTE = Tk.Button(addFileTE, text = "Clear", command = (lambda: clearField(eAddFileTE)))
    bClearEntryTE.pack(side = Tk.LEFT)
    
    bAddFileBrowseTE = Tk.Button(addFileTE, text = "Browse", command = (lambda: eAddFileTE.insert(0,Fd.askopenfilename(initialdir=recentfile ))))
    bAddFileBrowseTE.pack(side = Tk.LEFT)
    bAddFileRecentFilesTE = Tk.Button(addFileTE, text = "Recent Files", command = (lambda: loadRecentFile(eAddFileTE)))
    bAddFileRecentFilesTE.pack(side = Tk.LEFT)

    
    ### Input Corresponding PTX file ###########
    addFileTEPTX = Tk.Frame(specChoicesTE, bg = 'white')
    addFileTEPTX.pack(side = Tk.TOP, anchor = Tk.W)
    lAddFileTEPTX = Tk.Label(addFileTEPTX, text= "Add Corresponding PTX File: ", bg= 'white')
    lAddFileTEPTX.pack(side = Tk.LEFT)
    eAddFileTEPTX = Tk.Entry(addFileTEPTX, width= 30, bg = 'white')
    eAddFileTEPTX.pack(side = Tk.LEFT)
    bClearEntryTEPTX = Tk.Button(addFileTEPTX, text = "Clear", command = (lambda: clearField(eAddFileTEPTX)))
    bClearEntryTEPTX.pack(side = Tk.LEFT)
    
    bAddFileBrowseTEPTX = Tk.Button(addFileTEPTX, text = "Browse", command = (lambda: eAddFileTEPTX.insert(0,Fd.askopenfilename(initialdir=recentfile )))) #"/home/taamodt/fpga_simulation/run/"
    bAddFileBrowseTEPTX.pack(side = Tk.LEFT)
    bAddFileRecentFilesTEPTX = Tk.Button(addFileTEPTX, text = "Recent Files", command = (lambda: loadRecentFile(eAddFileTEPTX)))
    bAddFileRecentFilesTEPTX.pack(side = Tk.LEFT)
    
    lNote = Tk.Label(addFileTEPTX, text = '*Must include at least PTX and Stat files before pressing the submit button', bg = 'white')
    lNote.pack(side = Tk.LEFT)
        
                  

    #### Input the corresponding stat file ##################
    addFileTEStat = Tk.Frame(specChoicesTE, bg = 'white')
    addFileTEStat.pack(side = Tk.TOP, anchor = Tk.W)
    lAddFileTEStat = Tk.Label(addFileTEStat, text= "Add Corresponding Stat File: ", bg= 'white')
    lAddFileTEStat.pack(side = Tk.LEFT)
    eAddFileTEStat = Tk.Entry(addFileTEStat, width= 30, bg = 'white')
    eAddFileTEStat.pack(side = Tk.LEFT)
    bClearEntryTEStat = Tk.Button(addFileTEStat, text = "Clear", command = (lambda: clearField(eAddFileTEStat)))
    bClearEntryTEStat.pack(side = Tk.LEFT)
    
    bAddFileBrowseTEStat = Tk.Button(addFileTEStat, text = "Browse", command = (lambda: eAddFileTEStat.insert(0,Fd.askopenfilename(initialdir=recentfile )))) #"/home/taamodt/fpga_simulation/run/"
    bAddFileBrowseTEStat.pack(side = Tk.LEFT)
    bAddFileRecentFilesTEStat = Tk.Button(addFileTEStat, text = "Recent Files", command = (lambda: loadRecentFile(eAddFileTEStat)))
    bAddFileRecentFilesTEStat.pack(side = Tk.LEFT)
    
    bAddFileSubmitTEStat = Tk.Button(addFileTEStat, text = "Add Files", command = lambda: addToListboxTE([cFilesAddedTE,cFilesAddedTEPTX, cFilesAddedTEStat],
        [eAddFileTE,eAddFileTEPTX, eAddFileTEStat]), bg = 'green')
    bAddFileSubmitTEStat.pack(side = Tk.LEFT)
    
    #### Display text file Chosen and stat file Chosen ###########
    
    #TEXT FILES CHOSEN
    filesAddedTE = Tk.Frame(specChoicesTE, bg = 'white')
    filesAddedTE.pack(side = Tk.TOP, anchor = Tk.W)
    lFilesAddedTE = Tk.Label(filesAddedTE, text = "CUDA Source Code File Added: ", bg = 'white')
    lFilesAddedTE.pack(side = Tk.LEFT)
    cFilesAddedTE = Tk.Listbox(filesAddedTE, width = 100, height = 3, bg = 'white')
    cFilesAddedTE.pack(side = Tk.LEFT)
    
    #Corresponding PTX File Chosen
    filesAddedTEPTX = Tk.Frame(specChoicesTE, bg = 'white')
    filesAddedTEPTX.pack(side = Tk.TOP, anchor = Tk.W)
    lFilesAddedTEPTX = Tk.Label(filesAddedTEPTX, text = "Corresponding PTX Files Added: ", bg = 'white')
    lFilesAddedTEPTX.pack(side = Tk.LEFT)
    cFilesAddedTEPTX = Tk.Listbox(filesAddedTEPTX, width = 100, height = 3, bg = 'white')
    cFilesAddedTEPTX.pack(side = Tk.LEFT, padx = 15)
    bFilesAddedRemTE = Tk.Button(filesAddedTE, text = "Remove", command = (lambda: delFileFromListbox(cFilesAdded)))
    bFilesAddedRemTE.pack(side = Tk.LEFT)
    
    
    #CORRESPONDING STAT FILES CHOSEN
    filesAddedTEStat = Tk.Frame(specChoicesTE, bg = 'white')
    filesAddedTEStat.pack(side = Tk.TOP, anchor = Tk.W)
    lFilesAddedTEStat = Tk.Label(filesAddedTEStat, text = "Corresponding Stat Files Added: ", bg = 'white')
    lFilesAddedTEStat.pack(side = Tk.LEFT)
    cFilesAddedTEStat = Tk.Listbox(filesAddedTEStat, width = 100, height = 3, bg = 'white')
    cFilesAddedTEStat.pack(side = Tk.LEFT, padx = 15)
    
    
    bSUBMIT = Tk.Button(root, text = "Submit", font = ("Gill Sans MT", 12, "bold"), width = 10, command = lambda: submitClicked(instance, num.get(), skipCFLogVar.get(), cflog2cuda.get(), [cFilesAddedTE, cFilesAddedTEPTX, cFilesAddedTEStat]))
    bSUBMIT.pack(pady = 5)
    
    
    instance.mainloop()

    
def loadRecentFile(entry):
    instance = Tk.Toplevel(bg = 'white')
    instance.title("Recent Files")
   
    try: 
        loadfile = open(os.path.join(userSettingPath, 'recentfiles.txt'), 'r')
        recentfiles = loadfile.readlines()
    except IOError,e:
        if e.errno == 2:
            recentfiles = ''
        else:
            raise e
    
    recentFileWindow = Tk.Frame(instance, bg = 'white')
    recentFileWindow.pack(side = Tk.TOP)
    scrollbar = Tk.Scrollbar(recentFileWindow, orient = Tk.VERTICAL)
    cRecentFile = Tk.Listbox(recentFileWindow, width = 100, height = 15, yscrollcommand = scrollbar.set)
    cRecentFile.bind("<Double-Button-1>", lambda(event): recentFileInsert(entry, cRecentFile.get('active'), instance))
    cRecentFile.pack(side = Tk.LEFT)
    scrollbar.config(command = cRecentFile.yview)
    scrollbar.pack(side = Tk.LEFT, fill = Tk.Y)
    
    tmp = []
    for x in range(len(recentfiles) - 1, 0, -1):
        try:
            tmp.index(recentfiles[x][0:-1])
            pass
        except:
            tmp.append(recentfiles[x][0:-1])
            
    for x in range(0,len(tmp)):
        cRecentFile.insert(Tk.END, tmp[x])
        
    belowRecentFileWindow = Tk.Frame(instance, bg = 'white')
    belowRecentFileWindow.pack(side = Tk.BOTTOM)
    bRecentFile = Tk.Button(belowRecentFileWindow , text = "Submit", command = lambda: recentFileInsert(entry, cRecentFile.get('active'), instance))
    bRecentFile.pack()
    bRecentFileCancel = Tk.Button(belowRecentFileWindow , text = 'Cancel', command = (lambda: instance.destroy()))
    bRecentFileCancel.pack()
    
def recentFileInsert(entry, string, window):
    window.destroy()
    entry.insert(0, string)
    
def clearField(entry):
    entry.delete(0,Tk.END)
    
    
def delFileFromListbox(filesListbox):
    for files in Filenames:
        if files[-80:] == filesListbox.get('active')[-80:]:
            Filenames.remove(files)
    filesListbox.delete(Tk.ANCHOR)

   
def addToListboxTE(listbox, entry):
    for iter in range(1,len(listbox)):
        try:
           test = open(entry[iter].get(), 'r')
        except:
            errorMsg('Could not open file ' + entry[iter].get())
            return
    
    for iter in range(0,len(listbox)):
        listbox[iter].insert(Tk.END, entry[iter].get())
        entry[iter].delete(0,Tk.END)
   
def addToListbox(listbox, string, entry):
    try:
        test = open(string, 'r')
        Filenames.append(string)
        listbox.insert(Tk.END, string)
        entry.delete(0,Tk.END)
    except:
        errorMsg('Could not open file')
        return 0
        
def addListToListbox(listbox,list):
    for file in list:
        try:
            string = os.path.abspath(file)
            if os.path.isfile(string):
                Filenames.append(string)
                listbox.insert(Tk.END, string)
            else:
                print 'Could not open file: ' + string
        except:
            print 'Could not open file: ' + file
            
        
def errorMsg(string):
    error = Tk.Toplevel(bg = 'white')
    error.title("Error Message")
    tError = Tk.Label(error, text = "Error", font = ("Gills Sans MT", 20, "underline", "bold"), bg = "red")
    tError.pack(side = Tk.TOP, pady = 20)
    lError = Tk.Label(error, text = string, font = ("Gills Sans MT", 15, "bold"), bg = 'white')
    lError.pack(pady = 10, padx = 10)
    bError = Tk.Button(error, text = "OK", font = ("Times New Roman", 14), command = (lambda: error.destroy()))
    bError.pack(pady = 10)
   
def submitClicked(instance, num, skipcflog, cflog2cuda, listboxes):
    
    for iter in range(0, len(listboxes)):
        if iter == 0:
            TEFiles = listboxes[iter].get(0, Tk.END)
        if iter == 1:
            TEPTXFiles = listboxes[iter].get(0, Tk.END)
        else:
            TEStatFiles = listboxes[iter].get(0, Tk.END)
   
    organizedata.skipCFLog = skipcflog
    lexyacc.skipCFLOGParsing = skipcflog
    organizedata.convertCFLog2CUDAsrc = cflog2cuda

    start = 0
    if (not os.path.exists(userSettingPath)):
        os.makedirs(userSettingPath)
    f_recentFiles = open(os.path.join(userSettingPath, 'recentfiles.txt'), 'a')
    for files in Filenames:
        f_recentFiles.write(files + '\n')

    for files in TEFiles:
        f_recentFiles.write(files + '\n')
    
    for files in TEPTXFiles:
        f_recentFiles.write(files + '\n')
    
    for files in TEStatFiles:
        f_recentFiles.write(files + '\n')

    f_recentFiles.close()
    if num == '1':
        res = 'small'  
    elif num == '2':
        res = 'medium'  
    else:
        res = 'big'
    instance.destroy()
    startup(res, [TEFiles, TEPTXFiles, TEStatFiles])

def graphAddTab(vars, graphTabs,res, entry):
    
    TabsForGraphs.append(guiclasses.formEntry(graphTabs, str(len(TabsForGraphs) + 1), vars, res, entry))
    entry.delete(0, Tk.END)
    entry.insert(0, 'TabTitle?')
    
def remTab(graphTabs):
    
    graphTabs.delete(Pmw.SELECT)
    
def destroy(instance, quit):
    quit.destroy()
    instance.destroy()
    

def tmpquit(instance):
    
    quit = Tk.Toplevel(bg = 'white')
    quit.title("...")
    tQuit = Tk.Label(quit, text = "Quit?", font = ("Gills Sans MT", 20, "underline", "bold"), bg = "white")
    tQuit.pack(side = Tk.TOP, pady = 20)
    lQuit = Tk.Label(quit, text = "Are you sure you want to quit?", font = ("Gills Sans MT", 15, "bold"), bg = 'white')
    lQuit.pack(side = Tk.TOP, pady = 20, padx = 10)
    bQuit = Tk.Button(quit, text = "Yes", font = ("Time New Roman", 13), command = (lambda: destroy(instance, quit)))
    bQuit.pack(side = Tk.LEFT, anchor = Tk.W, pady = 5, padx = 5)
    bNo = Tk.Button(quit, text = "No", font= ("Time New Roman", 13), command = (lambda: quit.destroy()))
    bNo.pack(side = Tk.RIGHT, pady = 5, padx = 5)
    
    
def startup(res, TEFILES):
    global vars
    # The Main Window Stuff
    
    # Instantiate the window
    instance = Tk.Tk();
    instance.title("AerialVision GPU Graphing Tool")
    

    #set the window size
    if res == 'small':
        root = Tk.Frame(instance, width = 1325, height = 850, bg = 'white');
    elif res == 'medium':
        root = Tk.Frame(instance, width = 1700, height = 1100, bg = 'white');
    else:
        root = Tk.Frame(instance, width = 1700, height = 1100, bg = 'white');
    root.pack_propagate(0);
    root.pack();

    # User can choose between a text editor or a visualizer
    chooseTextVisualizer = Pmw.NoteBook(root)
    chooseTextVisualizer.pack(fill= 'both', expand = 'true')
    
    visualizer = chooseTextVisualizer.add('Visualizer')
    textEditor = chooseTextVisualizer.add('Source Code View')
    

    #INITIALIZING THE VISUALIZER

    #The top frame for the control panel
    # Frame for Control Panel
    if res == 'small':
        controlPanel = Tk.Frame(visualizer, width=1250, height= 50, bg ="beige", borderwidth = 5, relief = Tk.GROOVE);
    elif res == 'medium':
        controlPanel = Tk.Frame(visualizer, width=1530, height= 50, bg ="beige", borderwidth = 5, relief = Tk.GROOVE);
    else:
        controlPanel = Tk.Frame(visualizer, width=1530, height= 50, bg ="beige", borderwidth = 5, relief = Tk.GROOVE);
    controlPanel.pack(anchor = Tk.N, pady = 5);
    controlPanel.pack_propagate(0)

    # Control Panel Title
    controlTitle = Tk.Frame(controlPanel, bg = 'beige')
    controlTitle.pack(side = Tk.LEFT)
    lControlTitle = Tk.Label(controlTitle, text='Control Panel:     ', font = ("Gills Sans MT", 15, "bold"), bg = "beige");
    lControlTitle.pack(side = Tk.LEFT)
    
    #Number of Tabs Frame)
    numbTabs = Tk.Frame(controlPanel, bg = 'beige')
    numbTabs.pack(side = Tk.LEFT)
    eAddTab = Tk.Entry(numbTabs)
    bRemTab = Tk.Button(numbTabs, text = "Rem Tab", command = (lambda: remTab(graphTabs)), bg = 'red')
    bRemTab.pack(side=Tk.LEFT)
    bAddTab = Tk.Button(numbTabs, text = "Add Tab", command = (lambda: graphAddTab(vars, graphTabs, res, eAddTab)))
    bAddTab.pack(side=Tk.LEFT)
    eAddTab.pack(side = Tk.LEFT)
    eAddTab.insert(0, 'TabTitle?')
    bManageFiles = Tk.Button(numbTabs, text = "Manage Files", command = lambda: manageFiles())
    bManageFiles.pack(side = Tk.LEFT)
    
    #Quit or Open up new Window Frame
    quitNew = Tk.Frame(controlPanel, bg = 'beige')
    quitNew.pack(side = Tk.RIGHT, padx = 10)
    bQuit = Tk.Button(quitNew, text = "Quit", bg = 'red', command = (lambda: tmpquit(instance)))
    bQuit.pack(side = Tk.LEFT)


    #The bottom Frame that contains tabs,graphs,etc...


    #Instantiating the Main frame
    #Frame for Graphing Area
    if res == 'small':
        graphMainFrame = Tk.Frame(visualizer, width = 1250, height = 750, borderwidth = 5, relief = Tk.GROOVE);
    elif res == 'medium':
        graphMainFrame = Tk.Frame(visualizer, width = 1615, height = 969, borderwidth = 5, relief = Tk.GROOVE);
    else:
        graphMainFrame = Tk.Frame(visualizer, width = 1615, height = 969, borderwidth = 5, relief = Tk.GROOVE);
    graphMainFrame.pack(pady = 5);
    graphMainFrame.pack_propagate(0);
    
    #Setting up the Tabs
    graphTabs = Pmw.NoteBook(graphMainFrame)
    graphTabs.pack(fill = 'both', expand = 'true')
    #Class newTab will take "graphTabs" which is the widget on top of graphMainFrame and create a new tab
    #for every instance of the class
    
    # Here we extract the available data that can be graphed by the user

    for files in Filenames:
        vars[files] = lexyacc.parseMe(files)
    
    markForDel = {}
    
     
    for files in vars:
        markForDel[files] = []
        for variables in vars[files]:
            if variables == 'CFLOG':
                continue
            if variables == 'EXTVARS':
                continue
            if checkEmpty(vars[files][variables].data) == 0:
                markForDel[files].append(variables)

    for files in markForDel:
        for variables in markForDel[files]:
            del vars[files][variables]

    organizedata.setCFLOGInfoFiles(TEFILES)
    for files in Filenames:
        vars[files] = organizedata.organizedata(vars[files])

    graphAddTab(vars, graphTabs, res, eAddTab)


    # INITIALIZING THE TEXT EDITOR
    
    if res == 'small':
        textControlPanel = Tk.Frame(textEditor, width = 1250, height = 50, bg = 'beige', borderwidth =  5, relief = Tk.GROOVE)
    elif res == 'medium':
        textControlPanel = Tk.Frame(textEditor, width = 1530, height = 50, bg = 'beige', borderwidth =  5, relief = Tk.GROOVE)
    else:
        textControlPanel = Tk.Frame(textEditor, width = 1530, height = 50, bg = 'beige', borderwidth =  5, relief = Tk.GROOVE)    
   
   
    textControlPanel.pack(anchor = Tk.N, pady = 5)
    textControlPanel.pack_propagate(0)
    
    lTextControlPanel = Tk.Label(textControlPanel, text = 'Control Panel:     ', font = ("Gills Sans MT", 15, "bold"), bg = "beige")
    lTextControlPanel.pack(side = Tk.LEFT)
    bTextRemTab = Tk.Button(textControlPanel, text = 'Rem Tab', command = (lambda: textRemTab(textTabs)), bg = 'red')
    bTextRemTab.pack(side = Tk.LEFT)
    bTextAddTab = Tk.Button(textControlPanel, text = 'AddTab', command = (lambda: textAddTab(textTabs,res, TEFILES)))
    bTextAddTab.pack(side= Tk.LEFT)
    bTextManageFiles = Tk.Button(textControlPanel, text = 'Manage Files', command = (lambda: textManageFiles()))
    bTextManageFiles.pack(side = Tk.LEFT)
    
    
    
    #Quit or Open up new Window Frame
    textquitNew = Tk.Frame(textControlPanel, bg = 'beige')
    textquitNew.pack(side = Tk.RIGHT, padx = 10)
    bTextQuit = Tk.Button(textquitNew, text = "Quit", bg = 'red', command = (lambda: tmpquit(instance)))
    bTextQuit.pack(side = Tk.LEFT)
    
    if res == 'small':
        textMainFrame = Tk.Frame(textEditor, width = 1250, height = 750, borderwidth = 5, relief = Tk.GROOVE)
    elif res == 'medium':
        textMainFrame = Tk.Frame(textEditor, width = 1615, height = 969, borderwidth = 5, relief = Tk.GROOVE)
    else:
        textMainFrame = Tk.Frame(textEditor, width = 1615, height = 969, borderwidth = 5, relief = Tk.GROOVE)
        
    textMainFrame.pack(pady = 5)
    textMainFrame.pack_propagate(0)
    
    textTabs = Pmw.NoteBook(textMainFrame)
    textTabs.pack(fill = 'both', expand = 'true')
    
    textAddTab(textTabs, res, TEFILES)
    
    instance.mainloop()
    
def textManageFiles():
    textManageFiles = Tk.Toplevel(bg = 'white')
    title = Tk.Label(textManageFiles, text = 'Manage Files', font = ("Gill Sans MT", 15, "bold", "underline"), bg= 'white' )
    title.pack(side = Tk.TOP)

    bottomFrameMaster = Tk.Frame(textManageFiles, bg= 'white')
    bottomFrameMaster.pack(side = Tk.TOP, padx = 20, pady = 20)
    bottomFrame1 = Tk.Frame(bottomFrameMaster, bg= 'white')
    bottomFrame1.pack(side = Tk.LEFT, padx = 20, pady = 20)
    bottomFrameOption = Tk.Frame(bottomFrameMaster, bg = 'white')
    bottomFrameOption.pack(side = Tk.LEFT)
    ltextCurrentFiles = Tk.Label(bottomFrame1, text= 'Current Files:     ', bg = 'white')
    ltextCurrentFiles.pack(side = Tk.LEFT)
    
    bottomFrame2 = Tk.Frame(textManageFiles, bg= 'white')
    bottomFrame2.pack(side = Tk.TOP, anchor= Tk.W)
    ctextCurrentFiles = Tk.Listbox(bottomFrame1, width = 100)
    ctextCurrentFiles.pack(side = Tk.LEFT)
    lSubmittedChanges= Tk.Label(bottomFrame2, text='Changes:  ', bg = 'white')
    lSubmittedChanges.pack(side = Tk.LEFT, padx=35, pady =15)
    cSubmittedChanges = Tk.Listbox(bottomFrame2, width = 100)
    cSubmittedChanges.pack(side = Tk.LEFT,pady = 15)
    for files in Filenames:
        ctextCurrentFiles.insert(Tk.END, files)
        
    btextAddFile = Tk.Button(bottomFrameOption, text = 'Add File', command = lambda: textAddFile(bottomFrameMaster,cSubmittedChanges, textManageFiles))
    btextAddFile.pack()
    btextRemFile = Tk.Button(bottomFrameOption, text= 'Remove File')
    btextRemFile.pack(side = Tk.LEFT)
    
    
    
    
def textAddFile(frame, listbox, master):
    addFileFrame = Tk.Frame(frame, bg = 'white')
    addFileFrame.pack(side = Tk.RIGHT,padx = 15)
    
    topFrame = Tk.Frame(addFileFrame, bg= 'white')
    topFrame.pack(side = Tk.TOP)
    bottomFrame = Tk.Frame(addFileFrame, bg = 'white')
    bottomFrame.pack(side = Tk.TOP)
    
    lSourceCode = Tk.Label(topFrame, text = 'Source Code File', bg = 'white')
    lSourceCode.pack(side = Tk.LEFT)
    
    eSourceCode = Tk.Entry(topFrame)
    eSourceCode.pack(side = Tk.LEFT)
    
    bSourceCodeClearEntry = Tk.Button(topFrame, text = "Clear", command = lambda: (eSourceCode.delete(0, Tk.END)))
    bSourceCodeClearEntry.pack(side = Tk.LEFT)
    
    bSourceCodeAddFileBrowse = Tk.Button(topFrame, text = "Browse", command = (lambda: eSourceCode.insert(0,Fd.askopenfilename())))
    bSourceCodeAddFileBrowse.pack(side = Tk.LEFT)
    
    bSourceCodeAddFileRecentFiles = Tk.Button(topFrame, text = "Recent Files", command = (lambda: loadRecentFile(eAddFile)))
    bSourceCodeAddFileRecentFiles.pack(side = Tk.LEFT)
    
   
    
    
    lStatFile = Tk.Label(bottomFrame, text= 'Corresponding Stat File', bg = 'white')
    lStatFile.pack(side = Tk.LEFT)

    eStatFile = Tk.Entry(bottomFrame)
    eStatFile.pack(side = Tk.LEFT)
    
    bSourceCodeClearEntry = Tk.Button(bottomFrame, text = "Clear", command = lambda: (eStatFile.delete(0, Tk.END)))
    bSourceCodeClearEntry.pack(side = Tk.LEFT)
    
    bSourceCodeAddFileBrowse = Tk.Button(bottomFrame, text = "Browse", command = (lambda: eStatFile.insert(0,Fd.askopenfilename())))
    bSourceCodeAddFileBrowse.pack(side = Tk.LEFT)
    
    bSourceCodeAddFileRecentFiles = Tk.Button(bottomFrame, text = "Recent Files", command = (lambda: loadRecentFile(eAddFile)))
    bSourceCodeAddFileRecentFiles.pack(side = Tk.LEFT)    

    bSubmit = Tk.Button(addFileFrame, text = "Submit", command = lambda: sourceCodeAddFileSubmit(eSourceCode, eStatFile, listbox, master))
    bSubmit.pack(side = Tk.BOTTOM)
    
def sourceCodeAddFileSubmit(eSourceCode, eStatFile, listbox, frame):
    source = open(eSourceCode.get(), 'r')
    stat = open(eStatFile.get(), 'r')
    SourceCode[eSourceCode.get()] = [source, stat]
    frame.destroy()

    
    
    
def textAddTab(textTabs,res, TEFILES):
    TabsForText.append(guiclasses.newTextTab(textTabs, str(len(TabsForText) + 1), res, TEFILES))
    
def textRemTab(textTabs):
    textTabs.delete(Pmw.SELECT)
    

def manageFiles():
    manageFilesWindow = Tk.Toplevel(bg = 'white')
    manageFilesWindow.title("Manage Files")
    titleFrame = Tk.Frame(manageFilesWindow, bg = 'white')
    titleFrame.pack(side = Tk.TOP)
    lTitle = Tk.Label(titleFrame, bg = 'white', text = "Manage Files" ,font = ("Gill Sans MT", 15, "bold", "underline"))
    lTitle.pack(side = Tk.LEFT)
    bHelp = Tk.Button(titleFrame, text = " ? ")
    bHelp.pack(side = Tk.LEFT, padx = 10)
    optionsFrame = Tk.Frame(manageFilesWindow, bg = 'white')
    optionsFrame.pack(side = Tk.TOP, padx = 20, pady = 20)
    lCurrentFiles = Tk.Label(optionsFrame, text = "Current Files: ", bg= 'white')
    lCurrentFiles.pack(side = Tk.LEFT)
    cCurrentFiles = Tk.Listbox(optionsFrame, width = 100)
    cCurrentFiles.pack(side = Tk.LEFT)
    for files in Filenames:
        cCurrentFiles.insert(Tk.END, files)
    buttonsFrame = Tk.Frame(optionsFrame, bg = 'white')
    buttonsFrame.pack(side = Tk.LEFT, padx = 20)
    bAdd = Tk.Button(buttonsFrame, text = "Add", width = 10, command = lambda: manageFilesAddFile(optionsFrame, cSubmittedChanges))
    bAdd.pack(side = Tk.TOP)
    bRemove = Tk.Button(buttonsFrame, text = "Remove", width = 10,  command = (lambda: manageFilesDelFile(cCurrentFiles, cSubmittedChanges)))
    bRemove.pack(side = Tk.TOP)
    bRefresh = Tk.Button(buttonsFrame, text = "Refresh", width = 10, command = (lambda: manageFilesRefreshFile(cCurrentFiles, cSubmittedChanges)))
    bRefresh.pack(side = Tk.TOP)
    bSubmit = Tk.Button(buttonsFrame, text = "Submit Changes",  width = 10, command = lambda: manageFilesSubmit(manageFilesWindow, cSubmittedChanges))
    bSubmit.pack(side = Tk.TOP)
    bCancel = Tk.Button(buttonsFrame, text = "Omit Changes", width  = 10, command = lambda: manageFilesOmitChanges(manageFilesWindow))
    bCancel.pack(side = Tk.LEFT)
    submittedChangesFrame = Tk.Frame(manageFilesWindow, bg= 'white')
    submittedChangesFrame.pack(side = Tk.TOP, anchor = Tk.W, pady = 10, padx = 20)
    lSubmittedChanges = Tk.Label(submittedChangesFrame, text = "Changes:       ", bg= 'white')
    lSubmittedChanges.pack(side = Tk.LEFT)
    cSubmittedChanges = Tk.Listbox(submittedChangesFrame, width = 100)
    cSubmittedChanges.pack(side = Tk.LEFT)
  
  
def manageFilesOmitChanges(window):
    window.destroy()


def manageFilesAddFile(frame, listbox):
    addFrame = Tk.Frame(frame, bg = 'white')
    addFrame.pack(side = Tk.LEFT, anchor = Tk.N)
    lTitle = Tk.Label(addFrame, text = "Add a New File", bg = 'white')
    lTitle.pack(side = Tk.TOP)
    widgetsForAddFrame = Tk.Frame(addFrame, bg = 'white')
    widgetsForAddFrame.pack(side = Tk.TOP)
    eAddFile = Tk.Entry(widgetsForAddFrame, width= 30, bg = 'white')
    eAddFile.pack(side = Tk.LEFT)
    bClearEntry = Tk.Button(widgetsForAddFrame, text = "Clear", command = lambda: (clearField(eAddFile)))
    bClearEntry.pack(side = Tk.LEFT)
    bAddFileSubmit = Tk.Button(widgetsForAddFrame, text = "Submit", command = lambda: manageFilesAddFileSubmit(eAddFile, listbox))
    bAddFileSubmit.pack(side = Tk.LEFT)
    bAddFileBrowse = Tk.Button(widgetsForAddFrame, text = "Browse", command = (lambda: eAddFile.insert(0,Fd.askopenfilename())))
    bAddFileBrowse.pack(side = Tk.LEFT)
    bAddFileRecentFiles = Tk.Button(widgetsForAddFrame, text = "Recent Files", command = (lambda: loadRecentFile(eAddFile)))
    bAddFileRecentFiles.pack(side = Tk.LEFT)
    bCancel = Tk.Button(addFrame, text = "<--", command = lambda: addFrame.destroy())
    bCancel.pack(side = Tk.TOP, anchor = Tk.W, pady = 20)
        
    
def manageFilesAddFileSubmit(entry, listbox):
    try:
        tmpList = listbox.get(0,Tk.END)
        index = tmpList.index('Add File: ' + entry.get())
        errorMsg("This request is already in the queue")
    except:
        listbox.insert(Tk.END, 'Add File: ' + entry.get())   
    
    entry.delete(0,Tk.END)
    
def manageFilesRefreshFile(filesListbox, listbox):
    try:
        tmpList = listbox.get(0,Tk.END)
        index = tmpList.index("Refresh File: " + filesListbox.get('active'))
        errorMsg("This request is already in the queue")
    except:
        listbox.insert(Tk.END, "Refresh File: " + filesListbox.get('active'))

def manageFilesDelFile(filesListbox, listbox):
    try:
        tmpList = listbox.get(0,Tk.END)
        index = tmpList.index("Delete File: " + filesListbox.get('active'))
        errorMsg("This request is already in the queue")
    except:
        listbox.insert(Tk.END, "Delete File: " + filesListbox.get('active'))

def manageFilesSubmit(window, listbox):
    global vars
    submittedEntries = listbox.get(0, Tk.END)
    count = 0
    for entries in submittedEntries:
        if entries[0:3] == 'Add':
            #try:
            test = open(entries[10:], 'r')
            Filenames.append(entries[10:])
            vars[entries[10:]] = lexyacc.parseMe(entries[10:])
            
            markForDel = []
            for variables in vars[entries[10:]]:
                if (variables != 'CFLOG' and checkEmpty(vars[entries[10:]][variables].data) == 0):
                    markForDel.append(variables)
    
            for variables in markForDel:
                del vars[entries[10:]][variables]
        
            vars[entries[10:]] = organizedata.organizedata(vars[entries[10:]])
                
                
                
            #except:
            #    errorMsg('Could not open file' + str(count))
        
        elif entries[0:7] == 'Refresh':
            del vars[entries[14:]]
            vars[entries[14:]] = lexyacc.parseMe(entries[14:])
            
            markForDel = []
            for variables in vars[entries[14:]]:
                if checkEmpty(vars[entries[14:]][variables].data) == 0:
                    markForDel.append(variables)
    
            for variables in markForDel:
                del vars[entries[14:]][variables]
        
            vars[entries[14:]] = organizedata.organizedata(vars[entries[14:]])
            
    
        elif entries[0:6] == 'Delete':
           del vars[entries[13:]]
           Filenames.remove(entries[13:])
        
        else:
            errorMsg('This is a bug... please submit bug report')
            
    window.destroy() 
    
   

            
    
