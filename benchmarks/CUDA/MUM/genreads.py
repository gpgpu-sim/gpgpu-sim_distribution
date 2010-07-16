#!/usr/bin/env python
# encoding: utf-8
"""
untitled.py

Created by Cole Trapnell on 2007-04-26.
Copyright (c) 2007 __MyCompanyName__. All rights reserved.
"""

import sys
import getopt
import random


help_message = """
genreads outputs a multi-FASTA file containing a random sampling of 
read-sized subsequences of the provided reference sequence.  

Usage:
	genreads [options] <reference file> <length of reads> <# of reads>	
	
Options:
	-m < % mismatch>   --mismatches=  the % of bases in each read that
	                                  don't match the reference
	-s <integer value> --seed=        the seed for the random number 
	                                  generator
"""


class Usage(Exception):
	def __init__(self, msg):
		self.msg = msg


def main(argv=None):

	if argv is None:
		argv = sys.argv
	try:
		try:
			opts, args = getopt.getopt(argv[1:], "hSd:m:s:v", ["help","distribution=", "mismatches=", "seed=", "sort"])
		except getopt.error, msg:
			raise Usage(msg)
	
		mismatch = 0.0
	
		seed_val = 0
		sorted = False
		# option processing
		for option, value in opts:
			if option == "-v":
				verbose = True
			if option in ("-h", "--help"):
				raise Usage(help_message)
			if option in ("-d", "--distribution"):
				dist = value
			if option in ("-m", "--mismatches"):
				mismatch = float(value)
			if option in ("-s", "--seed"):
				seed_val = long(value)
			if option in ("-S", "--sort"):
				sorted = True
			
		random.seed(seed_val)
		
		if len(args) != 3:
			raise Usage(help_message)
	
		fasta_input = args[0]
		read_length = int(args[1])
		num_reads = int(args[2])
	
		#fasta_input = "NC_003997.fna"
		#read_length = 47
		#num_reads = 10000
		
		f = open(fasta_input, "r")
		lines = f.readlines()
		
		if lines[0].find(">") == -1:
			raise Usage("File is not FASTA format")
	
	
		seq = "".join([line.strip() for line in lines[1:]])
		
#		for line in lines[1:]:
#			line = line.strip()
#			if line.find(">") != -1:
#				raise("Multi-FASTA format not supported")
#			seq += line



		#print seq
		L = len(seq)
	
		#allowable = range(0, L - read_length)
		
		rid = 0
		
		base_mismatcher = {
			"A" : "C",
			"C"	: "G",
			"G" : "T",
			"T" : "A"
		}

		reads = []
		
		for i in range(0, num_reads):
			start = random.randint(0, L - read_length)
			#assert start <= L - read_length
			end = start + read_length

			rid +=1
			print ">" + "rid" + str(rid)
			print seq[start:end]
			
	except Usage, err:
		print >> sys.stderr, sys.argv[0].split("/")[-1] + ": " + str(err.msg)
		return 2


if __name__ == "__main__":
	sys.exit(main())
