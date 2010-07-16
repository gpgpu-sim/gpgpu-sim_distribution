# (c) Copyright 2007 The Board of Trustees of the University of Illinois.

from binaryfilecompare import eof, many

# Rename the builtin 'float' variable to avoid name conflicts
builtin_float = float

def verbatim(f):
	"""Read a line of text from file 'f'."""
	line = f.readline()
	return line

def float(f):
	"""Read a line of text from file 'f' as a single floating-point
	number."""
	words = f.readline().split()
	if len(words) != 1:
		raise ValueError, "Expecting line to contain a single number"
	return builtin_float(words[0])

def floats(f):
	"""Read a line of text from file 'f' as a list of floating-point
	numbers."""
	words = f.readline().split()
	return [builtin_float(x) for x in words]


