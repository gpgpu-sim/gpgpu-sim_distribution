# (c) Copyright 2007 The Board of Trustees of the University of Illinois.

import sys

# A monad for comparing two files.  Comparing proceeds until the first
# mismatch occurs, at which point comparison stops and the error is
# reported.
class CompareMonad(object):
	def checkType(x):
		if not isinstance(x, CompareMonad):
			raise TypeError, "Not a CompareMonad instance"
	checkType = staticmethod(checkType)

	def run(self, ref_file, out_file):
		raise NotImplementedError

# The >>= operator.  Executes 'fst', then executes 'snd' with the result
# of 'fst'.
class Bind(CompareMonad):
	
	def __init__(self, fst, snd):
		CompareMonad.checkType(fst)
		self.fst = fst
		self.snd = snd

	def run(self, ref_file, out_file):
		(ok, value) = self.fst.run(ref_file, out_file)
		if ok:
			sndMonad = self.snd(value)
			CompareMonad.checkType(sndMonad)
			return sndMonad.run(ref_file, out_file)
		return (False, None)

# The >> operator.  Executes 'fst', then executes 'snd'.
class Then(CompareMonad):
	def __init__(self, fst, snd):
		CompareMonad.checkType(fst)
		CompareMonad.checkType(snd)
		self.fst = fst
		self.snd = snd

	def run(self, ref_file, out_file):
		(ok, value) = self.fst.run(ref_file, out_file)
		if ok:
			return self.snd.run(ref_file, out_file)
		return (False, None)

# The 'return' operator.  Returns a value that is accessible
# within the monad.
class Return(CompareMonad):
	def __init__(self, value):
		self.value = value

	def run(self, ref_file, out_file):
		return (True, self.value)

# Run a list of monad instances and return their results as
# a list.
class Sequence(CompareMonad):
	def __init__(self, ms):
		self.ms = ms

	def run(self, ref_file, out_file):
		values = []
		for m in self.ms:
			CompareMonad.checkType(m)
			(ok, value) = m.run(ref_file, out_file)
			if not ok: return (False, None)
			values.append(value)

		return (True, values)

# Run a list of monad instances, ignoring their results.
class Sequence_(CompareMonad):
	def __init__(self, ms):
		self.ms = ms

	def run(self, ref_file, out_file):
		for m in self.ms:
			CompareMonad.checkType(m)
			(ok, value) = m.run(ref_file, out_file)
			if not ok: return (False, None)

		return (True, None)

# The basic CompareMonad routine.  This reads a value from both input
# files, compares the result, and, if the comparison was successful,
# returns the value.
class Compare(CompareMonad):
	"""Read an item from both input files and compare it."""

	def __init__(self,
			read = file.read,
			equal = lambda x, y: x == y,
			message = "Output does not match the expected output"):
		self.read = read
		self.equal = equal
		self.message = message

	def run(self, ref_file, out_file):
		try:
			x = self.read(ref_file)
		except ValueError, e:
			sys.stderr.write("Malformed reference file!\n")
			return (False, None)
		except EOFError:
			sys.stderr.write("Unexpected end of reference file!\n")
			return (False, None)
		try:
			y = self.read(out_file)
		except ValueError, e:
			sys.stderr.write("Malformed output file;\n")
			sys.stderr.write(str(e))
			sys.stderr.write('\n')
			return (False, None)
		except EOFError:
			sys.stderr.write("Unexpected end of output file\n")
			return (False, None)

		# Compare reference data to result data
		if self.equal(x, y):
			return (True, x)
		else:
			sys.stderr.write(self.message)
			sys.stderr.write('\n')
			return (False, None)

def open_or_abort(filename):
	try: f = file(filename, "r")
	except:
		sys.stderr.write("Cannot open file '" + filename + "'\n")
		sys.exit(-1)
	return f


def default_main(comparison_routine):
	"""Default main() routine.  Read file names from sys.argv
	and compare the files."""
	
	if len(sys.argv) != 3:
		sys.stderr.write("Usage: compare-output <from-file> <to-file>\n")
		sys.exit(-1)

	ref = open_or_abort(sys.argv[1])
	out = open_or_abort(sys.argv[2])

	(ok, _) = comparison_routine.run(ref, out)
	if ok:
		sys.stdout.write("Pass\n")
		sys.exit(0)
	else:
		sys.stdout.write("Mismatch\n")
		sys.exit(-1)

