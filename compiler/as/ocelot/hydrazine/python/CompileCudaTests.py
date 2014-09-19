#! /usr/bin/env python

################################################################################
#	\file CompileCuda.py
#	\author Gregory Diamos <gregory.diamos@gatech.edu>
#	\date Wednesday May 12, 2010
#	\brief A script that performs CUDA to C++ conversion
################################################################################

import os
from optparse import OptionParser
import re

class CudaSource:
	def __init__(self, path, ptx):
		self.filename = os.path.abspath(path)
		if ptx:
			self.outfile = self.filename[:-3] + ".ptx"
		else:
			self.outfile = self.filename + ".cpp"
		
def getAllCudaSources(path, ptx):
	sources = []
	for dirpath, dirnames, filenames in os.walk(path):
		for filename in filenames:
			name = os.path.join(dirpath, filename)
			if os.path.isfile(name):
				split = name.rsplit('.', 1)
				extension = ""
				if len(split) == 2:
					extension = split[1]
				if extension == "cu":
					sources.append(CudaSource(name, ptx))
	return sources

def printAutomakeBlock(dirpath, sources):
	exe = sources['exe'][0][0:1].capitalize() + sources['exe'][0][1:]
	cppFiles = [os.path.join(dirpath,x) for x in sources['cpp'] ]
	cudaFiles = [os.path.join(dirpath, x + '.cpp') for x in sources['cuSources'] ]
	print "################################################################################"
	print "##", exe
	print exe + '_CXXFLAGS = $(CUDA_CFLAGS) $(SDK_CFLAGS)'
	print exe + '_SOURCES = ' + ' \\\n\t'.join(cppFiles + cudaFiles)
	print exe + '_LDADD = $(OCELOT_LIBS)'
	print exe + '_LDFLAGS = -static'
	print "################################################################################\n"
	return exe

# Obtain everything up to a token
#
# [x.strip("") for x in re.split("\s", line) if x.strip() != '']
#
def splitLineUpToComment(line):
	tokens = [x.strip("") for x in re.split("\s", line) if x.strip() != '']
	result = []
	for token in tokens:
		if token != '#':
			result.append(token)
		else:
			break;
	return result

#
#
def getAllCudaSourcesMakefile(path, ptx, automake=False, apps=False):
	""" scrapes CUDA dependencies by examining a Makefile located in the application's source directory """
	cuFiles = []		
	programs = []
	for dirpath, dirnames, filenames in os.walk(path):
		for filename in filenames:
			name = os.path.join(dirpath, filename)
			if os.path.isfile(name) and os.path.basename(name) == 'Makefile':
				sources = {'cpp': [], 'cuDeps': [], 'cuSources': [], 'exe': []}
				tokenStateMap = {'CCFILES': 'cpp', 'CFILES': 'cpp', 'CUDEPS': 'cuDeps', 'CU_DEPS': 'cuDeps', \
					'CUFILES': 'cuSources', 'CUFILES_sm_10': 'cuSources', 'CUFILES_sm_11': 'cuSources', \
					'CUFILES_sm_20': 'cuSources', 'CUFILES_sm_12': 'cuSources', 'CUFILES_sm_13': 'cuSources', \
					'END' :'none', 'MPICXX': 'none', 'EXECUTABLE': 'exe'}
				tokens = []
				lines = [x for x in open(os.path.abspath(name), "r").readlines()]
				for line in lines:
					if line[0] == '#':
						if len(tokens) == 0 or tokens[-1] != 'END':
							tokens.append('END')
					else:
						tokens += splitLineUpToComment(line)
				state = 'none'
				for token in tokens:
					if token in tokenStateMap.keys():
						state = tokenStateMap[token]
					else:
						regex = """[/0-9\-a-zA-Z_]+\.[a-z]+"""
						if state == 'exe' and token != 'EXECUTABLE':
							regex = """[/0-9\-a-zA-Z_]+"""
						g = re.search(regex, token)
						if g and state in sources.keys():
							sources[state].append(g.group(0))
				cuFiles += [CudaSource(os.path.join(dirpath,x), ptx) for x in sources['cuSources']]
				
				#
				# print Automake components while we're at it
				#
				if len(sources['exe']) and automake:
					programs.append(printAutomakeBlock(dirpath, sources))
					
	if apps and automake and len(programs):
		print "check_PROGRAMS =", " \\\n\t".join(programs)
	return cuFiles

#
#
#
def compileSources(commandBase, sources, continueOnError, inferShaderModel = False):
	for source in sources:
		if not os.path.isfile(source.outfile):
			archFlags = ""
			if len(source.filename) > 8 and inferShaderModel:
				suffix = source.filename[-8:].lower()
				flags = {'_sm10.cu': '-arch=sm_10', '_sm13.cu': '-arch=sm_13', \
					'_sm12.cu' : '-arch=sm_12', '_sm20.cu': '-arch=sm_20', '_sm21.cu': '-arch=sm_21' }
				if suffix in flags.keys():
					archFlags = ' ' + flags[suffix] + ' '
			command = commandBase + archFlags + " -I" + os.path.dirname(source.filename) \
				+ " -I" + os.path.dirname(source.filename) + "/../inc"\
				+ " -I./inc -Isdk -Ishared " \
				+ " -o " + source.outfile + " " + source.filename
			print command
			os.system(command)
			if not os.path.isfile(source.outfile) and not continueOnError:
				print 'error - compiling \'' + source.filename \
					+ '\' failed. aborting...\n'
				break

def sanitizeSources(sources):
	for source in sources:
		if os.path.isfile(source.outfile):
			print "sanitizing " + os.path.basename(source.outfile)
			inFile = open(source.outfile, 'r')
			text = inFile.read()
			inFile.close()
			text = text.replace('stdarg', 'va')
			text = text.replace('auto', '')
			outFile = open(source.outfile, 'w')
			outFile.write(text)

def clean(sources):
	for source in sources:
		if os.path.isfile(source.outfile):
			print "deleting file " + source.outfile
			os.remove(source.outfile)
		
def main():
	parser = OptionParser()
	
	parser.add_option("-d", "--directory", action="store", 
		default=".", dest="directory", help="The directory to run on.")
	parser.add_option("-p", "--ptx", action="store_true", 
		default=False, dest="ptx", 
		help="Generate PTX rather than .cu.cpp sources.")
	parser.add_option("-a", "--arguments", action="store", 
		default="-I ~/checkout/thrust -I ./sdk", dest="arguments", 
		help="NVCC options.")
	parser.add_option("-c", "--clean", action="store_true", 
		dest="clean", default=False, help="Delete all .cu.cpp files.")
	parser.add_option("-s", "--sanitize", action="store_true", 
		dest="sanitize", default=False, help="Only sanitize .cu.cpp files.")
	parser.add_option("-m", "--makefile", action="store_true",
		dest="makefile", help="Print components of Makefile.am for each application to stdout")
	parser.add_option("-L", "--list", action="store_true",
		dest="listapps", help="Print a listing of applications for Makefile.am")
	parser.add_option("-n", "--nocompile", action="store_true",
		dest="nocompile", help="Do not compile")
	parser.add_option("-r", "--shader-model", action="store_true",
		dest="shadermodel", help="Infers shader model from suffix of filename (e.g. _sm13.cu)")
	
	(options, args) = parser.parse_args()
	
	if options.ptx:
		command = "nvcc --ptx " + options.arguments
	else:
		command = "nvcc --cuda " + options.arguments
	
	path = os.getcwd()
	
	if options.makefile:
		sources = getAllCudaSourcesMakefile(options.directory, options.ptx,
			options.makefile, options.listapps)
	else:
		sources = getAllCudaSources(options.directory, options.ptx)
	
	if options.ptx:
		compileSources(command, sources, True, options.shadermodel)
	else:
		if options.clean:
			clean(sources)
		elif options.sanitize:
			sanitizeSources(sources)
		else:
			if not options.nocompile:
				compileSources(command, sources, False, options.shadermodel)
			if not options.makefile:
				sanitizeSources(sources)
	
if __name__ == "__main__":
    main()
	
