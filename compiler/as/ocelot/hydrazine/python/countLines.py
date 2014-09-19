##
# @file countLines.py
#
# @brief 	This program will take a file name specifying
#			a root directory and count the number of lines in all source files
#			with specific extension.  
#
##

from optparse import OptionParser

import os
import re

def countLines( filename, comment, blockStart, blockEnd, verbose ):

	if verbose:
		print "Counting lines in", filename
	if os.path.isfile( filename ):
		currentFile = open( filename, 'r' )
		blockComment = 0;

		commentDetector = []
	
		for i in comment:
			commentDetector.append( " *" + i )

		if blockStart != "":
			commentDetector.append(blockStart)

		lines = 0;
	
		for currentLine in currentFile:
	
			m = re.compile(" *\n").match(currentLine)
	
			if not blockComment and not m:
				anyFound = 0
				for i in commentDetector:
					m = re.match(i, currentLine)
					if m:
						anyFound = 1
						break
			
				if not anyFound:
					lines += 1
			
				index = 0
				m = re.compile(blockStart).search( currentLine, index )
			
				while m:

					blockComment = 1
					index = m.end()

					m = re.compile(blockEnd).search( currentLine, index )
				
					if not m:
						break
					else:
						index = m.end()
						blockComment = 0
						m = re.compile(blockStart).search( currentLine, index )
							
			else: 
				if not m:
					index = 0
					m = re.compile(blockEnd).search( currentLine, index )
			
					if m:
			
						blockComment = 0
						m = re.compile(blockStart).search( currentLine, index )
			
						while m:

							blockComment = 1
							index = m.end()

							m = re.compile(blockEnd).search( currentLine, index )
						
							if not m:
								break
							else:
								index = m.end()
								blockComment = 0
								m = re.compile(blockStart).search( currentLine, index )
		if verbose:
			print "Lines in", filename, lines						
		return lines
	else:
		return 0

def exclude( pattern, files ):
	
	for i in pattern:
	
		matches = []
	
		for j in files:
			m = re.search(i,j)
			if m:
				matches.append( m.string )
						
		for j in matches:
			files.remove( j )
		
def filterExtension( extension, files ):

	matches = []

	for i in extension:

		i = i + "$"

		for j in files:
			m = re.search(i,j)
			if m:
				matches.append(m.string)
	
	return matches

def getFiles( directory, result ):
	if os.path.isdir(directory):
		names = os.listdir(directory)
		def joinPaths(name, path=directory):
			return os.path.join(path, name)

		names = map(joinPaths, names)
		result.extend(names)

		def directoryAndNotLink(a): return os.path.isdir(a) and not os.path.islink(a)
		localDirectories = filter(directoryAndNotLink, names)

		for i in localDirectories:
			getFiles(i, result)

	else:
		if os.path.isfile(directory):
			result.append(directory)

def main():
	
	parser = OptionParser()
	parser.add_option( "-d", "--directory", help="Specify the directory to parse.", action = "store", type = "string" )
	parser.add_option( "-e", "--exclude", help="Specify a regular expression to exclude from the files being examined.", action = "append", type = "string" )
	parser.add_option( "-x", "--extension", help="Specify the extensions of files to examine.", action = "append", type = "string" )
	parser.add_option( "-c", "--comment", help="Specify the start of a comment line.", action = "append", type = "string" )
	parser.add_option( "-b", "--block_start", help="Specify the start of a comment block", action="store", type = "string" )
	parser.add_option( "-n", "--block_end", help="Specify the end of a comment block", action="store", type = "string" )
	parser.add_option( "-v", "--verbose", help="Print out extra information as comments are being counted", action="store_true")
	
	parser.set_defaults(verbose=0)
	parser.set_defaults(block_start="/\*")
	parser.set_defaults(block_end="\*/")
	parser.set_defaults(comment=["//"])
	parser.set_defaults(exclude=[])
	parser.set_defaults(extension=["\.h", "\.cpp"])
	parser.set_defaults(directory=".")
	
	(options, args) = parser.parse_args()
	
	if len(options.comment) > 1:
		options.comment.pop(0)
	
	if len(options.extension) > 2:
		options.extension.pop(0)
		options.extension.pop(0)

	dirs = []

	getFiles( options.directory, dirs )
	
	if not (len(options.exclude) == 0):
		exclude( options.exclude, dirs )
		
	dirs = filterExtension( options.extension, dirs )

	lines = 0;

	for i in dirs:
		lines += countLines( i, options.comment, options.block_start, options.block_end, options.verbose )
		
	print "The total number of lines was", lines

if __name__ == '__main__':
	main()

