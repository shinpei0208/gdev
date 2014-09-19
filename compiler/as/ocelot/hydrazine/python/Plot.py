################################################################################
##	\file RunRegression.py
##	\author Gregory Diamos
##	\date July 19, 2008
##	\brief A class and script for parsing a list of data elements and 
##	identifiers, and plotting them using Matplotlib
################################################################################

from optparse import OptionParser

import matplotlib.pyplot as plot
import matplotlib.text as text
import matplotlib.font_manager as font_manager
import numpy
import re
import scipy.stats

################################################################################
## Comments
def isComment( string ):
	regularExpression = re.compile( "[ \t]*//" )
	return regularExpression.match( string ) != None

################################################################################

################################################################################
## Element - A data element to be plotted
class Element:
	def __init__(self, data, error):
		self.data = data
		self.error = error

################################################################################

################################################################################
## Plot - A class that parses an input file and creates a plot
class Plot:
	def __init__( self, path, barWidth, color, verbose ) :
		self.path = path
		self.verbose = verbose
		self.barWidth = barWidth
		self.xlabel=""
		self.ylabel=""
		self.title=""
		self.barWidth=.35
		self.setNames=[]
		self.arguments=[]
		self.colors=[]
		self.defaultColor=color
		self.log = False
		self.yerror = False
		self.normalize = False
		self.position = 'best'
		self.sort = True
		
	def parse( self ):
		self.names = {}				
		inputs = open( self.path, 'r' )
		self.parseNames( inputs )
		self.parseArguments( inputs )
		self.parseData( inputs )
		self.partition()
		assert self.size > 0
		while len( self.arguments ) < self.size:
			self.arguments.append("")
		while len( self.colors ) < self.size:
			self.colors.append(self.defaultColor)
		assert len( self.arguments ) == self.size
		self.barWidth=min(self.barWidth, 1/(self.size + 1.0))
		
	def parseNames( self, inputs ):
		while True:
			temp = inputs.readline();
			if isComment( temp ):
				continue
			elif len( temp) == 0:
				break
			elif temp == '\n':
				continue
			elif temp.startswith("xlabel ") :
				self.xlabel = temp[7:]
			elif temp.startswith("ylabel ") :
				self.ylabel = temp[7:]
			elif temp.startswith("position ") :
				self.position = temp[9:].strip('\n')
			elif temp.startswith("title ") :
				self.title = temp[6:]
			elif temp.startswith("barwidth ") :
				self.barWidth = float(temp[9:])
			elif temp.startswith("labels ") :
				for name in temp[7:].split():
					self.setNames.append(name)
			elif temp.startswith("colors ") :
				for name in temp[7:].split():
					self.colors.append(name)
			elif temp.startswith("log ") :
				if temp.find( "True" ) > 0:
					self.log = True
				else:
					self.log = False
			elif temp.startswith("sorting ") :
				if temp.find( "True" ) > 0:
					self.sort = True
				else:
					self.sort = False
			elif temp.startswith("normalize ") :
				if temp.find( "True" ) > 0:
					self.normalize = True
				else:
					self.normalize = False
			elif temp.startswith("errorBars ") :
				if temp.find( "True" ) > 0:
					self.yerror = True
				else:
					self.yerror = False
			elif temp.find( "--arguments--" ) != -1 :
				break
				
	def parseArguments( self, inputs ):
		while True:
			temp = inputs.readline();
			if isComment( temp ):
				continue
			elif len( temp) == 0:
				break
			elif temp == '\n':
				continue
			elif temp.find( "--data--" ) != -1 :
				break
			else:
				self.arguments.append(temp)

	def combine( self, string ):
		inElement = False
		elements = [ ]
		identifier = string.split(' ', 1)
		data = [ identifier[ 0 ] ]
		if len(identifier) > 1:
			for word in identifier[1].split():
				if inElement:
					if word == ']':
						inElement = False
						data.append( numpy.mean( numpy.array( elements ) ) )
						del elements[:]
					else:
						elements.append( float( word ) )
				else:
					if word == '[':
						inElement = True
					else:
						data.append( float( word ) )
		if self.normalize:
			factor = data[1]
			for i in range(1, len(data)):
				data[i] = data[i] / factor
		for i in range(1, len(data)):
			data[i] = str( data[i] )
		return data
	
	def computeErrorBound( self, string ):
		inElement = False
		data = [ ]
		elements = [ ]
		for word in string.split()[1:]:
			if inElement:
				if word == ']':
					inElement = False
					if self.normalize:
						factor = elements[ 0 ]
						for i in range(0,len(elements)):
							elements[i] = elements[i] / factor
					data.append( numpy.std( numpy.array( elements ) ) )
					del elements[:]
				else:
					elements.append( float( word ) )
			else:
				if word == '[':
					inElement = True
				else:
					data.append( 0 )
		for i in range(0,len(data)):
			data[ i ] = data[i] * ( 1 - scipy.stats.t.cdf(.975, len(data)))

		return data
			
	def parseData( self, inputs ):
		count = 0
		index = 0
		self.names = [ ]
		self.namemap = { }
		self.names.append( { } )
		self.size = -1
		for name in inputs:
			if isComment( name ):
				continue
			if name == "\n" :
				continue
			elif name.startswith( "--new set--" ):
				index = 0
				count += 1
				self.names.append( { } )
				continue
			items = self.combine( name )
			self.namemap[ items[ 0 ] ] = index
			if items[ 0 ] in self.names:
				raise Exception, "Duplicate type " + items[ 0 ] + " declared"
			error = self.computeErrorBound( name )
			self.names[ count ][ items[ 0 ] ] = [ ]
			if self.size == -1 :
				self.size = len( items ) - 1
			if self.size != len( items ) - 1:
				raise Exception, "Label " + items[ 0 ] + " only has " \
					+ str( len( items ) - 1 ) + " elements, expecting " \
					+ str( self.size )
			for i in range( 1, len( items ) ):
				self.names[ count ][ items[ 0 ] ].append( \
					Element( float( items[ i ] ), float( error[ i - 1 ] ) ) )
			index += 1
	
	def arrange( self, names ):
		indexmap = {}
		for name in names:
			indexmap[ self.namemap[ name ] ] = name
	
		names = []
		for i in range( 0, len( indexmap ) ):
			names.append( indexmap[ i ] )
		return names
	
	def partition( self ):
		self.labels = [ ]
		self.dataVector = [ ]
		self.errorVector = [ ]
		totalElements = 0
		count = 0
		for nameSet in self.names :
			names = nameSet.keys()
			if self.sort:
				names.sort()
			else:
				names = self.arrange( names )
			totalElements += len( names )
			for name in names :
				self.labels.append( name )
				data = nameSet[ name ]
				if count == 0:
					for i in data:
						self.dataVector.append( [ ] )
						self.errorVector.append( [ ] )
				index = 0
				for i in data:
					self.dataVector[ index ].append( i.data )
					self.errorVector[ index ].append( i.error )
					index += 1
				count += 1
		self.indicies = range( totalElements )
		
	def display( self ):
		self.parse()
		plot.figure( 1 )
		plots = [ ]
		count = 0
		for data in self.dataVector:
			error = None
			if self.yerror:
				error = numpy.array( self.errorVector[ count ] )
			plots.append( plot.bar( numpy.array( self.indicies ) 
				+ count * self.barWidth, numpy.array( data ), 
				self.barWidth, color = self.colors[ count ], log = self.log, 
				yerr = error )[0] )
			count += 1
		plot.xlabel( self.xlabel )
		plot.ylabel( self.ylabel )
		plot.title( self.title )
		error = None
		plot.xticks( numpy.array(self.indicies) + numpy.array(self.barWidth)
			* ( self.size / 2.0 ), self.labels, rotation = 'vertical' )
		if len( self.setNames ) == len( plots ):
			plot.legend( plots, self.setNames, self.position )
		plot.show()

################################################################################


################################################################################
## Main
def main():
	parser = OptionParser()
	
	parser.add_option( "-i", "--inputFile", \
		default="plot.in" )
	parser.add_option( "-v", "--verbose", default = False, \
		action = "store_true" )
	parser.add_option( "-b", "--barWidth", default = .35 )
	parser.add_option( "-c", "--default_color", default="k" )
	
	( options, arguments ) = parser.parse_args()
	
	plot = Plot( options.inputFile, options.barWidth, options.default_color, 
		options.verbose )
	plot.display()

################################################################################

################################################################################
## Guard Main
if __name__ == "__main__":
	main()

################################################################################

