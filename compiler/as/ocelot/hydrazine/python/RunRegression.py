#!/usr/bin/python

import sys

if sys.version_info >= (3, 0):
    raise "Python version 3.0 or higher not supported"

################################################################################
##
##
##	\file RunRegression.py
##
##	\author Gregory Diamos
##
##	\date July 19, 2008
##
##	\brief A class and script for parsing a list of tests, and running them
##			one by one, finally logging the results to a file.
##
################################################################################

import os
import logging
from optparse import OptionParser
import re
import random
import string
import time
from Queue import Queue
from Queue import Empty
from Queue import Full
from threading import Thread
from threading import Lock

def detectCPUs():
 """
 Detects the number of CPUs on a system. Cribbed from pp.
 """
 # Linux, Unix and MacOS:
 if hasattr(os, "sysconf"):
     if os.sysconf_names.has_key("SC_NPROCESSORS_ONLN"):
         # Linux & Unix:
         ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
         if isinstance(ncpus, int) and ncpus > 0:
             return ncpus
     else: # OSX:
         return int(os.popen2("sysctl -n hw.ncpu")[1].read())
 # Windows:
 if os.environ.has_key("NUMBER_OF_PROCESSORS"):
         ncpus = int(os.environ["NUMBER_OF_PROCESSORS"]);
         if ncpus > 0:
             return ncpus
 return 1 # Default

################################################################################
## Run Regression Class

class Test:
	def __init__( self, path, parameters, lock ):
		self.parameters = parameters
		self.path = path
		self.style = "default"
		self.passed = False
		self.status = ""
		self.time = 0.0
		self.lock = lock
		self.arguments = ""
	
	def parseParameters( self ):
		for parameter in self.parameters:
			if parameter.find("CUDA_SDK_STYLE") != -1:
				self.style = "CUDA_SDK_STYLE"
			elif parameter.find("PARBOIL_STYLE") != -1:
				self.style = "PARBOIL_STYLE"
			else:
				self.arguments += parameter + " "
				
	def parseDefault( self, message ):	
		self.status = "Did not complete."	
		for line in message.splitlines():
			line = line.strip("\n")
			if( re.search( "Pass/Fail : Pass", line ) != None or re.search("PASSED", line) != None):
				self.passed = True
				self.status = "Passed"
			elif ( re.search( "Pass/Fail : Fail", line ) != None or re.search("FAILED", line) != None):
				self.passed = False
				self.status = "Failed"
				break
		
	def parseCudaSdkStyle( self, message ):
		self.status = "Did not complete."
		
		for line in message.splitlines():
			line = line.strip("\n")
			if( re.search( "Test PASSED", line ) != None ):
				self.passed = True
				self.status = "Passed"
			elif ( re.search( "TEST PASSED", line ) != None ):
				self.passed = True
				self.status = "Passed"
			elif ( re.search( "Test FAILED", line ) != None ):
				self.passed = False
				self.status = "Failed"
				break
			elif ( re.search( "TEST FAILED", line ) != None ):
				self.passed = False
				self.status = "Failed"
				break
	
	def parseParboilStyle( self, message ):
		self.status = "Did not complete."
		self.passed = False
		
		for line in message.splitlines():
			line = line.strip("\n")
			if( re.search( "GPU:", line ) != None ):
				self.passed = True
				self.status = "Passed"
	
	def parseResult( self, message ):
		if self.style == "CUDA_SDK_STYLE":
			self.parseCudaSdkStyle(message)
		elif self.style == "PARBOIL_STYLE":
			self.parseParboilStyle(message)
		else:
			self.parseDefault(message)
							
	def run( self ):
		self.parseParameters()
		self.lock.acquire()
		logging.debug( "Set style to " + self.style)
		logging.info( "Running test program " + self.path \
			+ " " + self.arguments )
		tempName = self.randomString() + ".tmp"
		command = "echo \"\u001b\" | " + self.path + " " + self.arguments
		if self.style != "CUDA_SDK_STYLE" and self.style != "PARBOIL_STYLE":
			command += " -v"
		command += " >" + tempName + " 2>&1"
		logging.debug( "The command was " + command )
		self.lock.release()
		begin = time.time()
		os.system( command )
		self.time = time.time() - begin
		self.lock.acquire()
		logging.info( "Test " + self.path )
		logging.info( "Test completed in " + str(self.time) + " seconds"  )
		try:
			tempFile = open( tempName, "rb" )
			message = tempFile.read()
			message.decode("utf8")	
			logging.info( " It produced the following output:\n" + message );
			self.lock.release()
			self.parseResult( message )
			tempFile.close()
		except IOError:
			logging.info( " It produced an invalid output file\n" );
			self.lock.release()
			self.parseResult( "" )
		except UnicodeError:
			logging.info( " Produced an output file with invalid encoding\n" );
			self.lock.release()
			self.parseResult( "" )
		if( os.path.isfile( tempName ) ):
			os.remove( tempName )

	def randomString( self ):
		return "".join(random.sample(string.letters+string.digits, 8))

class TestThread(Thread):
	def __init__( self, queue ):
		Thread.__init__( self )
		self.queue = queue
		
	def run( self ):
		running = True
		while running:
			try:
				test = self.queue.get(False)
				test.run()
			except Empty:
				running = False

class RunRegression:
	
	def __init__( self, logFile, testFile, debug, testDirectory, jobs ):
		self.logFile = os.path.abspath( logFile )
		self.testFile = os.path.abspath( testFile )
		self.baseDirectory = os.getcwd()
		self.createLogFile( debug )
		self.testDirectory = testDirectory
		self.parseTestFile( )
		self.jobs = int(jobs)
	
	def createLogFile( self, debug ):
		if debug:
			logging.basicConfig(level = logging.DEBUG,
								format = '%(levelname)-8s %(message)s',
								filename = self.logFile,
								filemode = "w"		
								)
		else:
			logging.basicConfig(level = logging.INFO,
								format = '%(levelname)-8s %(message)s',
								filename = self.logFile,
								filemode = "w"		
								)
					
	# Each line in the test file is the relative path
	# to a test
	def parseTestFile( self ):
		self.tests = set()
		self.passed = {}
		self.failed = {}
		self.noTest = set()
		lock = Lock()
		if len( self.testDirectory ) == 0:
			testBase = os.path.dirname( self.testFile )
		else:
			testBase = self.testDirectory
		logging.debug( "Changing directory to " + testBase )
		os.chdir( testBase )
		logging.info( "Reading in test file " + self.testFile )
		tests = open( self.testFile, 'r' )
		logging.info( " Found the following tests:" )
		for currentTest in tests:
			parameters = currentTest.split();
			if len( parameters ) == 0 :
				continue
			if parameters[0] != "" and parameters[0][0] != '#':
				logging.info( "  " + os.path.abspath( parameters[0] ) )
				if( os.path.isfile( os.path.abspath( parameters[0] ) ) ):
					self.tests.add( Test( os.path.abspath( parameters[0] ), \
						parameters[1:], lock ) )
				else:
					logging.error( "Could not find test program " + \
						os.path.abspath( parameters[0] ) )
					self.noTest.add( os.path.abspath( parameters[0] ) )
		tests.close()
		logging.info( "==== INDIVIDUAL TEST RESULTS ====\n" )
		logging.debug( "Returning to base directory " + self.baseDirectory )
		os.chdir( self.baseDirectory )
		
	def run( self ):
		self.passed.clear()
		self.failed.clear()		
		queue = Queue()		
		for test in self.tests:
			queue.put(test, False)		
		threads = []	
		threadCount = min(detectCPUs(), self.jobs)
		for i in range( threadCount ):
			thread = TestThread( queue )
			threads.append( thread )
			threads[i].start()
		for thread in threads:
			thread.join()
		for test in self.tests:
			if test.passed:
				self.passed[test.path] = test
			else:
				self.failed[test.path] = test
	
	def report( self, verbose ):
		print str( len( self.passed ) ) + " out of " + str( len( self.tests ) \
			+ len( self.noTest ) ) + " tests passed."
		string = "\nPassing tests:\n"
		for ( path, test ) in self.passed.iteritems():
			string += " (%3.3f" % test.time + "s) : " + test.path + " : " \
				+ test.status + "\n"
		if len( self.failed ) != 0:
			string += "\nFailing tests:\n"
			for ( path, test ) in self.failed.iteritems():
				string += " (%3.3f" % test.time + "s) : " + test.path + " : " \
					+ test.status + "\n"
		if len( self.noTest ) != 0:
			string += "\nNon-Existent tests:\n"
			for test in self.noTest:
				string += " " + test + "\n" 
		logging.info( string )
		if( verbose ):
			print string
		

################################################################################


################################################################################
## Main
def main():
	parser = OptionParser()
	
	parser.add_option( "-l", "--logFile", \
		default="python/regression/regressionResults.txt" )
	parser.add_option( "-t", "--testFile", \
		default="python/regression/regressionTests.txt" )
	parser.add_option( "-v", "--verbose", default = False, \
		action = "store_true" )
	parser.add_option( "-d", "--debug", default = False, action = "store_true" )
	parser.add_option( "-j", "--jobs", default = 1024 )
	parser.add_option( "-p", "--test_path", default="" )	
	( options, arguments ) = parser.parse_args()
	
	regression = RunRegression( options.logFile, options.testFile, \
		options.debug, options.test_path, options.jobs )
	regression.run()
	regression.report( options.verbose )	

################################################################################

################################################################################
## Guard Main
if __name__ == "__main__":
	main()

################################################################################

