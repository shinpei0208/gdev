#!/usr/bin/env python

################################################################################
#	\file   build.py
#	\author Gregory Diamos <gregory.diamos@gatech.edu>
#   \date   Sunday March 13, 2011
#	\brief  The Ocelot build script to direct scons builds and run unit tests 
################################################################################

import os
import re
import subprocess
import time
from optparse import OptionParser
import sys

################################################################################
## Build Ocelot
def build(options):
	command = "scons -Q"

	if options.clean:
		command += " -c"

	if options.debug:
		command += " mode=debug"

	if options.no_werr:
		command += " Werror=false"
	
	if options.no_wall:
		command += " Wall=false"
	
	if options.no_llvm:
		command += " enable_llvm=false"
	
	if options.no_opengl:
		command += " enable_opengl=false"

	if options.no_cuda_runtime:
		command += " enable_cuda_runtime=false"
	
	if options.static:
		command += " library=static"
	
	if options.build_deb:
		if not options.install:
			print "Install must be set for a debian build, setting it"
			options.install = True
		command += " debian"

	if options.install:
		command += " install=true"

	if options.install_prefix:
		command += " install_path=" + options.install_prefix

	if options.build_target != '':
		if options.debug:
			command += " .debug_build/"
		else:
			command += " .release_build/"
		command += options.build_target
		
	if options.test_level != 'none':
		command += " tests test_level=" + options.test_level

	if options.threads > 1:
		command += " -j" + str(options.threads)

	# Run SCons
	print command
	
	# Flush above message as on Windows it is not appearing until
	# after following subprocess completes.
	sys.stdout.flush()

	scons = subprocess.Popen(command, shell=True)

	return scons.wait() == 0
	
################################################################################

################################################################################
## Run Unit Tests
def runUnitTests(options, buildSucceeded):
	if not buildSucceeded:
		print "Build failed..."
		return False

 	if options.clean:
		print "Build cleaned..."
 		return False
	
	if options.test_level == 'none':
		return False
	
	command = "python hydrazine/python/RunRegression.py -v"
	
	if options.debug:
		command += " -p .debug_build/"
		prefix = "debug"
	else:
		command += " -p .release_build/"
		prefix = "release"

	if options.test_level == 'basic':
		log = "regression/" + prefix + "-basic.log"
		command += " -t regression/basic.level"
	elif options.test_level == 'full':
		log = "regression/" + prefix + "-full.log"
		command += " -t regression/full.level"
	elif options.test_level == 'sass':
		log = "regression/" + prefix + "-sass.log"
		command += " -t regression/sass.level"
	else:
		print "Unsupported test_level of '" + options.test_level + "'"
		return False

	command += " -l " + log
	
	print '\nRunning Ocelot Unit Tests...'
	print command
	status = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT).stdout.read()
 	print status
	
	# Check for any failing/missing tests
	if re.search('Failing tests|Non-Existent tests', status):
		return False
	else:
		return True
		
################################################################################

################################################################################
## Submit to SVN
def submit(options, testPassed):
	if not options.submit:
		return
		
	if len(options.message) == 0:
		print "Log message not specified (use -m)"
		return
	
	if not testPassed:
		print "Regression tests failed or not run, commit prohibited."
 		return
		
	command = "svn commit -m \"" + options.message + "\""
	
	os.system(command)
################################################################################

################################################################################
## Main
def main():
	parser = OptionParser()
	
	parser.add_option( "-c", "--clean", \
		default = False, action = "store_true",
		help = "delete all build results except previously installed files" )
	parser.add_option( "-d", "--debug", \
		default = False, action = "store_true", \
		help = "build Ocelot in debug mode." )
	parser.add_option( "-t", "--test_level", default = "none", \
		help = "set the test level (none, basic, full, sass)" )
	parser.add_option( "-j", "--threads", "--jobs", dest="threads",
		type="int", default = "1" )
	parser.add_option( "-s", "--submit", \
		default = False, action = "store_true" )
	parser.add_option( "-S", "--static", \
		default = False, action = "store_true",
		help = "Statically link ocelot." )
	parser.add_option( "-i", "--install", \
		default = False, action = "store_true", help = "Install ocelot." )
	parser.add_option( "-b", "--build_target", \
		default = "", help = "build a specific target." )
	parser.add_option( "-a", "--no_wall", \
		default = False, action = "store_true", help =
			"don't display all warnings." )
	parser.add_option( "-w", "--no_werr", \
		default = False, action = "store_true", help =
			"don't turn warnings into errors." )
	parser.add_option( "-p", "--install_prefix", \
		help = "The base path to install ocelot in." )
	parser.add_option( "--build_deb", \
		default = False, action = "store_true",
		help = "Build a .deb package of Ocelot." )
	parser.add_option( "--no_llvm", \
		default = False, action = "store_true", help = "Disable llvm support." )
	parser.add_option( "--no_opengl", \
		default = False, action = "store_true", help = "Disable opengl support." )
	parser.add_option( "--no_cuda_runtime", \
		default = False, action = "store_true",
		help = "Disable exporting cuda runtime symbols." )
	parser.add_option( "-m", "--message", default = "", \
		help = "the message describing the changes being committed." )
	
	( options, arguments ) = parser.parse_args()
	
	if options.submit:
		if options.test_level != 'full':
			print "Full test level required for a submit."
		options.test_level = 'full'
		options.build_target = ''

	# Do the build
	buildSucceeded = build(options)

	# Run unit tests
	testsPassed = runUnitTests(options, buildSucceeded)

	# Submit if the tests pass
	submit(options, testsPassed)
	
	if (buildSucceeded and (options.clean or 
		(options.test_level == 'none') or testsPassed)):
		sys.exit(0)
	else:
		print "Build failed"
		sys.exit(1)

################################################################################

################################################################################
## Guard Main
if __name__ == "__main__":
	main()

################################################################################

