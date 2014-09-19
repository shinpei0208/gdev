#!/usr/bin/env python

################################################################################
#	\file   create_config.py
#	\author Gregory Diamos <gregory.diamos@gatech.edu>
#   \date   Friday April 20, 2012
#	\brief  Provides functionality for creating configure.ocelot files.
################################################################################

import os
import re
import subprocess
import time
from optparse import OptionParser
import sys

import StringIO

	
################################################################################
## A classes representing a configuration file
class TraceConfig:
	def __init__(self):
		self.enableDebugger         = False
		self.debuggerKernelFilter   = ""
		self.alwaysAttachDebugger   = True
		self.enableRaceDetector     = False
		self.ignoreIrrelevantWrites = False
		self.enableMemoryChecker    = True
		self.checkInitialization    = False
		self.enableTimer            = True
		self.database               = "traces/database.trace"
		self.timerOutputFile        = "kernel-times.json"

	def write(self, openFile):
		openFile.write("\ttrace: {\n")
		openFile.write("\t\tdatabase: \"" + self.database + "\",\n")

		openFile.write("\t\tmemoryChecker: {\n")
		openFile.write("\t\t\tenabled:             " + str(self.enableMemoryChecker) + ",\n")
		openFile.write("\t\t\tcheckInitialization: " + str(self.checkInitialization) + "\n")
		openFile.write("\t\t},\n")

		openFile.write("\t\traceDetector: {\n")
		openFile.write("\t\t\tenabled:                " + str(self.enableRaceDetector) + ",\n")
		openFile.write("\t\t\tignoreIrrelevantWrites: " + str(self.ignoreIrrelevantWrites) + "\n")
		openFile.write("\t\t},\n")

		openFile.write("\t\tkernelTimer: {\n")
		openFile.write("\t\t\tenabled:    " + str(self.enableTimer)     + ",\n")
		openFile.write("\t\t\toutputFile: \"" + str(self.timerOutputFile) + "\"\n" )
		openFile.write("\t\t},\n")

		openFile.write("\t\tdebugger: {\n")
		openFile.write("\t\t\tenabled:      " + str(self.enableDebugger) + ",\n")
		openFile.write("\t\t\tkernelFilter: \"" + self.debuggerKernelFilter + "\",\n")
		openFile.write("\t\t\talwaysAttach: " + str(self.alwaysAttachDebugger) + "\n")
		openFile.write("\t\t}\n")

		openFile.write("\t}")


class CudaConfig:
	def __init__(self):
		self.implementation = "CudaRuntime"
		self.tracePath      = "trace/CudaAPI.trace"

	def write(self, openFile):
		openFile.write("\tcuda: {\n")
		openFile.write("\t\timplementation: \"" + self.implementation + "\",\n")
		openFile.write("\t\ttracePath:      \"" + self.tracePath + "\"\n")
		openFile.write("\t}")

class ExecutiveConfig:
	def __init__(self):
		self.defaultDeviceID          = 0
		self.devices                  = "[\"nvidia\"]"
		self.preferredISA             = "nvidia"
		self.optimizationLevel        = "full"
		self.isaRequired              = False
		self.asynchronousKernelLaunch = True
		self.port                     = 2011
		self.host                     = "127.0.0.1"
		self.reconvergenceMechanism   = "ipdom"
		self.workerThreadLimit        = 8
		self.warpSize                 = 32

	def write(self, openFile):
		openFile.write("\texecutive: {\n")
		openFile.write("\t\tdevices:                  " + self.devices + ",\n")
		openFile.write("\t\tpreferredISA:             \"" + self.preferredISA + "\",\n")
		openFile.write("\t\toptimizationLevel:        \"" + self.optimizationLevel + "\",\n")
		openFile.write("\t\treconvergenceMechanism:   \"" + self.reconvergenceMechanism + "\",\n")
		openFile.write("\t\tdefaultDeviceID:          " + str(self.defaultDeviceID) + ",\n")
		openFile.write("\t\trequired:                 " + str(self.isaRequired) + ",\n")
		openFile.write("\t\tasynchronousKernelLaunch: " + str(self.asynchronousKernelLaunch) + ",\n")
		openFile.write("\t\tport:                     " + str(self.port) + ",\n")
		openFile.write("\t\thost:                     \"" + self.host + "\",\n")
		openFile.write("\t\tworkerThreadLimit:        " + str(self.workerThreadLimit) + ",\n")
		openFile.write("\t\twarpSize:                 " + str(self.warpSize) + "\n")
		openFile.write("\t}")

class OptimizationsConfig:
	def __init__(self):
		self.subkernelSize            = 10000
		self.structuralTransform      = False
		self.predicateToSelect        = False
		self.linearScanAllocation     = False
		self.mimdThreadScheduling     = False
		self.syncElimination          = False
		self.hoistSpecialValues       = False
		self.simplifyCFG              = True
		self.enforceLockStepExecution = False

	def write(self, openFile):
		openFile.write("\toptimizations: {\n")
		openFile.write("\t\tsubkernelSize:            " + str(self.subkernelSize) + ",\n")
		openFile.write("\t\tsimplifyCFG:              " + str(self.simplifyCFG) + ",\n")
		openFile.write("\t\tstructuralTransform:      " + str(self.structuralTransform) + ",\n")
		openFile.write("\t\tpredicateToSelect:        " + str(self.predicateToSelect) + ",\n")
		openFile.write("\t\tlinearScanAllocation:     " + str(self.linearScanAllocation) + ",\n")
		openFile.write("\t\tmimdThreadScheduling:     " + str(self.mimdThreadScheduling) + ",\n")
		openFile.write("\t\tsyncElimination:          " + str(self.syncElimination) + ",\n")
		openFile.write("\t\thoistSpecialValues:       " + str(self.hoistSpecialValues) + ",\n")
		openFile.write("\t\tenforceLockStepExecution: " + str(self.enforceLockStepExecution) + "\n")
		openFile.write("\t}")
	
class CheckpointConfig:
	def __init__(self):
		self.enabled = False
		self.path    = "."
		self.prefix  = "kernel_trace_"
		self.suffix  = ".trace"
		self.verify  = False
		
	def write(self, openFile):
		openFile.write("\tcheckpoint: {\n")
		openFile.write("\t\tenabled:  " + str(self.enabled) + ",\n")
		openFile.write("\t\tpath:   \"" + str(self.path) + "\",\n")
		openFile.write("\t\tprefix: \"" + self.prefix + "\",\n")
		openFile.write("\t\tsuffix: \"" + self.suffix + "\",\n")
		openFile.write("\t\tverify:   " + str(self.verify) + "\n")
		openFile.write("\t}")


class ConfigFile:
	def __init__(self):
		self.ocelotName    = "ocelot"
		self.version       = ""
		self.trace         = TraceConfig()
		self.cuda          = CudaConfig()
		self.executive     = ExecutiveConfig()
		self.checkpoint    = CheckpointConfig()
		self.optimizations = OptimizationsConfig()

	def write(self, openFile):
		openFile.write("{\n")

		openFile.write("\tocelot: \"" + self.ocelotName + "\",\n")
		openFile.write("\tversion: \"" + self.version + "\",\n")

		self.trace.write(openFile)
		openFile.write(",\n")

		self.cuda.write(openFile)
		openFile.write(",\n")

		self.executive.write(openFile)
		openFile.write(",\n")

		self.checkpoint.write(openFile)
		openFile.write(",\n")

		self.optimizations.write(openFile)
		openFile.write("\n")

		openFile.write("}\n")

	def __str__(self):
		output = StringIO.StringIO()
		self.write(output)
		
		return output.getvalue()


################################################################################

