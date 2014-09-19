EnsureSConsVersion(1,2)

import os

import inspect
import platform
import re
import subprocess
from SCons import SConf

def getDebianArchitecture():
	"""Determines the debian architecture
	
	return {deb_arch}
	"""

	# check for supported OS
	if os.name != 'posix':
		raise ValueError, 'Error: unknown OS.  Can only build .deb on linux.'
	
	try:
		dpkg_arch_path = which('dpkg-architecture')
	except:
		raise ValueError, "Failed to find 'dpkg-architecture' needed for .deb" \
			". Try installing dpkg-dev"

	# setup .deb environment variables
	arch = os.popen( \
		'dpkg-architecture -c \'echo $DEB_BUILD_ARCH\'').read().split()

	if len(arch) == 0:
		raise ValueError, 'Failed to get architecture from dpkg-architecture'

	return arch[0]

def getCudaPaths():
	"""Determines CUDA {bin,lib,include} paths
	
	returns (bin_path,lib_path,inc_path)
	"""

	# determine defaults
	if os.name == 'nt':
		bin_path = 'C:/CUDA/bin'
		lib_path = 'C:/CUDA/lib'
		inc_path = 'C:/CUDA/include'
	elif os.name == 'posix':
		bin_path = '/usr/local/cuda/bin'
		lib_path = '/usr/local/cuda/lib'
		inc_path = '/usr/local/cuda/include'
	else:
		raise ValueError, 'Error: unknown OS.  Where is nvcc installed?'
	 
	if platform.machine()[-2:] == '64':
		lib_path += '64'

	# override with environement variables
	if 'CUDA_BIN_PATH' in os.environ:
		bin_path = os.path.abspath(os.environ['CUDA_BIN_PATH'])
	if 'CUDA_LIB_PATH' in os.environ:
		lib_path = os.path.abspath(os.environ['CUDA_LIB_PATH'])
	if 'CUDA_INC_PATH' in os.environ:
		inc_path = os.path.abspath(os.environ['CUDA_INC_PATH'])

	return (bin_path,lib_path,inc_path)

def getBoost(env):
	"""Determines BOOST {bin_path,lib_path,include_path,libs}
	
	returns (bin_path,lib_path,inc_path,libs)
	"""

	# determine defaults
	if os.name == 'posix':
		bin_path = '/usr/bin'
		lib_path = '/usr/lib'
		inc_path = '/usr/include'
	elif os.name == 'nt':
		bitness = 32;
		# If Visual Studio then MSCV builder will have set TARGET_ARCH
		if 'TARGET_ARCH' in env:
			if env['TARGET_ARCH'] == 'x86_64':
				bitness = 64
		boost_path = '../../tools/boost_1_46_1'
		inc_path = boost_path + "/"
		bin_path = boost_path + "/bin"
		lib_path = boost_path + "/lib/win" + str(bitness)
	else:
		raise ValueError, 'Error: unknown OS.  Where is boost installed?'

	# override with environement variables
	if 'BOOST_BIN_PATH' in os.environ:
		bin_path = os.path.abspath(os.environ['BOOST_BIN_PATH'])
	if 'BOOST_LIB_PATH' in os.environ:
		lib_path = os.path.abspath(os.environ['BOOST_LIB_PATH'])
	if 'BOOST_INC_PATH' in os.environ:
		inc_path = os.path.abspath(os.environ['BOOST_INC_PATH'])

	libs = [];
	if os.name == 'nt':
		if env['mode'] == 'debug':
			debug_str = "gd"
		else:
			debug_str = ""
			lib_files = Flatten([Glob(os.path.join(lib_path,
				'libboost_system-vc100-mt-s' + debug_str + '-*')),
			Glob(os.path.join(lib_path,
				'libboost_filesystem-vc100-mt-s' + debug_str + '-*')),
			Glob(os.path.join(lib_path,
				'libboost_thread-vc100-mt-s' + debug_str + '-*'))])
			for f in lib_files:
				libs.append(os.path.basename(f.abspath))
			
	elif os.name == 'posix':
		ext = ""
		if os.path.exists(os.path.join(lib_path, 'libboost_system-mt.so')):
			ext = "-mt"
			
		libs = ['-lboost_system' + ext, '-lboost_filesystem' + ext,
			'-lboost_thread' + ext]
	else:
		raise ValueError, 'Error: unknown OS.  What are Boost library names?'

	return (bin_path,lib_path,inc_path,libs)

def getFlexPaths(env):
	"""Determines Flex {include} paths

	returns (inc_path)
	"""

	# determine defaults
	if os.name == 'posix':
		inc_path = ['/usr/include']
	elif os.name == 'nt':
		inc_path = ['../../tools/MinGW/msys/1.0/include']
	else:
		raise ValueError, 'Error: unknown OS.  Where is FLEX installed?'

	# override with environement variables
	if 'FLEX_INC_PATH' in os.environ:
		inc_path = os.path.abspath(os.environ['FLEX_INC_PATH'])

	return (inc_path)

def getGLEW(env, enabled):
	"""Determines GLEW {bin_path,lib_path,include_path,libs} and is it installed?

	returns (have_glew,bin_path,lib_path,inc_path,libs)
	"""
	
	configure = Configure(env)
	glew = configure.CheckLib('GLEW')		
	
	if not enabled:
		print "Glew disabled by user"
		return (False, [], [], [], [])

	if not glew:
		print "Glew disabled: not found"
		return (glew, [], [], [], [])

	# determine defaults
	if os.name == 'posix':
		bin_path = ['/usr/bin']
		lib_path = ['/usr/lib']
		inc_path = ['/usr/include']
	elif os.name == 'nt':
		bin_path = []
		lib_path = []
		inc_path = []
	else:
		raise ValueError, 'Error: unknown OS.  Where is GLEW installed?'

	# override with environement variables
	if 'GLEW_BIN_PATH' in os.environ:
		bin_path = [os.path.abspath(os.environ['GLEW_BIN_PATH'])]
	if 'GLEW_LIB_PATH' in os.environ:
		lib_path = [os.path.abspath(os.environ['GLEW_LIB_PATH'])]
	if 'GLEW_INC_PATH' in os.environ:
		inc_path = [os.path.abspath(os.environ['GLEW_INC_PATH'])]

	libs = []
	if os.name == 'nt':
		libs.append('opengl32.lib');
	elif os.name == 'posix':
		libs.append('-lGLEW');
		if platform.system() == 'Darwin':
			libs.append('-lGL')
			lib_path.append('/usr/X11/lib')
 
	return (glew,bin_path,lib_path,inc_path,libs)

def getLLVMPaths(enabled):
	"""Determines LLVM {have,bin,lib,include,cflags,lflags,libs} paths
	
	from user specified variables
	
	returns (have,bin_path,lib_path,inc_path,cflags,lflags,libs)
	"""

	# determine defaults
	bin_path = ['/usr/local/bin']
	lib_path = ['/usr/local/lib']
	inc_path = ['/usr/local/include']
	cflags   = ['-I/usr/local/include', '-D_DEBUG', '-D_GNU_SOURCE',
	            '-D__STDC_CONSTANT_MACROS', '-D__STDC_FORMAT_MACROS',
	            '-D__STDC_LIMIT_MACROS']
	libs     = ['-lLLVMAsmParser', '-lLLVMX86CodeGen', '-lLLVMSelectionDAG',
	            '-lLLVMAsmPrinter', '-lLLVMX86AsmParser', '-lLLVMMCParser',
	            '-lLLVMX86Disassembler', '-lLLVMX86Desc', '-lLLVMX86Info',
	            '-lLLVMX86AsmPrinter', '-lLLVMX86Utils', '-lLLVMJIT',
	            '-lLLVMRuntimeDyld', '-lLLVMExecutionEngine', '-lLLVMCodeGen',
	            '-lLLVMScalarOpts', '-lLLVMInstCombine', '-lLLVMTransformUtils',
	            '-lLLVMipa', '-lLLVMAnalysis', '-lLLVMTarget', '-lLLVMMC',
	            '-lLLVMObject', '-lLLVMCore', '-lLLVMSupport']
	
	if os.name == 'nt':
		lflags =  []
		cflags =  []
	# Do we need this?
	#	libs   += ['Shell32', 'Advapi32']
	else:
		lflags = ['-L/usr/local/lib', '-lpthread', '-ldl', '-lm']
		libs   = ['-l' + l for l in libs]

	# override defaults
	if 'LLVM_BIN_PATH' in os.environ:
		bin_path = [os.path.abspath(os.environ['LLVM_BIN_PATH'])]
	if 'LLVM_LIB_PATH' in os.environ:
		lib_path = [os.path.abspath(os.environ['LLVM_LIB_PATH'])]
	if 'LLVM_INC_PATH' in os.environ:
		inc_path = [os.path.abspath(os.environ['LLVM_INC_PATH'])]
	if 'LLVM_CFLAGS' in os.environ:
		cflags   = os.environ['LLVM_CFLAGS'].split()
	if 'LLVM_LFLAGS' in os.environ:
		lflags   = os.environ['LLVM_LFLAGS'].split()
	if 'LLVM_LIBS' in os.environ:
		libs     = os.environ['LLVM_LIBS'].split()
	
	return (True,bin_path,lib_path,inc_path,cflags,lflags,libs)

def getLLVMPaths(enabled):
	"""Determines LLVM {have,bin,lib,include,cflags,lflags,libs} paths
	
	returns (have,bin_path,lib_path,inc_path,cflags,lflags,libs)
	"""
	
	if not enabled:
		return (False, [], [], [], [], [], [])
	
	if 'USER_SPECIFIED_LLVM_PATHS' in os.environ:
		return getUserSpecifiedLLVMPaths()
	else:
		try:
			llvm_config_path = which('llvm-config')
		except:
			print 'Failed to find llvm-config'
			return (False, [], [], [], [], [], [])
	
	# determine defaults
	bin_path = os.popen('llvm-config --bindir').read().split()
	lib_path = os.popen('llvm-config --libdir').read().split()
	inc_path = os.popen('llvm-config --includedir').read().split()
	cflags   = os.popen('llvm-config --cppflags').read().split()
	lflags   = os.popen('llvm-config --ldflags').read().split()
	libs     = os.popen('llvm-config --libs core jit native \
		asmparser instcombine').read().split()
	
	# remove -DNDEBUG
	if '-DNDEBUG' in cflags:
		cflags.remove('-DNDEBUG')

	# remove lib_path from libs
	for lib in libs:
		if lib[0:2] == "-L":
			libs.remove(lib)

	# remove inc_path from cflags
	for flag in cflags:
		if flag[0:2] == "-I":
			cflags.remove(flag)

	return (True,bin_path,lib_path,inc_path,cflags,lflags,libs)
	
def getTools():
	result = []
	if os.name == 'nt':
		result = ['default', 'msvc']
	elif os.name == 'posix':
		result = ['default', 'c++', 'g++']
	else:
		result = ['default']

	return result;


OldEnvironment = Environment;


# this dictionary maps the name of a compiler program to a dictionary mapping the name of
# a compiler switch of interest to the specific switch implementing the feature
gCompilerOptions = {
		'gcc' : {'warn_all' : '-Wall',
			'warn_errors' : '-Werror',
			'optimization' : '-O2', 'debug' : '-g', 
			'exception_handling' : '', 'standard': ''},
		'g++' : {'warn_all' : '-Wall',
			'warn_errors' : '-Werror',
			'optimization' : '-O2', 'debug' : '-g', 
			'exception_handling' : '', 'standard': '-std=c++0x'},
		'c++' : {'warn_all' : '-Wall',
			'warn_errors' : '-Werror',
			'optimization' : '-O2', 'debug' : '-g',
			'exception_handling' : '',
			'standard': ['-stdlib=libc++', '-std=c++0x', '-pthread']},
		'cl'  : {'warn_all' : '/Wall',
			'warn_errors' : '/WX',
			'optimization' : ['/Zi', '/Ox', '/DNDEBUG'],
			'debug' : ['/Zi', '/Od', '/D_DEBUG', '/RTC1'],
			'exception_handling': '/EHsc',
			'standard': ['/GS', '/GR', '/Gd', '/fp:precise',
				'/Zc:wchar_t','/Zc:forScope', '/DYY_NO_UNISTD_H',
				'/D_WIN32_WINNT=0x0601', '/wd4482', '/wd4503']}
	}

def updateCompilerOptionsFromEnv(env):
	if 'CRT_LIB_OPTION' in env:
		gCompilerOptions['cl']['optimization'].append(env['CRT_LIB_OPTION'])
		gCompilerOptions['cl']['debug'].append(env['CRT_LIB_OPTION']+'d')

# this dictionary maps the name of a linker program to a dictionary mapping the name of
# a linker switch of interest to the specific switch implementing the feature
gLinkerOptions = {
		'gcc'  : {'debug' : ''},
		'g++'  : {'debug' : ''},
		'c++'  : {'debug' : ''},
		'link' : {'debug' : '/debug'}
	}


def getCFLAGS(mode, warn, warnings_as_errors, CC):
	result = []
	if mode == 'release':
		# turn on optimization
		result.append(gCompilerOptions[CC]['optimization'])
	elif mode == 'debug':
		# turn on debug mode
		result.append(gCompilerOptions[CC]['debug'])
		result.append('-DOCELOT_DEBUG')

	if warn:
		# turn on all warnings
		result.append(gCompilerOptions[CC]['warn_all'])

	if warnings_as_errors:
		# treat warnings as errors
		result.append(gCompilerOptions[CC]['warn_errors'])

	result.append(gCompilerOptions[CC]['standard'])

	return result

def getLibCXXPaths():
	"""Determines libc++ path

	returns (inc_path, lib_path)
	"""

	# determine defaults
	if os.name == 'posix':
		inc_path = '/usr/include'
		lib_path = '/usr/lib/libc++.so'
	else:
		raise ValueError, 'Error: unknown OS.  Where is libc++ installed?'

	# override with environement variables
	if 'LIBCXX_INC_PATH' in os.environ:
		inc_path = os.path.abspath(os.environ['LIBCXX_INC_PATH'])
	if 'LIBCXX_LIB_PATH' in os.environ:
		lib_path = os.path.abspath(os.environ['LIBCXX_LIB_PATH'])

	return (inc_path, lib_path)

def getCXXFLAGS(mode, warn, warnings_as_errors, CXX):
	if not CXX in gCompilerOptions:
		raise ValueError, 'Error: No support for compiler \'' + CXX + '\''
	
	result = []
	if mode == 'release':
		# turn on optimization
		result.append(gCompilerOptions[CXX]['optimization'])
	elif mode == 'debug':
		# turn on debug mode
		result.append(gCompilerOptions[CXX]['debug'])
	# enable exception handling
	result.append(gCompilerOptions[CXX]['exception_handling'])

	if warn:
		# turn on all warnings
		result.append(gCompilerOptions[CXX]['warn_all'])

	if warnings_as_errors:
		# treat warnings as errors
		result.append(gCompilerOptions[CXX]['warn_errors'])

	result.append(gCompilerOptions[CXX]['standard'])

	return result

def getLINKFLAGS(mode, LINK):
	result = []
	if mode == 'debug':
		# turn on debug mode
		result.append(gLinkerOptions[LINK]['debug'])

	if LINK == 'c++':
		result.append(getLibCXXPaths()[1]);

	return result

def getVersion(base):
	try:
		svn_path = which('svn')
	except:
		print 'Failed to get subversion revision'
		return base + '.0'

	process = subprocess.Popen('svn info ..', shell=True,
		stdout=subprocess.PIPE, stderr=subprocess.PIPE)

	(svn_info, std_err_data) = process.communicate()
	
	match = re.search('Revision: ', svn_info)
	revision = 'unknown'
	if match:
		end = re.search('\n', svn_info[match.end():])
		if end:
			revision = svn_info[match.end():match.end()+end.start()]
	else:
		print 'Failed to get SVN repository version!'

	return base + '.' + revision

def fixPath(path):
	if (os.name == 'nt'):
		return path.replace('\\', '\\\\')
	else:
		return path

import collections

def flatten(l):
	for el in l:
		if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
			for sub in flatten(el):
				yield sub
		else:
			yield el

def defineConfigFlags(env):
	
	include_path = os.path.join(env['INSTALL_PATH'], "include")
	library_path = os.path.join(env['INSTALL_PATH'], "lib")
	bin_path     = os.path.join(env['INSTALL_PATH'], "bin")

	configFlags = env['CXXFLAGS'] + " ".join( 
		["%s\"\\\"%s\\\"\"" % x for x in (
			('-DOCELOT_CXXFLAGS=', " ".join(flatten(env['CXXFLAGS']))),
			('-DPACKAGE=', 'ocelot'),
			('-DVERSION=', env['VERSION']),
			('-DOCELOT_PREFIX_PATH=', fixPath(env['INSTALL_PATH'])),
			('-DOCELOT_LDFLAGS=', fixPath(env['OCELOT_LDFLAGS'])),
			('-L', fixPath(library_path)),
			('-DOCELOT_INCLUDE_PATH=', fixPath(include_path)),
			('-DOCELOT_LIB_PATH=', fixPath(library_path)),
			('-DOCELOT_BIN_PATH=', fixPath(bin_path))
		)])

	env.Replace(OCELOT_CONFIG_FLAGS = configFlags)

def importEnvironment():
	env = {  }
	
	if 'PATH' in os.environ:
		env['PATH'] = os.environ['PATH']
	
	if 'CXX' in os.environ:
		env['CXX'] = os.environ['CXX']
	
	if 'CC' in os.environ:
		env['CC'] = os.environ['CC']
	
	if 'TMP' in os.environ:
		env['TMP'] = os.environ['TMP']
	
	if 'LD_LIBRARY_PATH' in os.environ:
		env['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']

	return env

def Environment():
	vars = Variables()

	# allow the user discretion to choose the MSVC version
	if os.name == 'nt':
		vars.Add(EnumVariable('MSVC_VERSION', 'MS Visual C++ version',
			'10.0', allowed_values=('8.0', '9.0', '10.0')))
		vars.Add(EnumVariable('TARGET_ARCH',
			'MS Visual C++ target architecture',
			'x86_64', allowed_values=('x86', 'x86_64')))
		vars.Add(EnumVariable('CRT_LIB_OPTION',
			'MS Visual C++ runtime library option for cl.exe ',
			'/MD', allowed_values=('/MD', '/MT')))


	# add a variable to handle RELEASE/DEBUG mode
	vars.Add(EnumVariable('mode', 'Release versus debug mode', 'release',
		allowed_values = ('release', 'debug')))

	# add a variable to treat warnings as errors
	vars.Add(BoolVariable('Werror', 'Treat warnings as errors', 1))

	# add a variable to handle warnings
	vars.Add(BoolVariable('Wall', 'Enable all compilation warnings', 1))
	
	# shared or static libraries
	libraryDefault = 'shared'
	if os.name == 'nt':
		libraryDefault = 'static'
	
	vars.Add(EnumVariable('library', 'Build shared or static library',
		libraryDefault, allowed_values = ('shared', 'static')))
	
	vars.Add(BoolVariable('enable_llvm',
		'Compile in support for LLVM if available', 1))
	
	vars.Add(BoolVariable('enable_opengl',
		'Compile in support for OpenGL if available', 1))
	
	vars.Add(BoolVariable('enable_cuda_runtime',
		'Export cuda runtime symbols (prevents linking Ocelot '
		'against another version of the CUDA runtime)', 1))

	# add a variable to compile the ocelot unit tests
	vars.Add(EnumVariable('test_level',
		'Build the ocelot unit tests at the given test level', 'none',
		allowed_values = ('none', 'basic', 'full', 'sass')))

	# add a variable to determine the install path
	if os.name == 'nt':
		if 'USERPROFILE' in os.environ:
			default_install_path = os.path.join(
				os.environ['USERPROFILE'], 'ocelot')
		else:
			default_install_path = '/ocelot'
	elif os.name == 'posix':
		default_install_path = '/usr/local'
	else:
		raise ValueError, 'Error: unknown OS.  " \
			"Where is Ocelot installed by default?'
	
	if 'OCELOT_INSTALL_PATH' in os.environ:
		default_install_path = os.environ['OCELOT_INSTALL_PATH']
		
	vars.Add(PathVariable('install_path', 'The ocelot install path',
		default_install_path, PathVariable.PathIsDirCreate))
 
	vars.Add(BoolVariable('install', 'Include ocelot install path in default '
		'targets that will be built and configure to install in the '
		'install_path (defaults to false unless one of the targets is '
		'"install")', 0))

	# create an Environment
	env = OldEnvironment(ENV = importEnvironment(), \
		tools = getTools(), variables = vars)

	# disable the cuda runtime
	if env['enable_cuda_runtime']:
		env.Replace(EXCLUDE_CUDA_RUNTIME = False)
	else:
		env.Replace(EXCLUDE_CUDA_RUNTIME = True)

	# update compiler options with variables in the environment
	# (such as those from the command line)
	updateCompilerOptionsFromEnv(env)

	# always link with the c++ compiler
	if os.name != 'nt':
		env['LINK'] = env['CXX']
	
	# set the version
	env.Replace(VERSION = getVersion("2.1"))
	
	# get the absolute path to the directory containing
	# this source file
	thisFile = inspect.getabsfile(Environment)
	thisDir = os.path.dirname(thisFile)

	# get C compiler switches
	env.AppendUnique(CFLAGS = getCFLAGS(env['mode'], env['Wall'], \
		env['Werror'], env.subst('$CC')))

	# get CXX compiler switches
	env.AppendUnique(CXXFLAGS = getCXXFLAGS(env['mode'], env['Wall'], \
		env['Werror'], env.subst('$CXX')))

	# get linker switches
	env.AppendUnique(LINKFLAGS = getLINKFLAGS(env['mode'], env.subst('$LINK')))

	# get bison switches
	env.AppendUnique(YACCFLAGS = "-d")
	
	# Install paths
	if env['install']:
		env.Replace(INSTALL_PATH = os.path.abspath(env['install_path']))
	else:
		env.Replace(INSTALL_PATH = os.path.abspath('.'))

	# Set the debian architecture
	if 'debian' in COMMAND_LINE_TARGETS:
		env.Replace(deb_arch = getDebianArchitecture())
	else:
		env.Replace(deb_arch = 'unknown')
		
	# get CUDA paths
	(cuda_exe_path, cuda_lib_path, cuda_inc_path)  = getCudaPaths()
	
	# CUDA builder
	env.Append(BUILDERS = {'Cuda': Builder(
		action=cuda_exe_path + '/nvcc -arch=sm_20 $SOURCE -c -o $TARGET',
		suffix = '.o',
		src_suffix = '.cu'
	)})

	# CUDA builder
	nvccPath = cuda_exe_path + ('/' if cuda_exe_path != '' else '')
	env.Append(BUILDERS = {'Cuda': Builder(
		action= nvccPath + 'nvcc -arch=sm_20 $SOURCE -c -o $TARGET',
		suffix = '.o',
		src_suffix = '.cu'
	)})
	
	# append the default VC++ paths
	if os.name == 'nt':
		env.Append(LIBPATH = str.split(os.environ['LIB'], ';'))
		env.Append(CPPPATH = str.split(os.environ['INCLUDE'], ';'))
	
	# set the build path
	env.Replace(BUILD_ROOT = str(env.Dir('.')))

	# get boost information
	(boost_exe_path,boost_lib_path,boost_inc_path,boost_libs) = getBoost(env)
	env.AppendUnique(LIBPATH = [boost_lib_path])
	env.AppendUnique(CPPPATH = [boost_inc_path])
	env.AppendUnique(EXTRA_LIBS = boost_libs) 
	

	# get GLEW information
	(glew,glew_exe_path,glew_lib_path,glew_inc_path,
		glew_libs) = getGLEW(env, env['enable_opengl'])
	if glew:
		env.AppendUnique(LIBPATH = glew_lib_path)
		env.AppendUnique(CPPPATH = glew_inc_path)
		env.AppendUnique(EXTRA_LIBS = glew_libs) 
	env.Replace(HAVE_GLEW = glew)

	# get Flex paths
	(flex_inc_path) = getFlexPaths(env)
	env.AppendUnique(CPPPATH = [flex_inc_path])

	# get libc++
	if env['CXX'] == 'c++':
		env.AppendUnique(CPPPATH = getLibCXXPaths()[0])

	# get llvm paths
	(llvm, llvm_exe_path,llvm_lib_path,llvm_inc_path,llvm_cflags,\
		llvm_lflags,llvm_libs) = getLLVMPaths(env['enable_llvm'])
	env.AppendUnique(LIBPATH = llvm_lib_path)
	env.AppendUnique(CPPPATH = llvm_inc_path)
	env.AppendUnique(CXXFLAGS = llvm_cflags)
	env.AppendUnique(LINKFLAGS = llvm_lflags)
	env.Replace(HAVE_LLVM = llvm)
	env.Replace(LLVM_LIBS = llvm_libs)

	# set ocelot include path
	env.Prepend(CPPPATH = os.path.dirname(thisDir))

	if env['install']:
		env.AppendUnique(LIBPATH = os.path.abspath(
			os.path.join(env['install_path'], 'lib')))
	
	if os.name == 'nt':      
               env.AppendUnique(CFLAGS   = "-DYY_NO_UNISTD_H")
               env.AppendUnique(CXXFLAGS = "-DYY_NO_UNISTD_H")


	# we need libdl on linux, and librt
	if os.name == 'posix':
		env.AppendUnique(EXTRA_LIBS = ['-ldl']) 
		if platform.system() != 'Darwin':
			env.AppendUnique(EXTRA_LIBS = ['-lrt', '-lpthread'])
	
	# set ocelot libs
	ocelot_libs = '-locelot'
	env.Replace(OCELOT_LDFLAGS=ocelot_libs)

	# generate OcelotConfig flags
	defineConfigFlags(env)

	# generate help text
	Help(vars.GenerateHelpText(env))

	return env

