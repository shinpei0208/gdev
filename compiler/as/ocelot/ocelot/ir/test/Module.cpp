/*!
	\file Module.cpp

	\author Andrew Kerr <arkerr@gatech.edu>

	\brief parses a set of PTX source files into modules and prints relevant information about
		each kernel
*/

#include <iostream>
#include <string>
#include <fstream>
#include <string>

#include <hydrazine/interface/ArgumentParser.h>
#include <hydrazine/interface/macros.h>
#include <hydrazine/interface/debug.h>

#include <ocelot/ir/interface/Module.h>
#include <ocelot/ir/interface/Kernel.h>
#include <ocelot/ir/interface/ControlFlowGraph.h>
#include <ocelot/parser/interface/PTXParser.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

void analyze(const char *filename) {
	using namespace std;
	using namespace ir;

	string fname = filename;

	ir::Module module(fname);

	cout << "Module: " << module.modulePath << ":\n";	

	Module::KernelMap::iterator k_it = module.begin(Instruction::PTX);

	for (; k_it != module.end(Instruction::PTX); ++k_it) {

		Kernel *kernel = k_it->second;
		cout << "  kernel " << kernel->name << "(\n";

		for (vector<Parameter>::iterator it = kernel->parameters.begin();
			it != kernel->parameters.end(); ++it) {
			cout << "    " << PTXOperand::toString((*it).type) << " " << (*it).name << ",\n";
		}
		cout << "  );\n\n";
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////

static void print_usage() {
	using namespace std;

	cout << "Module <input path>:\n\n";
	cout << " - parses the input file(s) and emits a list of kernels with their parameters\n";
}

int main(int argc, char **argv) {
	using namespace std;
	using namespace ir;

	if (argc == 1) {
		print_usage();
	}
	else {
		for (int i = 1; i < argc; i++) {
			analyze(argv[i]);
		}
	}

	return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

