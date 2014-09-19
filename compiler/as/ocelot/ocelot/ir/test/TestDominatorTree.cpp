/*!
	\file TestDominatorTree.cpp

	\author Andrew Kerr <arkerr@gatech.edu>

	\brief tests CFG analysis by forming a CFG from a sequence of instructions known to produce
		a certain CFG and then comparing output
*/

#include <memory.h>
#include <sstream>
#include <fstream>
#include <hydrazine/interface/Test.h>

#include <hydrazine/interface/ArgumentParser.h>
#include <hydrazine/interface/macros.h>
#include <hydrazine/interface/debug.h>

#include <ocelot/ir/interface/Kernel.h>
#include <ocelot/ir/interface/ControlFlowGraph.h>
#include <ocelot/ir/interface/DominatorTree.h>
#include <ocelot/ir/interface/PostdominatorTree.h>
#include <ocelot/ir/test/SamplePTX.h>

namespace test {

class TestDominatorTree: public Test {
public:
	bool verbose;
	std::stringstream output;
	char filename[256];

	TestDominatorTree() {
		memset(filename, 0, 256);
		verbose = false;
		Name = "TestDominatorTree";

		Description = "Constructs a dominator tree from a control flow graph produced from ";
		Description += "a statically constructed kernel.";
	}

	bool doTest( std::string & status ) {
		bool result = true;
		if (std::string(filename) != "") {
			result = test4(std::string(filename));
		}
		else {
			result = test1() && test2();
		}
		return result;
	}


private:

	bool test1() {
		using namespace std;
		using namespace ir;

		bool result = true;
		ir::Kernel::PTXStatementVector statements;
		ir::Kernel::PTXInstructionVector instructions;
		ir::ControlFlowGraph *cfg;

		statements = Sample_cfg();
		cfg = new ControlFlowGraph;

		Kernel::constructCFG(*cfg, instructions, statements.begin(), statements.end());

		{
			// dominator tree
			DominatorTree dtree(cfg);

		}
		delete cfg;

		return result;

	}

	bool test2() {
		using namespace std;
		using namespace ir;

		bool result = true;
		ir::Kernel::PTXStatementVector statements;
		ir::Kernel::PTXInstructionVector instructions;
		ir::ControlFlowGraph *cfg = new ControlFlowGraph;

		statements = Sample_cfg();

		Kernel::constructCFG(*cfg, instructions, statements.begin(), statements.end());

		{
			// dominator tree
			PostdominatorTree dtree(cfg);
		}
		delete cfg;

		return result;
	}

	bool test3() {
		using namespace std;
		using namespace ir;

		bool result = true;
		ir::Kernel::PTXStatementVector statements;
		ir::Kernel::PTXInstructionVector instructions;
		
		ir::ControlFlowGraph *cfg = new ControlFlowGraph;

		statements = Sample_Divergence();
		Kernel::constructCFG(*cfg, instructions, statements.begin(), statements.end());
		{
			vector< BasicBlock *> kernel = cfg->executable_sequence();
			ofstream file_cfg("vis/kernel.txt");
			for (vector<BasicBlock*>::iterator it = kernel.begin(); it != kernel.end(); ++it) {
				file_cfg << (*it)->label << ":\n";
			}
		}
		{
			DominatorTree dtree(cfg);
			PostdominatorTree pdtree(cfg);
		}
		delete cfg;

		return result;
	}

	bool test4(std::string filename) {
		bool result = true;
		return result;
	}
};

}

int main(int argc, char **argv) {
	using namespace std;
	using namespace ir;
	using namespace test;

	hydrazine::ArgumentParser parser( argc, argv );
	test::TestDominatorTree test;

	parser.description( test.description() );

	bool help;

	parser.parse( "-h", help, "Display this help message." );
	parser.parse( "-v", test.verbose, "Print out info after the test." );
/*	parser.parse( "-i", test., "Load a PTX document, perform CFG analysis on each kernel, "
		"and emit .dot files in created directories named after each kernel."); */

	if( help ) {
		std::cout << parser.help();
		return 2;
	}

	test.test();

	if( test.verbose )	{
		std::cout << test.toString();
		std::cout << test.output.str() << "\n";
	}

	return test.passed();
}

