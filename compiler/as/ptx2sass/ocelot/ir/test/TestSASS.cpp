/*!
 * \file SASSTest.cpp
 */

#include <iostream>
#include <fstream>
#include <string>
#include <hydrazine/interface/Test.h>
#include <ocelot/ir/interface/Module.h>
#include <ocelot/ir/interface/SASSKernel.h>
#include <ocelot/ir/interface/SASSStatement.h>
#include <ocelot/ir/interface/SASSInstruction.h>

namespace test {
class TestSASS : public Test {
public:
	TestSASS() {
		verbose = true;
		name = "TestSASS";
		description = "simple tests for SASS.";
	}

	bool doTest() {
		bool ret = true;
		status << "result of tests" << std::endl;
		ret &= testNop();
		ret &= testIadd();
		ret &= testBra();
		ret &= testStatement();
		ret &= testKernel();
		return ret;
	}

private:
	bool testNop() {
		ir::SASSInstruction instr(ir::SASSInstruction::Nop);
		instr.addOperand("CC.T");
		status << "  Instr:Nop = " << instr.toString() << std::endl;
		return (instr.toString() == "NOP CC.T;") ? true:false;
	}

	bool testIadd() {
		ir::SASSInstruction instr(ir::SASSInstruction::Iadd);
		instr.addModifier("X");
		instr.addOperand("R0");
		instr.addOperand("R1");
		instr.addOperand("R2");
		status << "  Instr:Iadd = " << instr.toString() << std::endl;
		return (instr.toString() == "IADD.X R0,R1,R2;") ? true:false;
	}

	bool testBra() {
		ir::SASSInstruction instr(ir::SASSInstruction::Bra);
		instr.setPredicate("@P0");
		instr.addOperand("0x120");
		status << "  Instr:Bra = " << instr.toString() << std::endl;
		return (instr.toString() == "@P0 BRA 0x120;") ? true:false;
	}

	bool testStatement() {
		ir::SASSStatement s(ir::SASSStatement::Instr,
			ir::SASSInstruction::Nop);
		status << "  Statement = " << s.toString() << std::endl;
		return true;
	}

	bool testKernel() {
		ir::Module module("../tests/ptx/thrust/Sort.ptx");
		for(ir::Module::KernelMap::const_iterator it =
			module.kernels().begin();
			it != module.kernels().end(); it++) {
			ir::PTXKernel* ptx =
				dynamic_cast<ir::PTXKernel*>(it->second);
			ir::SASSKernel sass(*ptx);
			ir::SASSStatement s(ir::SASSStatement::Instr,
				ir::SASSInstruction::Iadd);
			sass.addStatement(s);
			status << "  SASSKernel = " << sass.toString() <<
				std::endl;
		}
		return true;
	}
};
}

int main(int argc, char *argv[]) {
	test::TestSASS test;
	test.test();
	return test.passed();
}

