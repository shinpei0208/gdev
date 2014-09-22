/*!
	\file RuntimeException.cpp

	\author Andrew Kerr <arkerr@gatech.edu>

	\brief defines an exception to be thrown at runtime during the emulation or execution of a
		kernel

	\date 29 Jan 2009
*/

#include <sstream>
#include <ocelot/executive/interface/RuntimeException.h>

executive::RuntimeException::RuntimeException( ): 
	message("Unspecified runtime exception"), PC(-1) {

}

executive::RuntimeException::~RuntimeException( ) throw() {

}

executive::RuntimeException::RuntimeException(std::string msg): 
	message(msg), PC(-1), thread(0), cta(0) {

}

executive::RuntimeException::RuntimeException(std::string msg, 
	ir::PTXInstruction instr):
	message(msg), PC(0), thread(0), cta(0), instruction(instr) {

}

executive::RuntimeException::RuntimeException(std::string msg, 
	int pc, ir::PTXInstruction instr): 
	message(msg), PC(pc), thread(0), cta(0), instruction(instr) {

}

executive::RuntimeException::RuntimeException(std::string msg, int pc, int t, 
	int c, ir::PTXInstruction instr): 
	message(msg), PC(pc), thread(t), cta(c), instruction(instr) {

}

std::string executive::RuntimeException::toString() const {
	using namespace std;
	string error = message;
	if (PC >= 0) {
		stringstream ss;
		ss << "[PC " << PC << "] ";
		ss << "[thread " << thread << "] ";
		ss << "[cta " << cta << "] ";
		ss << instruction.toString() << " - " << message;
		error = ss.str();
	}
	return error;
}

const char* executive::RuntimeException::what() const  throw() {
	return message.c_str();
}
