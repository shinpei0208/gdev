/*!
 * \file SASSKernel.h
 */

#ifndef __SASS_KERNEL_H__
#define __SASS_KERNEL_H__

#include <map>
#include <vector>
#include <ocelot/ir/interface/IRKernel.h>
#include <ocelot/ir/interface/PTXKernel.h>
#include <ocelot/ir/interface/Instruction.h>
#include <ocelot/ir/interface/SASSStatement.h>

namespace ir {
	class SASSKernel : public IRKernel {
	public:
		class Param {
		public:
			Param(unsigned int o = 0, unsigned int s = 0);
			unsigned int offset, size;
		};

		void assemble();
		const std::string& code() const;

		SASSKernel();
		SASSKernel(const IRKernel &k);
		std::string toString() const;

		void addStatement(SASSStatement s);
		void setParameters(const PTXKernel &k);
		unsigned int getParameterOffset(std::string name);
		unsigned int getParameterSize(std::string name);
		bool isParameter(std::string name);
		void setLabel(std::string name);
		void setSharedMemory(const PTXKernel *k);
		unsigned int getLocalSharedMemory(std::string name);
		unsigned int size();
		unsigned int instrBytes();
		unsigned int address();

	private:
		std::vector<SASSStatement> _statements;
		std::string _arch;
		unsigned int _mach;
		std::map<std::string, Param> _params;
		std::map<std::string, unsigned int> _labels;
		unsigned int _shared;
		std::map<std::string, unsigned int> _shared_gmap;
		std::map<std::string, unsigned int> _shared_lmap;
		std::string _code;
	};
}

#endif /* __SASS_KERNEL_H__ */
