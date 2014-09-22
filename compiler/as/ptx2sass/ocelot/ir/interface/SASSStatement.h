/*!
 * \file SASSStatement.h
 */

#ifndef __SASS_STATEMENT_H__
#define __SASS_STATEMENT_H__

#include <string>
#include <vector>
#include <ocelot/ir/interface/SASSInstruction.h>

namespace ir {
	class SASSStatement {
	public:
		enum Type {
			Type_Invalid = -1,
			Entry,
			Param,
			Instr,
		};

		Type type;
		SASSInstruction instr;

		SASSStatement(Type type = Type_Invalid,
			SASSInstruction::Opcode op = SASSInstruction::Opcode_Invalid);
		~SASSStatement();
		std::string toString() const;
		const SASSStatement& operator=(const SASSStatement& s);
		static std::string toString(Type type);
	};
}

#endif /* __SASS_STATEMENT_H__ */
