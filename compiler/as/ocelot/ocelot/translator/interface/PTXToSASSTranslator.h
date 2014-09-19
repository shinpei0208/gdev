/*!
 * \file PTXToSASSTranslator.h
 * \brief header of PTXToSASSTranslator.cpp.
 */

#ifndef __PTX_TO_SASS_TRANSLATOR_H__
#define __PTX_TO_SASS_TRANSLATOR_H__

#include <string>
#include <map>
#include <ocelot/analysis/interface/ControlTree.h>
#include <ocelot/ir/interface/PTXKernel.h>
#include <ocelot/ir/interface/SASSKernel.h>
#include <ocelot/translator/interface/Translator.h>

typedef analysis::ControlTree ControlTree;

namespace translator {
	/*! \brief A translator from PTX to SASS */
	class PTXToSASSTranslator : public Translator {
	public:
		typedef ir::Instruction::RegisterType RegType;

		static const RegType sassRegStart = 2;
		static const RegType sassRegEnd = 63;
		static const RegType ptx64flag = (1<<31);
		static const RegType sassInvalid = (1<<31);

		class RegisterRange {
		public:
			int head, tail; // range(line#)
			RegType sass;
			RegisterRange(int h = -1, int t = -1, RegType s = sassInvalid);
			bool isAssigned(int l);
			bool isObsoleted(int l);
		};

		PTXToSASSTranslator(OptimizationLevel l = NoOptimization);
		ir::Kernel *translate(const ir::Kernel *k);
		ir::Kernel* translate(ir::PTXKernel* ptx);
		static std::string makeImmediate(int n, bool rz = true);

	private:
		std::map<RegType, RegType> _reg32;
		std::map<RegType, RegType> _pred;
		std::map<RegType, RegType> _reg64;
		std::map<RegType, RegisterRange> _rangemap;
		int _line;
		ir::SASSKernel* _sass;
		ir::PTXKernel* _ptx;

		void _translate(const ControlTree::Node* n);
		void _translate(const ControlTree::InstNode* n);
		void _translate(const ControlTree::BlockNode* n);
		void _translate(const ControlTree::IfThenNode* n);
		void _translate(const ControlTree::NaturalNode* n);
		void _translate(const ir::PTXInstruction &i); 
		void _translateAdd(const ir::PTXInstruction &i);
		void _translateAnd(const ir::PTXInstruction &i);
		void _translateBar(const ir::PTXInstruction &i);
		void _translateBra(const ir::PTXInstruction &i);
		void _translateCvta(const ir::PTXInstruction &i);
		void _translateDiv(const ir::PTXInstruction &i);
		void _translateFma(const ir::PTXInstruction &i);
		void _translateLd(const ir::PTXInstruction &i);
		void _translateMad(const ir::PTXInstruction &i);
		void _translateMov(const ir::PTXInstruction &i);
		void _translateMul(const ir::PTXInstruction &i);
		void _translateNeg(const ir::PTXInstruction &i);
		void _translateOr(const ir::PTXInstruction &i);
		void _translateRet(const ir::PTXInstruction &i);
		void _translateSelp(const ir::PTXInstruction &i);
		void _translateSetp(const ir::PTXInstruction &i);
		void _translateShl(const ir::PTXInstruction &i);
		void _translateShr(const ir::PTXInstruction &i);
		void _translateSt(const ir::PTXInstruction &i);
		void _translateSub(const ir::PTXInstruction &i);

		std::string makeReg(ir::PTXOperand o, int a = 0, std::string mod = "");
		std::string makeAddress(ir::PTXOperand o, int a = 0);
		std::string makeImmAddress(int n);
		std::string makeSpecial(ir::PTXOperand o);
		std::string makeModifier(ir::PTXInstruction::Modifier m);
		std::string makeComparison(ir::PTXInstruction::CmpOp c);

		void _makeRange();
		void _makeRange(const ControlTree::Node* n,
			int *from = NULL, int *to = NULL);
		void _makeRange(const ControlTree::InstNode* n,
			int *from = NULL, int *to = NULL);
		void _makeRange(const ControlTree::BlockNode* n,
			int *from = NULL, int *to = NULL);
		void _makeRange(const ControlTree::IfThenNode* n,
			int *from = NULL, int *to = NULL);
		void _makeRange(const ControlTree::NaturalNode* n,
			int *from = NULL, int *to = NULL);
		void _makeRange(const ir::PTXInstruction &i, 
			int *from = NULL, int *to = NULL);
		void _makeRange(const ir::PTXOperand &o);
		void _renewRange(int from, int to);
		void _refreshRange();
		RegType _assignRange32(RegType ptx);
		RegType _assignRange64(RegType ptx);
	};
}

#endif /* __PTX_TO_SASS_TRANSLATOR_H__ */
