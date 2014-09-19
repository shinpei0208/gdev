/*! \file PTXToILTranslator.h
 *  \author Rodrigo Dominguez <rdomingu@ece.neu.edu>
 *  \date April 19, 2010
 *  \brief The header file for the PTX to IL Translator class.
 */

#ifndef PTX_TO_IL_TRANSLATOR_H_INCLUDED
#define PTX_TO_IL_TRANSLATOR_H_INCLUDED

//C++ standard library includes
#include <string>
#include <map>

//Ocelot includes
#include <ocelot/translator/interface/Translator.h>
#include <ocelot/ir/interface/ILKernel.h>
#include <ocelot/executive/interface/ATIExecutableKernel.h>
#include <ocelot/analysis/interface/ControlTree.h>

typedef analysis::ControlTree CT;
typedef executive::ATIExecutableKernel ATIExecutableKernel;

namespace translator
{
	/*! \brief A translator from PTX to IL */
	class PTXToILTranslator : public Translator
	{
		public:
			/*! \brief Default constructor */
			PTXToILTranslator(OptimizationLevel l = NoOptimization);

			/*! \brief Translate a module from PTX to IL */
			ir::Kernel *translate(const ir::Kernel *k);
			ir::Kernel *translate(const ATIExecutableKernel *k);

		private:
			typedef std::map<int, ir::ILOperand> LiteralMap;
			typedef std::map<ir::PTXOperand::RegisterType, 
					ir::ILOperand> RegisterMap;

			LiteralMap  _literals;
			RegisterMap _registers;

			/*! \brief Pointer to the translated kernel */
			ir::ILKernel *_ilKernel;

			/*! \brief Maximum number of registers */
			static const unsigned int _maxRegs = 5000;

			/*! \brief Temporary register count */
			unsigned int _temps;

			void _translateInstructions();
			void _addKernelPrefix(const ATIExecutableKernel *k);

			void _translate(const CT::Node*        node);
			void _translate(const CT::InstNode*    node);
			void _translate(const CT::BlockNode*   node);
			void _translate(const CT::IfThenNode*  node);
			void _translate(const CT::NaturalNode* node);

			void _translate(const ir::PTXInstruction &i); 

			ir::ILOperand _translate(const ir::PTXOperand &o);
			ir::ILOperand _translateArrayDst(const ir::PTXOperand::Array a);
			ir::ILOperand _translateArraySrc(const ir::PTXOperand::Array a);
			ir::ILOperand _translateMemMask(unsigned int i);
			ir::ILOperand _translate(const ir::PTXOperand::SpecialRegister &s, 
					const ir::PTXOperand::VectorIndex& d);

			ir::ILInstruction::DataType _translate(
					const ir::PTXOperand::DataType d);

			void _translateAbs   (const ir::PTXInstruction& i);
			void _translateAdd   (const ir::PTXInstruction& i);
			void _translateAnd   (const ir::PTXInstruction& i);
			void _translateAtom  (const ir::PTXInstruction& i);
			void _translateBar   (const ir::PTXInstruction& i);
			void _translateBra   (const ir::PTXInstruction& i);
			void _translateClz   (const ir::PTXInstruction& i);
			void _translateCos   (const ir::PTXInstruction &i);
			void _translateCvt   (const ir::PTXInstruction& i);
			void _translateDiv   (const ir::PTXInstruction& i);
			void _translateEx2   (const ir::PTXInstruction& i);
			void _translateExit  (const ir::PTXInstruction& i);
			void _translateFma   (const ir::PTXInstruction& i);
			void _translateLd    (const ir::PTXInstruction& i);
			void _translateLdu   (const ir::PTXInstruction& i);
			void _translateLg2   (const ir::PTXInstruction& i);
			void _translateMad   (const ir::PTXInstruction& i);
			void _translateMax   (const ir::PTXInstruction& i);
			void _translateMembar(const ir::PTXInstruction& i);
			void _translateMin   (const ir::PTXInstruction& i);
			void _translateMov   (const ir::PTXInstruction& i);
			void _translateMul   (const ir::PTXInstruction& i);
			void _translateMul24 (const ir::PTXInstruction& i);
			void _translateNeg   (const ir::PTXInstruction& i);
			void _translateNot   (const ir::PTXInstruction& i);
			void _translateOr    (const ir::PTXInstruction& i);
			void _translatePopc  (const ir::PTXInstruction& i);
			void _translateRcp   (const ir::PTXInstruction& i);
			void _translateRem   (const ir::PTXInstruction& i);
			void _translateRsqrt (const ir::PTXInstruction& i);
			void _translateSelP  (const ir::PTXInstruction& i);
			void _translateSet   (const ir::PTXInstruction& i);
			void _translateSetP  (const ir::PTXInstruction& i);
			void _translateShl   (const ir::PTXInstruction& i);
			void _translateShr   (const ir::PTXInstruction& i);
			void _translateSlct  (const ir::PTXInstruction& i);
			void _translateSin   (const ir::PTXInstruction &i);
			void _translateSqrt  (const ir::PTXInstruction& i);
			void _translateSt    (const ir::PTXInstruction& i);
			void _translateSub   (const ir::PTXInstruction& i);
			void _translateVote  (const ir::PTXInstruction& i);
			void _translateXor   (const ir::PTXInstruction& i);

			void _translateLdParam      (const ir::PTXInstruction& i);
			void _translateLdSharedByte (const ir::PTXInstruction& i);
			void _translateLdSharedWord (const ir::PTXInstruction &i);
			void _translateLdSharedDword(const ir::PTXInstruction& i);
			void _translateStSharedByte (const ir::PTXInstruction& i);
			void _translateStSharedWord (const ir::PTXInstruction& i);
			void _translateStSharedDword(const ir::PTXInstruction& i);

			void _translateIDiv(const ir::PTXInstruction& i);
			void _translateUDiv(const ir::PTXInstruction& i);
			void _translateFDiv(const ir::PTXInstruction& i);

			void _translateIRem(const ir::PTXInstruction& i);
			void _translateURem(const ir::PTXInstruction& i);

			void _translateISetP(const ir::PTXInstruction& i);
			void _translateFSetP(const ir::PTXInstruction& i);

			ir::ILOperand _translateLiteral(int l);
			ir::ILOperand _translateConstantBuffer(const ir::PTXOperand o, unsigned int offset);

			void _convertSrc(const ir::PTXInstruction &i, ir::ILOperand& a);
			void _convert(const ir::PTXInstruction &i);
			void _convert(const ir::PTXInstruction &i, const ir::ILOperand a, ir::ILOperand& d);
			void _convertDst(const ir::PTXInstruction &i, const ir::ILOperand d);

			ir::ILOperand _tempRegister();

			void _add(const ir::ILInstruction &i);
	};

	/*! \brief returns the last instruction in a control tree node */
	ir::PTXInstruction* getLastIns(const CT::Node* node);

	/*! \brief returns true if n is a power of 2 */
	bool _isPowerOf2(unsigned int n);

	/*! \brief returns the log base 2 of a power of 2 number */
	int _Log2(unsigned int n);
}

#endif

