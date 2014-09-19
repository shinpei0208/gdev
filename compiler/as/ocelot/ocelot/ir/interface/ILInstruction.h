/*! \file ILInstruction.h
 *  \author Rodrigo Dominguez <rdomingu@ece.neu.edu>
 *  \date April 27, 2010
 *  \brief The header file for the IL Instruction class.
 */

#ifndef IL_INSTRUCTION_H_INCLUDED
#define IL_INSTRUCTION_H_INCLUDED

// Ocelot includes
#include <ocelot/ir/interface/Instruction.h>
#include <ocelot/ir/interface/ILOperand.h>

namespace ir
{
	/*! \brief A class used to represent any IL Instruction */
	class ILInstruction : public Instruction
	{
		public:
			/*! \brief The opcode of the instruction */
			enum Opcode
			{
				Abs,
				Add,
				And,
				Break,
				Cmov_Logical,
				Cos_Vec,
				Div,
				Else,
				End,
				EndIf,
				EndLoop,
				Eq,
				Exp_Vec,
				Ffb_Hi,
				Fence,
				Fma,
				FtoI,
				FtoU,
				Ge,
				Iadd,
				Iand,
				Icbits,
				Ieq,
				IfLogicalNZ,
				IfLogicalZ,
				Ige,
				Ilt,
				Imax,
				Imin,
				Imul,
				Ine,
				Inegate,
				Inot,
				Ior,
				Ishl,
				Ishr,
				ItoF,
				Ixor,
				Lds_And_Id,
				Lds_Load_Id,
				Lds_Or_Id,
				Lds_Read_Add_Id,
				Lds_Store_Id,
				Log_Vec,
				Lt,
				Mad,
				Mov,
				Mul,
				Ne,
				Rcp,
				Round_Nearest,
				Round_Neginf,
				Rsq_Vec,
				Sin_Vec,
				Sub,
				Sqrt_Vec,
				Uav_Arena_Load_Id,
				Uav_Arena_Store_Id,
				Uav_Raw_Load_Id,
				Uav_Raw_Store_Id,
				Uav_Read_Add_Id,
				Uav_Read_Max_Id,
				Uav_Read_Min_Id,
				Uav_Read_Xchg_Id,
				Udiv,
				Umul,
				Umul24,
				Ushr,
				UtoF,
				WhileLoop,
				InvalidOpcode
			};

			enum DataType
			{
				Byte,
				Short,
				Dword
			};

		public:
			/*! \brief Default constructor */
			ILInstruction(Opcode op = InvalidOpcode);

			/*****************************//**
			 * \name Parsable IL strings
			 ********************************/
			//@{
			static  std::string toString(Opcode o);
			static  std::string toString(DataType d);
			virtual std::string toString() const;
			//@}

			/*! \brief Return a pointer to a new Instruction */
			virtual Instruction* clone(bool copy=true) const = 0;
			
		    /*! \brief Determines if the instruction is valid */
			virtual std::string valid() const;

		public:
			/*! \brief Opcode of the instruction */
			const Opcode opcode;
	};

	/*! \brief A generic 1 operand instruction */
	class ILUnaryInstruction : public ILInstruction
	{
		public:
			/*! \brief Default constructor */
			ILUnaryInstruction(Opcode op = InvalidOpcode);
			
			/*! \brief Parsable IL strings */
			virtual std::string toString() const;

			/*! \brief Return a pointer to a new Instruction */
			virtual Instruction* clone(bool copy = true) const = 0;

		public:
			/*! \brief The destination operand */
			ILOperand d;
			
			/*! \brief The source operand */
			ILOperand a;
	};
	
	/*! \brief A generic 2 operand instruction */
	class ILBinaryInstruction : public ILInstruction
	{
		public:
			/*! \brief Default constructor */
			ILBinaryInstruction(Opcode op = InvalidOpcode);
			
			/*! \brief Parsable IL strings */
			virtual std::string toString() const;

			/*! \brief Return a pointer to a new Instruction */
			virtual Instruction* clone(bool copy=true) const = 0;

		public:
			/*! \brief The destination operand */
			ILOperand d;
			
			/*! \brief The first source operand */
			ILOperand a;

			/*! \brief The second source operand */
			ILOperand b;
	};

	/*! \brief A generic 3 operand instruction */
	class ILTrinaryInstruction : public ILInstruction
	{
		public:
			/*! \brief Default constructor */
			ILTrinaryInstruction(Opcode op = InvalidOpcode);
			
			/*! \brief Parsable IL strings */
			virtual std::string toString() const;

			/*! \brief Return a pointer to a new Instruction */
			virtual Instruction* clone(bool copy=true) const = 0;

		public:
			/*! \brief The destination operand */
			ILOperand d;
			
			/*! \brief The first source operand */
			ILOperand a;

			/*! \brief The second source operand */
			ILOperand b;

			/*! \brief The third source operand */
			ILOperand c;
	};

	class ILAbs: public ILUnaryInstruction
	{
		public:
			ILAbs();
			Instruction *clone(bool copy=true) const;
	};

	class ILAdd : public ILBinaryInstruction
	{
		public:
			ILAdd();
			Instruction *clone(bool copy=true) const;
	};

	class ILAnd : public ILBinaryInstruction
	{
		public:
			ILAnd();
			Instruction *clone(bool copy=true) const;
	};

	class ILBreak: public ILInstruction
	{
		public:
			ILBreak();
			Instruction *clone(bool copy=true) const;
	};

	class ILCmov_Logical : public ILTrinaryInstruction
	{
		public:
			ILCmov_Logical();
			Instruction *clone(bool copy=true) const;
	};

	class ILCos_Vec : public ILUnaryInstruction
	{
		public:
			ILCos_Vec();
			Instruction *clone(bool copy=true) const;
	};

	class ILDiv : public ILBinaryInstruction
	{
		public:
			ILDiv();
			Instruction *clone(bool copy=true) const;
	};

	class ILElse : public ILInstruction
	{
		public:
			ILElse();
			Instruction *clone(bool copy=true) const;
	};

	class ILEnd : public ILInstruction
	{
		public:
			ILEnd();
			Instruction *clone(bool copy=true) const;
	};

	class ILEndIf : public ILInstruction
	{
		public:
			ILEndIf();
			Instruction *clone(bool copy=true) const;
	};

	class ILEndLoop : public ILInstruction
	{
		public:
			ILEndLoop();
			Instruction *clone(bool copy=true) const;
	};

	class ILEq : public ILBinaryInstruction
	{
		public:
			ILEq();
			Instruction *clone(bool copy=true) const;
	};

	class ILExp_Vec: public ILUnaryInstruction
	{
		public:
			ILExp_Vec();
			Instruction *clone(bool copy=true) const;
	};

	class ILFfb_Hi: public ILUnaryInstruction
	{
		public:
			ILFfb_Hi();
			Instruction *clone(bool copy=true) const;
	};

	class ILFence: public ILInstruction
	{
		public:
			ILFence();
			Instruction *clone(bool copy=true) const;

			/*! \brief Set/unset threads flag */
			void threads(bool value = true);

			/*! \brief Set/unset lds flag */
			void lds(bool value = true);

			/*! \brief Set/unset memory flag */
			void memory(bool value = true);

			/*! \brief Parsable IL strings */
			std::string toString() const;

		private:
			/*! \brief threads, lds, memory flags */
			bool _threads, _lds, _memory;
	};

	class ILFma : public ILTrinaryInstruction
	{
		public:
			ILFma();
			Instruction *clone(bool copy=true) const;
	};

	class ILFtoI: public ILUnaryInstruction
	{
		public:
			ILFtoI();
			Instruction *clone(bool copy=true) const;
	};

	class ILFtoU: public ILUnaryInstruction
	{
		public:
			ILFtoU();
			Instruction *clone(bool copy=true) const;
	};

	class ILGe : public ILBinaryInstruction
	{
		public:
			ILGe();
			Instruction *clone(bool copy=true) const;
	};

	class ILIadd : public ILBinaryInstruction
	{
		public:
			ILIadd();
			Instruction *clone(bool copy=true) const;
	};

	class ILIand : public ILBinaryInstruction
	{
		public:
			ILIand();
			Instruction *clone(bool copy=true) const;
	};

	class ILIcbits: public ILUnaryInstruction
	{
		public:
			ILIcbits();
			Instruction *clone(bool copy=true) const;
	};

	class ILIeq : public ILBinaryInstruction
	{
		public:
			ILIeq();
			Instruction *clone(bool copy=true) const;
	};

	class ILIfLogicalNZ : public ILInstruction
	{
		public:
			ILIfLogicalNZ();
			Instruction *clone(bool copy=true) const;

			/*! \brief The first source operand */
			ILOperand a;

			/*! \brief Parsable IL strings */
			std::string toString() const;
	};

	class ILIfLogicalZ : public ILInstruction
	{
		public:
			ILIfLogicalZ();
			Instruction *clone(bool copy=true) const;

			/*! \brief The first source operand */
			ILOperand a;

			/*! \brief Parsable IL strings */
			std::string toString() const;
	};

	class ILIge : public ILBinaryInstruction
	{
		public:
			ILIge();
			Instruction *clone(bool copy=true) const;
	};

	class ILIlt : public ILBinaryInstruction
	{
		public:
			ILIlt();
			Instruction *clone(bool copy=true) const;
	};

	class ILImax : public ILBinaryInstruction
	{
		public:
			ILImax();
			Instruction *clone(bool copy=true) const;
	};

	class ILImin : public ILBinaryInstruction
	{
		public:
			ILImin();
			Instruction *clone(bool copy=true) const;
	};

	class ILImul : public ILBinaryInstruction
	{
		public:
			ILImul();
			Instruction *clone(bool copy=true) const;
	};

	class ILIne : public ILBinaryInstruction
	{
		public:
			ILIne();
			Instruction *clone(bool copy=true) const;
	};

	class ILInegate : public ILUnaryInstruction
	{
		public:
			ILInegate();
			Instruction *clone(bool copy=true) const;
	};

	class ILInot : public ILUnaryInstruction
	{
		public:
			ILInot();
			Instruction *clone(bool copy=true) const;
	};

	class ILIor : public ILBinaryInstruction
	{
		public:
			ILIor();
			Instruction *clone(bool copy=true) const;
	};

	class ILIshl : public ILBinaryInstruction
	{
		public:
			ILIshl();
			Instruction *clone(bool copy=true) const;
	};

	class ILIshr : public ILBinaryInstruction
	{
		public:
			ILIshr();
			Instruction *clone(bool copy=true) const;
	};

	class ILItoF: public ILUnaryInstruction
	{
		public:
			ILItoF();
			Instruction *clone(bool copy=true) const;
	};

	class ILIxor : public ILBinaryInstruction
	{
		public:
			ILIxor();
			Instruction *clone(bool copy=true) const;
	};

	class ILLds_And_Id : public ILInstruction
	{
		public:
			ILLds_And_Id();
			Instruction *clone(bool copy=true) const;

			/*! \brief The first source operand */
			ILOperand a;

			/*! \brief The second source operand */
			ILOperand b;

			/*! \brief Parsable IL strings */
			std::string toString() const;
	};

	class ILLds_Load_Id : public ILUnaryInstruction
	{
		public:
			ILLds_Load_Id();
			Instruction *clone(bool copy=true) const;
	};

	class ILLds_Or_Id : public ILInstruction
	{
		public:
			ILLds_Or_Id();
			Instruction *clone(bool copy=true) const;

			/*! \brief The first source operand */
			ILOperand a;

			/*! \brief The second source operand */
			ILOperand b;

			/*! \brief Parsable IL strings */
			std::string toString() const;
	};

	class ILLds_Read_Add_Id : public ILBinaryInstruction
	{
		public:
			ILLds_Read_Add_Id();
			Instruction *clone(bool copy=true) const;
	};

	class ILLds_Store_Id : public ILInstruction
	{
		public:
			ILLds_Store_Id();
			Instruction *clone(bool copy=true) const;

			/*! \brief The first source operand */
			ILOperand a;

			/*! \brief The second source operand */
			ILOperand b;

			/*! \brief Parsable IL strings */
			std::string toString() const;
	};

	class ILLog_Vec : public ILUnaryInstruction
	{
		public:
			ILLog_Vec();
			Instruction *clone(bool copy=true) const;
	};

	class ILLt : public ILBinaryInstruction
	{
		public:
			ILLt();
			Instruction *clone(bool copy=true) const;
	};

	class ILMad : public ILTrinaryInstruction
	{
		public:
			ILMad();
			Instruction *clone(bool copy=true) const;
	};

	class ILMov : public ILUnaryInstruction
	{
		public:
			ILMov();
			Instruction *clone(bool copy=true) const;
	};

	class ILMul : public ILBinaryInstruction
	{
		public:
			ILMul();
			Instruction *clone(bool copy=true) const;
	};

	class ILNe : public ILBinaryInstruction
	{
		public:
			ILNe();
			Instruction *clone(bool copy=true) const;
	};

	class ILRcp : public ILUnaryInstruction
	{
		public:
			ILRcp();
			Instruction *clone(bool copy=true) const;
	};

	class ILRound_Nearest : public ILUnaryInstruction
	{
		public:
			ILRound_Nearest();
			Instruction *clone(bool copy=true) const;
	};

	class ILRound_Neginf: public ILUnaryInstruction
	{
		public:
			ILRound_Neginf();
			Instruction *clone(bool copy=true) const;
	};

	class ILRsq_Vec : public ILUnaryInstruction
	{
		public:
			ILRsq_Vec();
			Instruction *clone(bool copy=true) const;
	};

	class ILSin_Vec : public ILUnaryInstruction
	{
		public:
			ILSin_Vec();
			Instruction *clone(bool copy=true) const;
	};

	class ILSub : public ILBinaryInstruction
	{
		public:
			ILSub();
			Instruction *clone(bool copy=true) const;
	};

	class ILSqrt_Vec: public ILUnaryInstruction
	{
		public:
			ILSqrt_Vec();
			Instruction *clone(bool copy=true) const;
	};

	class ILUav_Arena_Load_Id : public ILUnaryInstruction
	{
		public:
			ILUav_Arena_Load_Id();
			Instruction *clone(bool copy=true) const;

			/*! \brief Parsable IL strings */
			std::string toString() const;

		public:
			DataType type;
	};

	class ILUav_Arena_Store_Id : public ILInstruction
	{
		public:
			ILUav_Arena_Store_Id();
			Instruction *clone(bool copy=true) const;

			/*! \brief The first source operand */
			ILOperand a;

			/*! \brief The second source operand */
			ILOperand b;

			/*! \brief Parsable IL strings */
			std::string toString() const;

		public:
			DataType type;
	};

	class ILUav_Raw_Load_Id : public ILUnaryInstruction
	{
		public:
			ILUav_Raw_Load_Id();
			Instruction *clone(bool copy=true) const;
	};

	class ILUav_Raw_Store_Id : public ILBinaryInstruction
	{
		public:
			ILUav_Raw_Store_Id();
			Instruction *clone(bool copy=true) const;
	};

	class ILUav_Read_Add_Id : public ILBinaryInstruction
	{
		public:
			ILUav_Read_Add_Id();
			Instruction *clone(bool copy=true) const;
	};

	class ILUav_Read_Max_Id : public ILBinaryInstruction
	{
		public:
			ILUav_Read_Max_Id();
			Instruction *clone(bool copy=true) const;
	};

	class ILUav_Read_Min_Id : public ILBinaryInstruction
	{
		public:
			ILUav_Read_Min_Id();
			Instruction *clone(bool copy=true) const;
	};

	class ILUav_Read_Xchg_Id : public ILBinaryInstruction
	{
		public:
			ILUav_Read_Xchg_Id();
			Instruction *clone(bool copy=true) const;
	};

	class ILUdiv : public ILBinaryInstruction
	{
		public:
			ILUdiv();
			Instruction *clone(bool copy=true) const;
	};

	class ILUmul : public ILBinaryInstruction
	{
		public:
			ILUmul();
			Instruction *clone(bool copy=true) const;
	};

	class ILUmul24 : public ILBinaryInstruction
	{
		public:
			ILUmul24();
			Instruction *clone(bool copy=true) const;
	};

	class ILUshr : public ILBinaryInstruction
	{
		public:
			ILUshr();
			Instruction *clone(bool copy=true) const;
	};

	class ILUtoF: public ILUnaryInstruction
	{
		public:
			ILUtoF();
			Instruction *clone(bool copy=true) const;
	};

	class ILWhileLoop : public ILInstruction
	{
		public:
			ILWhileLoop();
			Instruction *clone(bool copy=true) const;
	};
}
#endif
