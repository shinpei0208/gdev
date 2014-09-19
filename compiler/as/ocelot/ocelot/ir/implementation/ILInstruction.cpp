/*! \file ILInstruction.cpp
 *  \author Rodrigo Dominguez <rdomingu@ece.neu.edu>
 *  \date April 27, 2010
 *  \brief The source file for the IL Instruction class.
 */

// Ocelot includes
#include <ocelot/ir/interface/ILInstruction.h>

// Hydrazine includes
#include <hydrazine/interface/debug.h>

namespace ir
{
	ILInstruction::ILInstruction(Opcode op) : opcode(op)
	{
	}

	std::string ILInstruction::toString(Opcode opcode)
	{
		switch(opcode)
		{
			case Abs:                   return "abs";
			case Add:                   return "add";
			case And:                   return "and";
			case Break:                 return "break";
			case Cmov_Logical:          return "cmov_logical";
			case Cos_Vec:               return "cos_vec";
			case Div:                   return "div";
			case Else:                  return "else";
			case End:                   return "end";
			case EndIf:                 return "endif";
			case EndLoop:               return "endloop";
			case Eq:                    return "eq";
			case Exp_Vec:               return "exp_vec";
			case Fence:                 return "fence";
			case Ffb_Hi:                return "ffb_hi";
			case Fma:                   return "fma";
			case FtoI:                  return "ftoi";
			case FtoU:                  return "ftou";
			case Ge:                    return "ge";
			case Iadd:                  return "iadd";
			case Iand:                  return "iand";
			case Icbits:                return "icbits";
			case Ieq:                   return "ieq";
			case IfLogicalNZ:           return "if_logicalnz";
			case IfLogicalZ:            return "if_logicalz";
			case Ige:                   return "ige";
			case Ilt:                   return "ilt";
			case Imax:                  return "imax";
			case Imin:                  return "imin";
			case Imul:                  return "imul";
			case Ine:                   return "ine";
			case Inegate:               return "inegate";
			case Inot:                  return "inot";
			case Ior:                   return "ior";
			case Ishl:                  return "ishl";
			case Ishr:                  return "ishr";
			case ItoF:                  return "itof";
			case Ixor:                  return "ixor";
			case Lds_And_Id:            return "lds_and_id(1)";
			case Lds_Load_Id:           return "lds_load_id(1)";
			case Lds_Or_Id:             return "lds_or_id(1)";
			case Lds_Read_Add_Id:       return "lds_read_add_id(1)";
			case Lds_Store_Id:          return "lds_store_id(1)";
			case Log_Vec:               return "log_vec";
			case Lt:                    return "lt";
			case Mad:                   return "mad";
			case Mov:                   return "mov";
			case Mul:                   return "mul";
			case Ne:                    return "ne";
			case Rcp:                   return "rcp";
			case Round_Nearest:         return "round_nearest";
			case Round_Neginf:          return "round_neginf";
			case Rsq_Vec:               return "rsq_vec";
			case Sin_Vec:               return "sin_vec";
			case Sub:                   return "sub";
			case Sqrt_Vec:              return "sqrt_vec";
			case Uav_Arena_Load_Id:     return "uav_arena_load_id(8)";
			case Uav_Arena_Store_Id:    return "uav_arena_store_id(8)";
			case Uav_Raw_Load_Id:       return "uav_raw_load_id(0)";
			case Uav_Raw_Store_Id:      return "uav_raw_store_id(0)";
			case Uav_Read_Add_Id:       return "uav_read_add_id(0)";
			case Uav_Read_Max_Id:       return "uav_read_max_id(0)";
			case Uav_Read_Min_Id:       return "uav_read_min_id(0)";
			case Uav_Read_Xchg_Id:      return "uav_read_xchg_id(0)";
			case Udiv:                  return "udiv";
			case Umul:                  return "umul";
			case Umul24:                return "umul24";
			case Ushr:                  return "ushr";
			case UtoF:                  return "utof";
			case WhileLoop:             return "whileloop";
			case InvalidOpcode:         return "INVALID_OPCODE";
			default:
			{
				assertM(false, "Opcode " << opcode << " not supported");
			}
		}

		assertM(false, "Unreachable line");
		return "";
	}

	std::string ILInstruction::toString(DataType d)
	{
		switch(d)
		{
			case Short: return "short";
			case Byte:  return "byte";
			case Dword: return "dword";
			default:
			{
				assertM(false, "DataType " << d << " not supported");
			}
		}
		
		return "";
	}

	std::string ILInstruction::toString() const
	{
		return toString(opcode);
	}

	std::string ILInstruction::valid() const
	{
		assertM(false, "Not implemented yet");
		return "";
	}

	ILUnaryInstruction::ILUnaryInstruction(Opcode op) : ILInstruction(op)
	{
	}

	std::string ILUnaryInstruction::toString() const
	{
		return ILInstruction::toString(opcode) +
			d.shiftString() + d.clampString() + 
			" " + d.dstString() + ", " + a.srcString();
	}

	ILBinaryInstruction::ILBinaryInstruction(Opcode op) : ILInstruction(op)
	{
	}

	std::string ILBinaryInstruction::toString() const
	{
		return ILInstruction::toString(opcode) + 
			d.shiftString() + d.clampString() + 
			" " + d.dstString() + ", " + a.srcString() + ", " + b.srcString();
	}

	ILTrinaryInstruction::ILTrinaryInstruction(Opcode op) : ILInstruction(op)
	{
	}

	std::string ILTrinaryInstruction::toString() const
	{
		return ILInstruction::toString(opcode) + 
			d.shiftString() + d.clampString() + 
			" " + d.dstString() + ", " + a.srcString() + ", " + b.srcString() + 
			", " + c.srcString();
	}

	ILAbs::ILAbs() : ILUnaryInstruction(Abs)
	{
	}

	Instruction *ILAbs::clone(bool copy) const
	{
		return new ILAbs(*this);
	}

	ILAdd::ILAdd() : ILBinaryInstruction(Add)
	{
	}

	Instruction *ILAdd::clone(bool copy) const
	{
		return new ILAdd(*this);
	}

	ILAnd::ILAnd() : ILBinaryInstruction(And)
	{
	}

	Instruction *ILAnd::clone(bool copy) const
	{
		return new ILAnd(*this);
	}

	ILBreak::ILBreak() : ILInstruction(Break)
	{
	}

	Instruction *ILBreak::clone(bool copy) const
	{
		return new ILBreak(*this);
	}

	ILCmov_Logical::ILCmov_Logical() : ILTrinaryInstruction(Cmov_Logical)
	{
	}

	ILCos_Vec::ILCos_Vec() : ILUnaryInstruction(Cos_Vec)
	{
	}

	Instruction *ILCos_Vec::clone(bool copy) const
	{
		return new ILCos_Vec(*this);
	}

	Instruction *ILCmov_Logical::clone(bool copy) const
	{
		return new ILCmov_Logical(*this);
	}

	ILDiv::ILDiv() : ILBinaryInstruction(Div)
	{
	}

	Instruction *ILDiv::clone(bool copy) const
	{
		return new ILDiv(*this);
	}

	ILElse::ILElse() : ILInstruction(Else)
	{
	}

	Instruction *ILElse::clone(bool copy) const
	{
		return new ILElse(*this);
	}

	ILEnd::ILEnd() : ILInstruction(End)
	{
	}

	Instruction *ILEnd::clone(bool copy) const
	{
		return new ILEnd(*this);
	}

	ILEndIf::ILEndIf() : ILInstruction(EndIf)
	{
	}

	Instruction *ILEndIf::clone(bool copy) const
	{
		return new ILEndIf(*this);
	}

	ILEndLoop::ILEndLoop() : ILInstruction(EndLoop)
	{
	}

	Instruction *ILEndLoop::clone(bool copy) const
	{
		return new ILEndLoop(*this);
	}

	ILEq::ILEq() : ILBinaryInstruction(Eq)
	{
	}

	Instruction *ILEq::clone(bool copy) const
	{
		return new ILEq(*this);
	}

	ILExp_Vec::ILExp_Vec() : ILUnaryInstruction(Exp_Vec)
	{
	}

	Instruction *ILExp_Vec::clone(bool copy) const
	{
		return new ILExp_Vec(*this);
	}

	ILFfb_Hi::ILFfb_Hi() : ILUnaryInstruction(Ffb_Hi)
	{
	}

	Instruction *ILFfb_Hi::clone(bool copy) const
	{
		return new ILFfb_Hi(*this);
	}

	ILFence::ILFence() : ILInstruction(Fence), _threads(true), _lds(false), 
		_memory(false)
	{
	}

	std::string ILFence::toString() const
	{
		return ILInstruction::toString(opcode) + (_threads ? "_threads" : "") +
			(_lds ? "_lds" : "") + (_memory ? "_memory" : "");
	}

	Instruction *ILFence::clone(bool copy) const
	{
		return new ILFence(*this);
	}

	void ILFence::threads(bool value)
	{
		_threads = value;
	}

	void ILFence::lds(bool value)
	{
		_lds = value;
	}

	void ILFence::memory(bool value)
	{
		_memory = value;
	}

	ILFma::ILFma() : ILTrinaryInstruction(Fma)
	{
	}

	Instruction *ILFma::clone(bool copy) const
	{
		return new ILFma(*this);
	}

	ILFtoI::ILFtoI() : ILUnaryInstruction(FtoI)
	{
	}

	Instruction *ILFtoI::clone(bool copy) const
	{
		return new ILFtoI(*this);
	}

	ILFtoU::ILFtoU() : ILUnaryInstruction(FtoU)
	{
	}

	Instruction *ILFtoU::clone(bool copy) const
	{
		return new ILFtoU(*this);
	}

	ILGe::ILGe() : ILBinaryInstruction(Ge)
	{
	}

	Instruction *ILGe::clone(bool copy) const
	{
		return new ILGe(*this);
	}

	ILIadd::ILIadd() : ILBinaryInstruction(Iadd)
	{
	}

	Instruction *ILIadd::clone(bool copy) const
	{
		return new ILIadd(*this);
	}

	ILIand::ILIand() : ILBinaryInstruction(Iand)
	{
	}

	Instruction *ILIand::clone(bool copy) const
	{
		return new ILIand(*this);
	}

	ILIcbits::ILIcbits() : ILUnaryInstruction(Icbits)
	{
	}

	Instruction *ILIcbits::clone(bool copy) const
	{
		return new ILIcbits(*this);
	}

	ILIeq::ILIeq() : ILBinaryInstruction(Ieq)
	{
	}

	Instruction *ILIeq::clone(bool copy) const
	{
		return new ILIeq(*this);
	}

	ILIfLogicalNZ::ILIfLogicalNZ() : ILInstruction(IfLogicalNZ)
	{
	}

	std::string ILIfLogicalNZ::toString() const
	{
		return ILInstruction::toString(opcode) + " " + a.srcString();
	}

	Instruction *ILIfLogicalNZ::clone(bool copy) const
	{
		return new ILIfLogicalNZ(*this);
	}

	ILIfLogicalZ::ILIfLogicalZ() : ILInstruction(IfLogicalZ)
	{
	}

	std::string ILIfLogicalZ::toString() const
	{
		return ILInstruction::toString(opcode) + " " + a.srcString();
	}

	Instruction *ILIfLogicalZ::clone(bool copy) const
	{
		return new ILIfLogicalZ(*this);
	}

	ILIge::ILIge() : ILBinaryInstruction(Ige)
	{
	}

	Instruction *ILIge::clone(bool copy) const
	{
		return new ILIge(*this);
	}

	ILIlt::ILIlt() : ILBinaryInstruction(Ilt)
	{
	}

	Instruction *ILIlt::clone(bool copy) const
	{
		return new ILIlt(*this);
	}

	ILImax::ILImax() : ILBinaryInstruction(Imax)
	{
	}

	Instruction *ILImax::clone(bool copy) const
	{
		return new ILImax(*this);
	}

	ILImin::ILImin() : ILBinaryInstruction(Imin)
	{
	}

	Instruction *ILImin::clone(bool copy) const
	{
		return new ILImin(*this);
	}

	ILImul::ILImul() : ILBinaryInstruction(Imul)
	{
	}

	Instruction *ILImul::clone(bool copy) const
	{
		return new ILImul(*this);
	}

	ILIne::ILIne() : ILBinaryInstruction(Ine)
	{
	}

	Instruction *ILIne::clone(bool copy) const
	{
		return new ILIne(*this);
	}

	ILInegate::ILInegate() : ILUnaryInstruction(Inegate)
	{
	}

	Instruction *ILInegate::clone(bool copy) const
	{
		return new ILInegate(*this);
	}

	ILInot::ILInot() : ILUnaryInstruction(Inot)
	{
	}

	Instruction *ILInot::clone(bool copy) const
	{
		return new ILInot(*this);
	}

	ILIor::ILIor() : ILBinaryInstruction(Ior)
	{
	}

	Instruction *ILIor::clone(bool copy) const
	{
		return new ILIor(*this);
	}

	ILIshl::ILIshl() : ILBinaryInstruction(Ishl)
	{
	}

	Instruction *ILIshl::clone(bool copy) const
	{
		return new ILIshl(*this);
	}

	ILIshr::ILIshr() : ILBinaryInstruction(Ishr)
	{
	}

	Instruction *ILIshr::clone(bool copy) const
	{
		return new ILIshr(*this);
	}

	ILItoF::ILItoF() : ILUnaryInstruction(ItoF)
	{
	}

	Instruction *ILItoF::clone(bool copy) const
	{
		return new ILItoF(*this);
	}

	ILIxor::ILIxor() : ILBinaryInstruction(Ixor)
	{
	}

	Instruction *ILIxor::clone(bool copy) const
	{
		return new ILIxor(*this);
	}

	ILLds_And_Id::ILLds_And_Id() 
		: ILInstruction(Lds_And_Id)
	{
	}

	Instruction *ILLds_And_Id::clone(bool copy) const
	{
		return new ILLds_And_Id(*this);
	}

	std::string ILLds_And_Id::toString() const
	{
		return ILInstruction::toString(opcode) + " " + a.srcString() + ", " + b.srcString();
	}

	ILLds_Load_Id::ILLds_Load_Id() : ILUnaryInstruction(Lds_Load_Id)
	{
	}

	Instruction *ILLds_Load_Id::clone(bool copy) const
	{
		return new ILLds_Load_Id(*this);
	}

	ILLds_Or_Id::ILLds_Or_Id() : ILInstruction(Lds_Or_Id)
	{
	}

	Instruction *ILLds_Or_Id::clone(bool copy) const
	{
		return new ILLds_Or_Id(*this);
	}

	std::string ILLds_Or_Id::toString() const
	{
		return ILInstruction::toString(opcode) + " " + a.srcString() + ", " + b.srcString();
	}

	ILLds_Read_Add_Id::ILLds_Read_Add_Id() 
		: ILBinaryInstruction(Lds_Read_Add_Id)
	{
	}

	Instruction *ILLds_Read_Add_Id::clone(bool copy) const
	{
		return new ILLds_Read_Add_Id(*this);
	}

	ILLds_Store_Id::ILLds_Store_Id() : ILInstruction(Lds_Store_Id)
	{
	}

	Instruction *ILLds_Store_Id::clone(bool copy) const
	{
		return new ILLds_Store_Id(*this);
	}

	std::string ILLds_Store_Id::toString() const
	{
		return ILInstruction::toString(opcode) + " " + a.srcString() + ", " + b.srcString();
	}

	ILLog_Vec::ILLog_Vec() : ILUnaryInstruction(Log_Vec)
	{
	}

	Instruction *ILLog_Vec::clone(bool copy) const
	{
		return new ILLog_Vec(*this);
	}

	ILLt::ILLt() : ILBinaryInstruction(Lt)
	{
	}

	Instruction *ILLt::clone(bool copy) const
	{
		return new ILLt(*this);
	}

	ILMad::ILMad() : ILTrinaryInstruction(Mad)
	{
	}

	Instruction *ILMad::clone(bool copy) const
	{
		return new ILMad(*this);
	}

	ILMov::ILMov() : ILUnaryInstruction(Mov)
	{
	}

	Instruction *ILMov::clone(bool copy) const
	{
		return new ILMov(*this);
	}

	ILMul::ILMul() : ILBinaryInstruction(Mul)
	{
	}

	Instruction *ILMul::clone(bool copy) const
	{
		return new ILMul(*this);
	}

	ILNe::ILNe() : ILBinaryInstruction(Ne)
	{
	}

	Instruction *ILNe::clone(bool copy) const
	{
		return new ILNe(*this);
	}

	ILRcp::ILRcp() : ILUnaryInstruction(Rcp)
	{
	}

	Instruction *ILRcp::clone(bool copy) const
	{
		return new ILRcp(*this);
	}

	ILRound_Nearest::ILRound_Nearest() : ILUnaryInstruction(Round_Nearest)
	{
	}

	Instruction *ILRound_Nearest::clone(bool copy) const
	{
		return new ILRound_Nearest(*this);
	}

	ILRound_Neginf::ILRound_Neginf() : ILUnaryInstruction(Round_Neginf)
	{
	}

	Instruction *ILRound_Neginf::clone(bool copy) const
	{
		return new ILRound_Neginf(*this);
	}

	ILRsq_Vec::ILRsq_Vec() : ILUnaryInstruction(Rsq_Vec)
	{
	}

	Instruction *ILRsq_Vec::clone(bool copy) const
	{
		return new ILRsq_Vec(*this);
	}

	ILSin_Vec::ILSin_Vec() : ILUnaryInstruction(Sin_Vec)
	{
	}

	Instruction *ILSin_Vec::clone(bool copy) const
	{
		return new ILSin_Vec(*this);
	}

	ILSub::ILSub() : ILBinaryInstruction(Sub)
	{
	}

	Instruction *ILSub::clone(bool copy) const
	{
		return new ILSub(*this);
	}

	ILSqrt_Vec::ILSqrt_Vec() : ILUnaryInstruction(Sqrt_Vec)
	{
	}

	Instruction *ILSqrt_Vec::clone(bool copy) const
	{
		return new ILSqrt_Vec(*this);
	}

	ILUav_Arena_Load_Id::ILUav_Arena_Load_Id() 
		: ILUnaryInstruction(Uav_Arena_Load_Id)
	{
	}

	std::string ILUav_Arena_Load_Id::toString() const
	{
		return ILInstruction::toString(opcode) + "_size(" 
			+ ILInstruction::toString(type) + ") " + d.dstString() + ", " 
			+ a.srcString();
	}

	Instruction *ILUav_Arena_Load_Id::clone(bool copy) const
	{
		return new ILUav_Arena_Load_Id(*this);
	}

	ILUav_Arena_Store_Id::ILUav_Arena_Store_Id() 
		: ILInstruction(Uav_Arena_Store_Id)
	{
	}

	Instruction *ILUav_Arena_Store_Id::clone(bool copy) const
	{
		return new ILUav_Arena_Store_Id(*this);
	}

	std::string ILUav_Arena_Store_Id::toString() const
	{
		return ILInstruction::toString(opcode) + "_size(" 
			+ ILInstruction::toString(type) + ") " + a.srcString() + ", " 
			+ b.srcString();
	}

	ILUav_Raw_Load_Id::ILUav_Raw_Load_Id() 
		: ILUnaryInstruction(Uav_Raw_Load_Id)
	{
	}

	Instruction *ILUav_Raw_Load_Id::clone(bool copy) const
	{
		return new ILUav_Raw_Load_Id(*this);
	}

	ILUav_Raw_Store_Id::ILUav_Raw_Store_Id() 
		: ILBinaryInstruction(Uav_Raw_Store_Id)
	{
	}

	Instruction *ILUav_Raw_Store_Id::clone(bool copy) const
	{
		return new ILUav_Raw_Store_Id(*this);
	}

	ILUav_Read_Add_Id::ILUav_Read_Add_Id() 
		: ILBinaryInstruction(Uav_Read_Add_Id)
	{
	}

	Instruction *ILUav_Read_Add_Id::clone(bool copy) const
	{
		return new ILUav_Read_Add_Id(*this);
	}

	ILUav_Read_Max_Id::ILUav_Read_Max_Id() 
		: ILBinaryInstruction(Uav_Read_Max_Id)
	{
	}

	Instruction *ILUav_Read_Max_Id::clone(bool copy) const
	{
		return new ILUav_Read_Max_Id(*this);
	}

	ILUav_Read_Min_Id::ILUav_Read_Min_Id() 
		: ILBinaryInstruction(Uav_Read_Min_Id)
	{
	}

	Instruction *ILUav_Read_Min_Id::clone(bool copy) const
	{
		return new ILUav_Read_Min_Id(*this);
	}

	ILUav_Read_Xchg_Id::ILUav_Read_Xchg_Id() 
		: ILBinaryInstruction(Uav_Read_Xchg_Id)
	{
	}

	Instruction *ILUav_Read_Xchg_Id::clone(bool copy) const
	{
		return new ILUav_Read_Xchg_Id(*this);
	}

	ILUdiv::ILUdiv() : ILBinaryInstruction(Udiv)
	{
	}

	Instruction *ILUdiv::clone(bool copy) const
	{
		return new ILUdiv(*this);
	}

	ILUmul::ILUmul() : ILBinaryInstruction(Umul)
	{
	}

	Instruction *ILUmul::clone(bool copy) const
	{
		return new ILUmul(*this);
	}

	ILUmul24::ILUmul24() : ILBinaryInstruction(Umul24)
	{
	}

	Instruction *ILUmul24::clone(bool copy) const
	{
		return new ILUmul24(*this);
	}

	ILUshr::ILUshr() : ILBinaryInstruction(Ushr)
	{
	}

	Instruction *ILUshr::clone(bool copy) const
	{
		return new ILUshr(*this);
	}

	ILUtoF::ILUtoF() : ILUnaryInstruction(UtoF)
	{
	}

	Instruction *ILUtoF::clone(bool copy) const
	{
		return new ILUtoF(*this);
	}

	ILWhileLoop::ILWhileLoop() : ILInstruction(WhileLoop)
	{
	}

	Instruction *ILWhileLoop::clone(bool copy) const
	{
		return new ILWhileLoop(*this);
	}
}


