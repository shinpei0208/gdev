/*! \file ILOperand.cpp
 *  \author Rodrigo Dominguez <rdomingu@ece.neu.edu>
 *  \date April 29, 2010
 *  \brief The implementation file for the IL Operand class.
 */

// Ocelot includes
#include <ocelot/ir/interface/ILOperand.h>

// Hydrazine includes
#include <hydrazine/interface/debug.h>

// Boost includes
#include <boost/lexical_cast.hpp>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace ir
{
	ILOperand::ILOperand(RegType t)
		:
			type(t),
			modifier_present(false),
			immediate_present(false)
	{
	}

	std::string ILOperand::toString(RegType rt)
	{
		switch (rt)
		{
			case RegType_Temp:               return "r";
			case RegType_Const_Buf:          return "cb";
			case RegType_Literal:            return "l";
			case RegType_Thread_Id_In_Group: return "vTidInGrp";
			case RegType_Thread_Group_Id:    return "vThreadGrpId";
			case RegType_Generic_Mem:        return "mem";
			default: assertM(false, "Invalid register type " << rt);
		}
		
		return "";
	}

	ILOperand::Dst_Mod::Dst_Mod() :
		component_x(ModComp_Invalid),
		component_y(ModComp_Invalid),
		component_z(ModComp_Invalid),
		component_w(ModComp_Invalid),
		clamp(false),
		shift_scale(Shift_Invalid)
	{
	}

	std::string ILOperand::Dst_Mod::toString(ModDstComponent dc)
	{
		switch (dc)
		{
			case ModComp_NoWrite: return "_";
			case ModComp_0:       return "0";
			case ModComp_1:       return "1";
			default: assertM(false, "Invalid dest component select " << dc);
		}
	}

	std::string ILOperand::Dst_Mod::toString() const
	{
		return "." + 
			(component_x == ModComp_Write ? "x" : toString(component_x)) + 
			(component_y == ModComp_Write ? "y" : toString(component_y)) + 
			(component_z == ModComp_Write ? "z" : toString(component_z)) + 
			(component_w == ModComp_Write ? "w" : toString(component_w));
	}

	ILOperand::Src_Mod::Src_Mod() : 
		swizzle_x(CompSel_Invalid), 
		swizzle_y(CompSel_Invalid), 
		swizzle_z(CompSel_Invalid), 
		swizzle_w(CompSel_Invalid), 
		negate_x(false), 
		negate_y(false), 
		negate_z(false),
		negate_w(false)
	{
	}

	std::string ILOperand::Src_Mod::toString(ComponentSelect c)
	{
		switch (c)
		{
			case CompSel_X: return "x";
			case CompSel_Y: return "y";
			case CompSel_Z: return "z";
			case CompSel_W: return "w";
			case CompSel_0: return "0";
			case CompSel_1: return "1";
			default: assertM(false, "Invalid component select " << c);
		}
	}

	std::string ILOperand::Src_Mod::swizzleString() const
	{
		return toString(swizzle_x) + toString(swizzle_y) + toString(swizzle_z) +
			toString(swizzle_w);
	}

	std::string ILOperand::Src_Mod::negateString() const
	{
		return (negate_x || negate_y || negate_z || negate_w ? 
				std::string("_neg(") + 
				(negate_x ? "x" : "") +
				(negate_y ? "y" : "") +
				(negate_z ? "z" : "") +
				(negate_w ? "w" : "") +
				")" : "");
	}

	std::string ILOperand::Src_Mod::toString() const
	{
		return "." + swizzleString() + negateString();
	}

	std::string ILOperand::immediateString() const
	{
		return "[" + boost::lexical_cast<std::string>(imm) + "]";
	}

	std::string ILOperand::dstString() const
	{
		return toString(type) + boost::lexical_cast<std::string>(num) +
			(modifier_present ? dst_mod.toString() : "");
	}

	std::string ILOperand::srcString() const
	{
		switch (type)
		{
			case RegType_Temp:      /* fall thru */
			case RegType_Const_Buf: /* fall thru */
			case RegType_Literal:
			{
				return toString(type) + 
					boost::lexical_cast<std::string>(num) +
					(immediate_present ? immediateString() : "") +
					(modifier_present ? src_mod.toString() : "");
			}
			case RegType_Thread_Id_In_Group:
			case RegType_Thread_Group_Id:
			{
				return toString(type) + 
					(modifier_present ? src_mod.toString() : "");
			}
			default: assertM(false, "Invalid register type " << type);
		}
		
		return toString(type) + (modifier_present ? src_mod.toString() : "");
	}

	std::string ILOperand::clampString() const
	{
		return modifier_present && dst_mod.clamp ? "_sat" : "";
	}

	std::string ILOperand::shiftString() const
	{
		if (!modifier_present) return "";

		switch (dst_mod.shift_scale)
		{
			case Dst_Mod::Shift_None: return "";
			case Dst_Mod::Shift_X2:   return "_x2";
			case Dst_Mod::Shift_X4:   return "_x4";
			case Dst_Mod::Shift_X8:   return "_x8";
			case Dst_Mod::Shift_D2:   return "_d2";
			case Dst_Mod::Shift_D4:   return "_d4";
			case Dst_Mod::Shift_D8:   return "_d8";
			default: assertM(false, "Invalid shift scale " << dst_mod.shift_scale);
		}
	}

	ILOperand ILOperand::x() const
	{
		ILOperand o(*this);
		o.modifier_present = true;

		o.dst_mod.component_x = Dst_Mod::ModComp_Write;
		o.dst_mod.component_y = Dst_Mod::ModComp_NoWrite;
		o.dst_mod.component_z = Dst_Mod::ModComp_NoWrite;
		o.dst_mod.component_w = Dst_Mod::ModComp_NoWrite;
		o.dst_mod.clamp = false;
		o.dst_mod.shift_scale = Dst_Mod::Shift_None;

		o.src_mod.swizzle_x = Src_Mod::CompSel_X;
		o.src_mod.swizzle_y = Src_Mod::CompSel_X;
		o.src_mod.swizzle_z = Src_Mod::CompSel_X;
		o.src_mod.swizzle_w = Src_Mod::CompSel_X;

		return o;
	}

	ILOperand ILOperand::y() const
	{
		ILOperand o(*this);
		o.modifier_present = true;

		o.dst_mod.component_x = Dst_Mod::ModComp_NoWrite;
		o.dst_mod.component_y = Dst_Mod::ModComp_Write;
		o.dst_mod.component_z = Dst_Mod::ModComp_NoWrite;
		o.dst_mod.component_w = Dst_Mod::ModComp_NoWrite;
		o.dst_mod.clamp = false;
		o.dst_mod.shift_scale = Dst_Mod::Shift_None;

		o.src_mod.swizzle_x = Src_Mod::CompSel_Y;
		o.src_mod.swizzle_y = Src_Mod::CompSel_Y;
		o.src_mod.swizzle_z = Src_Mod::CompSel_Y;
		o.src_mod.swizzle_w = Src_Mod::CompSel_Y;

		return o;
	}

	ILOperand ILOperand::z() const
	{
		ILOperand o(*this);
		o.modifier_present = true;

		o.dst_mod.component_x = Dst_Mod::ModComp_NoWrite;
		o.dst_mod.component_y = Dst_Mod::ModComp_NoWrite;
		o.dst_mod.component_z = Dst_Mod::ModComp_Write;
		o.dst_mod.component_w = Dst_Mod::ModComp_NoWrite;
		o.dst_mod.clamp = false;
		o.dst_mod.shift_scale = Dst_Mod::Shift_None;

		o.src_mod.swizzle_x = Src_Mod::CompSel_Z;
		o.src_mod.swizzle_y = Src_Mod::CompSel_Z;
		o.src_mod.swizzle_z = Src_Mod::CompSel_Z;
		o.src_mod.swizzle_w = Src_Mod::CompSel_Z;

		return o;
	}

	ILOperand ILOperand::w() const
	{
		ILOperand o(*this);
		o.modifier_present = true;

		o.dst_mod.component_x = Dst_Mod::ModComp_NoWrite;
		o.dst_mod.component_y = Dst_Mod::ModComp_NoWrite;
		o.dst_mod.component_z = Dst_Mod::ModComp_NoWrite;
		o.dst_mod.component_w = Dst_Mod::ModComp_Write;
		o.dst_mod.clamp = false;
		o.dst_mod.shift_scale = Dst_Mod::Shift_None;

		o.src_mod.swizzle_x = Src_Mod::CompSel_W;
		o.src_mod.swizzle_y = Src_Mod::CompSel_W;
		o.src_mod.swizzle_z = Src_Mod::CompSel_W;
		o.src_mod.swizzle_w = Src_Mod::CompSel_W;

		return o;
	}

	ILOperand ILOperand::xy() const
	{
		ILOperand o(*this);
		o.modifier_present = true;

		o.dst_mod.component_x = Dst_Mod::ModComp_Write;
		o.dst_mod.component_y = Dst_Mod::ModComp_Write;
		o.dst_mod.component_z = Dst_Mod::ModComp_NoWrite;
		o.dst_mod.component_w = Dst_Mod::ModComp_NoWrite;
		o.dst_mod.clamp = false;
		o.dst_mod.shift_scale = Dst_Mod::Shift_None;

		o.src_mod.swizzle_x = Src_Mod::CompSel_X;
		o.src_mod.swizzle_y = Src_Mod::CompSel_Y;
		o.src_mod.swizzle_z = Src_Mod::CompSel_X;
		o.src_mod.swizzle_w = Src_Mod::CompSel_X;

		return o;
	}

	ILOperand ILOperand::neg() const
	{
		ILOperand o(*this);
		o.modifier_present = true;

		o.dst_mod.component_x = Dst_Mod::ModComp_Write;
		o.dst_mod.component_y = Dst_Mod::ModComp_Write;
		o.dst_mod.component_z = Dst_Mod::ModComp_Write;
		o.dst_mod.component_w = Dst_Mod::ModComp_Write;
		o.dst_mod.clamp = false;
		o.dst_mod.shift_scale = Dst_Mod::Shift_None;

		o.src_mod.swizzle_x = Src_Mod::CompSel_X;
		o.src_mod.swizzle_y = Src_Mod::CompSel_Y;
		o.src_mod.swizzle_z = Src_Mod::CompSel_Z;
		o.src_mod.swizzle_w = Src_Mod::CompSel_W;

		o.src_mod.negate_x = true;
		o.src_mod.negate_y = true;
		o.src_mod.negate_z = true;
		o.src_mod.negate_w = true;

		return o;
	}

	ILOperand ILOperand::clamp() const
	{
		ILOperand o(*this);
		o.modifier_present = true;

		o.dst_mod.component_x = Dst_Mod::ModComp_Write;
		o.dst_mod.component_y = Dst_Mod::ModComp_Write;
		o.dst_mod.component_z = Dst_Mod::ModComp_Write;
		o.dst_mod.component_w = Dst_Mod::ModComp_Write;
		o.dst_mod.clamp = true;
		o.dst_mod.shift_scale = Dst_Mod::Shift_None;

		o.src_mod.swizzle_x = Src_Mod::CompSel_X;
		o.src_mod.swizzle_y = Src_Mod::CompSel_Y;
		o.src_mod.swizzle_z = Src_Mod::CompSel_Z;
		o.src_mod.swizzle_w = Src_Mod::CompSel_W;

		o.src_mod.negate_x = false;
		o.src_mod.negate_y = false;
		o.src_mod.negate_z = false;
		o.src_mod.negate_w = false;

		return o;

	}
}
