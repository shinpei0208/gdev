/*! \file ILOperand.h
 *  \author Rodrigo Dominguez <rdomingu@ece.neu.edu>
 *  \date April 29, 2010
 *  \brief The header file for the IL Operand class.
 */

#ifndef IL_OPERAND_H_INCLUDED
#define IL_OPERAND_H_INCLUDED

// Ocelot includes
#include <ocelot/ir/interface/Instruction.h>

// C++ standard library includes
#include <string>

namespace ir
{
	/*! \brief A class for a basic IL Operand */
	class ILOperand {
		public:
			/*! \brief Register type */
			enum RegType {
				RegType_Temp,
				RegType_Const_Buf,
				RegType_Literal,
				RegType_Thread_Id_In_Group,
				RegType_Thread_Group_Id,
				RegType_Generic_Mem,
				RegType_Invalid
			};

			/*! \brief Destination modifier */
			class Dst_Mod {
				public:
					Dst_Mod();

					std::string toString() const;

					/*! \brief Destination component */
					enum ModDstComponent {
						ModComp_NoWrite, // do not write this component
						ModComp_Write,   // write the result to this component
						ModComp_0,       // force the component to float 0.0
						ModComp_1,       // force the component to float 1.0
						ModComp_Invalid
					};

					/*! \brief Shift scale */
					enum ShiftScale {
						Shift_None,
						Shift_X2,        // shift value left by 1 bit
						Shift_X4,        // shift value left by 2 bits
						Shift_X8,        // shift value left by 3 bits
						Shift_D2,        // shift value right by 1 bit
						Shift_D4,        // shift value right by 2 bits
						Shift_D8,        // shift value right by 3 bits
						Shift_Invalid
					};

					static std::string toString(ModDstComponent dc);

					ModDstComponent component_x;
					ModDstComponent component_y;
					ModDstComponent component_z;
					ModDstComponent component_w;
					bool clamp;
					ShiftScale shift_scale;
			};
			
			/*! \brief Source modifier */
			class Src_Mod {
				public:
					Src_Mod();

					std::string toString() const;

					/*! \brief Source component */
					enum ComponentSelect {
						CompSel_X,       // select the 1st component
						CompSel_Y,       // select the 2nd component
						CompSel_Z,       // select the 3rd component
						CompSel_W,       // select the 4th component
						CompSel_0,       // force the component to float 0.0
						CompSel_1,       // force the component to float 1.0
						CompSel_Invalid
					};

					ComponentSelect swizzle_x;
					ComponentSelect swizzle_y;
					ComponentSelect swizzle_z;
					ComponentSelect swizzle_w;
					bool negate_x;
					bool negate_y;
					bool negate_z;
					bool negate_w;

				private:
					std::string swizzleString() const;
					std::string negateString() const;
					static std::string toString(ComponentSelect c);
			};

			/*! \brief Default constructor */
			ILOperand(RegType type = RegType_Invalid);

			/*****************************//**
			 * \name Operand information
			 ********************************/
			//@{
			unsigned int num;
			RegType type;
			bool modifier_present;
			bool immediate_present;
			Dst_Mod dst_mod; // destination modifier
			Src_Mod src_mod; // source modifier
			unsigned int imm;
			//@}

			std::string immediateString() const;
			static std::string toString(RegType rt);

			std::string dstString() const;
			std::string srcString() const;
			std::string clampString() const;
			std::string shiftString() const;

			/*****************************//**
			 * \name Swizzles
			 *
			 * Returns a copy of the operand.
			 ********************************/
			//@{
			ILOperand x() const;
			ILOperand y() const;
			ILOperand z() const;
			ILOperand w() const;
			ILOperand xy() const;
			//@}
			
			/*! \brief Negate */
			ILOperand neg() const;
			/*! \brief Clamp */
			ILOperand clamp() const;
	};
}

#endif
