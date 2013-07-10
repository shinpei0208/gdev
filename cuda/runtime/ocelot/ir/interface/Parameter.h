/*! \file Parameter.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\date Jan 15, 2009
	\brief describes a parameter declaration
*/

#ifndef IR_PARAMETER_H_INCLUDED
#define IR_PARAMETER_H_INCLUDED

#include <sstream>
#include <vector>

#include <ocelot/ir/interface/PTXEmitter.h>
#include <ocelot/ir/interface/PTXOperand.h>
#include <ocelot/ir/interface/PTXStatement.h>

namespace ir {

	class Parameter {
	public:
		/*! Union of possible variables containing each value */		
		union ValueType {
			PTXU8 val_u8;
			PTXU16 val_u16;
			PTXU32 val_u32;
			PTXU64 val_u64;

			PTXS8 val_s8;
			PTXS16 val_s16;
			PTXS32 val_s32;
			PTXS64 val_s64;

			PTXF32 val_f32;
			PTXF64 val_f64;

			PTXB8 val_b8;
			PTXB16 val_b16;
			PTXB32 val_b32;
			PTXB64 val_b64;
		};
		
		/*! A vector of values */
		typedef std::vector<ValueType> ValueVector;

	public:
		/*!	\brief Get value */
		static std::string value(const Parameter& p);

	public:
		Parameter(const PTXStatement& statement,
			bool arg, bool isReturn = false);
		explicit Parameter(const std::string& s = "",
			ir::PTXOperand::DataType t = ir::PTXOperand::u64,
			unsigned int alignment = 0,
			ir::PTXInstruction::Vec v = ir::PTXOperand::v1,
			bool argument = false, bool returnArgument = false);
		~Parameter();

		/*!	Returns the size of a parameter */
		unsigned int getSize() const;

		/*!	Returns the size of a single element of a parameter */
		unsigned int getElementSize() const;
		
		/*! \brief Return the alignment restriction of the parameter */
		unsigned int getAlignment() const;

		/*! \brief Is this parameter a kernel argument? */ 
		bool isArgument() const;
		
		/*! \brief Return a parsable string representing the parameter */
		std::string toString(PTXEmitter::Target = PTXEmitter::Target_OcelotIR) const;
		
		/*! \brief returns true if the parameter is declared to be a pointer to
			some state space (i.e., ptrAddressSpace != AddressSpace_invalid */
		bool isPtrDeclaration() const;

	public:
	
		/*!	Data type of parameter */
		PTXOperand::DataType type;

		/*!	Name of parameter */
		std::string name;

		/*! \brief Alignment attribute */
		unsigned int alignment;

		/*! \brief Vector attribute */
		ir::PTXInstruction::Vec vector;

		/*! \brief Is this kernel argument */
		bool argument;
		
		/*! \brief is this a return argument? */
		bool returnArgument;
		
		/*! \brief if not equal to AddressSpace_invalid, indicates parameter is a ptr to
			the indicated address space*/
		PTXInstruction::AddressSpace ptrAddressSpace;

		/*	Runtime bindings */
	public:
		/*!	Offset in bytes from the beginning of the parameter block */
		unsigned int offset;

		/*!	Array containing union of values */
		ValueVector arrayValues;
	};

}

#endif
