/*! \file Local.h
	\author Gregory Diamos <gregory.diamos@gatech>
	\date Thursday September 17, 2009
	\brief The header file for the Local class
*/

#ifndef LOCAL_H_INCLUDED
#define LOCAL_H_INCLUDED

#include <ocelot/ir/interface/PTXInstruction.h>
#include <ocelot/ir/interface/PTXStatement.h>

namespace ir
{
	/*! \brief A class to represent a variable with local scope */
	class Local
	{
		public:
			/*! \brief The address space of the local */
			PTXInstruction::AddressSpace space;

			/*!	Data type of parameter */
			PTXOperand::DataType type;

			/*!	Name of parameter */
			std::string name;

			/*! \brief Alignment attribute */
			unsigned int alignment;

			/*! \brief Vector attribute */
			PTXInstruction::Vec vector;
			
			/*! \brief The number of elements if this is an array */
			unsigned int elements;
			
			/*! \brief Attribute */
			PTXStatement::Attribute attribute;

		public:
			/*! \brief Where this variable is allocated in memory */
			size_t offset;
			
		public:
			explicit Local(const PTXStatement& statement);
			Local();

			/*!	Returns the size of a local */
			unsigned int getSize() const;

			/*!	Returns the size of a single element of a local */
			unsigned int getElementSize() const;
		
			/*! \brief Return the alignment restriction of the local */
			unsigned int getAlignment() const;
					
			/*! \brief Return a PTX statement representing the local */
			PTXStatement statement() const;
		
			/*! \brief Return a parsable string representing the local */
			std::string toString() const;
			

	};
}

#endif

