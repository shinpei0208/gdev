/*! \file Global.h
	
	\date Tuesday March 31, 2009
	\author Gregory Diamos
	
	\brief The header file for the Global class.	
*/

#ifndef GLOBAL_H_INCLUDED
#define GLOBAL_H_INCLUDED

#include <ocelot/ir/interface/PTXStatement.h>

namespace ir
{

	/*! \brief A class for referencing preinitialized global variables */
	class Global
	{
		public:	
			bool local; //! Is variable set via an initializer
			void* pointer; //! Pointer to memory base
			void* reference; //! dummy pointer used for lookups
			ir::PTXStatement statement; //! Statement declaring the variable
	
		public:
			/*! \brief Constructor */
			Global();
			
			/*! \brief Initializing constructor */
			Global( char* );
			
			/*!	\brief Construct from Statement */
			Global( const ir::PTXStatement& );

			/*!	\brief Copy constructor */
			Global( const Global& );

			/*!	\brief Destructor for preinitialized globals */
			~Global();
			
			/*!	\brief Assignment */
			Global& operator=(const Global&);
			
			/*! \brief Get the address space of the global */
			PTXInstruction::AddressSpace space() const;
			
			/*! \brief Get the identifier of the global */
			const std::string& name() const;
	
	};

}

#endif

