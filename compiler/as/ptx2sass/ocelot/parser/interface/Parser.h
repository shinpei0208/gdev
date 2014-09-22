/*!
	\file Parser.h
	\date Wednesday January 14, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the Parser class.
*/

#ifndef PARSER_H_INCLUDED
#define PARSER_H_INCLUDED

#include <ocelot/ir/interface/Instruction.h>

/*!
	\brief A namespace for parser related classes
*/
namespace parser
{

	/*!
		\brief An interface that parses a text or binary file and generates
			an internal representation of a program.
	*/
	class Parser
	{
		public:		
			/*! \brief Name of the file being parsed */
			std::string fileName;
					
		public:
		
			/*! \brief Virtual destructor */
			virtual ~Parser() {};
	
			/*!
			
				\brief Parse a file, generating an internal representation of
					the program.
			
				\param input The stream being parsed
				
				\param language What is the language of the file being parsed?

				\return A module containing the the internal representation
					of the parsed program
							
			*/
			virtual void parse( std::istream& input, 
				ir::Instruction::Architecture 
				language = ir::Instruction::PTX ) = 0;
	
	};

}

#endif

