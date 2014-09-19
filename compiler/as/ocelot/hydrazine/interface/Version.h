/*! \file Version.h
	\date Saturday January 17, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the Version class
*/

#ifndef VERSION_H_INCLUDED
#define VERSION_H_INCLUDED

#include <string>

namespace hydrazine
{

	/*!	\brief A class to represent a version of a package */
	class Version
	{
	
		public:
			/*!	\brief A type for a version number */
			typedef unsigned int Number;
		
		private:
			/*!	\brief Parse string representation of the version
			
				\param version A string representation of the version
			*/
			void parse( const std::string& version );
			
		public:
			/*! \brief The major feature set */		
			Number major;
			
			/*! \brief The minor feature set */
			Number minor;
			
			/*! \brief The changelist number */
			Number changeList;
					
		public:
			/*! \brief Initialize the version from the specified config file */		
			Version();

			/*!
				\brief Return a string representation of the version
			
				\return Said representation
			*/
			std::string toString() const;
		
			/*! \brief Greater than comparison */
			bool operator>( const Version& );
	
	};

}

#endif

