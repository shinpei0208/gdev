/*! \file   PassFactory.h
	\date   Wednesday May 2, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the PassFactory class.
	
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace transforms { class Pass; }

namespace transforms
{

/*! \brief Used to create passes by name */
class PassFactory
{
public:
	/*! \brief Create a pass object from the specified name */
	static Pass* createPass(const std::string& name);
};

}


