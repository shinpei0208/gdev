/*! \file   compression.h
	\date   Wednesday December 5, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file the hydrazine compression abstraction
*/

#pragma once

// Standard Library Includes
#include <cstdint>

namespace hydrazine
{

void decompress(void* output, uint64_t& outputSize, const void* input,
	uint64_t inputSize);

}


