/*! \file   AnalysisFactory.h
	\date   Wednesday October 3, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the AnallysisFactory class.
	
*/

#pragma once

// Ocelot Includes
#include <ocelot/analysis/interface/Analysis.h>

namespace analysis
{

/*! \brief Used to create passes by name */
class AnalysisFactory
{
public:
	typedef Analysis::StringVector StringVector;

public:
	/*! \brief Create a analysis object from the specified name */
	static Analysis* createAnalysis(const std::string& name,
		const StringVector& options = StringVector());

};

}


