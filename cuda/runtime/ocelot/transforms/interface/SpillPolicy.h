/*
* SpillPolicy.h
*
*	Created on: Mar 19, 2012
*			Author: undead
*/

#ifndef SPILLPOLICY_H_
#define SPILLPOLICY_H_

// Ocelot Includes
#include <ocelot/transforms/interface/CoalescedRegister.h>

// Standard Library Includes
#include <set>
#include <map>

namespace transforms
{

class SpillPolicy
{
public:
	/*! \brief A unique ID for a register */
	typedef analysis::DataflowGraph::RegisterId RegisterId;
	/*! \brief A map of coalesced registers pointers */
	typedef std::map<RegisterId, CoalescedRegister*> CoalescedRegisterVector;
	/*! \brief A map from SSA registers to Coalesced registers */
	typedef std::map<RegisterId, RegisterId> CoalescedRegisterMap;
	/*! \brief A set of Coalesced registers */
	typedef std::set<RegisterId> RegisterSet;
	/*! \brief A map from SSA registers to Coalesced registers */
	typedef std::multimap<unsigned int, RegisterId> RegisterMap;

	/* \brief Less Recently Used policy */
	RegisterMap LRU(const CoalescedRegisterVector &crv,
		const RegisterSet &in,
		const Interval::Point &p) const;
	/* \brief Less [Times] Used policy */
	RegisterMap LU(const CoalescedRegisterVector &crv,
		const RegisterSet &in) const;
};

}

#endif /* SPILLPOLICY_H_ */

