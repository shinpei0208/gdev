/*
 * SpillPolicy.cpp
 *
 *	Created on: Mar 19, 2012
 *			Author: undead
 */

#include <ocelot/transforms/interface/SpillPolicy.h>

namespace transforms
{

SpillPolicy::RegisterMap SpillPolicy::LRU(const CoalescedRegisterVector &crv,
	const RegisterSet &in,
	const Interval::Point &p) const
{
	RegisterMap sorted;
	RegisterSet::const_iterator r = in.begin();
	RegisterSet::const_iterator end = in.end();
	
	for(; r != end; r++)
	{
		assertM(crv.find(*r) != crv.end(), "ihhhhhhhhhhh");
		const CoalescedRegister &reg = *crv.find(*r)->second;
		unsigned int value = (unsigned int)(-1) - reg.readDistance(p+1);
		sorted.insert(std::make_pair(value, reg.reg()));;
	}

	return sorted;
}

SpillPolicy::RegisterMap SpillPolicy::LU(const CoalescedRegisterVector &crv,
	const RegisterSet &in) const
{
	RegisterMap sorted;
	
	RegisterSet::const_iterator r = in.begin();
	RegisterSet::const_iterator end = in.end();
	for(; r != end; r++)
	{
		const CoalescedRegister &reg = *crv.find(*r)->second;
		unsigned int value = (unsigned int)(-1) - reg.readsCount;
		sorted.insert(std::make_pair(value, reg.reg()));;
	}

	return sorted;
}

}

