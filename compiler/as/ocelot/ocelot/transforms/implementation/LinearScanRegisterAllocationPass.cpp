/*! \file LinearScanRegisterAllocationPass.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Monday December 28, 2009
	\brief The source file for the LinearScanRegisterAllocationPass class.

	Updated by Diogo Nunes Sampaio on April 23, 2012
 */

#ifndef LINEAR_SCAN_REGISTER_ALLOCATION_PASS_CPP_INCLUDED
#define LINEAR_SCAN_REGISTER_ALLOCATION_PASS_CPP_INCLUDED

// Ocelot Includes
#include <ocelot/transforms/interface/LinearScanRegisterAllocationPass.h>
#include <ocelot/ir/interface/Module.h>

// Standard Library Includes
#include <stack>
#include <algorithm>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#ifdef NDEBUG
#undef NDEBUG
#endif

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif
#ifdef INFO
#undef INFO
#endif

#ifdef DEBUG
#undef DEBUG
#endif

#ifdef DEBUG_DETAILS
#undef DEBUG_DETAILS
#endif

#ifdef REPORT_ERROR_LEVEL
#undef REPORT_ERROR_LEVEL
#endif

#define REPORT_ERROR_LEVEL 5
#define REPORT_BASE 0
#if REPORT_BASE == 0
#define NDEBUG 1
#endif
#define INFO 4
#define DEBUG 3
#define DEBUG_DETAILS 2

#if REPORT_BASE
#include <sstream>
#endif

namespace transforms
{

const ir::PTXOperand::DataType LinearScanRegisterAllocationPass::selectType(
	const ir::PTXOperand::DataType &a, const ir::PTXOperand::DataType &b) const
{
	if(a == b) return a;

	if(ir::PTXOperand::valid(a, b)) return a;

	if(ir::PTXOperand::valid(b, a)) return b;

	bool hasFloat = ir::PTXOperand::isFloat(a) || ir::PTXOperand::isFloat(b);
	bool hasInt = ir::PTXOperand::isInt(a) || ir::PTXOperand::isFloat(b);

	bool isBinary = (hasFloat && hasInt) ||
		(((a <= ir::PTXOperand::DataType::b64) &&
		(a >= ir::PTXOperand::DataType::b8))) ||
		(((b <= ir::PTXOperand::DataType::b64) &&
		(b >= ir::PTXOperand::DataType::b8)));

	ir::PTXOperand::DataType out =
		ir::PTXOperand::DataType::TypeSpecifier_invalid;

	int size = std::max(ir::PTXOperand::bytes(a), ir::PTXOperand::bytes(b));

	size >>= 4;

	if(isBinary)
	{
		out = ir::PTXOperand::DataType::b8;
	}
	else if(hasFloat)
	{
		assertM(size > 0, "Invalid float size");
		out = ir::PTXOperand::DataType::f16;
		size >>= 1;
	}
	else
	{
		if(!(ir::PTXOperand::isSigned(a) && ir::PTXOperand::isSigned(b)))
		{
			out = ir::PTXOperand::DataType::u8;
		}
		else
		{
			out = ir::PTXOperand::DataType::s8;
		}
	}

	while(size > 0)
	{
		out = (ir::PTXOperand::DataType) ((int) (out) + 1);
		size >>= 1;
	}

	return out;
}

LinearScanRegisterAllocationPass::LinearScanRegisterAllocationPass(
	unsigned regs)
: KernelPass(StringVector({"DataflowGraphAnalysis",
		"DominatorTreeAnalysis","PostDominatorTreeAnalysis"}),
		"LinearScanRegisterAllocationPass"),
	_memoryStack("ocelot_ls_stack"), _registers(regs)
{
	
}
LinearScanRegisterAllocationPass::LinearScanRegisterAllocationPass(
	unsigned regs, const Analysis::StringVector& analysis, unsigned reserved)
: KernelPass(analysis, "LinearScanRegisterAllocationPass"),
	_memoryStack("ocelot_ls_stack"), _registers(regs)
{
	
}

void LinearScanRegisterAllocationPass::initialize(const ir::Module& m)
{
	reportE(INFO, "Initializing linear scan register allocation "
		"pass for module " << m.path());
}

/*TODO: Run on a kernel making different linear sequences,
	and choosing the one with lowest amount of spills
 *TODO: Accept profiling, with basic block access counts,
 	and chooses sequence with lowest runtime loads and stores*/
void LinearScanRegisterAllocationPass::runOnKernel(ir::IRKernel& k)
{
	reportE(INFO, "Running on kernel " << k.name << " with "
		<< _registers << " physical registers");

	// put the dataflow graph in the correct form
	auto dfg = static_cast<analysis::DataflowGraph*>(
		getAnalysis("DataflowGraphAnalysis"));
	
	dfg->convertToSSAType(analysis::DataflowGraph::Minimal);
	
	_clear();
	_kernel = &k;
	_coalesce();
	_computeIntervals();
	_allocate();
	_spill();
	_extendStack();
}

void LinearScanRegisterAllocationPass::finalize()
{
	reportE(INFO, "Finalizing linear scan pass.");
}

void LinearScanRegisterAllocationPass::setRegisterCount(unsigned count)
{
	_registers = count;
}

void LinearScanRegisterAllocationPass::_clear()
{
	_kernel = NULL;
	
	while(_coalesced.size() > 0)
	{
		free(_coalesced.begin()->second);
		_coalesced.erase(_coalesced.begin());
	}
	
	_ssa.clear();
	_writes.clear();
	_instructions.clear();
	_blocks.clear();
	_memoryStack.clear();
	_aliveIns.clear();
	_sequence.clear();
	_aliveIns.clear();
	_onRegisters.clear();
}

void LinearScanRegisterAllocationPass::_coalesce()
{
	typedef std::map<RegisterId, analysis::DataflowGraph::Type> TypeMap;
	typedef std::set<RegisterId> Connections;
	typedef std::map<RegisterId, Connections> Graph;
	typedef std::set<RegisterId> RegisterSet;
	typedef std::vector<RegisterId> RegisterVector;
	typedef std::stack<RegisterId> Stack;

	reportE(DEBUG, " Coalescing SSA registers into single values.");

	Graph graph;
	TypeMap types;
	getAnalysis("DominatorTreeAnalysis");
	getAnalysis("PostDominatorTreeAnalysis");

	for(auto block = _dfg().begin(); block != _dfg().end(); ++block)
	{
		/* Map all variables of each phi-instruction
			into a single node of the graph */
		for(auto phi = block->phis().begin(); phi != block->phis().end(); ++phi)
		{
			/* The graph node name is given by the
				destination variable of the phi-instruction */
			graph[phi->d.id].insert(phi->d.id);
			types[phi->d.id] = phi->d.type;
			reportE(DEBUG, "	" << phi->d.id << " <- "
				<< phi->d.id << ":" << ir::PTXOperand::toString(phi->d.type));
			for(auto r = phi->s.begin(); r != phi->s.end(); ++r)
			{
				graph[phi->d.id].insert(r->id);
				graph[r->id].insert(phi->d.id);
				reportE(DEBUG, "	" << phi->d.id << " <- "
					<< r->id << ":" << ir::PTXOperand::toString(r->type));
				reportE(DEBUG,
					"	Select type for " << ir::PTXOperand::toString(r->type)
					<< " and " << ir::PTXOperand::toString(types[phi->d.id]));
				types[phi->d.id] = selectType(r->type, types[phi->d.id]);
				reportE(DEBUG, "	Selected type for "
					<< ir::PTXOperand::toString(types[phi->d.id]));
				types[r->id] = r->type;
			}
		}

		/* Map every instruction result variable as a node */
		for(auto i = block->instructions().begin();
			i != block->instructions().end(); ++i)
		{
			for(auto d = i->d.begin(); d != i->d.end(); ++d)
			{
				graph[*d->pointer].insert(*d->pointer);
				reportE(DEBUG, "	" << *d->pointer << " <- "
					<< *d->pointer << ":" << ir::PTXOperand::toString(d->type));
				types[*d->pointer] = d->type;
			}
		}
	}

	RegisterSet allocated;

	for(Graph::iterator node = graph.begin(); node != graph.end(); ++node)
	{
		if(allocated.insert(node->first).second)
		{
			reportE(DEBUG, "	Examining node " << node->first
				<< " it is connected to: ");

			RegisterVector vector(1, node->first);
			Stack stack;
			stack.push(node->first);

			while(!stack.empty())
			{
				Graph::iterator next = graph.find(stack.top());
				assert( next != graph.end());
				stack.pop();

				for(Connections::iterator connection = next->second.begin();
					connection != next->second.end(); ++connection)
				{
					if(allocated.insert(*connection).second)
					{
						reportE(DEBUG, "	 " << *connection);
						stack.push(*connection);
						vector.push_back(*connection);
					}
				}
			}

			for(auto r = vector.begin(); r != vector.end(); ++r)
			{
				_ssa.insert(std::make_pair(*r, _coalesced.size()));
			}
			
			assertM(types.find(node->first) != types.end(),
				"Variable type not present");

			_addCoalesced(_coalesced.size(), types[node->first]);
		}
	}
}

/* This compute interval function does not take into account:
 * TODO: Holes where variables are not alive
 * TODO: Make a memory stack that take life ranges, so variables spilled that
 * don't interfere with each other can go in same memory position
 * TODO: 64bit register require 2 consequent 32 bit registers,
 	starting on a odd valued register */
void LinearScanRegisterAllocationPass::_computeIntervals()
{
	reportE(DEBUG, "Computing live intervals.");

	_sequence = _dfg().executableSequence();

	//Initially find each occurrence of each coalesced register on sequence
	Interval::Point count = 0;
	for(auto block = _sequence.begin(); block != _sequence.end(); ++block)
	{
		if((*block)->instructions().size() == 0) continue;

		unsigned int blockStart = ++count;
		unsigned int blockEnd = count + (*block)->instructions().size();
		_aliveIns.insert(blockStart);
		reportE(DEBUG, "\t---Point:" << blockStart << " marked as aliveIn");
		_writes[blockStart] = RegisterSet();
		_blocks[blockStart] = *block;
		reportE(DEBUG, "\tChecking variables:");

		for(auto r = (*block)->aliveIn().begin();
				r != (*block)->aliveIn().end(); r++)
		{
			reportE(DEBUG, "\t\tVariable: " << r->id);
			CoalescedRegisterMap::iterator index = _ssa.find(r->id);
			assertM(index != _ssa.end(), "Variable has not been mapped "
				"as a coalesced variable");
			reportE(DEBUG, "\t\t\tMapped as coalesced " << index->second);
			_writes[blockStart].insert(index->second);
		
			if((_coalesced[index->second]->interval.begin == 0) ||
				(_coalesced[index->second]->interval.begin > blockStart))
			{
				reportE(DEBUG, "\t\t\tDefining life range start as "
					<< blockStart);
				_coalesced[index->second]->interval.begin = blockStart;
			}
		}
		reportE(DEBUG, "\t---Point:" << blockEnd << " is aliveOut");

		for(auto r = (*block)->aliveOut().begin();
				r != (*block)->aliveOut().end(); r++)
		{
			reportE(DEBUG, "\t\tVariable: " << r->id);
			CoalescedRegisterMap::iterator index = _ssa.find(r->id);
			assertM(index != _ssa.end(),
				"Variable has not been mapped as as coalesced variable");

			reportE(DEBUG, "\t\t\tMapped as coalesced "
				<< _coalesced[index->second]->reg());
			
			if((_coalesced[index->second]->interval.end < blockEnd))
			{
				_coalesced[index->second]->interval.end = blockEnd;
				reportE(DEBUG, "\t\t\tDefining life range end as " << blockEnd);
			}
		}
		
		reportE(DEBUG, "\tCheckin instructions");
		for(auto instruction = (*block)->instructions().begin();
				instruction != (*block)->instructions().end(); ++instruction)
		{
			++count;
			reportE(DEBUG, "\t\t---Point:" << count
				<< ", instruction " << instruction->i->toString());

			_writes[count] = RegisterSet();
			_instructions[count] = instruction;
			_blocks[count] = *block;

			reportE(DEBUG, "\t\tCheckin destination variables:");
			for(auto d = instruction->d.begin();
					d != instruction->d.end(); ++d)
			{
				reportE(DEBUG, "\t\t\tVariable:" << *d->pointer);
				CoalescedRegisterMap::iterator index = _ssa.find(*d->pointer);
				assertM(index != _ssa.end(), "Variable has not "
					"been mapped as as coalesced variable");
				reportE(DEBUG, "\t\t\tMapped as coalesced " << index->second);
				assertM(index->second <= _coalesced.size(),
					"Mapped value too high");
				CoalescedRegister *cr = _coalesced[index->second];
				_writes[count].insert(index->second);
				cr->rw[count] = CoalescedRegister::RW::WRITE;
				cr->writesCount++;
				if((cr->interval.begin == 0) || (cr->interval.begin > count))
				{
					cr->interval.begin = count;
					reportE(DEBUG, "\t\t\tDefining life range start as " << count);
				}
			}
			
			reportE(DEBUG, "\t\tCheckin source variables:");
			
			for(auto s = instruction->s.begin(); s != instruction->s.end(); ++s)
			{
				reportE(DEBUG, "\t\t\tVariable:" << *s->pointer);
				CoalescedRegisterMap::iterator index = _ssa.find(*s->pointer);
				
				assertM(index != _ssa.end(), "Variable has not been "
					"mapped as as coalesced variable");
				reportE(DEBUG, "\t\t\t\tMapped as coalesced " << index->second);
				assertM(index->second <= _coalesced.size(),
					"Mapped value too high");
					
				CoalescedRegister *cr = _coalesced[index->second];
				cr->rw[count] = CoalescedRegister::RW::READ;
				cr->readsCount++;
				
				if(cr->interval.end < count)
				{
					cr->interval.end = count;
					reportE(DEBUG, "\t\t\t\tDefining life range end " << count);
				}
			}
		}
	}
}

/*TODO Change spill policy so:
 *2	Prefers variables that:
 *2.1	 Aren't predicates (require more load / store instructions)
 *2.2	 Are 64 bit
 *2.3	 Already been spilled or that can spill to same place of
 *       a spilled variable (life range with holes) 
 */
void LinearScanRegisterAllocationPass::_allocate()
{
	RegisterList available;

	reportE(DEBUG, "Allocating")
	reportE(DEBUG, "\tThese registers are available: ");

	for(RegisterId r = 0; r != _registers; ++r)
	{
		available.push_back(r);
		reportE(DEBUG, "\t" << r);
	}
	
	for(auto point = _blocks.begin(); point != _blocks.end(); point++)
	{
		if(_aliveIns.find(point->first) != _aliveIns.end())
		{
			_treatAliveInPoint(available, point->first);
		}
		else
		{
			_treatPoint(available, point->first);
		}
	}
}

void LinearScanRegisterAllocationPass::_treatAliveInPoint(
	RegisterList &available, const Interval::Point pointNum)
{
	reportE(DEBUG, "---Point:" << pointNum
		<< ", is aliveIn for block " << _blocks[pointNum]->label());

#if REPORT_BASE
	std::stringstream a;
	for(RegisterSet::iterator id = _writes[pointNum].begin();
		id != _writes[pointNum].end(); id++)
	{
		a << *id << " ";
	}
	reportE(DEBUG, "\taliveIn list: " << a.str());
#endif

	if(_registers > available.size())
	{
		reportE(DEBUG, "\tRemoving all variables from registers");
		RegisterSet onRegister = _onRegisters[pointNum];
		
		while(!onRegister.empty())
		{
			CoalescedRegister& active = *_coalesced[*onRegister.begin()];
			reportE(DEBUG, "\t\tRemoving variable " << active.reg());
			
			while(active.allocated.size() > 0)
			{
				reportE(DEBUG, "\t\t\tFreeing physical register "
					<< *active.allocated.begin());
				available.push_back(*active.allocated.begin());
				active.allocated.erase(*active.allocated.begin());
			}
			
			onRegister.erase(onRegister.begin());
		}
		
		assertM(_registers == available.size(), "Has max regs ("
			<< _registers << ") and available (" << available.size() << ')' );
	}

	/* Allocate registers based on spill policy, spill the variables
		that won't fit, and don't allocate to already spilled */
	reportE(DEBUG, "\tMapping variables by spill policy");

	RegisterMap workList = _spillPolicy.LRU(_coalesced,
		_writes[pointNum], pointNum);
	RegisterSet onRegister;
	
	reportE(DEBUG, "\tAllocating registers");
	RegisterMap::iterator best = workList.end();
	RegisterMap::iterator worst = workList.begin();
	
	while(best != worst)
	{
		best--;
		CoalescedRegister& var = *_coalesced[best->second];
		reportE(DEBUG, "\t\tVariable " << var.reg()
			<< " that requires " << var.size()
			<< " physical registers, while available are "
			<< available.size());
		
		if(var.spilled())
		{
			reportE(DEBUG, "\t\t\tIs already spilled, ignoring it");
		}
		else
		{
			reportE(DEBUG, "\t\t\tVariable " << var.reg()
				<< " that requires " << var.size()
				<< " physical registers, while available are "
				<< available.size());
			
			if(available.size() < var.size())
			{
				reportE(DEBUG, "\t\t\t\tSpilling variable");
				var.spill();
			}
			else
			{
				reportE(DEBUG, "\t\t\tMarking variable "
					<< var.reg() << " as on register");
				
				onRegister.insert(var.reg());
				
				while(var.size() > var.allocated.size())
				{
					var.allocated.insert(available.front());
					reportE(DEBUG, "\t\t\t\tAllocating physical register "
						<< available.front());
					available.pop_front();
					reportE(DEBUG,
						"\t\t\t\t\tRemoving available physical register");
				}
			}
		}
	}
	
	_onRegisters[pointNum + 1] = onRegister;
}

void LinearScanRegisterAllocationPass::_treatPoint(
	RegisterList &available, const Interval::Point pointNum)
{

	unsigned int used = 0;
	RegisterSet created;
	reportE(DEBUG, "---Point:" << pointNum << ", instruction "
		<< _instructions[pointNum]->i->toString());
	assertM(_onRegisters.find(pointNum) != _onRegisters.end(),
		"Previous point not computed yet");
	
	RegisterSet onRegister = _onRegisters[pointNum];
	reportE(DEBUG, "\tChecking for expired variables");
	
	for(RegisterSet::iterator active = onRegister.begin();
		active != onRegister.end();)
	{
		if(_coalesced[*active]->interval.end > pointNum)
		{
			used += _coalesced[*active]->allocated.size();
			active++;
			continue;
		}
		
		CoalescedRegister& expired = *_coalesced[*active];
		reportE(DEBUG, "\t\t\tRegister " << expired.reg()
			<< " expired, freeing physical registers:");
		assertM(expired.size() == expired.allocated.size(),
			"Expired, has " << expired.allocated.size()
			<< " allocated physical registers, should have " << expired.size());

		while(expired.allocated.size() > 0)
		{
			reportE(DEBUG, "\t\t\t" << *expired.allocated.begin());
		
			available.push_back(*expired.allocated.begin());
			expired.allocated.erase(expired.allocated.begin());
		}
		
		RegisterSet::iterator remove = active;
		active++;
		onRegister.erase(remove);
	}

	reportE(DEBUG, "\t\tFree register count: " << available.size());
	assertM((used + available.size()) == _registers,
		"Used registers: " << used <<", available: " << available.size()
		<< " should have " << _registers);

	if(_writes[pointNum].empty())
	{
		reportE(DEBUG, "No variables created, skipping new variables");
		_onRegisters[pointNum + 1] = onRegister;
		reportE(DEBUG, "Variables count on register for next instruction: "
			<< onRegister.size());
		return;
	}

	unsigned registersReq = 0;
	reportE(DEBUG, "\tCounting number of required registers")

	for(RegisterSet::iterator id = _writes[pointNum].begin();
		id != _writes[pointNum].end(); id++)
	{
		CoalescedRegister& current = *_coalesced[*id];
		reportE(DEBUG, "\t\tVariable " << current.reg() << ", ("
			<< current.interval.begin << ", " << current.interval.end
			<< "), that uses " << current.size() << " registers");
			
		assertM(created.insert(*id).second,
			"Duplicated variable on a interval");

		if(!current.isAllocated()) registersReq += current.size();
	}

	reportE(DEBUG, "\tNumber of output registers: " << registersReq);
	assertM(registersReq <= (_registers),
		"More registers are required in output than registers available");

	if(available.size() < registersReq)
	{
		reportE(DEBUG, "\tOut of free registers, spill required, "
			"checking variables on register");
		
		RegisterMap workList = _spillPolicy.LRU(_coalesced,
			onRegister, pointNum);
		
		while(available.size() < registersReq)
		{
			reportE(DEBUG, "\t\tRemoving variable "
				<< _coalesced[workList.begin()->second]->reg() << " with size "
				<< _coalesced[workList.begin()->second]->size());

			CoalescedRegister *removed = _coalesced[workList.begin()->second];

			if(!removed->spilled())
			{
				reportE(DEBUG, "\t\t\tSpilling it");
				removed->spill();
			}
			
			while(removed->allocated.size() > 0)
			{
				available.push_back(*removed->allocated.begin());
				removed->allocated.erase(removed->allocated.begin());
			}

			onRegister.erase(workList.begin()->second);
			workList.erase(workList.begin());
		}
	}

	reportE(DEBUG, "\tAllocating physical registers");

	while(created.size() > 0)
	{
		CoalescedRegister& current = *_coalesced[*created.begin()];
		onRegister.insert(current.reg());
		reportE(DEBUG, "\t\tVariable:" << *created.begin());
		reportE(DEBUG, "\t\t\tMapped as coalesced " << current.reg()
			<< " with size = " << current.size());

		while(current.allocated.size() != current.size())
		{
			reportE(DEBUG, "\t\t\tPhysical register:" << available.front());
			assertM(!available.empty(), "No registers available");
			current.allocated.insert(available.front());
			available.pop_front();
		}

		created.erase(created.begin());
	}

	_onRegisters[pointNum + 1] = onRegister;
}

void LinearScanRegisterAllocationPass::_spill()
{
	reportE(DEBUG, "Inserting spill code");
	for(auto point = _blocks.begin(); point != _blocks.end(); point++)
	{
		if(_aliveIns.find(point->first) != _aliveIns.end()) continue;

		auto &i = _instructions[point->first];

		reportE(DEBUG, "\t---Point:" << point->first
			<< ", instruction " << i->i->toString());

		CoalescedRegister::InstructionList list;
		const RegisterSet &onRegister = _onRegisters[point->first];

		for(auto s = i->s.begin(); s != i->s.end(); ++s)
		{
			CoalescedRegisterMap::iterator mapping = _ssa.find(*s->pointer);
			assertM( mapping != _ssa.end(), "Variable not coalesced mapped");

			CoalescedRegister& coalesced = *_coalesced[mapping->second];
			reportE(DEBUG, "\t\tSource variable " << *s->pointer);
			reportE(DEBUG, "\t\t\tMapped as coalesced " << coalesced.reg());
			if(coalesced.spilled())
			{
				reportE(DEBUG, "\t\t\tSpilled");
				if(onRegister.find(coalesced.reg()) != onRegister.end())
				{
					reportE(DEBUG,
						"\t\t\tBut still on register, no load required");
				}
				else
				{
					coalesced.load(_dfg(), list, *s->pointer);
				}
			}
			else
			{
				assertM(onRegister.find(coalesced.reg()) != onRegister.end(),
					"Source variable not spilled but "
					"not marked as on register");
			}
		}

		while(list.size() > 0)
		{
			_dfg().insert(_blocks[point->first], list.front(), i);
			reportE(DEBUG, "\t\t\tAdd new load instructions: "
				<< list.front().toString());
			list.pop_front();
		}

		for(auto d = i->d.begin(); d != i->d.end(); ++d)
		{
			CoalescedRegisterMap::iterator mapping = _ssa.find(*d->pointer);
			assertM(mapping != _ssa.end(), "Variable "
				<< *d->pointer << " not mapped");
			assert(mapping->second < _coalesced.size());

			CoalescedRegister& coalesced = *_coalesced[mapping->second];
			reportE(DEBUG, "\t\tDestination variable " << *d->pointer);
			reportE(DEBUG, "\t\t\tMapped as coalesced " << coalesced.reg());

			if(coalesced.spilled())
			{
				coalesced.store(_dfg(), list, *d->pointer);
				reportE(DEBUG, "\t\t\tStored in memory");
			}
		}

		i++;
		if(list.size() > 0)
		{
			while(list.size() > 0)
			{
				reportE(DEBUG, "\t\t\tAdd new store instructions: "
					<< list.front().toString());
				_dfg().insert(_blocks[point->first], list.front(), i);
				list.pop_front();
			}
		}
	}
}

void LinearScanRegisterAllocationPass::_extendStack()
{
	_memoryStack.declaration(_kernel->locals);
}

analysis::DataflowGraph& LinearScanRegisterAllocationPass::_dfg()
{
	Analysis* dfg_structure = getAnalysis("DataflowGraphAnalysis");
	assert(dfg_structure != 0);

	return *static_cast<analysis::DataflowGraph*>(dfg_structure);
}

void LinearScanRegisterAllocationPass::_addCoalesced(const RegisterId id,
	const analysis::DataflowGraph::Type type)
{
	_coalesced[id] = new CoalescedRegister(id, type, &_memoryStack);
}

}

#endif

