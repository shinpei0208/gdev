/*! \file SSAGraph.cpp
	\date Saturday June 27, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The source file for the SSAGraph class.	
*/

#ifndef SSA_GRAPH_CPP_INCLUDED
#define SSA_GRAPH_CPP_INCLUDED

// Ocelot Includes
#include <ocelot/analysis/interface/SSAGraph.h>
#include <ocelot/analysis/interface/PostdominatorTree.h>
#include <ocelot/analysis/interface/DivergenceGraph.h>
// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <stack>
#include <set>

// Preprocessor Macros
#ifdef INFO
#undef INFO
#endif

#define INFO 4

#ifdef DEBUG
#undef DEBUG
#endif

#define DEBUG 3

#ifdef DEBUG_DETAILS
#undef DEBUG_DETAILS
#endif

#define DEBUG_DETAILS 2

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

#ifdef NDEBUG
#undef NDEBUG
#endif

#if REPORT_BASE == 0
#define NDEBUG 1
#endif

#ifdef REPORT_ERROR_LEVEL
#undef REPORT_ERROR_LEVEL
#endif

#define REPORT_ERROR_LEVEL 1

namespace analysis {

void SSAGraph::_initialize( Block& b, DataflowGraph::iterator block, 
	DataflowGraph::RegisterId& current )
{
	report( "	Initializing block: " << block->label() );
	b.regs.clear();
	for( DataflowGraph::RegisterSet::iterator 
		reg = block->_aliveIn.begin(); 
		reg != block->_aliveIn.end(); ++reg )
	{
		report( "	 Mapping alive in register r" << reg->id 
			<< " to r" << current );
		b.regs.insert( std::make_pair( *reg, 
			DataflowGraph::Register( current++, reg->type ) ) );
	}
	
	b.aliveInMap = b.regs;
	
	for( DataflowGraph::InstructionVector::iterator 
		instruction = block->_instructions.begin(); 
		instruction != block->_instructions.end(); ++instruction )
	{
		report( "	 Initializing instruction: "
			<< instruction->i->toString() );
		for( DataflowGraph::RegisterPointerVector::iterator 
			reg = instruction->s.begin(); 
			reg != instruction->s.end(); ++reg )
		{
			RegisterMap::iterator mapping = b.regs.find( *reg->pointer );
			assert( mapping != b.regs.end() );
			report( "		Mapping source register r" << *reg->pointer 
				<< " to r" << mapping->second.id );
			*reg->pointer = mapping->second.id;
		}
		
		for( DataflowGraph::RegisterPointerVector::iterator 
			reg = instruction->d.begin(); 
			reg != instruction->d.end(); ++reg )
		{
			RegisterMap::iterator mapping = b.regs.find( *reg->pointer );
			if( mapping == b.regs.end() )
			{
				report( "		Mapping destination register r"
					<< *reg->pointer 
					<< " to r" << current );
				mapping = b.regs.insert( std::make_pair( *reg, 
					DataflowGraph::Register( current++, 
					reg->type ) ) ).first;
			}
			else
			{
				report( "	 ReMapping destination register r" 
					<< *reg->pointer 
					<< " from r" << mapping->second.id
					<< " to r" << current );
				mapping->second.id = current++;
			}
			*reg->pointer = mapping->second.id;
		}
	}
}

void SSAGraph::_insertPhis()
{
	typedef std::vector< DataflowGraph::Register > IdVector;
	typedef std::unordered_map< DataflowGraph::Register, IdVector > IdMap;
	
	report( " Inserting Phi instructions." );
	
	for( BlockMap::iterator block = _blocks.begin(); 
		block != _blocks.end(); ++block )
	{
		IdMap map;
		
		report( "	For block " << block->first->label() );
		report( "	 Processing " << block->first->_predecessors.size() 
			<< " predecessors." );
		
		for( DataflowGraph::BlockPointerSet::iterator 
			predecessor = block->first->_predecessors.begin(); 
			predecessor != block->first->_predecessors.end(); 
			++predecessor )
		{
			report( "		For predecessor " << (*predecessor)->label() );
			BlockMap::iterator predecessorBlock 
				= _blocks.find( *predecessor );
			assert( predecessorBlock != _blocks.end() );
					
			for( DataflowGraph::RegisterSet::iterator 
				reg = (*predecessor)->_aliveOut.begin(); 
				reg != (*predecessor)->_aliveOut.end(); ++reg )
			{
				DataflowGraph::RegisterSet::iterator in 
					= block->first->_aliveIn.find( reg->id );
				if( in != block->first->_aliveIn.end() )
				{
					RegisterMap::iterator mapping 
						= block->second.aliveInMap.find( reg->id );
					assert( mapping != block->second.aliveInMap.end() );
					IdMap::iterator phi = map.find( mapping->second.id );
					if( phi == map.end() )
					{
						phi = map.insert( std::make_pair( 
							mapping->second, IdVector() ) ).first;
					}
					RegisterMap::iterator remapping 
						= predecessorBlock->second.regs.find( *reg );
					assert( remapping 
						!= predecessorBlock->second.regs.end() );
					report( "		 Mapping phi source r" 
						<< remapping->second.id 
						<< " to destination r" << mapping->second.id );
					phi->second.push_back( remapping->second );
				}
			}
		}
		
		report( "		 Mapping phis with no producer." );
		
		for( DataflowGraph::RegisterSet::iterator 
			reg = block->first->_aliveIn.begin(); 
			reg != block->first->_aliveIn.end(); ++reg )
		{
			RegisterMap::iterator mapping 
				= block->second.aliveInMap.find( reg->id );
			assert( mapping != block->second.aliveInMap.end() );
			IdMap::iterator phi = map.find( mapping->second );
			if( phi == map.end() )
			{
				phi = map.insert( std::make_pair( mapping->second, 
					IdVector() ) ).first;
				report( "			Mapping phi source r" << mapping->second.id 
					<< " to destination r" << mapping->second.id );
				phi->second.push_back( mapping->second );
			}
		}
		
		for( IdMap::iterator phi = map.begin(); phi != map.end(); ++phi )
		{
			DataflowGraph::PhiInstruction instruction;
			instruction.d = phi->first;
			for( IdVector::iterator source = phi->second.begin(); 
				source != phi->second.end(); ++source )
			{
				assertM(
					ir::PTXOperand::valid(instruction.d.type, source->type),
					" PHI destination register %r" << instruction.d.id
					<< " type "
					<< ir::PTXOperand::toString(instruction.d.type)
					<< " does not match source register %r" << source->id
					<< " type "
					<< ir::PTXOperand::toString(source->type) );
				instruction.s.push_back( *source );
			}
			block->first->_phis.push_back( instruction );
		}
	}
}

void SSAGraph::_updateIn()
{
	report( " Updating AliveIn for " << _blocks.size() << " blocks." );
	for( BlockMap::iterator block = _blocks.begin(); 
		block != _blocks.end(); ++block )
	{
		report( "	Updating AliveIn for block " << block->first->label() );
		block->first->_aliveIn.clear();
		
		for( DataflowGraph::PhiInstructionVector::iterator 
			phi = block->first->_phis.begin(); 
			phi != block->first->_phis.end(); ++phi )
		{
			for( DataflowGraph::RegisterVector::iterator 
				reg = phi->s.begin(); reg != phi->s.end(); ++reg )
			{
				report( "	 Adding register r" << reg->id << " ("
					<< ir::PTXOperand::toString( reg->type ) << ")" );
				block->first->_aliveIn.insert( *reg );
			}
		}
	}
}

void SSAGraph::_updateOut()
{
	report( " Updating AliveOut for " << _blocks.size() << " blocks." );
	for( BlockMap::iterator block = _blocks.begin(); 
		block != _blocks.end(); ++block )
	{
		report( "	Updating AliveOut for block " << block->first->label() );
		DataflowGraph::RegisterSet newAliveOut;
		for( DataflowGraph::RegisterSet::iterator 
			reg = block->first->_aliveOut.begin(); 
			reg != block->first->_aliveOut.end(); ++reg )
		{
			RegisterMap::iterator mapping = block->second.regs.find( *reg );
			assert( mapping != block->second.regs.end() );
			report( "	 Mapping alive out register r" << mapping->first.id 
				<< " to r" << mapping->second.id << " ("
				<< ir::PTXOperand::toString( mapping->second.type ) << ")" );
			newAliveOut.insert( mapping->second );
		}
		block->first->_aliveOut = std::move( newAliveOut );
	}
}

SSAGraph::SSAGraph(DataflowGraph& graph, DataflowGraph::SsaType form) :
		_graph(graph), _form(form)
{
}

void SSAGraph::toSsa()
{
	reportE(INFO,	"Converting dataflow graph SSA form");

	assert( !_graph._ssa);
	_graph._ssa = _form;
	_blocks.clear();
	DataflowGraph::RegisterId current = 0;

	reportE(INFO,	" Initializing " << _graph.size() << " blocks");
	for(DataflowGraph::iterator fi = _graph.begin();
		fi != _graph.end(); ++fi)
	{
		BlockMap::iterator block = _blocks.insert(
			std::make_pair(fi, Block())).first;
		_initialize(block->second, block->first, current);
	}

	_insertPhis();
	_updateIn();
	_updateOut();
	_graph._maxRegister = current;

	if(_form == DataflowGraph::SsaType::Default)
	{
		reportE(INFO,	" Graph is now in full SSA form.");
		return;
	}

	_minimize();

	if(_form == DataflowGraph::SsaType::Minimal)
	{
		reportE(DEBUG,	" Graph is now in minimal SSA form.");
		return;
	}

	_gssa(current);
	_graph._maxRegister = current;
	reportE(DEBUG,	" Graph is now in gated SSA form.");
}

void SSAGraph::fromSsa()
{
	report( "Converting dataflow graph out of pure SSA form" );
	
	typedef std::unordered_set< DataflowGraph::Register > RegisterSet;
	typedef std::unordered_map< DataflowGraph::Register, 
		RegisterSet > RegisterSetMap;
	typedef std::unordered_map< DataflowGraph::Register, 
		DataflowGraph::Register > RegisterMap;
	typedef std::stack< DataflowGraph::Register > RegisterStack;
	
	assert( _graph._ssa != DataflowGraph::SsaType::None);
	_graph._ssa = DataflowGraph::SsaType::None;
	
	RegisterMap    map;
	RegisterSetMap registerGraph;
	
	report( "Coalescing phi instructions." );
	for( DataflowGraph::iterator fi = _graph.begin(); 
		fi != _graph.end(); ++fi )
	{
		for( DataflowGraph::PhiInstructionVector::iterator 
			phi = fi->_phis.begin(); phi != fi->_phis.end(); ++phi )
		{
			for( DataflowGraph::RegisterVector::iterator s = phi->s.begin();
				s != phi->s.end(); ++s )
			{
				registerGraph[ phi->d ].insert( *s );
				registerGraph[ *s ].insert( phi->d );
			}
		}
		
		fi->_phis.clear();
	}

	RegisterSet encountered;

	for( RegisterSetMap::iterator node = registerGraph.begin(); 
		node != registerGraph.end(); ++node )
	{
		if( encountered.insert( node->first ).second )
		{
			report(" Subgraph for r" << node->first.id << " ("
				<< ir::PTXOperand::toString( node->first.type ) << ")");
			RegisterStack stack;
			DataflowGraph::Register reg = node->first;
			
			for( RegisterSet::iterator connection = node->second.begin();
				connection != node->second.end(); ++connection )
			{
				if( encountered.insert( *connection ).second )
				{
					stack.push( *connection );
				}
			}
			while( !stack.empty() )
			{
				DataflowGraph::Register nextReg = stack.top();
				stack.pop();
				map[ nextReg ] = reg;
				report( "	contains r" << nextReg.id << " ("
					<< ir::PTXOperand::toString( nextReg.type ) << ")" );
				RegisterSetMap::iterator 
					next = registerGraph.find( nextReg );
				assert( next != registerGraph.end() );
				for( RegisterSet::iterator 
					connection = next->second.begin();
					connection != next->second.end(); ++connection )
				{
					if( encountered.insert( *connection ).second )
					{
						stack.push( *connection );
					}
				}
			}
		}
	}

	report( "Updating instructions." );
	for( DataflowGraph::iterator fi = _graph.begin(); 
		fi != _graph.end(); ++fi )
	{
		report( " Examining block " << fi->label() );
		for( DataflowGraph::InstructionVector::iterator 
			instruction = fi->_instructions.begin(); 
			instruction != fi->_instructions.end(); ++instruction )
		{
			report( "	Examining instruction " 
				<< instruction->i->toString() );
			for( DataflowGraph::RegisterPointerVector::iterator 
				reg = instruction->s.begin(); 
				reg != instruction->s.end(); ++reg )
			{
				RegisterMap::iterator mapping = map.find( *reg->pointer );
				if( mapping != map.end() )
				{
					report( "	 Mapping r" << *reg->pointer 
						<< " to r" << mapping->second.id );
					*reg->pointer = mapping->second.id;
				}
			}

			for( DataflowGraph::RegisterPointerVector::iterator 
				reg = instruction->d.begin(); 
				reg != instruction->d.end(); ++reg )
			{
				RegisterMap::iterator mapping = map.find( *reg->pointer );
				if( mapping != map.end() )
				{
					report( "	 Mapping r" << *reg->pointer 
						<< " to r" << mapping->second.id );
					*reg->pointer = mapping->second.id;
				}
			}
			report( "	 Modified instruction to " 
				<< instruction->i->toString() );
		}
		
		DataflowGraph::RegisterSet newAlive;
		
		report(" Updating alive out set.");
		for( DataflowGraph::RegisterSet::iterator 
			reg = fi->_aliveOut.begin(); 
			reg != fi->_aliveOut.end(); ++reg )
		{
			RegisterMap::iterator mapping = map.find( *reg );
			if( mapping != map.end() )
			{
				report( "	r" << mapping->second.id << " ("
					<< ir::PTXOperand::toString( mapping->second.type )
					<< ")" );
				newAlive.insert( mapping->second );
			}
			else
			{
				report( "	r" << reg->id << " ("
					<< ir::PTXOperand::toString( reg->type ) << ")" );
				newAlive.insert( *reg );
			}
		}
		fi->_aliveOut = std::move( newAlive );
		
		newAlive.clear();
		
		report(" Updating alive in set.");
		for( DataflowGraph::RegisterSet::iterator 
			reg = fi->_aliveIn.begin(); 
			reg != fi->_aliveIn.end(); ++reg )
		{
			RegisterMap::iterator mapping = map.find( *reg );
			if( mapping != map.end() )
			{
				report( "	r" << mapping->second.id << " ("
					<< ir::PTXOperand::toString( mapping->second.type )
					<< ")" );
				newAlive.insert( mapping->second );
			}
			else
			{
				report( "	r" << reg->id << " ("
					<< ir::PTXOperand::toString( reg->type ) << ")" );
				newAlive.insert( *reg );
			}
		}
		
		fi->_aliveIn = std::move( newAlive );
	}
	
}

void SSAGraph::_minimize() 
{
	RegisterIdMap fromToMap;
	bool hasPhiChanges = true;
	std::map<DataflowGraph::RegisterId, bool> clearedReg;
	std::set<DataflowGraph::RegisterId> keptVariables;

	reportE(DEBUG, "#Starting SSA phi removing");
	while(hasPhiChanges)
	{
		reportE(DEBUG, "_minimize loop start++++++");
		hasPhiChanges = false;
		for(DataflowGraph::iterator fi = _graph.begin();
			fi != _graph.end(); ++fi)
		{
			reportE(DEBUG, "\tBlock:" << fi->label());
			for(auto phi = fi->phis().begin(); phi != fi->phis().end();)
			{
				keptVariables.insert(phi->d.id);
				/* Remove single phis */
				assertM(phi->s.size() > 0, "Empty phi");
				if(phi->s.size() == 1)
				{
					if(phi->d.id == phi->s.begin()->id)
					{
						reportE(DEBUG, "\tRemoving phi " << *phi);
						
						auto predecessor = *fi->predecessors().begin();
						
						if((fi->predecessors().size() == 1) &&
							predecessor->predecessors().size() == 0)
						{
						
						}
						else
						{
							auto erasePhi = phi++;
							reportE(DEBUG, "\t\tErasing phi " << *erasePhi);
							fi->phis().erase(erasePhi);
							hasPhiChanges = true;
							continue;
						}
					}
					else if(clearedReg.find(phi->d.id) == clearedReg.end())
					{
						reportE(DEBUG, "\tMapping phi " << *phi);
						clearedReg[phi->d.id] = false;
						fromToMap[phi->d.id] = (*phi->s.begin());
						hasPhiChanges = true;
					}
					else if(clearedReg[phi->d.id])
					{
						hasPhiChanges = true;
						keptVariables.erase(phi->d.id);
						reportE(DEBUG, "\tRemoving phi " << *phi);
						auto erasePhi = phi++;
						fi->phis().erase(erasePhi);
						continue;
					}
				}
				else
				{
					/* Check every phi source for duplicates or renaming */
					auto source    = phi->s.begin();
					auto sourceEnd = phi->s.end();
					
					DataflowGraph::RegisterSet newSources;
					bool hasNewSources = false;
					
					reportE(DEBUG, "\tFor phi " << *phi);
					
					for(; source != sourceEnd; source++)
					{
						RegisterIdMap::iterator final =
							fromToMap.find(source->id);

						if(final != fromToMap.end())
						{
							while(fromToMap.find(final->second.id) !=
								fromToMap.end())
							{
								final = fromToMap.find(final->second.id);
							}
							
							reportE(DEBUG, "\t\tChange source "
								<< source->id << " to " << final->second.id);
							
							newSources.insert(final->second);
							hasNewSources = true;
						}
						else
						{
							newSources.insert(*source);
						}
					}
					if(hasNewSources)
					{
						if(newSources.size() == 0)
						{
							reportE(DEBUG, "Removing phi " << *phi
							<< " with no new sources ");
							auto erasePhi = phi++;
							fi->phis().erase(erasePhi);
							hasPhiChanges = true;
							continue;
						}
						
						reportE(DEBUG, "Changing phi " << *phi << " to ");
						hasPhiChanges = true;
						phi->s.clear();

						while(newSources.size() > 0)
						{
							phi->s.push_back(*newSources.begin());
							newSources.erase(newSources.begin());
						}
						
						reportE(DEBUG, "" << *phi);
					}
				}
				phi++;
			}
		}

		for(DataflowGraph::iterator fi = _graph.begin();
			fi != _graph.end(); ++fi)
		{
			/* First insert new regs in alive in so they can
				be used in the instructions */
			reportE(DEBUG, "Block " << fi->label());
			reportE(DEBUG, "\tAliveIn:");

			for(auto in = fi->aliveIn().begin(); in != fi->aliveIn().end(); )
			{
				RegisterIdMap::iterator final = fromToMap.find(in->id);
				keptVariables.insert(in->id);
				
				if(final != fromToMap.end())
				{
					keptVariables.erase(in->id);
					
					while(fromToMap.find(final->second.id) != fromToMap.end())
					{
						final = fromToMap.find(final->second.id);
					}
					
					reportE(DEBUG, "\t\t - r" << in->id << " + r"
						<< final->second.id);
					
					if(fi->aliveIn().find(final->second) == fi->aliveIn().end())
					{
						fi->aliveIn().insert(final->second);
						in = fi->aliveIn().begin(); //???TODO: WTF?
					}
				}
				in++;
			}
			reportE(DEBUG, "\tInstruction:");
			/*Remove mapped variables from instructions sources */
			for(auto inst = fi->instructions().begin();
				inst != fi->instructions().end(); )
			{
				reportE(DEBUG, "\t\t" << inst->i->toString());
				auto d = inst->d.begin();
				auto dE = inst->d.end();
				
				for(; d != dE; d++)
				{
					keptVariables.insert(*d->pointer);
				}

				for(auto s = inst->s.begin(); s != inst->s.end(); s++)
				{
					RegisterIdMap::iterator final = fromToMap.find(*s->pointer);
					keptVariables.insert(*s->pointer);
					if(final != fromToMap.end())
					{
						keptVariables.erase(*s->pointer);
						while(fromToMap.count(final->second.id) != 0)
						{
							final = fromToMap.find(final->second.id);
						}
						*s->pointer = final->second.id;
					}
				}
				inst++;
			}

			/*Remove mapped variables from alive in */
			reportE(DEBUG, "Block " << fi->label());
			reportE(DEBUG, "\tAliveIn:");
			for(auto in = fi->aliveIn().begin(); in != fi->aliveIn().end(); )
			{
				RegisterIdMap::iterator final = fromToMap.find(in->id);
				keptVariables.insert(in->id);
				
				if(final != fromToMap.end())
				{
					while(fromToMap.find(final->second.id) != fromToMap.end())
					{
						final = fromToMap.find(final->second.id);
					}

					keptVariables.erase(in->id);
					reportE(DEBUG, "\t\t - r" << in->id
						<< " + r" << final->second.id);

					DataflowGraph::RegisterSet::iterator inErase = in++;
					fi->aliveIn().erase(inErase);
					fi->aliveIn().insert(final->second);
					in = fi->aliveIn().begin();
				}
				else
				{
					in++;
				}
			}

			/*Remove mapped variables from alive out */
			reportE(DEBUG, "\tAliveOut:");
			for(auto out = fi->aliveOut().begin();
				out != fi->aliveOut().end(); )
			{
				RegisterIdMap::iterator final = fromToMap.find(out->id);
				keptVariables.insert(out->id);
			
				if(final != fromToMap.end())
				{
					keptVariables.erase(out->id);
					while(fromToMap.find(final->second.id) != fromToMap.end())
					{
						final = fromToMap.find(final->second.id);
					}
					
					reportE(DEBUG, "\t\t - r" << out->id
						<< " + r" << final->second.id);
					
					auto outErase = out++;
					fi->aliveOut().erase(outErase);
					
					if(fi->aliveOut().count(final->second) == 0)
					{
						fi->aliveOut().insert(final->second);
					}
					
					continue;
				}
				out++;
			}
		}
		
		auto clear    = clearedReg.begin();
		auto endClear = clearedReg.end();

		for(; clear != endClear; clear++)
		{
			clear->second = true;
		}
	}
}

bool SSAGraph::_isPossibleDivBranch(
	const DataflowGraph::InstructionVector::iterator &instruction) const
{
	if(typeid(ir::PTXInstruction) == typeid(*(instruction->i)))
	{
		const ir::PTXInstruction& ptxI =
			*(static_cast<ir::PTXInstruction*>(instruction->i));

		return ((ptxI.opcode == ir::PTXInstruction::Bra) &&
			(!ptxI.uni) && (instruction->s.size() > 0));
	}

	return false;
}

void SSAGraph::_gssa(DataflowGraph::RegisterId& current)
{
	_graph._phiPredicateMap.clear();

	PostdominatorTree& pdt = static_cast<PostdominatorTree&>(
		*_graph.getAnalysis("PostDominatorTreeAnalysis"));
	assert(&pdt != nullptr);
	
	reportE(DEBUG, "Building GSSA");
	for(DataflowGraph::iterator fi = _graph.begin(); fi != _graph.end(); ++fi)
	{
		reportE(DEBUG, " Block " << fi->label());
		auto branch = --(fi->instructions().end());
		if((fi->instructions().empty()) || (!_isPossibleDivBranch(branch)))
		{
			reportE(DEBUG, "	Does not end on a possible divergent branch");
			continue;
		}
		
		reportE(DEBUG, "	Ends on branch:" << branch->i->toString());
		DataflowGraph::iterator pd = _graph.getCFGtoDFGMap().find(
			pdt.getPostDominator(fi->block()))->second;
		
		if(pd == _graph.end())
		{
			reportE(DEBUG, "	But does not have a postdominator block");
			continue;
		}
		assertM(branch->s.size() == 1, "Wrong, branch instruction with "
			<< branch->s.size() << " sources instead of 1");
		
		DataflowGraph::RegisterId predicate = *branch->s.begin()->pointer;
		reportE(DEBUG, "	Does have a postdominator block: " << pd->label());
		reportE(DEBUG, "	Uses predicate: " << predicate);
		/* If the block ends on a possible divergent branch:
		 * Step 1:
		 * find phis before the immediate postdominator block that depends on
		 * variables generated after the branch and mark them as predicate
		 * dependent.
		 *
		 * Step 2:
		 * Based on data dependency, any variable on the postdominator block
		 * that depends on variables generated on at least one of the sides of
		 * the branch, are predicate dependent. If it doesn't exists a phi
		 * instruction that uses this variables, split their life range
		 * adding a single source phi */
		
		std::set<DataflowGraph::Block*> workload;
		std::set<DataflowGraph::Block*> visited;
		std::set<DataflowGraph::RegisterId> newValues;
		std::set<DataflowGraph::PhiInstruction*> phis;
		DivergenceGraph dg;
		bool dontLoop = false;

		if(fi->fallthrough() != pd)
		{
			if(fi->fallthrough() != _graph.end())
			{
				workload.insert(&(*fi->fallthrough()));
				reportE(DEBUG, "	 Added fallthrough block "
					<< fi->fallthrough()->label() << " on workload");
			}
		}
		else
		{
			dontLoop = true;
			reportE(DEBUG, "	 Fallthrough block "
				<< fi->fallthrough()->label() << " is postdominator");
		}

		for(DataflowGraph::BlockPointerSet::iterator t = fi->targets().begin();
			t != fi->targets().end(); t++)
		{
			if(*t != pd)
			{
				if(*t != _graph.end())
				{
					workload.insert(&(*(*t)));
					reportE(DEBUG, "	 Added target block "
						<< (*t)->label() << " on workload");
				}
			}
			else
			{
				dontLoop = true;
				reportE(DEBUG, "	 Target block "
					<< (*t)->label() << " is postdominator");
			}
		}

		while(workload.size() > 0)
		{
			DataflowGraph::Block *block = *workload.begin();
			reportE(DEBUG, "		Working on block " << block->label());

			if((block->fallthrough() != pd) &&
				(block->fallthrough() != _graph.end()))
			{
				reportE(DEBUG, "		 Added fallthrough block "
					<< block->fallthrough()->label() << " on workload");
				workload.insert(&(*block->fallthrough()));
			}

			if(visited.find(block) == visited.end())
			{
				visited.insert(block);
				auto t = block->targets().begin();
				for(; t != block->targets().end(); t++)
				{
					if((*t != pd) && (*t != _graph.end()))
					{
						workload.insert(&(*(*t)));
						reportE(DEBUG, "		 Added target block "
							<< (*t)->label() << " on workload");
					}
				}
			}
			
			workload.erase(block);
			/* Find all phi instruction before the postdominator */
			reportE(DEBUG, "		 Compute phis:");
			
			auto phi  = block->phis().begin();
			auto phiE = block->phis().end();
			
			for(; phi != phiE; phi++)
			{
				/* If we have a possible one side loop, don't add the phis of the
				 * head block */
				if(!(dontLoop && (block == &(*fi))))
				{
					phis.insert(&(*phi));
					reportE(DEBUG, "			Added " << *phi
						<< " as possible dependent.");
				}
				else
				{
					reportE(DEBUG, "			Not adding " << *phi
						<< " as possible dependent.");
				}
				
				auto s  = phi->s.begin();
				auto sE = phi->s.end();
				
				reportE(DEBUG, "			 Adding data dependency for "
					<< *phi);
					
				for(; s != sE; s++)
				{
					dg.insertEdge(s->id, phi->d.id);
					reportE(DEBUG, "			 "
						<< s->id << " -> " << phi->d.id);
				}
			}

			reportE(DEBUG, "		 Compute instructions:");

			reportE((dontLoop && (block == &(*fi))) && DEBUG,
				"			Back into the header block, not adding "
				"instructions destinations as predicate dependent");

			auto ins    = block->instructions().begin();
			auto endIns = block->instructions().end();

			for(; ins != endIns; ins++)
			{
				reportE(DEBUG, "			Computing instruction " << *ins);
				auto destination    = ins->d.begin();
				auto endDestination = ins->d.end();

				for(; destination != endDestination; destination++)
				{
					if(!(dontLoop && (block == &(*fi))))
					{
						reportE(DEBUG, "			 Inserting "
							<< *destination->pointer
							<< " as predicate dependent.");
						/*TODO: To deal with unstructured CFG it is required to
						 * test phi nodes before the postdominator, and add them
						 * as predicate dependent only if they depend on
						 * variables from both sides of a branch instruction.
						 * It requires flags propagating from each side of each
						 * possible divergent branch A fast solution, but too
						 * conservative, is to add every phi instruction before
						 * the postdominator as dependent on the predicate. */
						dg.setAsDiv(*destination->pointer);
					}
					
					auto source    = ins->s.begin();
					auto sourceEnd = ins->s.end();
					
					for(; source != sourceEnd; source++)
					{
						dg.insertEdge(*source->pointer, *destination->pointer);
						reportE(DEBUG, "			 " << *source->pointer
							<< " -> " << *destination->pointer);
					}
				}
			}
		}

		reportE(DEBUG, "	Compute dependency of phis "
			"before the postdominator");

		/* Now, we locate every phi that depends on a newValue */
		while(phis.size() > 0)
		{
			DataflowGraph::PhiInstruction*          phi = *phis.begin();
			DataflowGraph::RegisterVector::iterator s   = phi->s.begin();
			DataflowGraph::RegisterVector::iterator sE  = phi->s.end();
			reportE(DEBUG, "	 " << *phi);

			for(; s != sE; s++)
			{
				auto val = newValues.find(s->id);
				if(val != newValues.end())
				{
					reportE(DEBUG, "		Is dependent on created value "
						<< s->id);
					if(_graph._phiPredicateMap.count(phi) == 0)
					{
						reportE(DEBUG, "		Creating map for phi" << *phi);
						std::set<DataflowGraph::RegisterId> predicates;
						_graph._phiPredicateMap[phi] = predicates;
					}
					
					_graph._phiPredicateMap[phi].insert(predicate);
				}
			}
			phis.erase(phi);
		}
		
		reportE(DEBUG, "	Coloring.");
		dg.computeDivergence();
		newValues.clear();
		reportE(DEBUG, "	Compute postdominator aliveIn colored variables.");
		
		for(auto var = pd->aliveIn().begin(); var != pd->aliveIn().end(); var++)
		{
			if(dg.isDivNode(var->id))
			{
				reportE(DEBUG, "	 aliveIn variable " << var->id
					<< " is predicate dependent");
				newValues.insert(var->id);
			}
			else
			{
				reportE(DEBUG, "	 aliveIn variable " << var->id
					<< " isn't predicate dependent");
			}
		}

		reportE(DEBUG, "	Compute postdominator phis with colored sources.");

		for(auto phi = pd->phis().begin(); phi != pd->phis().end(); phi++)
		{
			reportE(DEBUG, "	 " << *phi);
			for(auto s = phi->s.begin(); s != phi->s.end(); s++)
			{
				if(newValues.find(s->id) != newValues.end())
				{
					reportE(DEBUG, "	 Is dependent on colored variable "
						<< s->id << " and " << "dependent on predicate "
						<< predicate);

					if(_graph._phiPredicateMap.count(&(*phi)) == 0)
					{
						reportE(DEBUG, "		Creating map for phi" << *phi);

						_graph._phiPredicateMap[&(*phi)] =
							std::set<DataflowGraph::RegisterId>();
					}

					_graph._phiPredicateMap[&(*phi)].insert(predicate);
					newValues.erase(s->id);
				}
			}
		}
		
		if(newValues.empty())
		{
			reportE(DEBUG, "	All aliveIn variables are used on "
				"phi instructions, going to next branch");
			continue;
		}
		
		std::map<DataflowGraph::RegisterId, DataflowGraph::Register*> varMap;
		reportE(DEBUG, "	There are " << newValues.size()
			<< " variables that need life splitting");

		while(newValues.size() > 0)
		{
			auto var  = pd->aliveIn().begin();
			auto varE = pd->aliveIn().end();
			for(; var != varE; var++)
			{
				if(newValues.find(var->id) == newValues.end()) continue;

				DataflowGraph::Register destination(current++, var->type);
				DataflowGraph::PhiInstruction newPhi;
				newPhi.d = destination;
				newPhi.s.push_back(*var);
				pd->_phis.push_back(newPhi);
				DataflowGraph::PhiInstruction *phip = &pd->_phis.back();
				reportE(DEBUG, "	 Adding phi " << newPhi);
				varMap[var->id] = &pd->_phis.back().d;
				newValues.erase(var->id);
				std::set<DataflowGraph::RegisterId> predicates;
				_graph._phiPredicateMap[phip] = predicates;
				_graph._phiPredicateMap[phip].insert(predicate);
			}
		}

		reportE(DEBUG, "	 Renaming variables from here on.");
		while(varMap.size() > 0)
		{
			reportE(DEBUG, "########################"
				"#################################");
			std::set<DataflowGraph::Block*> altered;
			workload.clear();
			workload.insert(&(*pd));
			DataflowGraph::Register source(varMap.begin()->first,
				varMap.begin()->second->type);
			DataflowGraph::Register& dest = *varMap.begin()->second;
			varMap.erase(varMap.begin());
			while(workload.size() > 0){
				reportE(DEBUG, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
					"@@@@@@@@@@@@@@@@@@@@@@@@@@@");
				DataflowGraph::Block *block = *workload.begin();
				reportE(DEBUG, "		Working on block " << block->label());
				workload.erase(workload.begin());
				altered.insert(block);
				auto alive = block->aliveIn().find(source);
				if(alive == block->aliveIn().end())
				{
					reportE(DEBUG, "		 Block doesn't have the variable "
						<< source.id << " on the aliveIn");
					continue;
				}
				bool backEdge = false;
				if(block == &(*pd))
				{
					reportE(DEBUG, "		 Not changing "
						"post dominator aliveIn");
				}
				else
				{
					if(visited.find(block) != visited.end())
					{
						backEdge = true;
						reportE(DEBUG, "		 This block came from "
							"a back edge, not removing aliveIn");
					}
					else
					{
						reportE(DEBUG, "		 Had the variable "
							<< source.id << " on the aliveIn");
						block->aliveIn().erase(alive);
					}
					block->aliveIn().insert(dest);
				}
				if(backEdge)
				{
					reportE(DEBUG, "		 Don't add blocks "
						"targets of a backedge block");
				}
				else
				{
					alive = block->aliveOut().find(source);
					if(alive == block->aliveOut().end())
					{
						reportE(DEBUG,
							"		 Block doesn't have the variable "
							<< source.id << " on the aliveOut, not "
							"adding new workload");
					}
					else
					{
						reportE(DEBUG, "		 Changing aliveOut");
						block->aliveOut().erase(alive);
						block->aliveOut().insert(dest);
						if((altered.count(&(*block->fallthrough())) == 0) &&
							(block->fallthrough() != _graph.end()))
						{
							reportE(DEBUG, "		 Added fallthrough block "
								<< block->fallthrough()->label()
								<< " on workload");
							workload.insert(&(*block->fallthrough()));
						}

						auto t = block->targets().begin();
						for(; t != block->targets().end(); t++)
						{
							if(altered.count(&(*(*t))) == 0 &&
								(*t != _graph.end()))
							{
								workload.insert(&(*(*t)));
								reportE(DEBUG, "		 Added target block "
									<< (*t)->label() << " on workload");
							}
						}
					}
				}

				bool destOnPhi = false;
				reportE(DEBUG, "		 Compute phi instructions:");
				
				auto phi  = block->phis().begin();
				auto phiE = block->phis().end();

				for(; phi != phiE; phi++)
				{
					reportE(DEBUG, "			For " << *phi);
					bool hasToHaveDest = false;
					bool alreadyHasDest = false;
					
					if(phi->d == source)
					{
						reportE(DEBUG, "			 Phi d " << phi->d.id
							<< " == " << source.id
							<< " might have to add dest");
						hasToHaveDest = true;
					}
					
					if(phi->s.size() == 1)
					{
						reportE(DEBUG, "			 Life split phi, "
							"don't do nothing");
						break;
					}
					
					DataflowGraph::RegisterVector::iterator s = phi->s.begin();
					DataflowGraph::RegisterVector::iterator sE = phi->s.end();
					
					for(; s != sE; s++)
					{
						if(*s == source)
						{
							reportE(DEBUG, "			 Phi s " << s->id
								<< " == " << source.id
								<< " might have to add dest");
							hasToHaveDest = true;
						}
						if(*s == dest)
						{
							reportE(DEBUG, "			 Phi s " << s->id
								<< " == " << dest.id
								<< " don't have to add dest");
							alreadyHasDest = true;
							destOnPhi = true;
						}
					}
					if(hasToHaveDest && !alreadyHasDest)
					{
						reportE(DEBUG, "			 " << *phi << " becomes:");
						phi->s.push_back(dest);
						destOnPhi = true;
						reportE(DEBUG, "				 " << *phi);
					}
				}
				
				if(destOnPhi)
				{
					reportE(DEBUG, "			Dest " << dest.id
						<< " was found on phi, going to next block");
					continue;
				}
				
				if(backEdge)
				{
					reportE(DEBUG,
						"		 On block " << block->label()
						<< " from a backedge, must insert a "
						"inverted dest/source phi");
					
					DataflowGraph::PhiInstruction newPhi;
					newPhi.s.push_back(dest);
					newPhi.d = source;
					reportE(DEBUG, "			" << newPhi);
					block->phis().push_back(newPhi);
					continue;
				}
				
				reportE(DEBUG, "		 Must check all instructions "
					"sources on block " << block->label());
					
				auto ins    = block->instructions().begin();
				auto endIns = block->instructions().end();

				for(; ins != endIns; ins++)
				{
					reportE(DEBUG, "			Instruction before:" << *ins);
					auto isource    = ins->s.begin();
					auto isourceEnd = ins->s.end();
					
					for(; isource != isourceEnd; isource++)
					{
						if(*isource->pointer == source.id)
						{
							*isource->pointer = dest.id;
						}
					}
					reportE(DEBUG, "			Instruction after :" << *ins);
				}
			}
		}
	}
}

}

#endif

