/*! \file   ReadableLayoutPass.cpp
	\author Gregory Diamos <diamos@nvidia.com>
	\date   Wednesday July 11, 2012
	\brief  The source file for the ReadableLayoutPass class.
*/

// Ocelot Includes
#include <ocelot/transforms/interface/ReadableLayoutPass.h>
#include <ocelot/analysis/interface/DominatorTree.h>
#include <ocelot/ir/interface/IRKernel.h>


// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <queue>
#include <unordered_set>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace transforms
{

ReadableLayoutPass::ReadableLayoutPass()
: KernelPass({"DominatorTreeAnalysis"}, "ReadableLayoutPass")
{

}

void ReadableLayoutPass::initialize(const ir::Module& m)
{

}


void ReadableLayoutPass::runOnKernel(ir::IRKernel& k)
{
	blocks.clear();

	typedef ir::ControlFlowGraph::iterator iterator;
	typedef std::list<iterator>            BlockChain;
	typedef std::list<BlockChain>          BlockChainList;
	typedef BlockChainList::iterator       chain_iterator;
	typedef std::unordered_set<iterator>   BlockSet;
	typedef std::queue<iterator>           BlockQueue;
	typedef std::unordered_map<iterator,
		chain_iterator> BlockToChainMap;
	
	// Form chains of basic blocks
	BlockChainList  chains;
	BlockToChainMap blockToChains;
	
	for(auto block = k.cfg()->begin(); block != k.cfg()->end(); ++block)
	{
		auto chain = chains.insert(chains.end(), BlockChain(1, block));
		blockToChains.insert(std::make_pair(block, chain));
	}
	
	bool changed = true;
	
	report("Merging chains of basic blocks");
	while(changed)
	{
		changed = false;
		
		for(auto chain = blockToChains.begin();
			chain != blockToChains.end(); ++chain)
		{
			auto chainTail = chain->second->back();
			
			for(auto successor = chainTail->successors.begin();
				successor != chainTail->successors.end(); ++successor)
			{
				bool isFallthrough = false;
				
				if(chainTail->has_fallthrough_edge())
				{
					isFallthrough =
						chainTail->get_fallthrough_edge()->tail == *successor;
				}
				
				if(!isFallthrough && ((*successor)->predecessors.size() != 1
					|| chainTail->successors.size() != 1))
				{
					continue;
				}
				
				auto successorChain = blockToChains.find(*successor);
				assert(successorChain != blockToChains.end());
				
				// don't merge chains already in the same chain
				if(successorChain->second == chain->second) continue;
				
				auto chainHead = successorChain->second->front();
				
				if(chainHead != *successor) continue;
				
				auto successorChainPointer = successorChain->second;
				
				for(auto chainMember = successorChainPointer->begin();
					chainMember != successorChainPointer->end(); ++chainMember)
				{
					auto member = blockToChains.find(*chainMember);
					assert(member != blockToChains.end());
				
					blockToChains.erase(member);
					
					blockToChains.insert(std::make_pair(*chainMember,
						chain->second));
				}
				
				report(" connecting " << chainTail->label() << " -> "
					<< chainHead->label());
				
				chain->second->splice(chain->second->end(),
					*successorChainPointer);
				
				changed = true;
			}
			
			if(changed) break;
		}
	}
	
	// Topologically schedule the chains
	BlockSet   scheduled;
	BlockQueue readyQueue;
	BlockSet   ready;
	
	ready.insert(k.cfg()->get_entry_block());
	readyQueue.push(*ready.begin());

	report("Scheduling basic blocks.");
	while(!ready.empty())
	{
		auto node = readyQueue.front();
		readyQueue.pop();
	
		auto readySetEntry = ready.find(node);
		assert(readySetEntry != ready.end());
		ready.erase(readySetEntry);

		auto chain = blockToChains.find(node);
		assert(chain != blockToChains.end());
		
		// emit the blocks in the chain, in order
		for(auto block = chain->second->begin();
			block != chain->second->end(); ++block)
		{
			scheduled.insert(*block);
			blocks.push_back(*block);
			report(" scheduled " << (*block)->label());
		}
		
		// free up any dependent blocks
		for(auto block = chain->second->begin();
			block != chain->second->end(); ++block)
		{
			for(auto successor = (*block)->successors.begin();
				successor != (*block)->successors.end(); ++successor)
			{
				if(scheduled.count(*successor) != 0) continue;
				
				// are all dependencies satisfied
				bool allDependenciesSatisfied = true;
				
				for(auto predecessor = (*successor)->predecessors.begin();
					predecessor != (*successor)->predecessors.end();
					++predecessor)
				{
					if(_isCyclicDependency(*predecessor, *successor)) continue;
					
					if(scheduled.count(*predecessor) == 0)
					{
						allDependenciesSatisfied = false;
						break;
					}
				}
				
				if(allDependenciesSatisfied)
				{
					if(ready.insert(*successor).second)
					{
						readyQueue.push(*successor);
						report("  " << (*successor)->label() << " is ready.");
					}
				}
			}
		}
	}
}

void ReadableLayoutPass::finalize()
{

}

bool ReadableLayoutPass::_isCyclicDependency(iterator predecessor,
	iterator successor)
{
	auto domAnalysis = getAnalysis("DominatorTreeAnalysis");
	assert(domAnalysis != 0);
	
	auto dominatorTree = static_cast<analysis::DominatorTree&>(*domAnalysis);
	
	return dominatorTree.dominates(successor, predecessor);	
}

}


