/*! \file   LoopAnalysis.h
	\date   Thursday April 19, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the LoopAnalysis class.
*/

#pragma once

// Ocelot Includes
#include <ocelot/analysis/interface/Analysis.h>

#include <ocelot/ir/interface/ControlFlowGraph.h>

// Forward Declarations
namespace analysis { class DominatorTree; }

namespace analysis
{

/*! \brief A class for identifying and classifying loops in a CFG. */
class LoopAnalysis : public KernelAnalysis
{
public:
	/*! \brief The IR representation of a loop */
	class Loop
	{
	public:
		typedef ir::ControlFlowGraph CFG;
		typedef CFG::BlockPointerVector BlockPointerVector;
		
		typedef CFG::pointer_iterator       block_pointer_iterator;
		typedef CFG::const_pointer_iterator const_block_pointer_iterator;
		
		typedef CFG::iterator       block_iterator;
		typedef CFG::const_iterator const_block_iterator;
		
		typedef std::vector<Loop*> LoopVector;
	
		typedef LoopVector::iterator       iterator;
		typedef LoopVector::const_iterator const_iterator;
	
	public:
		Loop();

	public:
		/*! \brief Get the loop header block */
		block_iterator getHeader() const;
		
	public:
		/*! \brief Get the parent loop, or 0 if this is the top level */
		Loop* getParentLoop() const;
	
	public:
		// Iterator interface for contained loops
		iterator begin();
		iterator end();
	
		const_iterator begin() const;
		const_iterator end()   const;
	
	public:
		size_t size()  const;
		bool   empty() const;

	public:
		/*! \brief The nesting level of this loop, starting from 0 */
		unsigned int getLoopDepth() const;
		
	public:
		// Iterator interface for contained loops
		block_pointer_iterator block_begin();
		block_pointer_iterator block_end();
		
		const_block_pointer_iterator block_begin() const;
		const_block_pointer_iterator block_end()   const;
		
		/*! \brief Get the total number of basic blocks contained in the loop */
		size_t numberOfBlocks() const;
	
	public:
		/*! \brief Does this loop contain a specific block? */
		bool contains(const_block_iterator) const;
		/*! \brief Does this loop contain another loop? */
		bool contains(const Loop*) const;
	
	public:	
		// Simple loop analysis, any of these may fail for non-canonical loops
		BlockPointerVector getExitBlocks();
		
		/*! \brief If there is a single exit block (outside the loop), get it */
		block_iterator getExitBlock();

		/*! \brief If there is a preheader for this loop, return it.  
		
			A loop has a preheader if there is only one edge to the header
			of the loop from outside of the loop.  
		*/
		block_iterator getLoopPreheader();
		
		/*! \brief If there is a single block that branches back to the header,
					it is the latch.
		 */
		block_iterator getLoopLatch();
		
		/*! \brief If there is a single predecessor to the header, return it
		
			Otherwise return null.
		*/
		block_iterator getLoopPredecessor();
	
	public:
		LoopVector         subLoops;
		Loop*              parent;
		BlockPointerVector blocks;
	};
	
	typedef std::list<Loop> LoopList;
	
	typedef LoopList::iterator       iterator;
	typedef LoopList::const_iterator const_iterator;

	typedef Loop::block_iterator       block_iterator;
	typedef Loop::const_block_iterator const_block_iterator;

	typedef Loop::block_pointer_iterator       block_pointer_iterator;
	typedef Loop::const_block_pointer_iterator const_block_pointer_iterator;
	
	typedef std::unordered_map<const_block_iterator, Loop*> BlockToLoopMap;

public:
	LoopAnalysis();
	
public:
	#ifndef _WIN32
	LoopAnalysis& operator=(const LoopAnalysis&) = delete;
	LoopAnalysis(const LoopAnalysis&) = delete;
	#endif
	
public:
	/*! \brief Run the analysis over a specified kernel */
	void analyze(ir::IRKernel& kernel);

public:
	// Iterator over discovered loops in the program
	iterator begin();
	iterator end();

	const_iterator begin() const;
	const_iterator end()   const;

public:
	bool isContainedInLoop(const_block_iterator block);

public:
	// Get inner-most loop containing a specified block
	Loop* _getLoopAt(const_block_iterator block);

private:
	bool _tryAddingLoop(ir::ControlFlowGraph::iterator bb,
		analysis::DominatorTree* dominatorTree);

	void _moveSiblingLoopInto(Loop* to, Loop* from);
	void _insertLoopInto(Loop* to, Loop* from);


private:
	LoopList       _loops;
	BlockToLoopMap _blockToLoopMap;
	ir::IRKernel*  _kernel;

};

}

