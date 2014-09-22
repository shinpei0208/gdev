/*! \file DivergenceAnalysis.h
	\date Jun 1, 2010
	\author Diogo Sampaio <dnsampaio@gmail.com>
	\brief The header file for the DivergenceAnalysis class
*/

#ifndef DIVERGINGENCEANALYSIS_H_
#define DIVERGINGENCEANALYSIS_H_

#include <ocelot/ir/interface/Module.h>
#include <ocelot/analysis/interface/BranchInfo.h>
#include <ocelot/transforms/interface/Pass.h>

namespace analysis 
{
/*!\brief DivergenceAnalysis implements divergence analysis. The divergence
   analysis goes over the program dataflow graph and finds all the variables
   that will always hold the same values for every thread.
   
   
   "This algorithm is described in the article 'Divergence Analysis', to appear
   in the ACM Transactions on Programming Languages and Systems (TOPLAS) 2014."
   
 */
class DivergenceAnalysis : public KernelAnalysis
{
public:
	typedef std::set<BranchInfo>              branch_set;
	typedef std::unordered_set<DataflowGraph::iterator> block_set;
	typedef DataflowGraph::InstructionVector::const_iterator
		const_instruction_iterator;

public:
	DivergenceAnalysis();
	
	virtual void analyze( ir::IRKernel& k );

	/*!\brief Tests if a block ends with a divergent branch instruction */
	bool isDivBlock(const DataflowGraph::const_iterator &block) const;
	/*!\brief Tests if a block ends with a divergent branch instruction */
	bool isDivBlock(const DataflowGraph::iterator &block) const;

	/*!\brief Tests if all threads enter the block in a convergent state */
	bool isEntryDiv(const DataflowGraph::iterator &block) const;
			
	/*!\brief Tests if a branch instruction is divergent */
	bool isDivBranch(const DataflowGraph::Instruction &instruction) const;
	/*!\brief Tests if a instruction uses divergent variables */
	bool isDivInstruction(
		const DataflowGraph::Instruction &instruction) const;
	/*!\brief Tests if a instruction is a possibly divergent branch */
	bool isPossibleDivBranch(const DataflowGraph::Instruction
		&instruction) const;

	const DivergenceGraph& getDivergenceGraph() const;
	const DataflowGraph* getDFG() const;

    /*! If doControlFlowAnalysis is set to false, then we have the
        variance analysis of Straton et al..
        Use it only for experiments.
    */
	void setControlFlowAnalysis(bool doControlFlowAnalysis);
    /*! Mark blocks as convergent even if they depend on a divergent branch
    	as long as doing so does not introduce conflicts.
    */
	void setConditionalConvergence(bool includeConditionalConvergence);
	
protected:
	/*!\brief Performs convergence analysis, identifies blocks that can
		never be divergent */
	void _convergenceAnalysis();
	/*!\brief Make the initial data-flow analysis */
	void _analyzeDataFlow();
	/*!\brief Makes the control-flow analysis, dependent on the results
		of the data-flow analysis */
	void _analyzeControlFlow();
	/*!\brief Updates convergence analysis, using current divergence
		analysis results */
	void _updateConvergenceAnalysis();
	/*!\brief Taints the destination of a phi instr with a predicate */
	void _addPredicate(const DataflowGraph::PhiInstruction &phi,
		const DivergenceGraph::node_type &predicate);
	/*!\brief Removes the dependence */
	void _removePredicate(const DataflowGraph::PhiInstruction &phi,
		const DivergenceGraph::node_type &predicate);
	
	
protected:
	/*! \brief Is an operand a function call operand? */
	bool _isOperandAnArgument( const ir::PTXOperand& operand );
	/*! \brief Does an operand reference local memory? */
	bool _doesOperandUseLocalMemory( const ir::PTXOperand& operand );
	/*! \brief Tests if a block can end with a divergent branch instruction,
		without using control dependence analysis */
	bool _isPossibleDivBlock(const DataflowGraph::iterator &block) const;
	/*! \brief Tests if this block has at most 1 path that does not reach
		the exit without executing another instruction */
	bool _hasTrivialPathToExit(const DataflowGraph::iterator &block) const;

	/*! \brief Get the number of successors with paths to the post dominator
		that do not encounter convergent blocks */
	unsigned int _numberOfDivergentPathsToPostDominator(
		const DataflowGraph::iterator &block) const;
	/*! \brief Gets the set of divergent blocks contained in a block's
		post-dominance frontier */
	block_set _getDivergentBlocksInPostdominanceFrontier(
		const DataflowGraph::iterator &block);
	
protected:
	/*! \brief Get the set of possibly divergent branches */
	void _findBranches(branch_set& branches);
	/*! \brief Add divergence graph edges for control dependences */
	void _propagateDivergenceAlongControlDependences(branch_set& branches);
	/*! \brief Attempt to prove that divergent branches are actually convergent,
		update the divergence graph on success */
	bool _promoteDivergentBranchesToConvergent(branch_set& branches);

protected:
	ir::IRKernel *_kernel;
	/*!\brief Holds the variables marks of divergent blocks */
	DivergenceGraph _divergGraph;
	/*!\brief Set with all not divergent blocks in the kernel*/
	block_set  _notDivergentBlocks;
	bool _doCFGanalysis;
	bool _includeConditionalConvergence;

};

}

#endif /* DIVERGINGENCEANALYSIS_H_ */

