/*! \file DivergenceAnalysis.h
	\date Jun 1, 2010
	\author Diogo Sampaio <dnsampaio@gmail.com>
	\brief The source file for the DivergenceAnalysis class
*/

// Ocelot Includes
#include <ocelot/analysis/interface/DivergenceAnalysis.h>
#include <ocelot/analysis/interface/PostdominatorTree.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <cassert>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0
#define REPORT_PTX  0
#define REPORT_ALL_DEPENDENCES 0

namespace analysis
{

/*! \brief Constructor, already making the analysis of a input kernel */
DivergenceAnalysis::DivergenceAnalysis()
: KernelAnalysis( "DivergenceAnalysis",
	{"DataflowGraphAnalysis",
	"PostDominatorTreeAnalysis"}), _doCFGanalysis(true),
	_includeConditionalConvergence(true)
{
}

/*! \brief Analyze the control and data flow graphs searching for divergent 
 *    variables and blocks
 *
 * 1) Makes data flow analysis that detects divergent variables and blocks 
 *    based on divergent sources, such as t.id, laneId
 * 2) Makes control flow analysis that detects new divergent variables based
 *    on the dependency of variables of variables created on divergent paths
 */
void DivergenceAnalysis::analyze(ir::IRKernel &k)
{
	Analysis* dfgAnalysis = getAnalysis("DataflowGraphAnalysis");
	assert(dfgAnalysis != 0);

	DataflowGraph &dfg = static_cast<DataflowGraph&>(*dfgAnalysis);

	dfg.convertToSSAType(DataflowGraph::Gated);

	assert(dfg.ssa());

	_divergGraph.clear();
	_notDivergentBlocks.clear();
	_kernel = &k;

	report("Running Divergence analysis on kernel '" << k.name << "'")
	#if REPORT_PTX > 0
	k.write(std::cout);
	#endif

	DivergenceGraph::node_set predicates;
		
	/* 1) Makes data flow analysis that detects divergent variables and blocks
		based on divergent sources, such as t.id, laneId */
	_analyzeDataFlow();
	/* 2) Makes control flow analysis that detects new divergent variables
		based on the dependency of variables of variables created on divergent
		paths */
	_analyzeControlFlow();
}

/*! \brief Tests if a block ends with a divergent branch
	instruction (isDivBranchInstr) */
bool DivergenceAnalysis::isDivBlock(
	const DataflowGraph::const_iterator &block) const
{
	if (block->instructions().size() == 0) {
		return false;
	}
	return isDivBranch(*--block->instructions().end());
}

/*! \brief Tests if a block ends with a divergent branch
	instruction (isDivBranchInstr) */
bool DivergenceAnalysis::isDivBlock(const DataflowGraph::iterator &block) const
{
	if (block->instructions().size() == 0) {
		return false;
	}
	return isDivBranch(*--block->instructions().end());
}

/*!\brief Tests if all threads enter the block in a convergent state */
bool DivergenceAnalysis::isEntryDiv(
	const DataflowGraph::iterator &block ) const {
	return _notDivergentBlocks.count(block) == 0;
}
		

/*! \brief Tests if the a instruction is a branch instruction
	instruction and is possible a divergent instruction (isDivInstruction) */
bool DivergenceAnalysis::isDivBranch(
	const DataflowGraph::Instruction& instruction) const
{
	return (isDivInstruction(instruction) && isPossibleDivBranch(instruction));
}

/*!\brief Tests if a instruction is a branch instruction
	with possibility of divergence */
bool DivergenceAnalysis::isPossibleDivBranch(
	const DataflowGraph::Instruction& instruction) const
{
	if(typeid(ir::PTXInstruction) == typeid(*(instruction.i))) {
		const ir::PTXInstruction &ptxI =
			*(static_cast<ir::PTXInstruction *> (instruction.i));

		// Must be a branch to be a potentially divergent branch
		if (ptxI.opcode != ir::PTXInstruction::Bra) return false;

		// Uniform branches may not be divergent
		if (ptxI.uni) return false;
		
		// Branches with only a single target cannot be divergent
		if (instruction.block->successors().size() < 2) return false;
		
		// Branches must have at least 2 non-convergent paths to be divergent
		if (_numberOfDivergentPathsToPostDominator(instruction.block) < 2) {
			return false;
		}
		
		return true;
	}
	
	return false;
}

/*! \brief Tests if any of the registers of a instruction
	is possible divergent */
bool DivergenceAnalysis::isDivInstruction(
	const DataflowGraph::Instruction &instruction) const
{
	bool isDivergent = false;
	DataflowGraph::RegisterPointerVector::const_iterator
		reg = instruction.d.begin();
	DataflowGraph::RegisterPointerVector::const_iterator
		endReg = instruction.d.end();

	for (; (!isDivergent) && (reg != endReg); reg++) {
		isDivergent |= _divergGraph.isDivNode(*reg->pointer);
	}

	reg = instruction.s.begin();
	endReg = instruction.s.end();

	for (; (!isDivergent) && (reg != endReg); reg++) {
		isDivergent |= _divergGraph.isDivNode(*reg->pointer);
	}

	return isDivergent;
}

const DivergenceGraph& DivergenceAnalysis::getDivergenceGraph() const
{
	return _divergGraph;
}

const DataflowGraph* DivergenceAnalysis::getDFG() const
{
	const Analysis* dfgAnalysis = getAnalysis("DataflowGraphAnalysis");
	assert(dfgAnalysis != 0);

	return static_cast<const DataflowGraph*>(dfgAnalysis);
}

void DivergenceAnalysis::setControlFlowAnalysis(bool doControlFlowAnalysis)
{
	_doCFGanalysis = doControlFlowAnalysis;
}

void DivergenceAnalysis::setConditionalConvergence(bool doIt)
{
	_includeConditionalConvergence = doIt;
}

void DivergenceAnalysis::_convergenceAnalysis()
{
	report("Running convergence analysis.");

	Analysis* dfgAnalysis = getAnalysis("DataflowGraphAnalysis");
	assert(dfgAnalysis != 0);

	DataflowGraph &dfg = static_cast<DataflowGraph&>(*dfgAnalysis);
	
	/* Assume all blocks are convergent */
	block_set divergentBlocks;
	
	/* 1) mark all blocks in the post-dominance frontier of divergent branches
		along paths that do not encounter known convergent points
		as divergent.  This is an optimistic analysis. */
	report(" Marking divergent blocks.");
	for (auto block = dfg.begin() ; block != dfg.end(); ++block) {
		if (!isDivBlock(block) || _hasTrivialPathToExit(block)) continue;
		
		block_set divergentBlocksInPostdominanceFrontier =
			_getDivergentBlocksInPostdominanceFrontier(block);
		
		divergentBlocks.insert(divergentBlocksInPostdominanceFrontier.begin(),
			divergentBlocksInPostdominanceFrontier.end());
	}
	
	report(" Marking convergent blocks.");
	_notDivergentBlocks.clear();
	for (auto block = dfg.begin(); block != dfg.end(); ++block) {
		if (divergentBlocks.count(block) == 0) {
			report("  " << block->label() << " is assumed convergent.");
			_notDivergentBlocks.insert(block);
		}
	}
}

static bool isDivergenceSource(const ir::PTXOperand& operand)
{
	if (operand.addressMode == ir::PTXOperand::Special) {
		if( (operand.special == ir::PTXOperand::tid &&
			(operand.vIndex == ir::PTXOperand::ix))
			|| (operand.special == ir::PTXOperand::laneId)) {
			return true;
		}
	}
	
	return false;
}

void DivergenceAnalysis::_analyzeDataFlow()
{
	Analysis* dfg = getAnalysis("DataflowGraphAnalysis");
	assert(dfg != 0);

	DataflowGraph &nonConstGraph = static_cast<DataflowGraph&>(*dfg);
	DataflowGraph::const_iterator block = nonConstGraph.begin();
	DataflowGraph::const_iterator endBlock = nonConstGraph.end();

	report("Analyzing data flow");

	/* 1) Analyze the data flow adding divergence sources */
	for (; block != endBlock; ++block) {
		report(" for block " << block->label());
		
		DataflowGraph::PhiInstructionVector::const_iterator
			phiInstruction = block->phis().begin();
		DataflowGraph::PhiInstructionVector::const_iterator
			endPhiInstruction = block->phis().end();
        /* Go over the phi functions and add their dependences to the
         * dependence graph. */
		for (; phiInstruction != endPhiInstruction; phiInstruction++) {
			for (DataflowGraph::RegisterVector::const_iterator
				si = phiInstruction->s.begin();
				si != phiInstruction->s.end(); ++si) {
				_divergGraph.insertEdge(si->id, phiInstruction->d.id);
				report("  phi r" << phiInstruction->d.id << " <- r" << si->id);
			}
		}

		DataflowGraph::InstructionVector::const_iterator
			ii = block->instructions().begin();
		DataflowGraph::InstructionVector::const_iterator
			iiEnd = block->instructions().end();
		for (; ii != iiEnd; ++ii) {

			ir::PTXInstruction *ptxInstruction = NULL;
			bool atom = false;
			bool functionStackArgument = false;
			bool localMemoryOperand = false;
			bool isCall = false;

			std::set<const ir::PTXOperand*> divergenceSources;

			/* First we populate divergenceSources with all the
			 * source operands that might diverge.
			 */
			if (typeid(ir::PTXInstruction) == typeid(*(ii->i))) {
				ptxInstruction = static_cast<ir::PTXInstruction*> (ii->i);
				if (isDivergenceSource(ptxInstruction->a)) {
					divergenceSources.insert(&ptxInstruction->a);
				}
				if (isDivergenceSource(ptxInstruction->b)) {
					divergenceSources.insert(&ptxInstruction->b);
				}
				if (isDivergenceSource(ptxInstruction->c)) {
					divergenceSources.insert(&ptxInstruction->c);
				}

				if (ptxInstruction->opcode == ir::PTXInstruction::Atom){
					atom = true;
				}
				
				if (ptxInstruction->mayHaveAddressableOperand()) {
					if (_doesOperandUseLocalMemory(ptxInstruction->a)) {
						localMemoryOperand = true;
					}
				}
				
				if (ptxInstruction->opcode == ir::PTXInstruction::Call){
					isCall = true;
				}
			}

			/* Second, if this is a function call, we populate divergenceSources
			 * with all the source operands that might diverge in a call.
			 */
			if (_kernel->function()) {
				if (typeid(ir::PTXInstruction) == typeid(*(ii->i))) {
					ptxInstruction = static_cast<ir::PTXInstruction*> (ii->i);
				
					if (ptxInstruction->mayHaveAddressableOperand()) {
						if (_isOperandAnArgument(ptxInstruction->a)) {
							functionStackArgument = true;
							report("  operand '" << ptxInstruction->a.toString()
								<< "' is a function call argument.");
						}
					}
				}
			}
						
			/* Third, we link the source operands to the
			 * destination operands, and check if the destination
			 * can diverge. This will only happen in case the
			 * instruction is atomic. */
			DataflowGraph::RegisterPointerVector::const_iterator
				destinationReg = ii->d.begin();
			DataflowGraph::RegisterPointerVector::const_iterator
				destinationEndReg = ii->d.end();

			for (; destinationReg != destinationEndReg; destinationReg++) {
				if (divergenceSources.size() != 0) {
					std::set<const ir::PTXOperand*>::iterator
						divergenceSource = divergenceSources.begin();
					std::set<const ir::PTXOperand*>::iterator
						divergenceSourceEnd = divergenceSources.end();

					for (; divergenceSource != divergenceSourceEnd;
						divergenceSource++) {
						report("  destination register r"
							<< *destinationReg->pointer
							<< " is derived from a divergence source r"
							<< *divergenceSource);
						_divergGraph.insertEdge(*divergenceSource,
							*destinationReg->pointer);
					}
				}

				DataflowGraph::RegisterPointerVector::const_iterator
					sourceReg = ii->s.begin();
				DataflowGraph::RegisterPointerVector::const_iterator
					sourceRegEnd = ii->s.end();

				for (; sourceReg != sourceRegEnd; sourceReg++) {
					_divergGraph.insertEdge(*sourceReg->pointer,
						*destinationReg->pointer);
					reportE(REPORT_ALL_DEPENDENCES,
						"  r" << *destinationReg->pointer
						<< " <- r" << *sourceReg->pointer);
				}

				if (atom || functionStackArgument ||
					localMemoryOperand || isCall) {
					
					report("  destination register r"
						<< *destinationReg->pointer
						<< " is a divergence source.");
					_divergGraph.insertNode(*destinationReg->pointer);
					_divergGraph.setAsDiv(*destinationReg->pointer);
				}
			}
		}
	}
	/* 2) Computes the divergence propagation */
	_divergGraph.computeDivergence();
}

/*! \brief  Does control flow analysis to detect new divergent variables.
 * 1) Obtain information of all possible divergent branches on the kernel
 * 2) Obtain all branch instructions that depend on a divergent predicate
 * 3) For each divergent branch
 * 3.1) Compute the controlflow dependency
 * 3.2) Search the postdominator block for new divergent variables
 * 3.3) If new divergent variables were found
 * 3.3.1) Re-compute the divergence spread by the new divergence/dataflow graph
 * 3.3.2) Search for new divergent branch instructions
 */
void DivergenceAnalysis::_analyzeControlFlow()
{
	/* Set of possible diverging branches */
	report(" Analyzing control flow");
	
	std::set<BranchInfo> branches;

	/*! 1) Find the set of all possibly divergent branches in the kernel */
	_findBranches(branches);

	/* 2) i)   Propagate divergence along control dependence edges.
	      ii)  Perform convergence analysis, these may prove
	           that some divergent branches are actually convergent.
	      iii) If any divergent branches were removed, update their influence.
	*/
	_propagateDivergenceAlongControlDependences(branches);
	
	bool changed = true;
	
	while (changed) {
		_convergenceAnalysis();
		changed = _promoteDivergentBranchesToConvergent(branches);
	}
}

void DivergenceAnalysis::_addPredicate(const DataflowGraph::PhiInstruction &phi,
    const DivergenceGraph::node_type &predicate)
{
	_divergGraph.insertEdge(predicate, phi.d.id);
}

void DivergenceAnalysis::_removePredicate(
	const DataflowGraph::PhiInstruction &phi,
    const DivergenceGraph::node_type &predicate)
{
	_divergGraph.eraseEdge(predicate, phi.d.id, true);
}

bool DivergenceAnalysis::_isOperandAnArgument( const ir::PTXOperand& operand ) {
	if (operand.addressMode != ir::PTXOperand::Address) {
		return false;
	}
	
	return _kernel->getParameter(operand.identifier) != 0;
}

bool DivergenceAnalysis::_doesOperandUseLocalMemory(
	const ir::PTXOperand& operand ) {
	
	if (operand.addressMode != ir::PTXOperand::Address) {
		return false;
	}
	
	ir::Kernel::LocalMap::const_iterator local = _kernel->locals.find(
		operand.identifier);
	
	if (_kernel->locals.end() != local) {	
		if (local->second.space == ir::PTXInstruction::Local) {
			return true;
		}
	}
	
	const ir::Global* global = _kernel->module->getGlobal(operand.identifier);
	
	if (global == 0) {
		return false;
	}
	
	return global->space() == ir::PTXInstruction::Local;
}

bool DivergenceAnalysis::_isPossibleDivBlock(
	const DataflowGraph::iterator &block) const {
	if (block->instructions().size() == 0) {
		return false;
	}

	return isPossibleDivBranch(*--block->instructions().end());
}

static bool hasBarrier(const DataflowGraph::iterator &block) {
	
	for (auto instruction = block->instructions().begin();
		instruction != block->instructions().end(); ++instruction) {

		if (typeid(ir::PTXInstruction) == typeid(*(instruction->i))) {
			auto ptxInstruction =
				static_cast<ir::PTXInstruction*>(instruction->i);
		
			if (ptxInstruction->opcode == ir::PTXInstruction::Bar) return true;
		
			// texture instruction intrinsics
			if (ptxInstruction->isCall()) {
				if (ptxInstruction->a.addressMode !=
					ir::PTXOperand::FunctionName) {
					 continue;
				}
				
				if (ptxInstruction->a.identifier.find(
					"_Z_intrinsic_pseudo_tex") != 0) {
					 continue;
				}
				
				return true;
			}
		}
	}
	
	return false;
}

static bool buildDivergentSubgraph(
	DivergenceAnalysis::block_set& graph,
	const DataflowGraph::iterator &block,
	const DataflowGraph::iterator &postDominator) {

	bool hitPostDominator = false;

	// don't include blocks with barriers
	if(hasBarrier(block)) return false;
	
	// skip loops
	if(!graph.insert(block).second) return false;
	
	for (auto successor = block->successors().begin();
		successor != block->successors().end(); ++successor) {
		
		// stop at the post dominator
		if (*successor == postDominator)
		{
			hitPostDominator = true;

			continue;
		}
		
		hitPostDominator |= buildDivergentSubgraph(graph,
			*successor, postDominator);
	}
	
	return hitPostDominator;
}

static bool doAnyDivergentPathsReachThePostDominator(
	DivergenceAnalysis::block_set& graph,
	const DataflowGraph::iterator &root,
	const DataflowGraph::iterator &postDominator) {
	
	// build the graph
	return buildDivergentSubgraph(graph, root, postDominator);
}

unsigned int DivergenceAnalysis::_numberOfDivergentPathsToPostDominator(
	const DataflowGraph::iterator &block) const {
	
	const Analysis* dfgAnalysis = getAnalysis("DataflowGraphAnalysis");
	assert(dfgAnalysis != 0);

	const DataflowGraph &cdfg =
		static_cast<const DataflowGraph&>(*dfgAnalysis);
	DataflowGraph &dfg = const_cast<DataflowGraph&>(cdfg);
	
	PostdominatorTree* dtree = (PostdominatorTree*)
		(getAnalysis("PostDominatorTreeAnalysis"));
	
	auto postDominator = dfg.getCFGtoDFGMap()[
		dtree->getPostDominator(block->block())];

	unsigned int divergentPaths = 0;

	for (auto successor = block->successors().begin();
		successor != block->successors().end(); ++successor) {
		if (*successor == postDominator) {
			++divergentPaths;
			continue;
		}
		
		block_set allDivergentPaths;
		
		if (doAnyDivergentPathsReachThePostDominator(allDivergentPaths,
			*successor, postDominator)) {
			++divergentPaths;
		}
	}
	
	report("  There are " << divergentPaths << " divergent paths from "
		<< block->label() << " to post-dominator " << postDominator->label());

	return divergentPaths;
}
		
DivergenceAnalysis::block_set
	DivergenceAnalysis::_getDivergentBlocksInPostdominanceFrontier(
	const DataflowGraph::iterator &block) {
	
	const Analysis* dfgAnalysis = getAnalysis("DataflowGraphAnalysis");
	assert(dfgAnalysis != 0);

	const DataflowGraph &cdfg =
		static_cast<const DataflowGraph&>(*dfgAnalysis);
	DataflowGraph &dfg = const_cast<DataflowGraph&>(cdfg);
	
	PostdominatorTree* dtree = (PostdominatorTree*)
		(getAnalysis("PostDominatorTreeAnalysis"));
	
	auto postDominator = dfg.getCFGtoDFGMap()[
		dtree->getPostDominator(block->block())];

	block_set divergentBlocks;

	for (auto successor = block->successors().begin();
		successor != block->successors().end(); ++successor) {
		if (*successor == postDominator) continue;
		
		block_set allDivergentPaths;
		
		buildDivergentSubgraph(allDivergentPaths, *successor, postDominator);
		
		divergentBlocks.insert(allDivergentPaths.begin(),
			allDivergentPaths.end());
	}
	
	return divergentBlocks;
}

bool DivergenceAnalysis::_hasTrivialPathToExit(
	const DataflowGraph::iterator &block) const {

	// We can ignore divergent threads that immediately exit
	unsigned int exitingPaths = 0;
	
	const Analysis* dfgAnalysis = getAnalysis("DataflowGraphAnalysis");
	assert(dfgAnalysis != 0);

	const DataflowGraph &dfg =
		static_cast<const DataflowGraph&>(*dfgAnalysis);

	auto exit = --dfg.end();

	for (auto successor = block->successors().begin();
		successor != block->successors().end(); ++successor) {
		auto path = *successor;
		
		while (true) {
			if (path == exit) {
				++exitingPaths;
				break;
			}
			if (path->successors().size() != 1) {
				break;
			}
			if (!path->instructions().empty()) {
				if (path->instructions().size() == 1) {
					const ir::PTXInstruction &ptxI =
						*(static_cast<ir::PTXInstruction *> (
						path->instructions().back().i));
				
					if (ptxI.isExit()) {
						++exitingPaths;
					}
				}
				break;
			}
			path = *path->successors().begin();
		}
	}

	if (block->successors().size() - exitingPaths < 2) {
		return true;
	}
	
	return false;
}

void DivergenceAnalysis::_findBranches(branch_set& branches)
{
	Analysis* dfgAnalysis = getAnalysis("DataflowGraphAnalysis");
	assert(dfgAnalysis != 0);

	DataflowGraph &dfg = static_cast<DataflowGraph&>(*dfgAnalysis);

	/* Create a list of branches that can be divergent, that is,
		they are not  bra.uni and have a predicate */
	DataflowGraph::iterator block = dfg.begin();
	DataflowGraph::iterator endBlock = dfg.end();

	/* Post-dominator tree */
	PostdominatorTree *dtree;
	dtree = (PostdominatorTree*) (getAnalysis("PostDominatorTreeAnalysis"));

	report(" Finding branches");
	for (; block != endBlock; ++block) {
		ir::PTXInstruction *ptxInstruction = NULL;

		if (block->instructions().size() > 0) {
			/* Branch instructions can only be the last
			instruction of a basic block */
			DataflowGraph::Instruction& lastInstruction =
			*(--block->instructions().end());

			if (typeid(ir::PTXInstruction) == typeid(*(lastInstruction.i))) {
				ptxInstruction =
					static_cast<ir::PTXInstruction*>(lastInstruction.i);

				if ((ptxInstruction->opcode == ir::PTXInstruction::Bra)) {
					report("  examining " << ptxInstruction->toString());
					
					if(ptxInstruction->uni == true) { 
						report("   eliminated, uniform...");
						continue;
					}
					
					if(lastInstruction.s.size() == 0) {
						report("   eliminated, wrong source count ("
							<< lastInstruction.s.size() << ")...");
						continue;
					}
					
					assert(lastInstruction.s.size() == 1);
					DataflowGraph::iterator postDomBlock =
						dfg.getCFGtoDFGMap()[
							dtree->getPostDominator(block->block())];
					if (postDomBlock != dfg.end()) {
						BranchInfo newBranch(&(*block), &(*postDomBlock), 
							lastInstruction, _divergGraph);
						branches.insert(newBranch);
						report("   is potentially divergent...");
					}
					else {
						report("   eliminated, no post-dominator...");
					}
				}
			}
		}
	}
}

void DivergenceAnalysis::_propagateDivergenceAlongControlDependences(
	branch_set& branches)
{
	typedef std::unordered_set<const DataflowGraph::Block*> BlockSet;
	
	/* 2) Obtain all branch instructions that depend on a divergent predicate
	* List of branches that are divergent, so their controlflow
	influence must be tested */
	std::set<BranchInfo> worklist;

	/* Populate the divergent branches set */
	std::set<BranchInfo>::iterator branch    = branches.begin();
	std::set<BranchInfo>::iterator endBranch = branches.end();

	report(" Finding divergent branches");
	while (branch != endBranch) {
		if (isDivBranch(branch->instruction()) ){
			report("   found " << branch->instruction().i->toString());
			std::set<BranchInfo>::iterator divBranch = branch--;
			worklist.insert(*divBranch);
			branches.erase(divBranch);
			endBranch = branches.end();
			branch = branches.begin();
			continue;
		} else {
			report("   " << branch->instruction().i->toString()
				<< " is not divergent.");
		}
		branch++;
	}
	
	/*  3) For each divergent branch
	 * Test for divergence on the post-dominator block of every
	 	divergent branch instruction */
	report(" Propagating divergence along control dependences");
	while (worklist.size() > 0) {
		BranchInfo branchInfo = *worklist.begin();
		report("  for branch " << branchInfo.instruction().i->toString());
		/* 3.1) Compute the controlflow dependency. populate is O(E) */
		branchInfo.populate();

		/* 3.2) Search the postdominator blocks for
			new divergent variables */
		bool newDivergences = false;
		const DataflowGraph::PhiInstructionVector&
				phis = branchInfo.postDominator()->phis();
		DataflowGraph::PhiInstructionVector::const_iterator
			phi = phis.begin();
		DataflowGraph::PhiInstructionVector::const_iterator
			endphi = phis.end();
	
		report("   checking block " <<
			branchInfo.postDominator()->label()
			<< " in the post-dominator frontier");
	
		for (; phi != endphi; phi++) {
			DataflowGraph::RegisterVector::const_iterator
				source = phi->s.begin();
			DataflowGraph::RegisterVector::const_iterator
				endSource = phi->s.end();

			for (; source != endSource; source++) {
				if (branchInfo.isTainted(source->id)) {
					report("    adding dependence r" << phi->d.id
						<< " <- r" << branchInfo.predicate());
					_addPredicate(*phi, branchInfo.predicate());
					newDivergences = true;
				}
			}
		}	
		
		worklist.erase(branchInfo);
		/* 3.3) If new divergent variables were found*/
		if (newDivergences) {
			/* 3.3.1) Re-compute the divergence spread by the
				new divergence/dataflow graph */
			_divergGraph.computeDivergence();
			branch = branches.begin();
			/* 3.3.2) Search for new divergent branch instructions */
			while (branch != endBranch) {
				if (isDivBranch(branch->instruction()) ){
					std::set<BranchInfo>::iterator divBranch = branch--;
					worklist.insert(*divBranch);
					branches.erase(divBranch);
					endBranch = branches.end();
					branch = branches.begin();
					continue;
				}
				branch++;
			}
		}
	}
}

bool DivergenceAnalysis::_promoteDivergentBranchesToConvergent(
	branch_set& branches)
{
	bool removedDivergences = false;
	
	// Remove non-divergent branches from the set
	for (auto branch = branches.begin(); branch != branches.end(); ) {
		if (!isDivBranch(branch->instruction())) {
			
			report("  promoting " << branch->instruction().i->toString()
				<< " to convergent.");
			
			const DataflowGraph::PhiInstructionVector&
				phis = branch->postDominator()->phis();

			for (auto phi = phis.begin(); phi != phis.end(); ++phi) {
				DataflowGraph::RegisterVector::const_iterator
					source = phi->s.begin();
				DataflowGraph::RegisterVector::const_iterator
					endSource = phi->s.end();

				for (; source != endSource; source++) {
					if (branch->isTainted(source->id)) {
						report("   removing dependence r" << phi->d.id
							<< " <- r" << branch->predicate());
						_removePredicate(*phi, branch->predicate());
						
					}
				}
			}
			
			_divergGraph.forceConvergent(branch->predicate());
			removedDivergences = true;
			
			auto temp = branch++;
			
			branches.erase(temp);
		}
		else {
			++branch;
		}
	}
	
	if (removedDivergences) {
		_divergGraph.computeDivergence();
	}
	
	return removedDivergences;
}

}


