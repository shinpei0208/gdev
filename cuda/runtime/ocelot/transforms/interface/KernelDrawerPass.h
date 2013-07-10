/*! \file KernelDrawer.h
	\author Diogo Sampaio <sampaio@dcc.ufmg.br>
	\brief The header file for the KernelDrawerPass class.
*/
#ifndef DIVERGENCEDRAWER_H_
#define DIVERGENCEDRAWER_H_

// Ocelot Includes
#include <ocelot/transforms/interface/Pass.h>

#include <ocelot/analysis/interface/DominatorTree.h>
#include <ocelot/analysis/interface/PostdominatorTree.h>
#include <ocelot/analysis/interface/AffineAnalysis.h>
#include <ocelot/analysis/interface/DivergenceAnalysis.h>

namespace transforms
{

/* This class draws many graphs for a kernel. */
class KernelDrawerPass: KernelPass
{
public:
	enum TODO
	{
		noGraphs	=	0,
		varsGraph =	1,
		cfgGraph	=	2,
		divGraph	=	4,
		fullGraph =	8,
		results	 = 16,
		all = (unsigned) (-1)
	};

private:
	typedef union variables_s
	{
		struct aff_div_s
		{	 /* Compare variables of affine analysis with divergence analysis
				 * Variables constant for the divergence analysis */
				unsigned long undCst; /* T |	T */
				unsigned long cstCst; /* 0 +	C */
				unsigned long uniCst; /* 0 +	B */
				unsigned long knwCst; /* C +	C */
				unsigned long affCst; /* C +	B */
				unsigned long divCst; /* B +	B */
				/* Variables divergent for the divergence analysis */
				unsigned long undDiv;	/* T |	T */
				unsigned long cstDiv;	/* 0 +	C */
				unsigned long uniDiv;	/* 0 +	B */
				unsigned long knwDiv;	/* C +	C */
				unsigned long affDiv;	/* C +	B */
				unsigned long divDiv;	/* B +	B */
		} aff_div;

		struct aff_s
		{	 /* Affine types */
				unsigned long und;	/* T |	T */
				unsigned long cst;	/* 0 +	C */
				unsigned long uni;	/* 0 +	B */
				unsigned long knw;	/* C +	C */
				unsigned long aff;	/* C +	B */
				unsigned long div;	/* B +	B */
		} aff;

		struct div_s
		{ /* Divergent types */
				unsigned long cst;
				unsigned long div;
		} div;
		
	} variables_t;

	unsigned _todo;
	unsigned _analysis;
	unsigned long instructionsCount;

	string _kernelName;
	string _path;
	stringstream _domStr;
	stringstream _pdomStr;

	variables_t _variables;

	ir::IRKernel *_k;
	analysis::DataflowGraph *_dfg;
	analysis::DivergenceAnalysis *_div;
	analysis::AffineAnalysis *_aff;
	analysis::DominatorTree *_dom;
	analysis::PostdominatorTree *_pdom;

	string _edges(const analysis::DataflowGraph::iterator &block,
			const bool isFullGraph = false) const;
	string _printAffineTransferFunction(const int & id) const;
	string _getBGColor(const unsigned int id) const;
	string _instructionColor(
		const analysis::DataflowGraph::InstructionVector::iterator & ins) const;
	string _blockColor(const analysis::DataflowGraph::iterator & bck) const;
	bool _hasDom(const analysis::DataflowGraph::iterator & bck) const;
	string _domName(const analysis::DataflowGraph::iterator & bck) const;
	bool _hasPdom(const analysis::DataflowGraph::iterator & bck) const;
	string _pdomName(const analysis::DataflowGraph::iterator & bck) const;
	void _printFullGraphHeader(ofstream &out) const;
	bool _branchNotUni(
		const analysis::DataflowGraph::InstructionVector::iterator & ins) const;
	string _blockName(const string & blockLabel) const;

public:
	virtual void initialize(const ir::Module& m);
	virtual void runOnKernel(ir::IRKernel& k);
	virtual void finalize();

	void computeResults();

	void drawVariablesGraph() const;
	void drawDivergenceGraph() const;
	void drawControlFlowGraph() const;
	void drawFullGraph() const;
	void draw() const;
	void printResults() const;
	KernelDrawerPass(const std::string &path, unsigned graphs,
			unsigned analysis);
};

}
#endif /* DIVERGENCEDRAWER_H_ */

