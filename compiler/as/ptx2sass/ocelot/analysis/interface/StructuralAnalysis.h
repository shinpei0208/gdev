/*! \file	 StructuralAnalysis.h
	\author Haicheng Wu <hwu36@gatech.edu>
	\date	 Monday April 4, 2011
	\brief	The header file for the StructuralAnalysis pass.
*/

//===----------------------------------------------------------------------===// 
// 
// This file defines the class of Structural Analysis which will return the 
// control tree and unstructured branches of a function 
// 
//===----------------------------------------------------------------------===// 
 
#ifndef ANALYSIS_STRUCTURALANALYSIS_H 
#define ANALYSIS_STRUCTURALANALYSIS_H

// Ocelot Includes
#include <ocelot/transforms/interface/Pass.h>
#include <ocelot/ir/interface/ControlFlowGraph.h>
#include <ocelot/ir/interface/PTXKernel.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <set>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>

namespace ir {
	class PTXKernel;
}

namespace analysis {

// StructuralAnalysis - This class holds all the methods and data structures 
class StructuralAnalysis : public KernelAnalysis {
public:
	typedef enum {TREE, FORWARD, BACK, CROSS} EdgeClass;
	typedef std::pair<ir::ControlFlowGraph::iterator,
		ir::ControlFlowGraph::iterator> EdgeLLVMTy;
	typedef std::vector<EdgeLLVMTy> EdgeVecTy;
	typedef std::unordered_set<ir::ControlFlowGraph::iterator> BBSetTy;
	typedef std::vector<ir::ControlFlowGraph::iterator> BBVecTy;

	// Types defined in Fig 7.38 of Muchnick book
	typedef enum {
		Nil,
		Block,
		IfThen,
		IfThenElse,
		Case,
		SelfLoop,
		NaturalLoop,
		Improper,
		Unreachable
	} RegionTy;

	// NodeTy - This type is used for the CFG node
	typedef struct Node {
		bool isCombined; // Whether it is an original or combined from original 
		// Map to the corresponding ir::BasicBlock * if it is original
		ir::ControlFlowGraph::iterator BB;	
		std::set<struct Node *> predNode; // Predecessor of the node
		std::set<struct Node *> succNode; // Successor of the node
		// If isCombined is true,it points to the entry of the nodeset
		struct Node *entryNode; 
		struct Node *parentNode; // Parent Node in Control Tree
		std::set<struct Node *> childNode; // Child Nodes in Control Tree
		EdgeVecTy outgoingBR;	// Outgoing Branches of a loop
		EdgeVecTy incomingBR;	// Incoming Branches of a loop
		ir::ControlFlowGraph::iterator entryBB;	 // The entry Basic Block
		ir::ControlFlowGraph::iterator exitBB;		// The exit Basic Block
		BBSetTy containedBB; // ir::BasicBlock * s contained in this node
		EdgeVecTy incomingForwardBR; // The shared code of unstructured branch 
		RegionTy nodeType;				// The type of the node
		bool isLoopHeader;
		bool isBackEdge;
		struct Node *loopExitNode;
		
		~Node();
	} NodeTy;

	// NodeSetTy - used to holds nodes in a set 
	typedef std::set<NodeTy *> NodeSetTy;
	typedef std::unordered_map<
		ir::ControlFlowGraph::iterator, NodeTy *> BB2NodeMapTy;
	typedef std::map<NodeTy *, NodeTy *> Node2NodeMapTy;
	typedef std::map<NodeTy *, RegionTy> Node2RegionTyMapTy;
	typedef std::map<NodeTy *, NodeSetTy *> Node2NodeSetMapTy;
	typedef std::pair<NodeTy *, NodeTy *> EdgeTy;
	typedef std::map<EdgeTy, EdgeClass> Edge2ClassMapTy;
	typedef std::set<EdgeTy> EdgeSetTy;
	typedef std::map<NodeTy *, bool> VisitMapTy;

public:
	~StructuralAnalysis();
	void analyze(ir::IRKernel& k);

	// Get a text representation of the analysis
	void write(std::ostream& stream) const;

public:
	NodeSetTy Net;

	// unstructuredBRVec - store the detected unstructured branches
	EdgeVecTy unstructuredBRVec;

	// BB2NodeMap - This var is used to find the Node from ir::BasicBlock * ;
	BB2NodeMapTy BB2NodeMap;

	NodeSetTy unreachableNodeSet;
 
private:
	ir::PTXKernel* _kernel;

	// postorder traversal of the flowgraph
	int postCtr, postMax, preMax;
	std::map<int, NodeTy *> post;
	VisitMapTy visit, visitPath;
	std::map<NodeTy *, int> preTree, postTree;

	// domin - Set of dominators of each node
	Node2NodeSetMapTy domin;

	// edge2ClassMap - map the edge to its class
	Edge2ClassMapTy edge2ClassMap;

	// buildSimpleCFG - Build a Simple CFG out of the LLVM CFG
	void buildSimpleCFG(NodeSetTy &N);

	// structuralAnalysis - Follow Fig 7.39 of Muchnick book
	void structuralAnalysis(NodeSetTy &N, NodeTy *entry, bool debug);

	// DFSPostorder - Follow Fig 7.40 of Muchnick book 
	void DFSPostorder(NodeSetTy &N, NodeTy *x);

	// acyclicRegionType - Follow Fig 7.41 of Muchnick book
	RegionTy acyclicRegionType(NodeSetTy &N, NodeTy *node,
		NodeSetTy &nset, NodeTy **entryNode, NodeTy **exitNode, NodeTy *entry);

	// cyclicRegionType - Follow Fig 7.42 of Muchnick book
	RegionTy cyclicRegionType(NodeSetTy &N, NodeSetTy &nset,
		NodeTy *loopHeaderNode, NodeTy *backEdgeNode, NodeTy **exitNode,
		NodeTy *entry);

	// reduce - Follow Fig 7.43 of Muchnick book
	NodeTy * reduce(NodeSetTy &N, RegionTy rType, NodeSetTy &nodeSet,
		NodeTy *entryNode, NodeTy *exitNode);

	// replace - Follow Fig 7.44 of Muchnick book
	void replace(NodeSetTy &N, NodeTy *node, NodeSetTy &nodeSet);

	// isImproper - Follow Fig 7.45 of Muchnick book
	bool isImproper(NodeSetTy &N, NodeSetTy &nset, NodeTy *loopHeaderNode,
		NodeTy *backEdgeNode, NodeTy **exitNode, NodeTy *entry);

	// pathBack - Check if there is a node k such that there is a path from
	// m to k that does not pass through n and an edge k->n that is a back edge
	NodeTy* pathBack(NodeTy *n, NodeSetTy &N, NodeSetTy &reachUnder);

	// isCaseWithDefault - Check if node leads a case block
	bool isCaseWithDefault(NodeSetTy &N, NodeTy * entryNode,
		NodeTy **exitNode, NodeTy *entry);

	// isCaseWithoutDefault - Check if node leads a case block
	bool isCaseWithoutDefault(NodeSetTy &N, NodeTy * entryNode,
		NodeTy **exitNode, NodeTy *entry);

	// isImproperCaseWithDefault - Check if node leads 
	// a case block with incoming edges
	bool isImproperCaseWithDefault(NodeSetTy &N,
		NodeTy * entryNode, NodeTy *entry);

	// isImproperCaseoutWithDefault - Check if node leads
	//	a case block with incoming edges
	bool isImproperCaseWithoutDefault(NodeSetTy &N, NodeTy * entryNode,
		NodeTy **exitNode, NodeTy *entry);

	// path(n, m, I) - Return true if there is a path from from n to m 
	//such that all the nodes in it are in I and false otherwise
	bool path(NodeTy *n, NodeTy *m, NodeSetTy &I, NodeTy *esc);

	// path(n, m, I, src, dst ) - Return true if there is a path from from
	// n to m such that all the nodes in it are in I without going through edge
	// src->dst and false otherwise
	bool path(NodeTy *n, NodeTy *m, NodeSetTy &I, NodeTy *src, NodeTy *dst);

	// compact - Compact nodes in nset into n;
	void compact(NodeSetTy &N, NodeTy *n, NodeSetTy &nset);

	//mapNode2BB - Return the corresponding ir::BasicBlock * of the node
	ir::ControlFlowGraph::iterator mapNode2BB(NodeTy *node);

	// mapBB2Node - Return the corresponding sturcture node of the basic block
	NodeTy *mapBB2Node(ir::ControlFlowGraph::iterator bb); 

	// dumpCTNode - Dump one Control Node
	void dumpCTNode(std::ostream& stream, NodeTy *n) const;

	// dumpNode - Dump one node
	void dumpNode(std::ostream& stream, NodeTy *node) const;

	// findUnstructuredBR - Record the branch and remove it from CFG 
	void findUnstructuredBR(NodeSetTy &N, NodeTy *srcNode,
		NodeTy *dstNode, bool needForwardCopy, bool isGoto);

	// findBB - put all Basic Blocks in node into nodeVec
	void findBB(NodeTy *node, BBSetTy *nodeVec) const;

	// findEntryBB - find the entry Basic Block of the node
	ir::ControlFlowGraph::iterator findEntryBB (NodeTy *node);

	// dumpUnstructuredBR - Dump all found unstructured branches
	void dumpUnstructuredBR(std::ostream& stream) const;

	// isStillReachableFrom entry -Return true if after erasing
	// edge src->dst, dst is still reachable from entry
	bool isStillReachableFromEntry(NodeSetTy &N, NodeTy *entry,
		NodeTy *dstNode, NodeTy *srcNode);

	// clean - fill in the element of incoming branches and outgoing branches
	void cleanup(NodeTy *node);

	void cleanupUnreachable();

	void reconstructUnreachable();

	// deleteUnreachableNode - delete nodes that is no longer 
	// reachable from the entry
	void deleteUnreachableNodes(NodeSetTy &N, NodeTy *entry);

	////Interface to PTXToILTranslator
	///*! \brief Returns a pointer to the root node of the control tree */
	const NodeTy* get_root_node() const;

	bool checkUnique(EdgeVecTy &edgeVec, ir::ControlFlowGraph::iterator srcBB, 
		ir::ControlFlowGraph::iterator dstBB);
};

}

#endif

