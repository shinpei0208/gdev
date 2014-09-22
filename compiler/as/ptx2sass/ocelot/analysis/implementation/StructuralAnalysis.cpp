/*! \file StructuralAnalysis.cpp
	\author Haicheng Wu <hwu36@gatech.edu>
	\date	 Monday April 4, 2011
	\brief	The source file for the StructuralAnalysis pass.
*/

//===----------------------------------------------------------------------===// 
// 
// This file defines the class of Structural Analysis which will return the 
// control tree and unstructured branches of a function 
// 
//===----------------------------------------------------------------------===// 

// Ocelot Includes 
#include <ocelot/analysis/interface/StructuralAnalysis.h>
#include <algorithm>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace analysis {

StructuralAnalysis::Node::~Node()
{
	for(auto n = childNode.begin(); n != childNode.end(); ++n)
	{
		delete *n;
	}
}

// buildSimpleCFG - Build a Simple CFG out of the LLVM CFG
void StructuralAnalysis::buildSimpleCFG(NodeSetTy &N) {
	// Create a simple CFG node for every Basic Block
	for(ir::ControlFlowGraph::iterator i = _kernel->cfg()->begin(), 
		e = _kernel->cfg()->end();
		i != e; ++i) {
		NodeTy *n = new NodeTy;
		n->isCombined = false;
		n->BB = i;
		n->containedBB.insert(i);
		n->isLoopHeader = false;
		n->loopExitNode = NULL;
		n->parentNode = NULL;
		n->isBackEdge = false;			
		N.insert(n);
		BB2NodeMap[i] = n;
	}

	// Setup the edges of the simple CFG
	for(ir::ControlFlowGraph::iterator i = _kernel->cfg()->begin(), 
		e = _kernel->cfg()->end(); i != e; ++i) {
		NodeTy *n = BB2NodeMap[i];

		// Setup the predecessor of every node
		ir::ControlFlowGraph::BlockPointerVector PredVec = i->predecessors;

		for (ir::ControlFlowGraph::BlockPointerVector::iterator
			PI = PredVec.begin(), E = PredVec.end(); PI != E; ++PI) {
			NodeTy *p = BB2NodeMap[*PI];
			n->predNode.insert(p);
		}

		// Setup the successor of every node
		ir::ControlFlowGraph::BlockPointerVector SuccVec = i->successors;

		for (ir::ControlFlowGraph::BlockPointerVector::iterator
			SI = SuccVec.begin(), E = SuccVec.end(); SI != E; ++SI) {
			NodeTy *s = BB2NodeMap[*SI];
			n->succNode.insert(s);
		}
	}

	// Remove unreachable node
	NodeTy *entry = BB2NodeMap[_kernel->cfg()->get_entry_block()];

	deleteUnreachableNodes(N, entry); 
}

// structuralAnalysis - Follow Fig 7.39 of Muchnick book
void StructuralAnalysis::structuralAnalysis(NodeSetTy &N,
	NodeTy *entry, bool debug) {
	NodeTy *n = NULL, *p = NULL, *entryNode = NULL, *exitNode = NULL;
	RegionTy rType;
	NodeSetTy nodeSet, reachUnder;

	unstructuredBRVec.clear();

	// Handle the case if the Function has only one Basic Block
	if (N.size() == 1) {
		NodeTy *node = new NodeTy;
		NodeTy *singleNode = *(N.begin());
		node->isCombined = true;
		node->childNode.insert(singleNode);
		node->entryNode = singleNode;
		node->exitBB = singleNode->BB;
		node->containedBB.insert(singleNode->BB);
		node->nodeType = Block;

		singleNode->parentNode = node;

		N.erase(singleNode);
		N.insert(node);

		return;
	}

	do {
		bool change = false;

		post.clear();
		preTree.clear();
		postTree.clear();

		visit.clear();
		postMax = 0;
		postCtr = 1;
		preMax = 0;

		DFSPostorder(N, entry);

		while (N.size() > 1 && postCtr <= postMax) {

			n = post[postCtr];

			if (N.count(n) == 0) continue;
 
			// Locate an acyclic region, if present
			if (n->isLoopHeader && n->loopExitNode) {
				visitPath.clear();

				if (path(n, n->loopExitNode, N, NULL)) {
					NodeTy *tmpNode = n->loopExitNode;
	
					while(tmpNode->parentNode) tmpNode = tmpNode->parentNode;
	
					n->succNode.erase(tmpNode);
					tmpNode->predNode.erase(n);
					n->loopExitNode = tmpNode;
				} else
					n->loopExitNode = NULL;
			}
				
			rType = acyclicRegionType(N, n, nodeSet,
				&entryNode, &exitNode, entry);

			if (n->isLoopHeader && n->loopExitNode) {
				n->succNode.insert(n->loopExitNode);
				n->loopExitNode->predNode.insert(n);
			}

			if (rType == Improper) {
				change = true;

				break;
			}
			else if (rType != Nil) {
				p = reduce(N, rType, nodeSet, entryNode, exitNode);
				change = true;

				if (nodeSet.count(entry)) 
					entry = p;
 
				break; 
			} 
			else {
				NodeTy *backEdgeNode;

				if ((backEdgeNode = pathBack(n, N, reachUnder))) {
					rType = cyclicRegionType(N, reachUnder,
						n, backEdgeNode, &exitNode, entry);

					if (rType == Improper) {
						change = true;

						break;
					} else if (rType != Nil) {
						change = true;
						p = reduce(N, rType, reachUnder, n, exitNode);

						if (reachUnder.count(entry)) entry = p;

						break;
					} else
						postCtr++;
				} 
				else 
					postCtr++;
			}
		}
		
		if (!change) {
			for (int i = 1; i <= postMax; i++) {
				NodeTy *node = post[i];

				if (node->predNode.size() > 1 && node->succNode.size() == 0) {
					int min = postMax + 1;

					for (NodeSetTy::iterator pi = node->predNode.begin(),
						pe = node->predNode.end(); pi != pe; ++pi) {
						NodeTy *predNode = *pi;

						if (postTree[predNode] < min) min = postTree[predNode];
					}

					for (NodeSetTy::iterator pi = node->predNode.begin(),
						pe = node->predNode.end(); pi != pe; ++pi) {
						NodeTy *predNode = *pi;

						if (postTree[predNode] != min) {
							if (isStillReachableFromEntry(N, entry, 
								node, predNode)) {
								findUnstructuredBR(N, predNode,
									node, true, true);
								change = true;
							}
						}
					}

					if (change) break;
				}
			} 
		}

		if (!change) {
			for (int i = 1; i <= postMax; i++) {
				NodeTy *node = post[i];
			
				if (node->predNode.size() > 1 && !node->isBackEdge) {
					NodeTy *tmpNode = NULL;
					bool processThisNode = true;

					for (NodeSetTy::iterator pi = node->predNode.begin(),
						pe = node->predNode.end(); pi != pe; ++pi) {
						NodeTy *predNode = *pi;

						if (edge2ClassMap[std::make_pair(predNode, node)]
							== BACK) {
							processThisNode = false;

							break;
						}

						if (tmpNode == NULL) 
							tmpNode = predNode;
						else {
							visitPath.clear();

							if (path(tmpNode, predNode, N, node)) 
								continue;
							else { 
								visitPath.clear();

								if (path(predNode, tmpNode, N, node))
									tmpNode = predNode;
								else {
									processThisNode = false;

									break;
								}
							}
						}
					}

					if (processThisNode) {
						for (NodeSetTy::iterator pi = node->predNode.begin(),
							pe = node->predNode.end(); pi != pe; ++pi) {
							NodeTy *predNode = *pi;

							if (predNode == tmpNode) {
								if (isStillReachableFromEntry(N, entry,
									node, predNode)) {
									findUnstructuredBR(N, predNode, node,
										true, true);
									change = true;
								}
							}
						}

						if (change) break;
					}
				}
			} 
		}

		if (!change) {
			for (int i = 1; i <= postMax; i++) {
				NodeTy *node = post[i];
			
				if (node->predNode.size() > 1 && !node->isBackEdge) {
					bool processThisNode = true;
					int min = postMax + 1;

					for (NodeSetTy::iterator pi = node->predNode.begin(),
						pe = node->predNode.end(); pi != pe; ++pi) {
						NodeTy *predNode = *pi;

						if (edge2ClassMap[std::make_pair(
							predNode, node)] == BACK) {
							processThisNode = false;

							break;
						}

						if (postTree[predNode] < min) min = postTree[predNode];
					}

					if (processThisNode) {

						for (NodeSetTy::iterator pi = node->predNode.begin(),
							pe = node->predNode.end(); pi != pe; ++pi) {
							NodeTy *predNode = *pi;
	
							if (postTree[predNode] != min) {
								if (isStillReachableFromEntry(N, entry,
									node, predNode)) {
									findUnstructuredBR(N, predNode, node,
										true, true);
									change = true;
								}
							}
						}

						if (change) break;
					}
				}
			} 
		}
		
		#if REPORT_BASE != 0
			std::stringstream stream;
			dumpCTNode(stream, p);
			dumpUnstructuredBR(stream);
			report(stream.str());
			report(N.size() << "\n");
			report("********************\n");
		#endif

		assertM(change != false, "Cannot reduce any more "
			"in structural analysis");
	} while (N.size() != 1);
}

// DFSPostorder - Follow Fig 7.40 of Muchnick book 
void StructuralAnalysis::DFSPostorder(NodeSetTy &N, NodeTy *x) {
	visit[x] = true;
	preTree[x] = ++preMax;
 
	for (NodeSetTy::iterator i = x->succNode.begin(),
		e = x->succNode.end(); i != e; ++i) { 
		NodeTy *y = *i;

		if (visit.count(y) == 0) {
			DFSPostorder(N, y);
			edge2ClassMap[std::make_pair(x, y)] = TREE;
		}
		else if (preTree[x] < preTree[y])
			edge2ClassMap[std::make_pair(x, y)] = FORWARD;
		else if (postTree[y] == 0 || preTree[x] == preTree[y])
			edge2ClassMap[std::make_pair(x, y)] = BACK;
		else 
			edge2ClassMap[std::make_pair(x, y)] = CROSS;
	}

	postMax++;
	post[postMax] = x;
	postTree[x] = postMax;
}

// acyclicRegionType - Follow Fig 7.41 of Muchnick book
StructuralAnalysis::RegionTy StructuralAnalysis::acyclicRegionType(
	NodeSetTy &N, NodeTy *node,
	NodeSetTy &nset, NodeTy **entryNode, NodeTy **exitNode, NodeTy *entry) {
	NodeTy *m, *n;
	bool p, s;

	nset.clear();

	// Check for a block containing node
	NodeTy *firstNode, *lastNode;
	firstNode = lastNode = n = node;
	p = true;
	s = (n->succNode.size() == 1);

	while (p && s) {
		lastNode = n;

		if (nset.count(n) == 0) 
			nset.insert(n);
		else 
			return Nil;

		n = *(n->succNode.begin());
		p = (n->predNode.size() == 1);
		s = (n->succNode.size() == 1);
	}

	if (p) { 
		if (nset.count(n) == 0) {
			nset.insert(n);
			lastNode = n;
		}
		else
			return Nil;
	}

	n = node;
	p = (n->predNode.size() == 1);
	s = true;

	while (p && s) {
		firstNode = n;

		if (nset.count(n) == 0	|| n == node)
			nset.insert(n);
		else
			return Nil;
		n = *(n->predNode.begin());
		p = (n->predNode.size() == 1);
		s = (n->succNode.size() == 1);
	}

	if (s) {
		if (nset.count(n) == 0 || n == node) {
			firstNode = n; 
			nset.insert(n);
		}
		else
			return Nil;
	}

	if (firstNode->predNode.count(lastNode) > 0) {
		if (nset.size() == 2)
			return Nil;
		else
			nset.erase(firstNode);
	} 

	*entryNode = n;

	if (nset.size() >= 2) {
		if (nset.count(*entryNode) == 0)
			for (NodeSetTy::iterator i = (*entryNode)->succNode.begin(), 
				e = (*entryNode)->succNode.end(); i != e; ++i) 
				if (nset.count(*i) > 0) 
					*entryNode = *i;

		*exitNode = lastNode;

		return Block; 
	}

	*entryNode = node;

	if ((*entryNode)->succNode.size() == 2) {
		NodeSetTy::iterator i = (*entryNode)->succNode.begin();
		m = *i;
		++i;
		n = *i;

		if (m == *entryNode || n == *entryNode) return Nil;

		if (edge2ClassMap[std::make_pair(*entryNode, m)] == BACK) return Nil;
		if (edge2ClassMap[std::make_pair(*entryNode, n)] == BACK) return Nil;

		// Check for an normal IfThenElse
		if (m->succNode.size() == 1 && n->succNode.size() == 1	
			&& m->predNode.size() == 1 && n->predNode.size() == 1
			&& *(m->succNode.begin()) == *(n->succNode.begin())
			&& *(m->succNode.begin()) != *entryNode) { 
		
			if (edge2ClassMap[std::make_pair(
				m, *entryNode)] == BACK) return Nil;
			if (edge2ClassMap[std::make_pair(
				n, *entryNode)] == BACK) return Nil;
 
					nset.insert(*entryNode);
					nset.insert(m);
					nset.insert(n);
					*exitNode = *(m->succNode.begin());
 
					return IfThenElse;
				}
		// Check for an IfThenElse with no exit block
		if (m->succNode.size() == 0 && n->succNode.size() == 0	
			&& m->predNode.size() == 1 && n->predNode.size() == 1) {
			nset.insert(*entryNode);
			nset.insert(m);
			nset.insert(n);
			*exitNode = NULL;

			return IfThenElse;
		}
		// Check for an IfThen
		// n is the Then part
		else if (n->succNode.size() == 1 && n->predNode.size() == 1
			&& m == *(n->succNode.begin())) {
			if (edge2ClassMap[std::make_pair(n, m)] != BACK) {
				if (edge2ClassMap[std::make_pair(
					n, *entryNode)] == BACK) return Nil;

				nset.insert(*entryNode);
				nset.insert(n);
				*exitNode = m;
	
				return IfThen;
			}
		} 
		// m is the Then part
		else if (m->succNode.size() == 1 && m->predNode.size() == 1
			&& n == *(m->succNode.begin())) {
			if (edge2ClassMap[std::make_pair(m, n)] != BACK) {
				if (edge2ClassMap[std::make_pair(
					m, *entryNode)] == BACK) return Nil;

				nset.insert(*entryNode);
				nset.insert(m);
				*exitNode = n;
	
				return IfThen;
			}
		}
		// n is the Then part w/o exiting edge
		else if (n->succNode.size() == 0 && n->predNode.size() == 1) {
			visitPath.clear();

			if (!path(m, *entryNode, N, NULL)) {
				nset.insert(*entryNode);
				nset.insert(n);
				*exitNode = NULL;

				return IfThen;
			} 
		}
		// m is the Then part w/o exiting edge
		else if (m->succNode.size() == 0 && m->predNode.size() == 1) {
			visitPath.clear();

			if (!path(n, *entryNode, N, NULL)) {
				nset.insert(*entryNode);
				nset.insert(m);
				*exitNode = NULL;

				return IfThen;
			}
		}
		// Check for an IfThenElse with incoming edges
		else if (m->succNode.size() == 1 && n->succNode.size() == 1	
			&& *(m->succNode.begin()) == *(n->succNode.begin()) 
			&& *(m->succNode.begin()) != *entryNode) {

			if (edge2ClassMap[std::make_pair(
				m, *entryNode)] == BACK) return Nil;
			if (edge2ClassMap[std::make_pair(
				n, *entryNode)] == BACK) return Nil;

			if (n->predNode.count(*(n->succNode.begin())) == 0 &&
				m->predNode.count(*(m->succNode.begin())) == 0) {
			
				if (m->predNode.count(*(m->succNode.begin())) > 0
					|| n->predNode.count(*(n->succNode.begin())) > 0)
					return Nil;

				bool improperFlag = false;

				if (m->predNode.size() > 1) {
					for (NodeSetTy::iterator pi = m->predNode.begin(),
						pe = m->predNode.end(); pi != pe; ++pi)	
						if (*pi != *entryNode
							&& isStillReachableFromEntry(N, entry, m, *pi) 
							&& edge2ClassMap[std::make_pair(*pi, m)] != BACK) {
							findUnstructuredBR(N, *pi, m, true, true);					
							improperFlag = true;
						}
				}
	
				if (n->predNode.size() > 1) {
					for (NodeSetTy::iterator pi = n->predNode.begin(),
						pe = n->predNode.end(); pi != pe; ++pi)	
						if (*pi != *entryNode
							&& isStillReachableFromEntry(N, entry, n, *pi) 
							&& edge2ClassMap[std::make_pair(*pi, n)] != BACK) {
							findUnstructuredBR(N, *pi, n, true, true);					
							improperFlag = true;
						}
				}
	 
				if(improperFlag) return Improper;
			}
		}
		// Check for an IfThen with incoming edges
		// n is the Then part
		else if (n->succNode.size() == 1 && n->predNode.size() > 1
			&& m == *(n->succNode.begin())) {
			if (edge2ClassMap[std::make_pair(n,
				*entryNode)] == BACK) return Nil;

			if (edge2ClassMap[std::make_pair(n, m)] != BACK) {
				if (n->predNode.count(m) > 0)
					return Nil;

				bool improperFlag = false;

				for (NodeSetTy::iterator pi = n->predNode.begin(),
					pe = n->predNode.end(); pi != pe; ++pi)	
					if (*pi != *entryNode
						&& isStillReachableFromEntry(N, entry, n, *pi) 
						&& edge2ClassMap[std::make_pair(*pi, n)] != BACK) {
						findUnstructuredBR(N, *pi, n, true, true);					
						improperFlag = true;
					}
				
				if(improperFlag) return Improper;
			}
		} 
		// m is the Then part
		else if (m->succNode.size() == 1 && m->predNode.size() > 1
			&& n == *(m->succNode.begin())) {
			if (edge2ClassMap[std::make_pair(m, *entryNode)]
				== BACK) return Nil;

			if (edge2ClassMap[std::make_pair(m, n)] != BACK) {
				if (m->predNode.count(n) > 0)
					return Nil;

				bool improperFlag = false;

				for (NodeSetTy::iterator pi = m->predNode.begin(),
					pe = m->predNode.end(); pi != pe; ++pi)	
					if (*pi != *entryNode
						&& isStillReachableFromEntry(N, entry, m, *pi) 
						&& edge2ClassMap[std::make_pair(*pi, m)] != BACK) {
						findUnstructuredBR(N, *pi, m, true, true);					
						improperFlag = true;
					}
				
				if(improperFlag) return Improper;
			}
		}
		// Check for an IfThenElse (w/o exit block) with incoming edges
		else if (m->succNode.size() == 0 && n->succNode.size() == 0) {
			bool improperFlag = false;

			if (m->predNode.size() > 1) {
				for (NodeSetTy::iterator pi = m->predNode.begin(),
					pe = m->predNode.end(); pi != pe; ++pi)	
					if (*pi != *entryNode
						&& isStillReachableFromEntry(N, entry, m, *pi)) {
						findUnstructuredBR(N, *pi, m, true, true);					
						improperFlag = true;
					}
			}

			if (n->predNode.size() > 1) {
				for (NodeSetTy::iterator pi = n->predNode.begin(),
					pe = n->predNode.end(); pi != pe; ++pi)	
					if (*pi != *entryNode
						&& isStillReachableFromEntry(N, entry, n, *pi)) {
						findUnstructuredBR(N, *pi, n, true, true);					
						improperFlag = true;
					}
			}
 
			if(improperFlag) return Improper;
		}
		// n is the Then part (w/o exiting edge) with incoming edges
		else if (n->succNode.size() == 0 && n->predNode.size() > 1) {
			visitPath.clear();

			if (!path(m, *entryNode, N, NULL)) {
				if (n->predNode.count(m) > 0)
					return Nil;

				bool improperFlag = false;

				for (NodeSetTy::iterator pi = n->predNode.begin(),
					pe = n->predNode.end(); pi != pe; ++pi)
					if (*pi != *entryNode
						&& isStillReachableFromEntry(N, entry, n, *pi)) {
						findUnstructuredBR(N, *pi, n, true, true);
						improperFlag = true;
					}
	 
				if(improperFlag) return Improper;
			}
		}
		// m is the Then part w/o exiting edge with incoming edges
		else if (m->succNode.size() == 0 && m->predNode.size() > 1) {
			visitPath.clear();

			if (!path(n, *entryNode, N, NULL)) {
				if (m->predNode.count(n) > 0)
					return Nil;

				bool improperFlag = false;

				for (NodeSetTy::iterator pi = m->predNode.begin(),
					pe = m->predNode.end(); pi != pe; ++pi)
					if (*pi != *entryNode
						&& isStillReachableFromEntry(N, entry, n, *pi)) {
						findUnstructuredBR(N, *pi, m, true, true);
						improperFlag = true;
					}
	 
				if(improperFlag) return Improper;
			}
		}
	}
	// Check for Case
	else if ((*entryNode)->succNode.size() > 2) {
		if(isCaseWithDefault(N, *entryNode, exitNode, entry)) {
			nset.insert(*entryNode);
			
			for (NodeSetTy::iterator i = (*entryNode)->succNode.begin(),
				e = (*entryNode)->succNode.end(); i != e; ++i) 
				nset.insert(*i);
	
			return Case;
		}
		else if(isCaseWithoutDefault(N, *entryNode, exitNode, entry)) {
			nset.insert(*entryNode);
			
			for (NodeSetTy::iterator i = (*entryNode)->succNode.begin(),
				e = (*entryNode)->succNode.end(); i != e; ++i) 
				if(*i != *exitNode) nset.insert(*i);
	
			return Case;
		}
		else if (isImproperCaseWithDefault(N, *entryNode, entry)) 
			return Improper;
		else if (isImproperCaseWithoutDefault(N, *entryNode, exitNode, entry))
			return Improper;
	}

	return Nil; 
}

// isCaseWithDefault - Check if node leads a case block
bool StructuralAnalysis::isCaseWithDefault(NodeSetTy &N, NodeTy * entryNode,
	NodeTy **exitNode, NodeTy *entry) {
	*exitNode = NULL; 

	for (NodeSetTy::iterator i = entryNode->succNode.begin(),
		e = entryNode->succNode.end(); i != e; ++i) { 
		// Check if every successor node has only one successor	 
		if ((*i)->succNode.size() > 1) return false;

		if (edge2ClassMap[std::make_pair(entryNode, *i)] == BACK) return false;
 
		// If successor has only one predessor, it has to be the entry node		
		if ((*i)->predNode.size() == 1) { 
			if (entryNode != *((*i)->predNode.begin())) 
				return false;
		}
		// If successor has two predessors, one has to be the entry node
		// and the other has to be another successor node
		else if ((*i)->predNode.size() == 2) {
			NodeSetTy::iterator pi = (*i)->predNode.begin();
			NodeTy *predNode1 = *pi;
			++pi;
			NodeTy *predNode2 = *pi;

			if (predNode1 != entryNode
				|| entryNode->succNode.count(predNode2) == 0)
				if (entryNode->succNode.count(predNode1) == 0
					|| predNode2 != entryNode)
					return false;
		}
		// The predecessor node number has to be less than 3
		else 
			return false;	 

		NodeTy *succNode = *((*i)->succNode.begin());

		if (succNode == NULL) continue;
		
		if (succNode == entryNode) return false;

		// Check if the successor of the successor node is not another successor 
		if (entryNode->succNode.count(succNode) == 0) {
			// Check if the successor of the successor is the only exit node
			if (!*exitNode) *exitNode = succNode;
			else if (*exitNode != succNode) return false; 
		}
		// There is no loop between successors
		else if (succNode->succNode.count(*i) > 0) return false;

	}

	for (NodeSetTy::iterator i = entryNode->succNode.begin(),
		e = entryNode->succNode.end(); i != e; ++i) { 
		if ((*i)->succNode.size() == 0) {
			visitPath.clear();
 
			if (path(*exitNode, entryNode, N, NULL))
				return false;
		}
	}

	return true; 
}

// isImproperCaseWithDefault - Check if node leads a case block
bool StructuralAnalysis::isImproperCaseWithDefault(NodeSetTy &N,
	NodeTy *entryNode, NodeTy *entry) {
	NodeTy *exitNode = NULL; 
	EdgeSetTy improperEdgeSet;	

	for (NodeSetTy::iterator i = entryNode->succNode.begin(),
		e = entryNode->succNode.end(); i != e; ++i) { 
		// Check if every successor node has only one successor	 
		if ((*i)->succNode.size() > 1) return false;

		if (edge2ClassMap[std::make_pair(entryNode, *i)] == BACK) return false;

		NodeTy *succNode = *((*i)->succNode.begin());

		if (succNode) { 
			if (succNode == entryNode) return false;
	
			// Check if the successor of the successor node 
			// is not another successor node
			if (entryNode->succNode.count(succNode) == 0) {
				// Is the successor of the successor node is the only exit node?
				if (!exitNode) exitNode = succNode;
				else if (exitNode != succNode) return false; 
			}
			// There is no loop between successors
			else if (succNode->succNode.count(*i) > 0) 
				return false;
		}

		// If successor has only one predessor, it has to be the entry node		
		if ((*i)->predNode.size() == 1) { 
			if (entryNode != *((*i)->predNode.begin())) 
				return false;
		}
		// If successor has two predessors, one has to be the entry node
		// and the other has to be another successor node
		else if ((*i)->predNode.size() == 2) {
			NodeSetTy::iterator pi = (*i)->predNode.begin();
			NodeTy *predNode1 = *pi;
			++pi;
			NodeTy *predNode2 = *pi;

			if (predNode1 != entryNode
				|| entryNode->succNode.count(predNode2) == 0)
				if (entryNode->succNode.count(predNode1) == 0
					|| predNode2 != entryNode)
					return false;
		}
		// The predecessor node number has to be less than 3
		else { 
			int insideIncomingNum = 0;

			for (NodeSetTy::iterator pi = (*i)->predNode.begin(),
				pe = (*i)->predNode.end(); pi != pe; ++pi) {

				if (edge2ClassMap[std::make_pair(*pi, *i)] != BACK
					&& *pi != exitNode && *pi != entryNode) { 
					if (entryNode->succNode.count(*pi) == 0) 
						improperEdgeSet.insert(std::make_pair(*pi, *i));
					else {
						insideIncomingNum++;
	
						if (insideIncomingNum > 1)
							improperEdgeSet.insert(std::make_pair(*pi, *i));
					}				
				} else 
					return false;
			}
		}
	}

	for (NodeSetTy::iterator i = entryNode->succNode.begin(),
		e = entryNode->succNode.end(); i != e; ++i) { 
		if ((*i)->succNode.size() == 0) {
			visitPath.clear();
 
			if (path(exitNode, entryNode, N, NULL))
				return false;
		}
	}

	bool improperFlag = false;

	for (EdgeSetTy::iterator i = improperEdgeSet.begin(),
		e = improperEdgeSet.end(); i != e; ++i) 
		if (isStillReachableFromEntry(N, entry, i->second, i->first)) {
			findUnstructuredBR(N, i->first, i->second, true, true);
			improperFlag = true;
		}

	return improperFlag; 
}

// isCaseWithoutDefault - Check if node leads a case block
bool StructuralAnalysis::isCaseWithoutDefault(NodeSetTy &N,
	NodeTy * entryNode, NodeTy **exitNode, NodeTy *entry) {
	// Find the exit node first
	*exitNode = NULL; 

	for (NodeSetTy::iterator i = entryNode->succNode.begin(),
		e = entryNode->succNode.end(); i != e; ++i) {
		NodeTy *node1 = *i;
		bool foundExit = true;
 
		// all of successors of exit node are not within the switch block
		for (NodeSetTy::iterator si = node1->succNode.begin(),
			se = node1->succNode.end(); si != se; ++si) {
			NodeTy *succNode = *si;

			if (succNode) {
				if (entryNode->succNode.count(succNode) > 0) {
					foundExit = false;
	
					break;
				} else if (succNode == entryNode) 
						return false;
			}
		}

		if (!foundExit) continue;

		foundExit = false;

		// at least one of predcessors of exit node comes from switch block
		for (NodeSetTy::iterator pi = node1->predNode.begin(),
			pe = node1->predNode.end(); pi != pe; ++pi) {
			NodeTy *predNode = *pi;

			if (predNode != entryNode
				&& entryNode->succNode.count(predNode) > 0) {
				foundExit = true;
			}
		}

		if (foundExit) {
			*exitNode = node1;

			break;
		} 
	}

	if (!(*exitNode)) return false;
 
	for (NodeSetTy::iterator i = entryNode->succNode.begin(),
		e = entryNode->succNode.end(); i != e; ++i) {
		if (*i == *exitNode) continue; 
 
		// Check if every successor node has only one successor	 
		if ((*i)->succNode.size() > 1) return false;

		NodeTy *succNode = *((*i)->succNode.begin());

		if (succNode) {
			if (succNode == NULL) continue;
			
			if (edge2ClassMap[std::make_pair(entryNode, *i)]
				== BACK) return false;
	
			// The successor of the successor node should be the the another
			// successor node of node
			if (entryNode->succNode.count(succNode) == 0) return false;
			// There is no loop between successors
			else if (succNode != *exitNode
				&& succNode->succNode.count(*i) > 0) return false;
		}

		// If successor has only one predessor, it has to be the entry node		
		if ((*i)->predNode.size() == 1) { 
			if (entryNode != *((*i)->predNode.begin())) 
				return false;
		}
		// If successor has two predessors, one has to be the entry node
		// and the other has to be another successor node
		else if ((*i)->predNode.size() == 2) {
			NodeSetTy::iterator pi = (*i)->predNode.begin();
			NodeTy *predNode1 = *pi;
			++pi;
			NodeTy *predNode2 = *pi;

			if (predNode1 != entryNode
				|| entryNode->succNode.count(predNode2) == 0)
				if (entryNode->succNode.count(predNode1) == 0
					|| predNode2 != entryNode)
					return false;
		}
		// The predecessor node number has to be less than 3
		else 
			return false;	 
	}

	for (NodeSetTy::iterator i = entryNode->succNode.begin(),
		e = entryNode->succNode.end();
		i != e; ++i) { 
		if ((*i)->succNode.size() == 0) {
			visitPath.clear();
 
			if (path(*exitNode, entryNode, N, NULL))
				return false;
		}
	}

	return true; 
}

// isImproperCaseoutWithDefault - Check if node leads a case block with incoming edges
bool StructuralAnalysis::isImproperCaseWithoutDefault(NodeSetTy &N,
	NodeTy *entryNode, NodeTy **exitNode, NodeTy *entry) {
	EdgeSetTy improperEdgeSet;

	// Find the exit node first
	*exitNode = NULL; 

	for (NodeSetTy::iterator i = entryNode->succNode.begin(),
		e = entryNode->succNode.end();
		i != e; ++i) {
		NodeTy *node1 = *i;
		bool foundExit = true;

		// all of successors of exit node are not within the switch block
		for (NodeSetTy::iterator si = node1->succNode.begin(),
			se = node1->succNode.end();
			si != se; ++si) {
			NodeTy *succNode = *si;

			if (succNode ) {
				if (entryNode->succNode.count(succNode) > 0) {
					foundExit = false;
	
					break;
				} else if (succNode == entryNode)
					return false;
			}
		}

		if (!foundExit) continue;

		foundExit = false;

		// at least one of predcessors of exit node comes from switch block
		for (NodeSetTy::iterator pi = node1->predNode.begin(),
			pe = node1->predNode.end();
			pi != pe; ++pi) {
			NodeTy *predNode = *pi;

			if (predNode != entryNode
				&& entryNode->succNode.count(predNode) > 0) {
				foundExit = true;
			}
		}

		if (foundExit) {
			*exitNode = node1;

			break;
		} 
	}

	if (!(*exitNode)) return false;
 
	for (NodeSetTy::iterator i = entryNode->succNode.begin(),
		e = entryNode->succNode.end();
		i != e; ++i) {
		if (*i == *exitNode) continue; 
 
		// Check if every successor node has only one successor	 
		if ((*i)->succNode.size() > 1) return false;

		if (edge2ClassMap[std::make_pair(entryNode, *i)] == BACK) return false;

		NodeTy *succNode = *((*i)->succNode.begin());

		if (succNode) {
			if (succNode == NULL) continue;
	
			// The successor of the successor node should be the the another
			// successor node of node
			if (entryNode->succNode.count(succNode) == 0) return false;
			// There is no loop between successors
			else if (succNode != *exitNode && succNode->succNode.count(*i) > 0) 
				return false;
		}

		// If successor has only one predessor, it has to be the entry node		
		if ((*i)->predNode.size() == 1) { 
			if (entryNode != *((*i)->predNode.begin())) 
				return false;
		}
		// If successor has two predessors, one has to be the entry node
		// and the other has to be another successor node
		else if ((*i)->predNode.size() == 2) {
			NodeSetTy::iterator pi = (*i)->predNode.begin();
			NodeTy *predNode1 = *pi;
			++pi;
			NodeTy *predNode2 = *pi;

			if (predNode1 != entryNode
				|| entryNode->succNode.count(predNode2) == 0)
				if (entryNode->succNode.count(predNode1) == 0
					|| predNode2 != entryNode)
					return false;

			if (predNode1 == *exitNode) 
				return false;

			if (predNode2 == *exitNode)
				return false;
		}
		// The predecessor node number has to be less than 3
		else {
			int insideIncomingNum = 0;

			for (NodeSetTy::iterator pi = (*i)->predNode.begin(),
				pe = (*i)->predNode.end();
				pi != pe; ++pi) {

				if (edge2ClassMap[std::make_pair(*pi, *i)] != BACK
					&& (*i) != *exitNode && (*pi)
						!= entryNode && (*pi) != *exitNode) {
					if (entryNode->succNode.count(*pi) == 0)
						improperEdgeSet.insert(std::make_pair(*pi, *i));
					else {
						insideIncomingNum++;
	
						if (insideIncomingNum > 1)
							improperEdgeSet.insert(std::make_pair(*pi, *i));
					}
				} else
				 return false;
			}
		}
	}

	for (NodeSetTy::iterator i = entryNode->succNode.begin(),
		e = entryNode->succNode.end();
		i != e; ++i) { 
		if ((*i)->succNode.size() == 0) {
			visitPath.clear();
 
			if (path(*exitNode, entryNode, N, NULL))
				return false;
		}
	}

	bool improperFlag = false;

	for (EdgeSetTy::iterator i = improperEdgeSet.begin(),
		e = improperEdgeSet.end(); i != e; ++i) 
		if (isStillReachableFromEntry(N, entry, i->second, i->first)) {
			findUnstructuredBR(N, i->first, i->second, true, true);
			improperFlag = true;
		}

	return improperFlag; 
}

// cyclicRegionType - Follow Fig 7.42 of Muchnick book
StructuralAnalysis::RegionTy StructuralAnalysis::cyclicRegionType(
	NodeSetTy &N, NodeSetTy &nset,
	NodeTy *loopHeaderNode, NodeTy *backEdgeNode, NodeTy **exitNode,
	NodeTy *entry) {
	// Check for a SelfLoop
	if (nset.size() == 1) { 
		if (loopHeaderNode == backEdgeNode) {
			*exitNode = *(backEdgeNode->succNode.begin());

			return SelfLoop;
		} else 
			return Nil;
	}

	if (isImproper(N, nset, loopHeaderNode, backEdgeNode, exitNode, entry)) 
		// It is an Improper region
		return Improper;

	if (nset.size() == 2) {
		if (backEdgeNode->succNode.size() == 1) {
			for (NodeSetTy::iterator i = loopHeaderNode->predNode.begin(),
				e = loopHeaderNode->predNode.end(); i != e; ++i) {
				NodeTy *pred = *i;
		 
				if (pred != backEdgeNode) {
					if (edge2ClassMap[std::make_pair(
						pred, loopHeaderNode)] == BACK)
						return Nil;
				}
			}
	
			for (NodeSetTy::iterator i = loopHeaderNode->succNode.begin(),
				e = loopHeaderNode->succNode.end(); i != e; ++i) {
				NodeTy *succ = *i;
		 
				if (succ != backEdgeNode) {
					if (edge2ClassMap[std::make_pair(
						loopHeaderNode, succ)] == BACK)
						return Nil;
				}
			}
	
			if (backEdgeNode->predNode.size() != 1
				|| backEdgeNode->succNode.size() != 1 )
				return Nil;
	
			return NaturalLoop;
		} else if (backEdgeNode->succNode.size() > 1) {
			for (NodeSetTy::iterator i = loopHeaderNode->predNode.begin(),
				e = loopHeaderNode->predNode.end(); i != e; ++i) {
				NodeTy *pred = *i;
		 
				if (pred != backEdgeNode) {
					if (edge2ClassMap[std::make_pair(
						pred, loopHeaderNode)] == BACK)
						return Nil;
				}
			}
	
			for (NodeSetTy::iterator i = loopHeaderNode->succNode.begin(),
				e = loopHeaderNode->succNode.end(); i != e; ++i) {
				NodeTy *succ = *i;
		 
				if (succ != backEdgeNode) {
					if (edge2ClassMap[std::make_pair(
						loopHeaderNode, succ)] == BACK)
						return Nil;
				}
			}
	
			if (backEdgeNode->predNode.size() != 1)
				return Nil;
	
			return NaturalLoop;
		}
	}
 
	return Nil;
}

// reduce - Follow Fig 7.43 of Muchnick book
StructuralAnalysis::NodeTy * StructuralAnalysis::reduce(
	NodeSetTy &N, RegionTy rType,
	NodeSetTy &nodeSet, NodeTy *entryNode, NodeTy *exitNode) {
	NodeTy *node = new NodeTy;

	replace(N, node, nodeSet/*, addSelfEdge*/);
	node->isCombined = true;

	if (entryNode) {
		node->entryNode = entryNode;
		node->entryBB = findEntryBB(entryNode);
	}

	node->isLoopHeader = false;
	node->loopExitNode = NULL;
	node->isBackEdge = false;
	node->parentNode = NULL;

	if (exitNode) node->exitBB = findEntryBB(exitNode);
	else node->exitBB = _kernel->cfg()->end();

	for (NodeSetTy::iterator i = nodeSet.begin(),
		e = nodeSet.end(); i != e; ++i)
		findBB(*i, &(node->containedBB));

	node->nodeType = rType;

	return node;
}

// replace - Follow Fig 7.44 of Muchnick book
void StructuralAnalysis::replace(NodeSetTy &N, NodeTy *node,
	NodeSetTy &nodeSet/*, bool addSelfEdge*/) {
	// Link region node into abstract flowgraph, adjust the postorder traversal
	// and predecessor and successor functions, and augment the control tree
	compact(N, node, nodeSet/*, addSelfEdge*/);

	for (NodeSetTy::iterator i = nodeSet.begin(),
		e = nodeSet.end(); i != e; ++i) { 
		node->childNode.insert(*i);
		(*i)->parentNode = node;
	}
}

// isImproper - Follow Fig 7.45 of Muchnick book
bool StructuralAnalysis::isImproper(NodeSetTy &N, NodeSetTy &nset,
	NodeTy *loopHeaderNode, NodeTy *backEdgeNode, NodeTy **exitNode,
	NodeTy *entry) {
	bool improperFlag = false;	

	// Check loopHeaderNode first
	for (NodeSetTy::iterator i = loopHeaderNode->predNode.begin(),
		e = loopHeaderNode->predNode.end();
		i != e; ++i) {
		NodeTy *predNode = *i;

		if (edge2ClassMap[std::make_pair(predNode, loopHeaderNode)] == BACK) {
			if (nset.count(predNode) == 0
				&& isStillReachableFromEntry(N, entry,
					loopHeaderNode, predNode)) {
				findUnstructuredBR(N, predNode, loopHeaderNode, true, true);
				improperFlag = true;
			} else if (nset.count(predNode) > 0 && predNode != backEdgeNode) { 
				findUnstructuredBR(N, predNode, loopHeaderNode, false, false);
				improperFlag = true;
			}
		}
	}

	// Check the incoming edges
	for (NodeSetTy::iterator i = nset.begin(), e = nset.end(); i != e; ++i) {
		NodeTy *node = *i;

		if (node != loopHeaderNode) 
			for (NodeSetTy::iterator ii = node->predNode.begin(),
				ee = node->predNode.end();
				ii != ee; ++ii) {
				if (nset.count(*ii) == 0
					/*&& isStillReachableFromEntry(N, entry, node, *ii)*/) {
					improperFlag = true;

					findUnstructuredBR(N, *ii, node, false, true);
					deleteUnreachableNodes(N, entry);
				}
			}
	}

	EdgeSetTy exitEdgeSet;
	NodeTy *exitNodeOfHeader = NULL;
	NodeTy *exitNodeOfBackEdge = NULL;
	NodeTy *mainExitNode = NULL;

	for (NodeSetTy::iterator i = nset.begin(), e = nset.end(); i != e; ++i) {
		NodeTy *node = *i;

		for (NodeSetTy::iterator ii = node->succNode.begin(),
			ee = node->succNode.end();
			ii != ee; ++ii) {
			if (nset.count(*ii) == 0) {
				exitEdgeSet.insert(std::make_pair(node, *ii)); 

				if (node == loopHeaderNode) {
					if (exitNodeOfHeader == NULL)
						exitNodeOfHeader = *ii;
				} else if (node == backEdgeNode) {
					if (exitNodeOfBackEdge == NULL)
						exitNodeOfBackEdge = *ii;
				}
			}
		} 
	}

	if (exitNodeOfHeader)
		mainExitNode = exitNodeOfHeader; 
	else if (exitNodeOfBackEdge)
		mainExitNode = exitNodeOfBackEdge;

	for (EdgeSetTy::iterator i = exitEdgeSet.begin(),
		e = exitEdgeSet.end(); i != e; ++i) {
		EdgeTy exitEdge = *i;

		if (exitEdge.second != mainExitNode) {
			findUnstructuredBR(N, exitEdge.first, exitEdge.second, false, true);
			deleteUnreachableNodes(N, entry);
			improperFlag = true;
		} 
	}

	if (exitNodeOfHeader) {
		for (EdgeSetTy::iterator i = exitEdgeSet.begin(),
			e = exitEdgeSet.end(); i != e; ++i) {
			if (i->first != loopHeaderNode && (*i).second == mainExitNode) {
				findUnstructuredBR(N, i->first, i->second, false, false);
				improperFlag = true;
			}
		}	
	} else if (exitNodeOfBackEdge) {
		for (EdgeSetTy::iterator i = exitEdgeSet.begin(),
			e = exitEdgeSet.end(); i != e; ++i) {
			if (i->first != backEdgeNode && i->second == mainExitNode) {
				findUnstructuredBR(N, i->first, i->second, false, false);
				improperFlag = true;
			}
		}	
	}
 
	*exitNode = mainExitNode;
	loopHeaderNode->isLoopHeader = true;
	backEdgeNode->isBackEdge = true;
	loopHeaderNode->loopExitNode = mainExitNode;

	return improperFlag; 
}

// pathBack - Check if there is a node k such that there is a path from
// m to k that does not pass through n and an edge k->n that is a back edge
StructuralAnalysis::NodeTy *StructuralAnalysis::pathBack(
	NodeTy *n, NodeSetTy &N,
	NodeSetTy &reachUnder) {
	NodeTy *backEdgeNode = NULL;

	reachUnder.clear();

	// Find backedge first
	for (NodeSetTy::iterator i = n->predNode.begin(),
		e = n->predNode.end(); 
		i != e; ++i) { 
		NodeTy *predNode = *i;

		if (edge2ClassMap[std::make_pair(predNode, n)] == BACK) {
			if (reachUnder.count(predNode) == 0) {
				backEdgeNode = predNode;

				//Locate a cyclic region, if present
				reachUnder.clear();
				reachUnder.insert(n);
				reachUnder.insert(backEdgeNode);
			
				for (NodeSetTy::iterator ii = N.begin(), ee = N.end();
					ii != ee; ++ii) {
					NodeTy *m = *ii;

					// Check if there is a path from m to loop exit node
					visitPath.clear();
					if (path(m, backEdgeNode, N, n)) {
						visitPath.clear();
 
						if (path(n, m, N, backEdgeNode)) 
							reachUnder.insert(m);
					} 
				}
			}
		}
	}

	return backEdgeNode;
}

// path(n, m, I) - Return true if there is a path from from n to m 
// such that all the nodes in it are in I and false otherwise
bool StructuralAnalysis::path(NodeTy *n, NodeTy *m,
	NodeSetTy &I, NodeTy *esc) {
	visitPath[n] = true;

	if (n == esc || m == esc) return false;

	if (n == m) return true;

	for (NodeSetTy::iterator i = n->succNode.begin(),
		e = n->succNode.end();
		i != e; ++i) 
		if (I.count(*i) > 0 && *i != esc && !visitPath[*i]) {
			if (*i == m) 
				return true;
			else 
				if (path(*i, m, I, esc)) return true;
		}

	return false;
}

// path(n, m, I, src, dst ) - Return true if there is a path from from n to m 
// such that all the nodes in it are in I without going through edge src->dst
// and false otherwise
bool StructuralAnalysis::path(NodeTy *n, NodeTy *m,
	NodeSetTy &I, NodeTy *src, NodeTy *dst) {
	visitPath[n] = true;

	if (n == m) return true;

	for (NodeSetTy::iterator i = n->succNode.begin(),
		e = n->succNode.end();
		i != e; ++i)
		if (I.count(*i) > 0 && !visitPath[*i]) {
			if (*i == dst && n == src) continue;

			if (*i == m)
				return true;
			else
				if (path(*i, m, I, src, dst)) return true;
		}

	return false;
}

// compact - Compact nodes in nset into n;
void StructuralAnalysis::compact(NodeSetTy &N, NodeTy *n,
	NodeSetTy &nset/*, bool addSelfEdge*/) {
	// Adds node n to N
	N.insert(n);

	// Remove the nodes in nset from both N and post()
	for (NodeSetTy::iterator i = nset.begin(), e = nset.end(); i != e; ++i) {
		for (NodeSetTy::iterator si = (*i)->succNode.begin(),
			se = (*i)->succNode.end(); si != se; ++si) 
			if(nset.count(*si) == 0) {
				n->succNode.insert(*si);
				(*si)->predNode.insert(n);
				(*si)->predNode.erase(*i);
			}
	 
		for (NodeSetTy::iterator pi = (*i)->predNode.begin(),
			pe = (*i)->predNode.end(); pi != pe; ++pi) 
			if(nset.count(*pi) == 0) {
				n->predNode.insert(*pi);
				(*pi)->succNode.insert(n);
				(*pi)->succNode.erase(*i);
			}

		N.erase(*i);
	}
}

// mapNode2BB - Return the corresponding ir::BasicBlock * of the node
ir::ControlFlowGraph::iterator StructuralAnalysis::mapNode2BB(NodeTy *node) {
	NodeTy *tmpNode = node;
	ir::ControlFlowGraph::iterator bb;

	while (tmpNode->isCombined) 
		tmpNode = tmpNode->entryNode;
	
	bb = tmpNode->BB;
	
	return bb; 
}

// mapBB2Node - Return the corresponding sturcture node of the basic block
StructuralAnalysis::NodeTy * StructuralAnalysis::mapBB2Node(
	ir::ControlFlowGraph::iterator bb) {
	NodeTy *node, *tmpNode;

	node = BB2NodeMap[bb];

	while ((tmpNode = node->parentNode) != NULL) 
		node = tmpNode;
	
	return node; 
}

// dumpCTNode - dump one CT node
void StructuralAnalysis::dumpCTNode(std::ostream& stream,
	NodeTy *node) const {
	if(!node->isCombined) return;

	stream << "\t"; 

	switch (node->nodeType) {
	case Block:
		stream << "Block      ";
		break;
	case IfThen:
		stream << "IfThen     ";
		break;
	case IfThenElse:
		stream << "IfThenElse ";
		break;
	case Case:
		stream << "Case       ";
		break;
	case SelfLoop:
		stream << "SelfLoop   ";
		break;
	case NaturalLoop:
		stream << "NaturalLoop";
		break;
	default:
		break;
	}
	
	stream << "\t";

	dumpNode(stream, node);

	stream << '\n';

	for (NodeSetTy::iterator i = node->childNode.begin(),
		e = node->childNode.end();
		i != e; ++i) {
		dumpCTNode(stream, *i);
	}
}

// dumpNode - dump one node
void StructuralAnalysis::dumpNode(std::ostream& stream, NodeTy *node) const {
	BBSetTy *BBVec = new BBSetTy;

	findBB(node, BBVec);

	for (BBSetTy::iterator i = BBVec->begin(), e = BBVec->end(); i != e; ++i)
		stream << (*i)->label() << "\t";
}

// findUnstructuredBR - Record the branch and remove it from CFG 
void StructuralAnalysis::findUnstructuredBR(NodeSetTy &N, NodeTy *srcNode,
	NodeTy *dstNode, bool needForwardCopy, bool isGoto) {
	BBSetTy *srcNodeVec = new BBSetTy;
	BBSetTy *dstNodeVec = new BBSetTy;

	findBB(srcNode, srcNodeVec);
	findBB(dstNode, dstNodeVec);

	for (BBSetTy::iterator SRCI = srcNodeVec->begin(), SRCE = srcNodeVec->end();
		SRCI != SRCE; ++SRCI) {
		ir::ControlFlowGraph::iterator srcBB= *SRCI;

		ir::ControlFlowGraph::BlockPointerVector SuccVec = srcBB->successors;

		for (ir::ControlFlowGraph::BlockPointerVector::iterator
			SI = SuccVec.begin(), E = SuccVec.end(); SI != E; ++SI) {
			ir::ControlFlowGraph::iterator succ = *SI;

			for (BBSetTy::iterator DSTI = dstNodeVec->begin(),
				DSTE = dstNodeVec->end(); DSTI != DSTE; ++DSTI) {
				ir::ControlFlowGraph::iterator dstBB = *DSTI;

				if (*DSTI == succ) {
					if (isGoto) {
						if (checkUnique(unstructuredBRVec, srcBB, dstBB))
							unstructuredBRVec.push_back(
								std::make_pair(srcBB, dstBB));
					}
					
					if (needForwardCopy) {
						if (checkUnique(
							dstNode->incomingForwardBR, srcBB, dstBB))
							dstNode->incomingForwardBR.push_back(
								std::make_pair(srcBB, dstBB));
					}
				}
			}
		}
	}	 

	srcNode->succNode.erase(dstNode);
	dstNode->predNode.erase(srcNode);

	if (edge2ClassMap[std::make_pair(srcNode, dstNode)] == BACK) {
		dstNode->isLoopHeader = false;
		dstNode->loopExitNode = NULL;
	}
}

// findBB - put all Basic Blocks in node into nodeVec
void StructuralAnalysis::findBB(NodeTy *node, BBSetTy *nodeVec) const {
	if (!node->isCombined) 
		nodeVec->insert(node->BB);
	else {
		NodeSetTy nodeSet = node->childNode;

		for (NodeSetTy::const_iterator i = nodeSet.begin(),
			e = nodeSet.end(); i != e; ++i) 
			findBB(*i, nodeVec);		
	}
}

// dumpUnstructuredBR - Dump all found unstructured branches
void StructuralAnalysis::dumpUnstructuredBR(std::ostream& stream) const {
	stream << "\nUnstructured Branches:\n";
	
	for (EdgeVecTy::const_iterator i = unstructuredBRVec.begin(),
		e = unstructuredBRVec.end(); i != e; ++i) {
		 stream << "\t" << i->first->label()
		 	<< "\t" << i->second->label() << "\n";
	}
	
	stream << "\n";
}

//True if after erasing edge src->dst, dst is still reachable from entry
bool StructuralAnalysis::isStillReachableFromEntry(NodeSetTy &N,
	NodeTy *entry, NodeTy *dstNode, NodeTy *srcNode) {
	visitPath.clear();

	return path(entry, dstNode, N, srcNode, dstNode);
}

//findEntryBB - find the entry Basic Block of the node
ir::ControlFlowGraph::iterator StructuralAnalysis::findEntryBB(NodeTy *node) {
	if (!node->isCombined) 
		return node->BB;
	else 
		return findEntryBB(node->entryNode);
}

void StructuralAnalysis::cleanupUnreachable() {
	for (NodeSetTy::iterator i = unreachableNodeSet.begin(),
		e = unreachableNodeSet.end(); i != e; ++i) {
		cleanup(*i);
	}
}

// clean - fill in the element of incoming branches and outgoing branches
void StructuralAnalysis::cleanup(NodeTy *node) {
	if (!node->isCombined) 
		return;
	else {
		if ((node->nodeType == NaturalLoop || node->nodeType == SelfLoop)
			&& node->containedBB.size() > 1) {
			for (BBSetTy::iterator i = node->containedBB.begin(),
				e = node->containedBB.end(); i != e; ++i) {
				ir::ControlFlowGraph::iterator BB = *i;

				if (BB != node->entryBB) {
					ir::ControlFlowGraph::BlockPointerVector
						PredVec = BB->predecessors;

					for (ir::ControlFlowGraph::BlockPointerVector::iterator
						PI = PredVec.begin(),
						E = PredVec.end(); PI != E; ++PI) {
						ir::ControlFlowGraph::iterator Pred = *PI;
 
						 if (node->containedBB.count(Pred) == 0)
							 node->incomingBR.push_back(
							 	std::make_pair(Pred, BB));
					}

					ir::ControlFlowGraph::BlockPointerVector
						SuccVec = BB->successors;

					for (ir::ControlFlowGraph::BlockPointerVector::iterator
						SI = SuccVec.begin(),
						E = SuccVec.end(); SI != E; ++SI) {
						ir::ControlFlowGraph::iterator Succ = *SI;

						if (node->containedBB.count(Succ) == 0
							&& Succ != node->exitBB)
							node->outgoingBR.push_back(
								std::make_pair(BB, Succ));
					}
				}
			}
		}

		NodeSetTy nodeSet = node->childNode;

		for (NodeSetTy::iterator i = nodeSet.begin(),
			e = nodeSet.end(); i != e; ++i) 
			cleanup(*i);	 
	}
}

// deleteUnreachableNode - delete nodes no longer reachable from the entry
void StructuralAnalysis::deleteUnreachableNodes(NodeSetTy &N, NodeTy *entry) {
	for(NodeSetTy::iterator i = N.begin(), e = N.end(); i != e; ++i) {
		visitPath.clear();
		NodeTy *node = *i;

		if (!path(entry, node, N, NULL)) {

			for (NodeSetTy::iterator pi = node->predNode.begin(),
				pe = node->predNode.end(); pi != pe; ++pi)
				(*pi)->succNode.erase(node);

			for (NodeSetTy::iterator si = node->succNode.begin(),
				se = node->succNode.end(); si != se; ++si)
				(*si)->predNode.erase(node);

			unreachableNodeSet.insert(node);

			N.erase(node);
		}
	}
}

void StructuralAnalysis::reconstructUnreachable() {
BEGIN:
	bool merge = false;

	for (NodeSetTy::iterator i = unreachableNodeSet.begin(),
		e = unreachableNodeSet.end(); i != e; ++i) {
		NodeTy *node1 = *i;

		for (NodeSetTy::iterator ii = unreachableNodeSet.begin(),
			ee = unreachableNodeSet.end(); ii != ee; ++ii) {
			NodeTy *node2 = *i;

			if (node1 == node2) continue;

			for (NodeSetTy::iterator pi = node1->predNode.begin(),
				pe = node1->predNode.end();
				pi != pe; ++pi) {
				NodeTy *pred = *pi;
				 
				if ((pred->isCombined
					&& node2->containedBB.count(pred->entryBB) > 0)
					|| (!pred->isCombined
						&& node2->containedBB.count(pred->BB) > 0)) {
					 pred->succNode.insert(node1);					
					 merge = true;
				}
			}	
		
			for (NodeSetTy::iterator si = node1->succNode.begin(),
				se = node1->succNode.end();
				si != se; ++si) {
				NodeTy *succ = *si;

				if ((succ->isCombined
					&& node2->containedBB.count(succ->entryBB) > 0)
					|| (!succ->isCombined
					&& node2->containedBB.count(succ->BB) > 0)) {
					succ->predNode.insert(node1);					
					merge = true; 
				}
			}	

			if (merge) {
				NodeSetTy nodeSet;
				
				nodeSet.insert(node1);
				nodeSet.insert(node2);

				reduce(unreachableNodeSet, Unreachable, nodeSet, NULL, NULL);

				goto BEGIN;
			} 
		}
	}
}

StructuralAnalysis::~StructuralAnalysis()
{
	for(NodeSetTy::iterator n = Net.begin(); n != Net.end(); ++n)
	{
		delete *n;
	}
}

void StructuralAnalysis::analyze(ir::IRKernel& k) {
	_kernel = static_cast<ir::PTXKernel*>(&k);

 // build a Simple CFG out of the PTX CFG
	buildSimpleCFG(Net);

	NodeTy *entry = BB2NodeMap[_kernel->cfg()->get_entry_block()];

	bool debug = false;
	
	// Follow the Fig 7.39 of Muchnick book
	structuralAnalysis(Net, entry, debug);

	cleanup(*(Net.begin()));

	reconstructUnreachable();

	cleanupUnreachable();

//		dumpCTNode(std::cout, *(Net.begin()));

//		dumpUnstructuredBR(std::cout);

}

void StructuralAnalysis::write(std::ostream& stream) const {
	stream << _kernel->name <<":\n";

	dumpCTNode(stream, *(Net.begin()));

	dumpUnstructuredBR(stream);
}

bool StructuralAnalysis::checkUnique(EdgeVecTy &edgeVec,
	ir::ControlFlowGraph::iterator srcBB,
	ir::ControlFlowGraph::iterator dstBB) {
	for (EdgeVecTy::iterator i = edgeVec.begin(),
			e = edgeVec.end(); i != e; ++i) {

		if (i->first == srcBB && i->second == dstBB)
			return false;
	}

	return true;
}

}

