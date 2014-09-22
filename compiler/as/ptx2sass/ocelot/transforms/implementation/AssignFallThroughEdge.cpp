//- StructuralAnalysis.h - Structural Analysis - *C++ -*-// 
// 
//                     
// 
//===----------------------------------------------------------------------===// 
// 
// This file defines the class of Structural Analysis which will return the 
// control tree and unstructured branches of a function 
// 
//===----------------------------------------------------------------------===// 
 
#include <ocelot/transforms/interface/AssignFallThroughEdge.h>

namespace transforms {  
  void AssignFallThroughEdge::replaceWithDummyEdge() {
    for(ir::ControlFlowGraph::iterator i = _kernel->cfg()->begin(),
                                       e = _kernel->cfg()->end();
                                       i != e; ++i) {
      ir::BasicBlock::EdgePointerVector edges = i->out_edges;

      for (ir::BasicBlock::EdgePointerVector::iterator ei = edges.begin(),
      	ee = edges.end(); ei != ee; ++ei) {
        ir::ControlFlowGraph::edge_iterator oldEdge = *ei;

        if (oldEdge->type == ir::Edge::Branch
        	|| oldEdge->type == ir::Edge::FallThrough) {
          ir::ControlFlowGraph::iterator oldDest = oldEdge->tail;
          _kernel->cfg()->remove_edge(oldEdge);
          _kernel->cfg()->insert_edge(ir::ControlFlowGraph::Edge(
          	i, oldDest, ir::Edge::Dummy));
        }
      }
    }
  }

  void AssignFallThroughEdge::buildSimpleCFG() {
    for(ir::ControlFlowGraph::iterator i = _kernel->cfg()->begin(),
                                       e = _kernel->cfg()->end();
                                       i != e; ++i) {
      NodeCFGTy *n = new NodeCFGTy;
      n->BB = i;
      N.insert(n);
      BB2NodeMap[i] = n;
    }

    for(ir::ControlFlowGraph::iterator i = _kernel->cfg()->begin(),
                                       e = _kernel->cfg()->end();
                                       i != e; ++i) {
      NodeCFGTy *n = BB2NodeMap[i];

      ir::ControlFlowGraph::BlockPointerVector PredVec = i->predecessors;

      for (ir::ControlFlowGraph::BlockPointerVector::iterator
      	PI = PredVec.begin(), E = PredVec.end(); PI != E; ++PI) {
        NodeCFGTy *p = BB2NodeMap[*PI];
        n->predNode.insert(p);
      }

      ir::ControlFlowGraph::BlockPointerVector SuccVec = i->successors;

      for (ir::ControlFlowGraph::BlockPointerVector::iterator
      	SI = SuccVec.begin(), E = SuccVec.end(); SI != E; ++SI) {
        NodeCFGTy *s = BB2NodeMap[*SI];
        n->succNode.insert(s);
      }
    }
  }

  void AssignFallThroughEdge::DFSPostorder(NodeCFGTy *x) {
    visit[x] = true;
    preTree[x] = ++preMax;

    for (NodeCFGSetTy::iterator i = x->succNode.begin(),
                             e = x->succNode.end();
                             i != e; ++i) {
      NodeCFGTy *y = *i;

      if (visit.count(y) == 0) 
        DFSPostorder(y);
      else if (postTree[y] == 0 || preTree[x] == preTree[y])
        backEdgeVec.push_back(std::make_pair(x, y));
    }

    postMax++;
    postTree[x] = postMax;
  }

  void AssignFallThroughEdge::assignExitNode() {
    for (ir::ControlFlowGraph::iterator i = _kernel->cfg()->begin(),
    	e = _kernel->cfg()->end(); i != e; ++i) {
      ir::ControlFlowGraph::BlockPointerVector SuccVec = i->successors;

      for (ir::ControlFlowGraph::BlockPointerVector::iterator
      	SI = SuccVec.begin(), E = SuccVec.end(); SI != E; ++SI) {
        ir::ControlFlowGraph::iterator succ = *SI;

        if (succ == _kernel->cfg()->get_exit_block()) {
          _kernel->cfg()->remove_edge(i->get_edge(succ));
          _kernel->cfg()->insert_edge(ir::ControlFlowGraph::Edge(
          	i, succ, ir::Edge::FallThrough));
          hasIncomingFallThroughNode.insert(succ);
        }
      } 
    }
  }

  void AssignFallThroughEdge::assignBackEdge() {
    for (EdgeCFGVecTy::iterator i = backEdgeVec.begin(),
    	e = backEdgeVec.end(); i != e; ++i) {
      NodeCFGTy *srcNode = i->first;
      NodeCFGTy *dstNode = i->second;
      ir::ControlFlowGraph::iterator srcBB = srcNode->BB;
      ir::ControlFlowGraph::iterator dstBB = dstNode->BB;

      srcNode->succNode.erase(dstNode);
      dstNode->predNode.erase(srcNode);
      
      ir::ControlFlowGraph::edge_iterator edge = srcBB->get_edge(dstBB); 
      _kernel->cfg()->remove_edge(edge);
      _kernel->cfg()->insert_edge(ir::ControlFlowGraph::Edge(
      	srcBB, dstBB, ir::Edge::Branch));
    }
  }

  void AssignFallThroughEdge::findEntryNode(NodeCFGSetTy &S) {
    for (NodeCFGSetTy::iterator i = N.begin(), e = N.end(); i != e; ++i) {
      if ((*i)->predNode.size() == 0)
        S.insert(*i); 
    }
  }

  AssignFallThroughEdge::NodeCFGTy * AssignFallThroughEdge::pickOneNode(
  	NodeCFGSetTy &S) {
    int min = preMax + 1;
    NodeCFGTy *tmpNode = NULL;

    for (NodeCFGSetTy::iterator i = S.begin(), e = S.end(); i != e; ++i) {
      if (preTree[*i] < min) {
        min = preTree[*i];
        tmpNode = *i;
      }
    }

    return tmpNode;
  }

  void AssignFallThroughEdge::topoSort() {
    NodeCFGSetTy S;

    findEntryNode(S);
    sortedNodes.clear();
  
    while (S.size() > 0) { 
      NodeCFGTy *n = pickOneNode(S);
      S.erase(n);     
 
      sortedNodes.push_back(n);
      sortedVal[n] = sortedMax++;

      for (NodeCFGSetTy::iterator i = n->succNode.begin(),
                               e = n->succNode.end();
                               i != e; ++i) { 
        NodeCFGTy *succ = *i;

        succ->predNode.erase(n); 

        if (succ->predNode.size() == 0) 
          S.insert(succ);
      }
      
      n->succNode.clear(); 
    }
  }

  void AssignFallThroughEdge::adjustBraInst() {
    for(ir::ControlFlowGraph::iterator i = _kernel->cfg()->begin(),
                                       e = _kernel->cfg()->end();
                                       i != e; ++i) {
      ir::PTXInstruction *term =
      	static_cast<ir::PTXInstruction *>(i->instructions.back());

      if (term->opcode == ir::PTXInstruction::Bra) {
        if (i->out_edges.size() == 2) {
          ir::ControlFlowGraph::edge_iterator braEdge = i->get_branch_edge();
          ir::ControlFlowGraph::iterator braDst = braEdge->tail;

          if (braDst->label() != term->d.identifier) {
            term->d = std::move(ir::PTXOperand(braDst->label()));

            switch(term->pg.condition) {
            case ir::PTXOperand::Pred:
              term->pg.condition = ir::PTXOperand::InvPred;
              break;
            case ir::PTXOperand::InvPred:
              term->pg.condition = ir::PTXOperand::Pred;
              break;
            case ir::PTXOperand::PT:
              term->pg.condition = ir::PTXOperand::nPT;
              break;
            case ir::PTXOperand::nPT:
              term->pg.condition = ir::PTXOperand::PT;
              break;
            }
          } 
        } else {
          if (i->has_branch_edge()) { 
            ir::ControlFlowGraph::edge_iterator braEdge = i->get_branch_edge();
            ir::ControlFlowGraph::iterator braDst = braEdge->tail;
            term->d = std::move(ir::PTXOperand(braDst->label()));
            term->uni = true;
          } else {
            i->instructions.pop_back();
          }
        }
      } else {
        if (i->has_branch_edge()) { 
          ir::ControlFlowGraph::edge_iterator braEdge = i->get_branch_edge();
          ir::ControlFlowGraph::iterator braDst = braEdge->tail;

          ir::PTXInstruction* branch =
          	new ir::PTXInstruction(ir::PTXInstruction::Bra);
          branch->uni = true;
          branch->d = std::move(ir::PTXOperand(braDst->label()));
          i->instructions.push_back(branch);
        } 
      }
    }
  }

  void AssignFallThroughEdge::assignEdges() {
    buildSimpleCFG();

    NodeCFGTy *entry = BB2NodeMap[_kernel->cfg()->get_entry_block()];

    postMax = preMax = 0;

    DFSPostorder(entry);

    hasIncomingFallThroughNode.clear();

    assignExitNode();

    assignBackEdge();

    sortedMax = 0;

    topoSort();

    for (NodeCFGVecTy::iterator i = sortedNodes.begin(),
    	e = sortedNodes.end(); i != e; ++i) {
      NodeCFGTy *node = *i;
      ir::ControlFlowGraph::iterator BB = node->BB;
      ir::BasicBlock::EdgePointerVector edges = BB->out_edges;

      if (edges.size() == 2) {
        ir::BasicBlock::EdgePointerVector::iterator ei = edges.begin();
        ir::ControlFlowGraph::edge_iterator edge1 = *ei;
        ++ei;
        ir::ControlFlowGraph::edge_iterator edge2 = *ei;

        if (edge1->type == ir::Edge::FallThrough) {
          _kernel->cfg()->remove_edge(edge2);
          _kernel->cfg()->insert_edge(ir::ControlFlowGraph::Edge(BB,
          	edge2->tail, ir::Edge::Branch));
        } else if (edge2->type == ir::Edge::FallThrough) {
          _kernel->cfg()->remove_edge(edge1);
          _kernel->cfg()->insert_edge(ir::ControlFlowGraph::Edge(
          	BB, edge1->tail, ir::Edge::Branch));
        } else if (edge1->type == ir::Edge::Dummy
        	&& edge2->type == ir::Edge::Dummy) {
          ir::ControlFlowGraph::iterator dstBB1 = edge1->tail;
          NodeCFGTy *dstNode1 = BB2NodeMap[dstBB1];
          ir::ControlFlowGraph::iterator dstBB2 = edge2->tail;
          NodeCFGTy *dstNode2 = BB2NodeMap[dstBB2];
    
          ir::ControlFlowGraph::iterator closeBB, remoteBB;
    
          if (sortedVal[dstNode1] < sortedVal[dstNode2]) {
            closeBB = dstBB1;
            remoteBB = dstBB2;
          } else {
            closeBB = dstBB2;
            remoteBB = dstBB1;
          }
    
          _kernel->cfg()->remove_edge(edge1);
          _kernel->cfg()->remove_edge(edge2);
      
          if (hasIncomingFallThroughNode.count(closeBB) == 0) {
            _kernel->cfg()->insert_edge(ir::ControlFlowGraph::Edge(BB,
            	closeBB, ir::Edge::FallThrough));
            _kernel->cfg()->insert_edge(ir::ControlFlowGraph::Edge(BB,
            	remoteBB, ir::Edge::Branch));
            hasIncomingFallThroughNode.insert(closeBB); 
          } else if (hasIncomingFallThroughNode.count(remoteBB) == 0) {
            _kernel->cfg()->insert_edge(ir::ControlFlowGraph::Edge(BB,
            	remoteBB, ir::Edge::FallThrough));
            _kernel->cfg()->insert_edge(ir::ControlFlowGraph::Edge(BB,
            	closeBB, ir::Edge::Branch));
            hasIncomingFallThroughNode.insert(remoteBB); 
          } else {
            ir::ControlFlowGraph::iterator NewBB = _kernel->cfg()->insert_block(
            	ir::BasicBlock(_kernel->cfg()->newId()));
            ir::PTXInstruction* branch = new ir::PTXInstruction(
            	ir::PTXInstruction::Bra);
            branch->uni = true;
            branch->d = std::move(ir::PTXOperand(closeBB->label()));
            NewBB->instructions.push_back(branch);

            _kernel->cfg()->insert_edge(ir::ControlFlowGraph::Edge(
            	NewBB, closeBB, ir::Edge::Branch));
            _kernel->cfg()->insert_edge(ir::ControlFlowGraph::Edge(
            	BB, NewBB, ir::Edge::FallThrough));
            _kernel->cfg()->insert_edge(ir::ControlFlowGraph::Edge(
            	BB, NewBB, ir::Edge::Branch));
          }
        } else {
          ir::ControlFlowGraph::iterator remainBB;

          if (edge1->type == ir::Edge::Branch) {
            remainBB = edge2->tail;
            _kernel->cfg()->remove_edge(edge2);
          }
          else if (edge2->type == ir::Edge::Branch) {
            remainBB = edge1->tail;          
            _kernel->cfg()->remove_edge(edge1);
          }

          if (hasIncomingFallThroughNode.count(remainBB) == 0) {
            _kernel->cfg()->insert_edge(ir::ControlFlowGraph::Edge(BB,
            	remainBB, ir::Edge::FallThrough));
            hasIncomingFallThroughNode.insert(remainBB); 
          } else {
            ir::ControlFlowGraph::iterator NewBB = _kernel->cfg()->insert_block(
            	ir::BasicBlock(_kernel->cfg()->newId()));
            ir::PTXInstruction* branch = new ir::PTXInstruction(
            	ir::PTXInstruction::Bra);
            branch->uni = true;
            branch->d = std::move(ir::PTXOperand(remainBB->label()));
            NewBB->instructions.push_back(branch);

            _kernel->cfg()->insert_edge(ir::ControlFlowGraph::Edge(
            	NewBB, remainBB, ir::Edge::Branch));
            _kernel->cfg()->insert_edge(ir::ControlFlowGraph::Edge(
            	BB, NewBB, ir::Edge::FallThrough));
          }
        }
      } else if (edges.size() == 1) {
        ir::ControlFlowGraph::edge_iterator edge = *(edges.begin());

        if (edge->type == ir::Edge::Dummy) {
          ir::ControlFlowGraph::iterator dstBB = edge->tail;
          _kernel->cfg()->remove_edge(edge);

          if (hasIncomingFallThroughNode.count(dstBB) == 0) {
            _kernel->cfg()->insert_edge(ir::ControlFlowGraph::Edge(BB,
            	dstBB, ir::Edge::FallThrough));
            hasIncomingFallThroughNode.insert(dstBB); 
          } else {
            _kernel->cfg()->insert_edge(ir::ControlFlowGraph::Edge(BB,
            	dstBB, ir::Edge::Branch));
            hasIncomingFallThroughNode.insert(dstBB); 
          }
        }
      }
    } 

    adjustBraInst(); 
  } 
}
