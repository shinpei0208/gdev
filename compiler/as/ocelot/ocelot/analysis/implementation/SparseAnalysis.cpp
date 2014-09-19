/*! \file AffineAnalysis.h
	\date May 21, 2011
	\author Fernando Pereira <fpereira@dcc.ufmg.br>
	\brief The source file for the SparseAnalysis class
 */

#include <set>
#include <map>
#include <vector>
#include <iostream>

#include <ocelot/analysis/interface/DataflowGraph.h>
#include <ocelot/analysis/interface/AffineAnalysis.h>
#include <hydrazine/interface/debug.h>

namespace analysis {

	unsigned PGNode::node_counter = 0;

	/*! \brief Constructor. Initalized the fields of the object.
	 */
	SparseAnalysis::SparseAnalysis
		(DataflowGraph* dfg, TransferFunction* tf) :
		_tf(tf) {
			buildPropagationGraph(dfg);
	}

	/*! \brief Creates the propagation graph. First reads instructions and
		phi-functions from the DFG, then build the def-use chains between these
		instructions. The code duplication in the implementation is happening
		because Ocelot does not use the same type for instructions and
		phi-functions.
	 */
	void SparseAnalysis::buildPropagationGraph(DataflowGraph* dfg)
	{
		DataflowGraph::const_iterator i_block = dfg->begin();
		DataflowGraph::const_iterator e_block = dfg->end();
		for (; i_block != e_block; i_block++) {
			// Start by adding the phi functions into the propagation graph:
			DataflowGraph::PhiInstructionVector::const_iterator
				i_phi = i_block->phis().begin();
			DataflowGraph::PhiInstructionVector::const_iterator
				e_phi = i_block->phis().end();
			for (; i_phi != e_phi; i_phi++) {
				// Create a new node, and update the map of variables to nodes.
				DataflowGraph::RegisterId r = i_phi->d.id;
				assert(_def_map.find(r) == _def_map.end());
				PGNode* pgn = new PGPhi(&*i_phi);
				_def_map[r] = pgn;
				_propagation_graph.insert(pgn);
				_workqueue.push(pgn);
			}
			// Now, add the ordinary instructions into the propagation graph:
			DataflowGraph::InstructionVector::const_iterator
				i_inst = i_block->instructions().begin();
			DataflowGraph::InstructionVector::const_iterator
				e_inst = i_block->instructions().end();
			for (; i_inst != e_inst; ++i_inst) {
				PGNode* pgn = new PGInst(&*i_inst);
				_propagation_graph.insert(pgn);
				_workqueue.push(pgn);
				// Update the map of variables to nodes. An instruction might define
				// many registers, so we must go over the vector of definitions.
				DataflowGraph::RegisterPointerVector::const_iterator i_r =
					i_inst->d.begin();
				DataflowGraph::RegisterPointerVector::const_iterator e_r =
					i_inst->d.end();
				for (; i_r != e_r; i_r++) {
					DataflowGraph::RegisterId r = *(i_r->pointer);
					assert(_def_map.find(r) == _def_map.end());
					_def_map[r] = pgn;
				}
			}
		}
		// Go over the nodes in the propagation graph, linking sources to
		// destinations based on def-use relations among variables:
		std::set<PGNode*>::iterator i_pgn = _propagation_graph.begin();
		std::set<PGNode*>::iterator e_pgn = _propagation_graph.end();
		for (; i_pgn != e_pgn; i_pgn++) {
			PGNode* pgn = *i_pgn;
			if (typeid(PGPhi) == typeid(*pgn)) {
				PGPhi *pgphi = static_cast<PGPhi*> (pgn);
				DataflowGraph::RegisterVector::const_iterator i =
					pgphi->_phi->s.begin();
				DataflowGraph:: RegisterVector::const_iterator e =
					pgphi->_phi->s.end();
				for (; i != e; i++) {
					DataflowGraph::RegisterId r = i->id;
					assertM(_def_map.find(r) != _def_map.end(),
							"Variable " << r << " undefined");
					PGNode* pgn_pred = _def_map[r];
					pgn_pred->add_successor(pgn);
				}
			} else {
				assert(typeid(PGInst) == typeid(*pgn));
				PGInst *pginst = static_cast<PGInst*> (pgn);
				DataflowGraph::RegisterPointerVector::const_iterator i =
					pginst->_inst->s.begin();
				DataflowGraph::RegisterPointerVector::const_iterator e =
					pginst->_inst->s.end();
				for (; i != e; i++) {
					DataflowGraph::RegisterId r = *(i->pointer);
					assertM(_def_map.find(r) != _def_map.end(),
							"Variable " << r << " undefined");
					PGNode* pgn_pred = _def_map[r];
					pgn_pred->add_successor(pgn);
				}
			}
		}
	}

	/*! \brief Generate a dot representation of the propagation graph.
	 */
	ostream& SparseAnalysis::printPropagationGraphInDot(ostream &out) const {
		std::set<PGNode*>::iterator i_pgn = _propagation_graph.begin();
		std::set<PGNode*>::iterator e_pgn = _propagation_graph.end();
		out << "digraph PropagationGraph {\n";
		for (; i_pgn != e_pgn; i_pgn++) {
			PGNode* pgn = *i_pgn;
			out << "	" << pgn->getNodeId() << " [label=\"" << *pgn << "\"]\n";
			set<PGNode*>::const_iterator i_succ = pgn->succ_begin();
			set<PGNode*>::const_iterator e_succ = pgn->succ_end();
			for (; i_succ != e_succ; i_succ++) {
				out << "	" << pgn->getNodeId() << " -> "
				<< (*i_succ)->getNodeId() << std::endl;
			}
		}
		out << "}\n";
		return out;
	}

	ostream& SparseAnalysis::operator<<(ostream& out) const{
		return printPropagationGraphInDot(out);
	}

	/*! \brief This algorithm finds a fix point to the set of constraints in
		this instance of sparse analysis.
	 */
	void SparseAnalysis::iterateToFixPoint() {
		while (!_workqueue.empty()) {
			PGNode* pgn = _workqueue.front();
			_workqueue.pop();
			pgn->_isActive = false;
			// Update the state of the variable defined by the instruction:
			bool hasChanged = false;
			if (typeid(PGPhi) == typeid(*pgn)) {
				PGPhi *pgphi = static_cast<PGPhi*> (pgn);
				if ((*_tf)(*(pgphi->_phi))) {
					hasChanged = true;
				}
			} else {
				assert(typeid(PGInst) == typeid(*pgn));
				PGInst *pgphi = static_cast<PGInst*> (pgn);
				if ((*_tf)(*(pgphi->_inst))) {
					hasChanged = true;
				}
			}
			if (hasChanged) {
				std::set<PGNode*>::iterator i_succ = pgn->succ_begin();
				std::set<PGNode*>::iterator e_succ = pgn->succ_end();
				for (; i_succ != e_succ; i_succ++) {
					PGNode* succ = *i_succ;
					if ( ! succ->_isActive ) {
						succ->_isActive = true;
						_workqueue.push(succ);
					}
				}
			}
		}
	}

	std::ostream& operator<<( std::ostream& out, const PGNode& inst ) {
		return inst.toStream(out);
	}

}
