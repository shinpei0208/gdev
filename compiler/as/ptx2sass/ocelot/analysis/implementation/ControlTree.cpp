/*! \file ControlTree.cpp
 *  \author Rodrigo Dominguez <rdomingu@ece.neu.edu>
 *  \date June 10, 2010
 *  \brief The source file for the ControlTree class.
 */

// Ocelot includes
#include <ocelot/analysis/interface/ControlTree.h>
#include <ocelot/ir/interface/Instruction.h>
#include <ocelot/ir/interface/IRKernel.h>

// Hydrazine includes
#include <hydrazine/interface/string.h>
#include <hydrazine/interface/debug.h>

// STL includes
#include <functional>
#include <algorithm>
#include <queue>
#include <iterator>

// Boost includes
#include <boost/bind.hpp>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace analysis
{
	ControlTree::ControlTree() 
		: 
			_nodes(NodeVector()),
			_post(NodeList()),
			_visit(NodeSet()),
			_root(0)
	{
		
	}

	ControlTree::~ControlTree()
	{
		for (NodeVector::iterator node = _nodes.begin() ; 
				node != _nodes.end() ; ++node) 
		{
			delete *node;
		}
		
		_nodes.clear();
	}
	
	void ControlTree::analyze(ir::IRKernel& k)
	{
		auto cfg = k.cfg();
	
		Node* start = 0;
		Node* end = 0;
		std::unordered_map<CFG::const_iterator, Node*> bmap;

		CFG::const_iterator bb;
		for (bb = cfg->begin() ; bb != cfg->end() ; bb++)
		{
			report("Inserting node " << bb->label());
			Node *node = _insert_node(new InstNode(bb));
			bmap[bb] = node;
		}

		start = bmap[cfg->get_entry_block()];
		end = bmap[cfg->get_exit_block()];

		CFG::const_edge_iterator e;
		for (e = cfg->edges_begin() ; e != cfg->edges_end() ; e++)
		{
			CFG::const_iterator h = e->head;
			CFG::const_iterator t = e->tail;

			bmap[h]->succs().insert(bmap[t]);
			bmap[t]->preds().insert(bmap[h]);

			if (e->type == CFG::Edge::FallThrough)
			{
				report("Add edge " << h->label() << " --> " << t->label());
				bmap[h]->fallthrough() = bmap[t];
			} else
			{
				report("Add edge " << h->label() << " -> " << t->label());
			}
		}

		assertM(start->preds().size() == 0, "Start shouldn't have predecessor");
		assertM(end->succs().size() == 0, "End shouldn't have successor");

		_structural_analysis(start);
	}

	ControlTree::Node* ControlTree::_insert_node(Node* node)
	{
		_nodes.push_back(node);
		return node;
	}

	ControlTree::Node::Node(const std::string& label, RegionType rtype, 
			const NodeList& children) 
		: _label(label), _rtype(rtype), _children(children), _fallthrough(0)
	{
	}

	ControlTree::Node::~Node()
	{
	}

	const std::string& ControlTree::Node::label() const
	{
		return _label;
	}

	ControlTree::RegionType ControlTree::Node::rtype() const
	{
		return _rtype;
	}

	const ControlTree::NodeList& ControlTree::Node::children() const
	{
		return _children;
	}

	ControlTree::NodeSet& ControlTree::Node::succs()
	{
		return _succs;
	}

	ControlTree::NodeSet& ControlTree::Node::preds()
	{
		return _preds;
	}

	ControlTree::Node*& ControlTree::Node::fallthrough()
	{
		return _fallthrough;
	}

	bool ControlTree::Node::has_branch_edge() const
	{
		if (_succs.size() > 1 || (_succs.size() == 1 && _fallthrough == 0)) 
			return true;

		return false;
	}

	ControlTree::Edge ControlTree::Node::get_branch_edge()
	{
		assertM(has_branch_edge(), "The node has no branch edge");

		if (*(_succs.begin()) != _fallthrough) 
			return Edge(this, *(_succs.begin()));

		return Edge(this, *(++_succs.begin()));
	}

	std::ostream& ControlTree::write(std::ostream& out) const
	{
		std::unordered_map<Node*, unsigned int> bmap;

		out << "digraph {" << std::endl;

		// emit nodes
		out << "  // nodes" << std::endl;

		int i = 0;
		for (NodeVector::const_iterator n = _nodes.begin() ; n != _nodes.end() ; 
				++n, ++i)
		{
			bmap[*n] = i;
			if ((*n)->rtype() == Inst)
			{
				out << "  bb_" << i;
				out	<< " [shape=record,label=\"{" << (*n)->label();

				// emit instructions
				InstNode* m = static_cast<InstNode *>(*n);
				for (InstructionList::const_iterator 
						ins = m->bb()->instructions.begin(), 
						end = m->bb()->instructions.end() ;
						ins != end ; ins++)
				{
					out << " | " << 
						hydrazine::toGraphVizParsableLabel((*ins)->toString());
				}

				out << "}\"];";
			} else 
			{
				out << "  bb_" << i;
				out << " [label=\"" << (*n)->label() << "\"];";
			}
			out << std::endl;
		}

		// emit edges
		out << std::endl << "  // edges" << std::endl;

		for (NodeVector::const_iterator n = _nodes.begin() ; 
				n != _nodes.end() ; ++n)
		{
			NodeList children = (*n)->children();
			NodeList::const_iterator child;
			for (child = children.begin() ; child != children.end() ; child++)
			{
				out << "  bb_" << bmap[*n];
				out	<< " -> ";
				out	<< "bb_" << bmap[*child];
				out << ";" << std::endl;
			}
		}
		out << "}" << std::endl;

		return out;
	}

	const ControlTree::Node* ControlTree::get_root_node() const
	{
		return _root;
	}

	ControlTree::InstNode::InstNode(const CFG::const_iterator& bb)
		: Node(bb->label(), Inst, NodeList()), _bb(bb)
	{
	}

	const CFG::const_iterator& ControlTree::InstNode::bb() const
	{
		return _bb;
	}

	ControlTree::BlockNode::BlockNode(const std::string& label, 
			const NodeList& children) : Node(label, Block, children)
	{
	}

	ControlTree::IfThenNode::IfThenNode(const std::string& label, 
			Node* cond, Node* ifTrue, Node* ifFalse) 
		: Node(label, IfThen, buildChildren(cond, ifTrue, ifFalse))
	{
	}

	const ControlTree::NodeList ControlTree::IfThenNode::buildChildren(
			Node* cond, Node* ifTrue, Node* ifFalse) const
	{
		assert(cond != NULL && ifTrue != NULL);
		
		NodeList children;
		children.push_back(cond);
		children.push_back(ifTrue);
		if (ifFalse != NULL) children.push_back(ifFalse);

		return children;
	}

	ControlTree::Node* ControlTree::IfThenNode::cond() const
	{
		return children().front();
	}

	ControlTree::Node* ControlTree::IfThenNode::ifTrue() const
	{
		return *(++(children().begin()));
	}

	ControlTree::Node* ControlTree::IfThenNode::ifFalse() const
	{
		if (children().size() == 3) return children().back();
		else return NULL;
	}

	ControlTree::NaturalNode::NaturalNode(const std::string& label, 
			const NodeList& children) : Node(label, Natural, children)
	{
	}

	void ControlTree::_dfs_postorder(Node* x)
	{
		_visit.insert(x);

		NodeSet::iterator y;
		for (y = x->succs().begin() ; y != x->succs().end() ; y++)
		{
			if (_visit.find(*y) != _visit.end()) continue;
			_dfs_postorder(*y);
		}
		_post.push_back(x);
		report("_dfs_postorder: Added " << x->label());
	}

	ControlTree::Node* ControlTree::_acyclic_region_type(Node* node, 
			NodeSet& nset)
	{
		Node* n;
		bool p, s;
		NodeList nodes; // TODO Implement nodes as an ordered set

		nset.clear();

		// check for a Block containing node
		n = node;
		p = true;
		s = (n->succs().size() == 1);

		while (p && s)
		{
			if (nset.insert(n).second) nodes.push_back(n);
			n = *(n->succs().begin());
			p = (n->preds().size() == 1);
			s = (n->succs().size() == 1);
		}

		if (p)
		{
			if (nset.insert(n).second) nodes.push_back(n);
		}

		n = node;
		p = (n->preds().size() == 1);
		s = true;

		while (p && s)
		{
			if (nset.insert(n).second) nodes.push_front(n);
			n = *(n->preds().begin());
			p = (n->preds().size() == 1);
			s = (n->succs().size() == 1);
		}

		if (s)
		{
			if (nset.insert(n).second) nodes.push_front(n);
		}

		node = n;
		if (nodes.size() >= 2)
		{
			std::string label("BlockNode_");

			std::stringstream ss;
			ss << _nodes.size();
			label += ss.str();

			report("Found " << label << ": " << nodes.front()->label() << "..."
					<< nodes.back()->label());

			return _insert_node(new BlockNode(label, nodes));
		} else if (node->succs().size() == 2)
		{
			Node *m;

			m = *(node->succs().begin());
			n = *(++(node->succs().begin()));

			// check for an IfThen (if node then n)
			if (n->succs().size() == 1 && n->preds().size() == 1 &&
					*(n->succs().begin()) == m)
			{
				nset.clear(); nset.insert(node); nset.insert(n);

				std::string label("IfThenNode_");

				std::stringstream ss;
				ss << _nodes.size();
				label += ss.str();

				report("Found " << label << ":" << " if " << node->label() 
						<< " then " << n->label());

				return _insert_node(new IfThenNode(label, node, n));
			}

			// check for an IfThen (if node then m)
			if (m->succs().size() == 1 && m->preds().size() == 1 &&
					*(m->succs().begin()) == n)
			{
				nset.clear(); nset.insert(node); nset.insert(m);

				std::string label("IfThenNode_");

				std::stringstream ss;
				ss << _nodes.size();
				label += ss.str();

				report("Found " << label << ":" << " if " << node->label()
						<< " then " << m->label());

				return _insert_node(new IfThenNode(label, node, m));
			}

			// check for an IfThen (if node then n else m)
			if (m->succs().size() == 1 && n->succs().size() == 1 &&
					m->preds().size() == 1 && n->preds().size() == 1 &&
					*(m->succs().begin()) == *(n->succs().begin()) &&
					node->fallthrough() == n)
			{
				nset.clear(); nset.insert(node); nset.insert(n); nset.insert(m);

				std::string label("IfThenNode_");

				std::stringstream ss;
				ss << _nodes.size();
				label += ss.str();

				report("Found " << label << ":" << " if " << node->label()
						<< " then " << n->label() << " else " << m->label());

				return _insert_node(new IfThenNode(label, node, n, m));
			}

			// check for an IfThen (if node then m else n)
			if (m->succs().size() == 1 && n->succs().size() == 1 &&
					m->preds().size() == 1 && n->preds().size() == 1 &&
					*(m->succs().begin()) == *(n->succs().begin()) &&
					node->fallthrough() == m)
			{
				nset.clear(); nset.insert(node); nset.insert(m); nset.insert(n);

				std::string label("IfThenNode_");

				std::stringstream ss;
				ss << _nodes.size();
				label += ss.str();

				report("Found " << label << ":" << " if " << node->label()
						<< " then " << m->label() << " else " << n->label());

				return _insert_node(new IfThenNode(label, node, m, n));
			}
		}

		report("Couldn't find any acyclic regions");
		return NULL;
	}

	bool ControlTree::_isCyclic(Node* node)
	{
		if (node->rtype() == Natural) return true;

		return false;
	}

	bool ControlTree::_isBackedge(const Edge& edge)
	{
		const Node* head = edge.first;
		const Node* tail = edge.second;

		// head->tail is a back-edge if tail dominates head
		// (tail dominates head if head appears first in the _post list)
		const Node* match[] = {head, tail};
		NodeList::iterator n = 
			find_first_of(_post.begin(), _post.end(), match, match + 2);
		
		if (*n == head) return true;
		if (*n == tail) return false;

		assertM(false, "Neither head nor tail are valid nodes");

		return false;
	}

	void ControlTree::_compact(Node* node, NodeSet nodeSet)
	{
		NodeList::iterator n, pos;
		for (n = _post.begin() ; n != _post.end() && !nodeSet.empty() ; )
		{
			if (!nodeSet.erase(*n))
			{
				n++;
				continue;
			}

			n = _post.erase(n);
			pos = n;
		}

		_postCtr = _post.insert(pos, node);
	}

	void ControlTree::_reduce(Node* node, NodeSet nodeSet)
	{
		NodeSet::iterator n;
		for (n = nodeSet.begin() ; n != nodeSet.end() ; n++)
		{
			NodeSet::iterator p;
			for (p = (*n)->preds().begin() ; p != (*n)->preds().end() ; p++)
			{
				// ignore edges between nodeSet nodes
				// (except for back-edges which we check below)
				if (nodeSet.find(*p) != nodeSet.end()) continue;

				report("Del " << (*p)->label() << " -> " << (*n)->label());
				(*p)->succs().erase(*n);

				(*p)->succs().insert(node);
				node->preds().insert(*p);

				if ((*p)->fallthrough() == *n)
				{
					report("Add " << (*p)->label() << " --> " << node->label());
					(*p)->fallthrough() = node;
				} else 
				{
					report("Add " << (*p)->label() << " -> " << node->label());
				}
			}

			NodeSet::iterator s;
			for (s = (*n)->succs().begin() ; s != (*n)->succs().end() ; s++)
			{
				// ignore edges between nodeSet nodes
				// (except for back-edges which we check below)
				if (nodeSet.find(*s) != nodeSet.end()) continue;

				report("Del " << (*n)->label() << " -> " << (*s)->label());
				(*s)->preds().erase(*n);

				(*s)->preds().insert(node);
				node->succs().insert(*s);

				if ((*n)->fallthrough() == *s) 
				{
					report("Add " << node->label() << " --> " << (*s)->label());
					node->fallthrough() = *s;
				} else
				{
					report("Add " << node->label() << " -> " << (*s)->label());
				}
			}
		}

		// if not cyclic then check for back-edges between nodeSet nodes
		if (!_isCyclic(node))
		{
			for (n = nodeSet.begin() ; n != nodeSet.end() ; n++)
			{
				bool shouldbreak = false;
				NodeSet::iterator p;
				for (NodeSet::iterator p = (*n)->preds().begin() ; 
						p != (*n)->preds().end() ; ++p)
				{
					if (nodeSet.find(*p) == nodeSet.end()) continue;

					if (_isBackedge(Edge(*p, *n))) 
					{
						// add back-edge region->region
						report("Add " << node->label() << " -> " 
								<< node->label());
						node->preds().insert(node);
						node->succs().insert(node);

						// no need to add more than one back-edge
						shouldbreak = true;
						break;
					}
				}
				if (shouldbreak) break;
			}
		}

		// adjust the postorder traversal
		_compact(node, nodeSet);
	}

	bool ControlTree::_path(Node* m, Node* k, Node* n)
	{
		if (m == n || _visit.find(m) != _visit.end()) return false;
		if (m == k) return true;

		_visit.insert(m);

		for (NodeSet::const_iterator s = m->succs().begin() ;
				s != m->succs().end() ; s++)
		{
			if (_path(*s, k, n)) return true;
		}

		return false;
	}

	bool ControlTree::_path_back(Node* m, Node* n)
	{
		for (NodeSet::const_iterator k = n->preds().begin() ;
				k != n->preds().end() ; k++)
		{
			if (_isBackedge(Edge(*k, n)))
			{
				_visit.clear();
				if (_path(m, *k, n)) return true;
			}
		}

		return false;
	}

	ControlTree::Node* ControlTree::_cyclic_region_type(Node* node, 
			NodeList& nset)
	{
		if (nset.size() == 1)
		{
			if (node->succs().find(node) != node->succs().end())
			{
				// Self loop is a special case of a Natural loop
				std::string label("NaturalNode_");

				std::stringstream ss;
				ss << _nodes.size();
				label += ss.str();

				report("Found " << label << ": " << node->label());
				return _insert_node(new NaturalNode(label, NodeList(1, node)));
			} else
			{
				report("Couldn't find any cyclic regions");
				return NULL;
			}
		}

		for (NodeList::const_iterator m = nset.begin() ;
				m != nset.end() ; m++)
		{
			_visit.clear();
			if (!_path(node, *m)) {
				// it's an Improper region
				// TODO Improper regions are not supported yet
				report("Found Improper region");
				return NULL;
			}
		}

		// check for a Natural loop (this includes While loops)
		NodeList::iterator m;
		for (m = nset.begin() ; m != nset.end() ; ++m)
		{
			if (*m == node && (*m)->preds().size() != 2) break;
			if (*m != node && (*m)->preds().size() != 1) break;
		}

		if (m == nset.end())
		{
			std::string label("NaturalNode_");

			std::stringstream ss;
			ss << _nodes.size();
			label += ss.str();

			report("Found " << label << ": " << nset.front()->label() << "..."
					<< nset.back()->label());

			return _insert_node(new NaturalNode(label, nset));
		}

		report("Couldn't find any cyclic regions");
		return NULL;
	}

	void ControlTree::_structural_analysis(Node* entry)
	{
		Node* n;
		NodeSet nodeSet;
		NodeList reachUnder; // TODO Implement reachUnder as an ordered set
		bool changed;

		do
		{
			report("Starting Structural Analysis...");

			changed = false;

			_post.clear(); 
			_visit.clear();

			report("DFS Postorder");
			_dfs_postorder(entry);

			_postCtr = _post.begin();

			while (_post.size() > 1 && _postCtr != _post.end())
			{
				n = *_postCtr;

				// locate an acyclic region, if present
				report("Looking for acyclic region from " << n->label());
				Node* region = _acyclic_region_type(n, nodeSet);

				if (region != NULL)
				{
					report("Replacing nodeSet for " << region->label());
					_reduce(region, nodeSet);

					changed = true;

					if (nodeSet.find(entry) != nodeSet.end())
					{
						entry = region;
					}
				} else
				{
					// locate a cyclic region, if present
					reachUnder.clear(); nodeSet.clear();

					for (NodeList::const_iterator m = _post.begin() ;
							m != _post.end() ; m++)
					{
						if (*m != n && _path_back(*m, n)) 
						{
							report("Add " << (*m)->label() 
									<< " to reachUnder of " << n->label());
							reachUnder.push_front(*m); nodeSet.insert(*m);
						}
					}
					reachUnder.push_front(n); nodeSet.insert(n);

					report("Looking for cyclic region from " << n->label());
					region = _cyclic_region_type(n, reachUnder);

					if (region != NULL)
					{
						report("Replacing nodeSet for " << region->label());
						_reduce(region, nodeSet);

						changed = true;

						if (nodeSet.find(entry) != nodeSet.end())
						{
							entry = region;
						}
					} else
					{
						_postCtr++;
					}
				}
			}

			if (!changed)
			{
				changed = _forward_copy(entry);
			}

			assertM(changed, "Irreducible CFG");
		} while (_post.size() > 1);

		_root = entry;
	}

	ControlTree::NodeVector ControlTree::_executable_sequence(Node* entry) 
	{
		NodeVector sequence;
		NodeSet unscheduled;

		for(NodeList::iterator i = _post.begin(); i != _post.end(); ++i)
		{
			unscheduled.insert(*i);
		}

		report("Getting executable sequence.");

		sequence.push_back(entry);
		unscheduled.erase(entry);
		report("Added " << entry->label());

		while (!unscheduled.empty()) 
		{
			if (sequence.back()->fallthrough() != 0) 
			{
				Node* tail = sequence.back()->fallthrough();
				sequence.push_back(tail);
				unscheduled.erase(tail);
			}
			else 
			{
				// find a new block, favor branch targets over random blocks
				Node* next = *unscheduled.begin();

				for (NodeSet::iterator succ = sequence.back()->succs().begin() ;
						succ != sequence.back()->succs().end() ; ++succ)
				{
					if (unscheduled.count(*succ) != 0)
					{
						next = *succ;
						break;
					}
				}

				// rewind through fallthrough edges to find the beginning of the 
				// next chain of fall throughs
				report("    Restarting at " << next->label());
				bool rewinding = true;
				while (rewinding) {
					rewinding = false;
					for (NodeSet::iterator pred = next->preds().begin() ;
							pred != next->preds().end() ; ++pred)
					{
						if ((*pred)->fallthrough() == next)
						{
							assertM(unscheduled.count(*pred) != 0,
									(*pred)->label()
									<< " has multiple fallthrough branches.");
							next = *pred;
							report("    Rewinding to " << next->label());
							rewinding = true;
							break;
						}
					}
				}
				sequence.push_back(next);
				unscheduled.erase(next);
			}

			report("Added " << sequence.back()->label());
		}

		return sequence;
	}

	ControlTree::EdgeVector ControlTree::_find_forward_branches()
	{
		assert(_lexical.size() != 0);

		EdgeVector fwdBranches;

		for (NodeVector::iterator node = _lexical.begin() ;
				node != _lexical.end() ; ++node)
		{
			if ((*node)->has_branch_edge())
			{
				Edge branch = (*node)->get_branch_edge();

				NodeVector::iterator head = node;
				NodeVector::iterator tail = find(++head, _lexical.end(), 
						branch.second);

				if (tail != _lexical.end()) 
				{
					report("Found forward branch " << (*node)->label() << " -> " 
							<< (*tail)->label());
					fwdBranches.push_back(branch);
				}
			}
		}

		return fwdBranches;
	}

	bool ControlTree::_lexicographical_compare(const Node* a, const Node* b)
	{
		return (find(_lexical.begin(), _lexical.end(), a) < 
				find(_lexical.begin(), _lexical.end(), b));
	}

	ControlTree::NodeVector ControlTree::_control_graph(const Edge& nb)
	{
		NodeVector::iterator head, tail;

		head = find(_lexical.begin(), _lexical.end(), nb.first);
		tail = find(_lexical.begin(), _lexical.end(), nb.second);

		if (head < tail) return NodeVector(head, ++tail);

		return NodeVector(tail, ++head);
	}

	bool ControlTree::_interact(const NodeVector& CGi0, const NodeVector& CGm0)
	{
		NodeVector result1;
		NodeVector result2;
		NodeVector result3;

		// partial intersection
		std::set_intersection(
				CGi0.begin(), CGi0.end(), 
				CGm0.begin(), CGm0.end(), 
				std::back_inserter(result1), 
				boost::bind(&analysis::ControlTree::_lexicographical_compare, 
					boost::ref(this), _1, _2));
		std::set_difference(
				CGi0.begin(), CGi0.end(), 
				CGm0.begin(), CGm0.end(), 
				std::back_inserter(result2),
				boost::bind(&analysis::ControlTree::_lexicographical_compare, 
					boost::ref(this), _1, _2));
		std::set_difference(
				CGm0.begin(), CGm0.end(), 
				CGi0.begin(), CGi0.end(), 
				std::back_inserter(result3),
				boost::bind(&analysis::ControlTree::_lexicographical_compare, 
					boost::ref(this), _1, _2));

		if (!result1.empty() && !result2.empty() && !result3.empty()) 
			return true;

		return false;
	}

	bool ControlTree::_interact(const EdgeVector::iterator& i0, 
			const EdgeVector::iterator& m0)
	{
		return _interact(_control_graph(*i0), _control_graph(*m0));
	}

	ControlTree::NodeVector ControlTree::_minimal_hammock_graph(const Edge& nb)
	{
		NodeVector mhg = _control_graph(nb);

		// TODO Consider keeping a vector of edges in the class
		for (NodeVector::iterator node = _lexical.begin() ;
				node != _lexical.end() ; ++node)
		{
			if ((*node)->has_branch_edge())
			{
				Edge ib = (*node)->get_branch_edge();
				NodeVector CGib = _control_graph(ib);

				if (_interact(CGib, mhg))
				{
					NodeVector result;
					std::set_union(
							mhg.begin(), mhg.end(), 
							CGib.begin(), CGib.end(), 
							std::back_inserter(result), 
							boost::bind(
								&analysis::ControlTree::_lexicographical_compare, 
								boost::ref(this), _1, _2));
					mhg = result;
				}
			}
		}

		report("MHG = " << mhg.front()->label() << " ... " 
				<< mhg.back()->label());

		return mhg;
	}

	ControlTree::Node* ControlTree::_clone_node(const Node* node)
	{
		report("Clonning " << node->label());

		switch(node->rtype())
		{
			case Inst:
			{
				assert(node->children().size() == 0);
				const InstNode* ifnode = static_cast<const InstNode*>(node);
				return _insert_node(new InstNode(ifnode->bb()));
			}
			case Block:
			{
				NodeList children;
				for (NodeList::const_iterator child = node->children().begin();
						child != node->children().end(); ++child)
				{
					_clone_node(*child);
				}

				const BlockNode* bnode = static_cast<const BlockNode*>(node);
				return _insert_node(new BlockNode(bnode->label(), children));
			}
			case IfThen:
			{
				const IfThenNode* ifnode = static_cast<const IfThenNode*>(node);

				_clone_node(ifnode->cond());
				_clone_node(ifnode->ifTrue());
				if (ifnode->ifFalse() != NULL) 
					_clone_node(ifnode->ifFalse());

				return _insert_node(new IfThenNode(ifnode->label(), 
							ifnode->cond(), ifnode->ifTrue(), 
							ifnode->ifFalse()));
			}
			case Natural:
			{
				NodeList children;
				for (NodeList::const_iterator child = node->children().begin();
						child != node->children().end(); ++child)
				{
					_clone_node(*child);
				}

				const NaturalNode* nnode = 
					static_cast<const NaturalNode*>(node);
				return _insert_node(new NaturalNode(nnode->label(), children));

			}
			default: 
			{
				assertM(false, "Invalid region type " << node->rtype());
			}
		}
	}

	void ControlTree::_forward_copy_transform(const Edge& iFwdBranch, 
			const NodeVector& true_part)
	{
		std::unordered_map<Node*, Node*> bmap;

		for (NodeVector::const_iterator node = true_part.begin() ;
				node != true_part.end() ; ++node)
		{
			// Clone the node
			bmap[*node] = _clone_node(*node);

			// adjust the postorder traversal
			_post.insert(find(_post.begin(), _post.end(), *node), bmap[*node]);
		}

		Node* iFwdNode = iFwdBranch.first;
		NodeVector::const_iterator node = true_part.begin();
		Node* clone = bmap[*node];

		report("Del " << iFwdNode->label() << " -> " << (*node)->label());
		iFwdNode->succs().erase(*node);
		(*node)->preds().erase(iFwdNode);

		report("Add " << iFwdNode->label() << " -> " << clone->label());
		iFwdNode->succs().insert(clone);
		clone->preds().insert(iFwdNode);

		for ( ; node != true_part.end() ; ++node)
		{
			clone = bmap[*node];
			for (NodeSet::iterator s = (*node)->succs().begin() ;
					s != (*node)->succs().end() ; ++s)
			{
				if (find(true_part.begin(), true_part.end(), *s) != 
						true_part.end())
				{
					clone->succs().insert(bmap[*s]);
					bmap[*s]->preds().insert(clone);

					if ((*node)->fallthrough() == *s) 
					{
						report("Add " << clone->label() << " --> " 
								<< (bmap[*s])->label());
						clone->fallthrough() = bmap[*s];
					}
					else
					{
						report("Add " << clone->label() << " -> " 
								<< (bmap[*s])->label());
					}
				}
				else
				{
					clone->succs().insert(*s);
					(*s)->preds().insert(clone);

					if ((*node)->fallthrough() == *s) 
					{
						report("Add " << clone->label() << " --> " 
								<< (*s)->label());
						clone->fallthrough() = *s;
					}
					else
					{
						report("Add " << clone->label() << " -> " 
								<< (*s)->label());
					}
				}
			}
		}
	}

	void ControlTree::_elim_unreach_code(ControlTree::Node* en)
	{
		report("Eliminating unreachable code");

		for (NodeList::iterator i = _post.begin() ; i != _post.end() ; )
		{
			_visit.clear();
			if (_path(en, *i)) ++i;
			else
			{
				report("Eliminating unreachable node " << (*i)->label());

				for (NodeSet::iterator p = (*i)->preds().begin() ; 
						p != (*i)->preds().end() ; ++p)
				{
					report("Del " << (*p)->label() << " -> " << (*i)->label());
					(*p)->succs().erase(*i);
				}

				for (NodeSet::iterator s = (*i)->succs().begin() ;
						s != (*i)->succs().end() ; ++ s)
				{
					report("Del " << (*i)->label() << " -> " << (*s)->label());
					(*s)->preds().erase(*i);
				}

				i = _post.erase(i);
			}
		}
	}

	bool ControlTree::_forward_copy(Node* entry)
	{
		report("Starting Forward Copy Pass...");

		bool changed = false;

		_lexical = _executable_sequence(entry);
		EdgeVector fwdBranches = _find_forward_branches();

		while (!fwdBranches.empty())
		{
			EdgeVector::iterator iFwdBranch = fwdBranches.begin();

			report("iFwdBranch = " 
					<< iFwdBranch->first->label() << " -> "
					<< iFwdBranch->second->label());

			for (EdgeVector::iterator fwdBranch = iFwdBranch + 1 ;
					fwdBranch != fwdBranches.end() ; ++fwdBranch)
			{
				if (_interact(iFwdBranch, fwdBranch))
				{
					report(iFwdBranch->first->label() << " -> " 
							<< iFwdBranch->second->label()
							<< " interacts with "
							<< fwdBranch->first->label() 
							<< " -> " << fwdBranch->second->label());

					NodeVector MHGif = _minimal_hammock_graph(*iFwdBranch);
					NodeVector CGif  = _control_graph(*iFwdBranch);
					NodeVector ie    = NodeVector(1, MHGif.back());
					NodeVector J     = NodeVector(1, iFwdBranch->second);

					// true_part = the shared statements =
					// (MHGif - CGif - {ie}) U J
					NodeVector result1, result2, true_part;
					std::set_difference(
							MHGif.begin(), MHGif.end(), 
							CGif.begin(), CGif.end(), 
							std::back_inserter(result1),
							boost::bind(
								&analysis::ControlTree::_lexicographical_compare, 
								boost::ref(this), _1, _2));
					std::set_difference(
							result1.begin(), result1.end(),
							ie.begin(), ie.end(), 
							std::back_inserter(result2),
							boost::bind(
								&analysis::ControlTree::_lexicographical_compare, 
								boost::ref(this), _1, _2));
					std::set_union(
							result2.begin(), result2.end(),
							J.begin(), J.end(), 
							std::back_inserter(true_part),
							boost::bind(
								&analysis::ControlTree::_lexicographical_compare,
								boost::ref(this), _1, _2));

					NodeVector false_part;
					NodeVector ifie = NodeVector(1, iFwdBranch->first); 
					ifie.push_back(MHGif.back());

					// false_part = MHGif - {if, ie}
					std::set_difference(
							MHGif.begin(), MHGif.end(),
							ifie.begin(), ifie.end(), 
							std::back_inserter(false_part),
							boost::bind(
								&analysis::ControlTree::_lexicographical_compare,
								boost::ref(this), _1, _2));

					// a forward-copy transformation is applied
					_forward_copy_transform(*iFwdBranch, true_part);

					// perform unreachable-code elimination
					_elim_unreach_code(entry);

					changed = true;

					break;
				}
			}

			fwdBranches.erase(iFwdBranch);
		}

		return changed;
	}
}
