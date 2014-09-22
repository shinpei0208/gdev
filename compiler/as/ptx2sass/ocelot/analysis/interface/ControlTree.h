/*! \file ControlTree.h
 *  \author Rodrigo Dominguez <rdomingu@ece.neu.edu>
 *  \date June 10, 2010
 *  \brief The header file for the ControlTree class.
 *
 *  Computes the Control Tree as defined in Muchnick's textbook (sec 7.7):
 *
 *  1. The root of the control tree is an abstract graph representing the
 *  original flowgraph.
 *  2. The leaves of the control tree are individual basic blocks.
 *  3. The nodes between the root and the leaves are abstract nodes
 *  representing regions of the flowgraph.
 *  4. The edges of the tree represent the relationship between each abstract
 *  node and the regions that are its descendants (and that were abstracted to
 *  form it).
 *
 *  In the case of unstructured control flow graphs, a forward copy
 *  transformation is applied to deal with interacting forward branches, as
 *  explained in [1].
 *
 *  [1] Fubo Zhang and Erik H. D.Hollander. Using hammock graphs to structure 
 *      programs. IEEE Trans. Softw. Eng., 30(4):231.245, 2004.
 */

#ifndef ANALYSIS_CONTROLTREE_H_INCLUDED
#define ANALYSIS_CONTROLTREE_H_INCLUDED

// Ocelot includes
#include <ocelot/ir/interface/ControlFlowGraph.h>

#include <ocelot/analysis/interface/Analysis.h>

// STL includes
#include <list>
#include <vector>
#include <unordered_set>

typedef ir::ControlFlowGraph CFG;

namespace analysis
{
	/*! \brief Computes the Control Tree as defined in Muchnick's textbook */
	class ControlTree : public KernelAnalysis
	{
		public:
			/*! \brief Construct ControlTree */
			ControlTree();
			/*! \brief Default destructor */
			~ControlTree();

		public:
			void analyze(ir::IRKernel& k);

			/*! \brief Region type */
			enum RegionType
			{
				Inst,           // Instructions (e.g. basic block)
				Block,          // Block of nodes
				IfThen,         // If-Then
				Natural         // Natural loop with side exits
			};

			class Node;
			typedef std::list<Node*> NodeList;
			typedef std::unordered_set<Node*> NodeSet;
			typedef std::vector<Node*> NodeVector;
			typedef std::pair<Node*, Node*> Edge;
			typedef std::vector<Edge> EdgeVector;

			/*! \brief A polymorphic base class that represents any node */
			class Node
			{
				public:
					/*! \brief Constructor */
					Node(const std::string& label, RegionType rtype, 
							const NodeList& children);

					/*! \brief Destructor */
					virtual ~Node() = 0;

					/*! \brief Get the label */
					const std::string& label() const;
					/*! \brief Get the region type */
					RegionType rtype() const;
					/*! \brief Get the children */
					const NodeList& children() const;
					/*! \brief Get successors from the abstract flowgraph */
					NodeSet& succs();
					/*! \brief Get predecessors from the abstract flowgraph */
					NodeSet& preds();
					/*! \brief Get fallthrough node */
					Node*& fallthrough();
					/*! \brief Does this node have a branch edge */
					bool has_branch_edge() const;
					/*! \brief Get the branch edge */
					Edge get_branch_edge();

				private:
					/*! \brief Node label */
					const std::string _label;
					/*! \brief Region type */
					const RegionType _rtype;
					/*! \brief Children in the control tree */
					const NodeList _children;
					/*! \brief Successors in the abstract flowgraph */
					NodeSet _succs;
					/*! \brief Predecessors in the abstract flowgraph */
					NodeSet _preds;
					/*! \brief Fallthrough node */
					Node* _fallthrough;
			};

			/*! \brief A representation of the cfg basic block */
			class InstNode : public Node
			{
				public:
					typedef CFG::InstructionList InstructionList;

					/*! \brief Constructor */
					InstNode(const CFG::const_iterator& bb);

					/*! /brief Get the basic block in the cfg */
					const CFG::const_iterator& bb() const;

				private:
					/*! \brief Iterator to the basic block in the cfg */
					const CFG::const_iterator _bb;
			};

			typedef InstNode::InstructionList InstructionList;

			/*! \brief A sequence of nodes */
			class BlockNode : public Node
			{
				public:
					/*! \brief Constructor */
					BlockNode(const std::string& label, 
							const NodeList& children);
			};

			/*! \brief If-Then node */
			class IfThenNode : public Node
			{
				public:
					/*! \brief Constructor */
					IfThenNode(const std::string& label, Node* cond, 
							Node* ifTrue, Node* ifFalse = NULL);

					/*! \brief Get condition node */
					Node* cond() const;
					/*! \brief Get if-true node */
					Node* ifTrue() const;
					/*! \brief Get if-false node */
					Node* ifFalse() const;

				private:
					const NodeList buildChildren(Node* cond, 
							Node* ifTrue, Node* ifFalse) const;
			};

			class NaturalNode : public Node
			{
				public:
					/*! \brief Constructor */
					NaturalNode(const std::string& label, 
							const NodeList& children);
			};

			/*! \brief write a graphviz-compatible file for visualizing the 
			 * control tree */
			std::ostream& write(std::ostream& out) const;

			/*! \brief returns the root node of the control tree */
			const Node* get_root_node() const;

		private:
			Node* _insert_node(Node* node);

			/*! \brief depth first search */
			void _dfs_postorder(Node* x);
			/*! \brief determines whether node is the entry node of an acyclic
			 * control structure and returns its region. Stores in nset the set
			 * of nodes in the identified control structure */
			Node* _acyclic_region_type(Node* node, NodeSet& nset);
			/*! \brief is this a cyclic region? */
			bool _isCyclic(Node* node);
			// bool _isBackedge(Node* head, Node* tail);
			/*! \brief is this a back edge? */
			bool _isBackedge(const Edge& edge);
			/*! \brief adds node to the control tree, inserts node into _post
			 * at the highest-numbered position of a node in nodeSet, removes
			 * the nodes in nodeSet from _post, compacts the remaining nodes at
			 * the beginning of _post, and sets _postCtr to the index of node
			 * in the resulting postorder */
			void _compact(Node* node, NodeSet nodeSet);
			/*! \brief link region node into abstract flowgraph, adjust the
			 * predecessor and successor functions, and augment the control
			 * tree */
			void _reduce(Node* node, NodeSet nodeSet);
			/*! \brief returns true if there is a (possibly empty) path from m
			 * to k that does not pass through n */
			bool _path(Node* m, Node* k, Node* n = NULL);
			/*! \brief returns true if there is a node k such that there is a
			 * (possibly empty) path from m to k that does not pass through n
			 * and an edge k->n that is a back edge, and false otherwise. */
			bool _path_back(Node* m, Node* n);
			/*! \brief determines whether node is the entry node of a cyclic
			 * control structure and returns its region. Stores in nset the set
			 * of nodes in the identified control structure */
			Node* _cyclic_region_type(Node* node, NodeList& nset);
			void _structural_analysis(Node* entry);

			NodeVector _executable_sequence(Node* entry);
			EdgeVector _find_forward_branches();
			bool _lexicographical_compare(const Node* a, const Node* b);
			NodeVector _control_graph(const Edge& nb);
			bool _interact(const NodeVector& CGi0, const NodeVector& CGm0);
			bool _interact(const EdgeVector::iterator& e1, 
					const EdgeVector::iterator& e2); 
			NodeVector _minimal_hammock_graph(const Edge& nb);
			Node* _clone_node(const Node* node);
			void _forward_copy_transform(const Edge& iFwdBranch, 
					const NodeVector& true_part);
			void _elim_unreach_code(ControlTree::Node* en);
			bool _forward_copy(Node* entry);

			NodeVector _nodes;
			NodeList _post;
			NodeList::iterator _postCtr;
			NodeSet _visit;
			Node* _root;
			NodeVector _lexical;
	};
}

namespace std
{
	template<> struct hash<
		analysis::ControlTree::NodeList::iterator >
		{
			inline size_t operator()(
					analysis::ControlTree::NodeList::iterator it ) const
			{
				return ( size_t)&( *it );
			}
		};

	template<> struct hash<
		analysis::ControlTree::NodeList::const_iterator >
		{
			inline size_t operator()(
					analysis::ControlTree::NodeList::const_iterator it ) const
			{
				return ( size_t)&( *it );
			}
		};
}
#endif

