/*!	\file ControlFlowGraph.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\brief Interface for ControlFlowGraph
	\date 28 September 2008; 21 Jan 2009
*/

#ifndef IR_CONTROL_FLOW_GRAPH_H
#define IR_CONTROL_FLOW_GRAPH_H

#include <iostream>
#include <deque>
#include <vector>
#include <list>
#include <unordered_map>

// Forward declarations
namespace ir { class IRKernel;         }
namespace ir { class ControlFlowGraph; }
namespace ir { class Instruction;      }

namespace ir {

/*! \brief A basic block contains a series of instructions 
	terminated by control flow */
class BasicBlock {
public:
	/*! \brief A list of blocks */
	typedef std::list< BasicBlock >   BlockList;
	typedef BlockList::iterator       Pointer;
	typedef BlockList::const_iterator ConstPointer;

	/*! \brief An edge connects two basic blocks */
	class Edge {
	public:
		enum Type {
			Branch,	//< target of a branch
			FallThrough, //< edge when bb continues execution without jump
			Dummy, //< does not actually represent true control flow
			Invalid //< edge is not valid
		};
	
	public:
		Edge(BlockList::iterator h = BlockList::iterator(), 
			BlockList::iterator t = BlockList::iterator(), 
			Type y = FallThrough);

		/*!	pointer to head node of edge */
		BlockList::iterator head;
		/*!	pointer to tail node of edge */
		BlockList::iterator tail;
		/*!	Edge properties */
		Type type;

	public:
		/*!	is this a fallthrough? */
		bool isFallthrough() const;
	};

	typedef std::list<Edge>                 EdgeList;
	typedef std::vector<Pointer>            BlockPointerVector;
	typedef std::vector<ConstPointer>       ConstBlockPointerVector;
	typedef std::vector<EdgeList::iterator> EdgePointerVector;
	typedef std::list<Instruction*>         InstructionList;
	typedef InstructionList::iterator       instruction_iterator;
	typedef InstructionList::const_iterator const_instruction_iterator;
	typedef unsigned int                    Id;

public:
	BasicBlock(ControlFlowGraph* cfg = 0, Id i = 0, 
		const InstructionList& instructions = InstructionList(),
		const std::string& c = "");
	BasicBlock(Id i,
		const InstructionList& instructions = InstructionList(),
		const std::string& c = "");
	~BasicBlock();

	/*! \brief Clear/delete all instructions owned by the block */
	void clear();

	/*! \brief Get the fallthrough edge */
	EdgeList::iterator get_fallthrough_edge();
	/*! \brief Get the fallthrough edge */
	EdgeList::const_iterator get_fallthrough_edge() const;
	/*! \brief Does this have a fallthrough edge */
	bool has_fallthrough_edge() const;

	/*! \brief Get the branch edge */
	EdgeList::iterator get_branch_edge();
	/*! \brief Get the branch edge */
	EdgeList::const_iterator get_branch_edge() const;
	/*! \brief Does this have a branch edge */
	bool has_branch_edge() const;

	/*! \brief Get the edge connecting to the specified block */
	EdgeList::iterator get_edge(BlockList::iterator b);
	/*! \brief Get the edge connecting to the specified block */
	EdgeList::const_iterator get_edge(BlockList::const_iterator b) const;

	/*! \brief Get a string representation of the block's label */
	std::string label() const;
	
public:
	/*! \brief Find an in-edge with specific head and tail pointers */
	EdgePointerVector::iterator find_in_edge(
		BlockList::const_iterator head);
	/*! \brief Find an out-edge with specific head and tail pointers */
	EdgePointerVector::iterator find_out_edge(
		BlockList::const_iterator tail);

public:
	/*!	list of instructions in BasicBlock. */
	InstructionList instructions;

	/*! \brief a comment appearing in the emitted PTX source file */
	std::string comment;
	/*! \brief Basic block unique identifier */
	Id id;

	/*! \brief Direct successor blocks */
	BlockPointerVector successors;
	/*! \brief Direct predecessor blocks */
	BlockPointerVector predecessors;

	/*! \brief Edges from direct predecessors */
	EdgePointerVector in_edges;
	/*! \brief Edges to direct successors */
	EdgePointerVector out_edges;

	/*! \brief A pointer to the owning CFG */
	ControlFlowGraph* cfg;

public:

	/*!
		\brief an object that formats the string representation of a basic 
			block used in the DOT output of the graph
	*/
	class DotFormatter {
	public:
		DotFormatter();
		virtual ~DotFormatter();

	public:		
		/*! \brief emits label for entry block */
		virtual std::string entryLabel(const BasicBlock *block);
		
		/*! \brief emits label for exit block */
		virtual std::string exitLabel(const BasicBlock *block);
	
		/*!	\brief prints string representation of */
		virtual std::string toString(const BasicBlock *block);

		/*! \brief emits DOT representation of an edge	*/
		virtual std::string toString(const Edge *edge);
	};
};

typedef BasicBlock::Edge Edge;

/*! Control flow graph */
class ControlFlowGraph {
public:
	typedef ir::BasicBlock BasicBlock;

	/*! \brief A list of basic blocks */
	typedef BasicBlock::BlockList BlockList;
	/*! \brief A list of edges */
	typedef BasicBlock::EdgeList EdgeList;
	/*! \brief A vector of edge pointers */
	typedef BasicBlock::EdgePointerVector EdgePointerVector;
	/*! \brief A vector of block pointers */
	typedef BasicBlock::BlockPointerVector BlockPointerVector;
	/*! \brief A const vector of block pointers */
	typedef BasicBlock::ConstBlockPointerVector ConstBlockPointerVector;
	
	/*! \brief An iterator over basic blocks */
	typedef BlockList::iterator iterator;
	/*! \brief A const iterator over basic blocks */
	typedef BlockList::const_iterator const_iterator;
	
	/*! \brief A pointer to an iterator */
	typedef BlockPointerVector::iterator pointer_iterator;
	/*! \brief A pointer to an iterator */
	typedef BlockPointerVector::const_iterator const_pointer_iterator;
	/*! \brief A pointer to an iterator */
	typedef BlockPointerVector::reverse_iterator reverse_pointer_iterator;	
	
	/*! \brief An iterator over edges */
	typedef EdgeList::iterator edge_iterator;
	/*! \brief A const iterator over edges */
	typedef EdgeList::const_iterator const_edge_iterator;
	/*! \brief Edge pair */
	typedef std::pair<edge_iterator, edge_iterator> EdgePair;

	/*! \brief A pointer to an edge iterator */
	typedef EdgePointerVector::iterator edge_pointer_iterator;
	/*! \brief A const pointer to an edge iterator */
	typedef EdgePointerVector::const_iterator const_edge_pointer_iterator;

	/*! \brief A map from a block pointer to an int */
	typedef std::unordered_map<const_iterator, unsigned int> BlockMap;
	/*! \brief The edge */
	typedef BasicBlock::Edge Edge;

	/*! \brief An instruction list */
	typedef BasicBlock::InstructionList InstructionList;
	/*! \brief An iterator over instructions */
	typedef InstructionList::iterator instruction_iterator;

	/*! \brief maps a basic block [by label] to a coloring */
	typedef std::unordered_map<std::string, unsigned int> BasicBlockColorMap;

public:
	ControlFlowGraph(ir::IRKernel* kernel = 0);
	~ControlFlowGraph();
	
	/*!	deep copy of ControlFlowGraph */
	ControlFlowGraph& operator=(const ControlFlowGraph &);

public:
	/*! \brief Get the name of an edge type */
	static std::string toString( Edge::Type t );
	
public:
	/*! Make sure that the next call to newId will return a unique id */
	void computeNewBlockId();

	/*! Get a unique identifier for a new block */
	BasicBlock::Id newId();

	/*!	Gets the number of blocks within the graph */
	size_t size() const;
	
	/*! \brief Get the number of instructions within the graph */
	size_t instructionCount() const;
	
	/*! \brie Is the graph empty? */
	bool empty() const;

	/*!	Inserts a basic block into the CFG */
	iterator insert_block(const BasicBlock& b);
	
	/*! Duplicates the selected block, inserts it in an unconnected state,
		returns an iterator to the newly created block */
	iterator clone_block(const_iterator block);
	
	/*! Removes a basic block and associated edges. Any blocks dominated by
		block are now unreachable but still part of the graph.
	
		\param block block to remove from graph
	*/
	void remove_block(iterator block);
	
	/*! Disconnect the block from all edges. Any blocks dominated by
		block are now unreachable but still part of the graph.
	
		\param block block to remove edges from graph
	*/
	void disconnect_block(iterator block);
	
	/*! Disconnect the block from all out-edges. Any blocks dominated by
		block are now unreachable but still part of the graph.
	
		\param block block to remove out-edges from graph
	*/
	void disconnect_block_out_edges(iterator block);

	/*! Creates an edge between given basic blocks
		\param edge edge to create
	*/
	edge_iterator insert_edge(const Edge& e);
	
	/*! Removes the edge which may exist from head->tail. This may render tail
		unreachable.
		
		\param edge to remove
	*/
	void remove_edge(edge_iterator edge);
	
	/*! Given an edge head->tail, retargets edge and creates new edge such that 
		the path head->newblock->tail exists. Inserts newblock into CFG and 
		returns newly created edge
		
		\param edge existing edge to split
		\param newblock new BasicBlock to insert into CFG and create an edge 
			from
		\return implicily created edges (head->newblock, newblock->tail) with 
			same type as edge [may need modifying]
	*/
	EdgePair split_edge(edge_iterator edge, const BasicBlock& newBlock);

	/*! \brief Splits a basic block into two such that there is a fallthrough
		edge from the original block to the newly split block.
	
		This function will map all out_edges of the first block to the second
		block.
		
		\param block The block being split
		\param the instruction within the block to perform the split, it ends
			up in the newly split block
		\param the label of the new block
		\return A pointer to the newly allocated second block
	*/
	iterator split_block(iterator block, instruction_iterator instruction, 
		Edge::Type type);

	/*!	Returns the entry block of a control flow graph */
	iterator get_entry_block();

	/*!	Returns the exit block of a control flow graph */
	iterator get_exit_block();
	
	/*! Returns the entry block of a control flow graph */
	const_iterator get_entry_block() const;

	/*!	Returns the exit block of a control flow graph */
	const_iterator get_exit_block() const;
	
	/*!	write a graphviz-compatible file for visualizing the CFG */
	std::ostream& write(std::ostream& out) const;

	/*!	\brief write a graphviz-compatible file for visualizing the CFG
	*/
	std::ostream& write(std::ostream& out, 
		BasicBlock::DotFormatter& blockFormatter) const;
	
	/*! \brief Clears all basic blocks and edges in the CFG.*/
	void clear();
	
	/*! returns an ordered sequence of the nodes of the CFG including entry 
		and exit that would be encountered by a pre order traversal
	*/
	BlockPointerVector pre_order_sequence();
	
	/*! returns an ordered sequence of the nodes of the CFG including entry 
		and exit that would be encountered by a post order traversal
	*/
	BlockPointerVector post_order_sequence();

	/*! returns an ordered sequence of the nodes of the CFG including entry 
		and exit that would be encountered by a reverse post order traversal
		This is equivalent to a topological order
	*/
	BlockPointerVector topological_sequence();

	/*! returns an ordered sequence of the nodes of the CFG including entry 
		and exit that would be encountered by a reverse post order traversal
		This is equivalent to a topological order
	*/
	BlockPointerVector reverse_topological_sequence();

	/*! Returns an ordered sequence of basic blocks such that the entry node 
		is first and all fall-through edges produce adjacencies
	*/
	BlockPointerVector      executable_sequence();
	ConstBlockPointerVector	executable_sequence() const;

public:
	/*! \brief Get an iterator to the first block */
	iterator begin();
	/*! \brief Get an iterator to the last block */
	iterator end();
	
	/*! \brief Get a const iterator to the first block */
	const_iterator begin() const;
	/*! \brief Get a const iterator the last block */
	const_iterator end() const;

	/*! \brief Get an iterator to the first edge */
	edge_iterator edges_begin();
	/*! \brief Get an iterator to the last edge */
	edge_iterator edges_end();

	/*! \brief Get a const iterator to the first edge */
	const_edge_iterator edges_begin() const;
	/*! \brief Get a const iterator to the last edge */
	const_edge_iterator edges_end() const;

public:
	/*! \brief The owning kernel */
	IRKernel* kernel;

public:
	BlockList _blocks;
	EdgeList  _edges;

	iterator _entry;
	iterator _exit;
	
	BasicBlock::Id _nextId;
};

}

namespace std
{
	template<> 
	struct hash<ir::ControlFlowGraph::iterator>
	{
	public:
		size_t operator()(const ir::ControlFlowGraph::iterator& it) const
		{
			return (size_t)it->id;
		}
	};

	template<> 
	struct hash<ir::ControlFlowGraph::const_iterator>
	{
	public:
		size_t operator()(const ir::ControlFlowGraph::const_iterator& it) const
		{
			return (size_t)it->id;
		}
	};
	
	template<> 
	struct hash<ir::ControlFlowGraph::edge_iterator>
	{
	public:
		size_t operator()(const ir::ControlFlowGraph::edge_iterator& it) const
		{
			return (size_t)(it->head->id ^ it->tail->id);
		}
	};

	template<> 
	struct hash<ir::ControlFlowGraph::const_edge_iterator>
	{
	public:
		size_t operator()(
			const ir::ControlFlowGraph::const_edge_iterator& it) const
		{
			return (size_t)(it->head->id ^ it->tail->id);
		}
	};

	template<> 
	struct hash<ir::ControlFlowGraph::InstructionList::iterator>
	{
	public:
		size_t operator()( 
			const ir::ControlFlowGraph::InstructionList::iterator& it) const
		{
			return (size_t)&(*it);
		}
	};

	template<> 
	struct hash<ir::ControlFlowGraph::InstructionList::const_iterator>
	{
	public:
		typedef ir::ControlFlowGraph::InstructionList::const_iterator type;
	
		size_t operator()(const type& it) const
		{
			return (size_t)&(*it);
		}
	};
}


#endif

