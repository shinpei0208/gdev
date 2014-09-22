/*! \file   DataflowGraph.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Tuesday June 23, 2009
	\file   The header file for the DataflowGraph class.
*/

#ifndef DATAFLOW_GRAPH_H_INCLUDED
#define DATAFLOW_GRAPH_H_INCLUDED

// Ocelot Includes
#include <ocelot/analysis/interface/Analysis.h>
#include <ocelot/ir/interface/ControlFlowGraph.h>
#include <ocelot/ir/interface/PTXInstruction.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <cassert>
#include <unordered_set>
#include <set>
#include <map>
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

// Forward Declarations
namespace analysis { class SSAGraph; }

/*! \brief A namespace for compiler analysis modules. */
namespace analysis
{
/*! \brief A class representing a graph of block of instructions, showing
		which registers are alive in each basic block.
*/
class DataflowGraph : public KernelAnalysis
{
	friend class SSAGraph;
	public:			
		/*! \brief Datatype, we use PTX */
		typedef ir::PTXOperand::DataType Type;
		/*! \brief A unique ID for a logical register */
		typedef unsigned int RegisterId;
		/*! \brief A list of instructions */
		typedef ir::ControlFlowGraph::InstructionList InstructionList;

		enum SsaType
		{
			None    = 0,
			Default = 1,
			Minimal = 2,
			Gated   = 4
		};

    	/*! \brief A register with type info */
		class RegisterPointer
		{
			public:
				/*! \brief The type of the register */
				Type type;
				 /*! \brief Register id pointer */
				RegisterId* pointer;
				
			public:
				/*! \brief Constructor with type and pointer */
				RegisterPointer( RegisterId* id, Type type );
				
			public:
				/*! \brief Needed for comparisons */
				bool operator==( const RegisterPointer& r ) const;
		};

		/*! \brief A register with type info */
		class Register
		{
			public:
				/*! \brief The type of the register */
				Type type;
				/*! \brief The id of the register */
				RegisterId id;
				
			public:
				/*! \brief Constructor with type and id */
				Register( RegisterId id = 0, 
					Type type = ir::PTXOperand::TypeSpecifier_invalid );
				/*! \brief Constructor from pointer */
				Register( const RegisterPointer& r );
				
			public:
				/*! \brief Needed for comparisons */
				bool operator==( const Register& r ) const;
		};

		struct RegisterHash
		{
			inline std::size_t operator()( Register r ) const
			{
				return r.id;
			}
		};
		
		/*! \brief A vector of register ID pointers */
		typedef std::vector< RegisterPointer > RegisterPointerVector;
		/*! \brief A vector of registers */
		typedef std::vector< Register > RegisterVector;
		/*! \brief register iterator */
		typedef RegisterVector::iterator register_iterator;
		/*! \brief register iterator */
		typedef RegisterPointerVector::iterator register_pointer_iterator;
		
		/*! \brief An exception for potentially uninitialized regs */
		class NoProducerException : public std::exception
		{
			private:
				std::string _message;
			
			public:
				NoProducerException( RegisterId reg );
				~NoProducerException() throw();
				const char* what() const throw();
		};

		
		class Block;
		class Instruction;
		class PhiInstruction;
		/*! \brief A vector of Instructions */
		typedef std::list< Instruction > InstructionVector;
		/*! \brief A vector of PhiInstructions */
		typedef std::list< PhiInstruction > PhiInstructionVector;
		/*! \brief A vector of blocks */
		typedef std::list< Block > BlockVector;
		/*! \brief An iterator over an instruction vector */
		typedef InstructionVector::iterator InstructionIterator;
		/*! \brief A vector of instruction iterators */
		typedef std::list< InstructionIterator > InstructionIteratorList;
		/*! \brief iterator */
		typedef BlockVector::iterator iterator;
		/*! \brief const_iterator */
		typedef BlockVector::const_iterator const_iterator;
		
		/*! \brief A class for referring to a generic instruction. */
		class Instruction
		{
			public:
				/*! \brief The pointer to the instruction */
				ir::Instruction* i;
				/*! \brief Destination registers */
				RegisterPointerVector d;
				/*! \brief Source registers */
				RegisterPointerVector s;
				/*! \brief A pointer to the owning DFG block */
				iterator block;

		};
		
		/*! \brief A class for referring to a phi instruction. */
		class PhiInstruction
		{
			public:
				PhiInstruction(const Register& d = 0,
					const RegisterVector& s = RegisterVector());
			
			public:
				/*! \brief Destination register */
				Register d;
				/*! \brief Source registers */
				RegisterVector s;
				
			public:
				std::string toString() const;
		};

		struct BlockVectorHash
		{
			inline std::size_t operator()( BlockVector::iterator it ) const
			{
				return (std::size_t)it->id();
			}
		};

		/*! \brief A vector of Block pointers */
		typedef std::unordered_set< BlockVector::iterator,
			BlockVectorHash > BlockPointerSet;

		/*! \brief A unique set of register Ids */
		typedef std::unordered_set< Register,
			RegisterHash > RegisterSet;		

		/*! \brief A class for referring to a generic basic block of 
				instructions.
		*/
		class Block
		{
			friend class SSAGraph;
			friend class DataflowGraph;
			friend class SyncEliminationPass;
			public:
				/*! \brief The type of block */
				enum Type
				{
					Entry,
					Exit,
					Body,
					Invalid
				};


			private:
				/*! \brief Registers that are alive entering the block */
				RegisterSet _aliveIn;
				/*! \brief Register that are alive exiting the block */
				RegisterSet _aliveOut;
				/*! \brief The fallthrough block */
				BlockVector::iterator _fallthrough;
				/*! \brief The target blocks */
				BlockPointerSet _targets;
				/*! \brief A list of predecessor blocks */
				BlockPointerSet _predecessors;
				/*! \brief A list of successors blocks */
				BlockPointerSet _successors;
				/*! \brief The type of block */
				Type _type;
				/*! \brief Ordered set of phi instructions in the block */
				PhiInstructionVector _phis;
				/*! \brief Ordered set of instructions in the block */
				InstructionVector _instructions;
				/*! \brief A pointer to the underlying 
					basic block in the cfg */
				ir::ControlFlowGraph::iterator _block;
				/*! \brief A pointer to the owning DFG */
				DataflowGraph* _dfg;

			private:
				/*! \brief Compare two register sets */
				static bool _equal( const RegisterSet& one, 
					const RegisterSet& two );

			private:
				/*! \brief Resolve a type ambiguity */
				void _resolveTypes( DataflowGraph::Type& d,
					const DataflowGraph::Type& s );
				/*! \brief Update the live ranges */
				bool compute( bool hasFallthrough );
		
			public:
				/*! \brief Constructor from a Control Flow Graph block */
				Block( DataflowGraph& dfg, 
					ir::ControlFlowGraph::iterator block );
				/*! \brief Default constructor */
				Block( DataflowGraph& dfg, Type t = Invalid );
				
			public:
				/*! \brief Get registers that are alive entering the block*/
				const RegisterSet& aliveIn() const;
				/*! \brief Get registers that are alive exiting the block */
				const RegisterSet& aliveOut() const;
				/*! \brief Get registers that are alive entering the block*/
				RegisterSet& aliveIn();
				/*! \brief Get registers that are alive exiting the block */
				RegisterSet& aliveOut();
				/*! \brief Get the fallthrough block */
				BlockVector::iterator fallthrough() const;
				/*! \brief Get the fallthrough block */
				bool hasFallthrough() const;
				/*! \brief Get a list of target blocks */
				const BlockPointerSet& targets() const;
				/*! \brief Get a list of predecessor blocks */
				const BlockPointerSet& predecessors() const;
				/*! \brief Get a list of successor blocks */
				const BlockPointerSet& successors() const;
				/*! \brief Get a list of target blocks */
				BlockPointerSet& targets();
				/*! \brief Get a list of successor blocks */
				BlockPointerSet& successors();
				/*! \brief Get a list of predecessor blocks */
				BlockPointerSet& predecessors();
				/*! \brief Get the type of the block */
				Type type() const;
				/*! \brief Get an ordered set of instructions in the block*/
				const InstructionVector& instructions() const;
				/*! \brief Get an ordered set of instructions in the block*/
				InstructionVector& instructions();
				/*! \brief Get an ordered set of phis in the block*/
				const PhiInstructionVector& phis() const;
				/*! \brief Get an ordered set of phis in the block*/
				PhiInstructionVector& phis();
				/*! \brief Get the block label */
				std::string label() const;
				/*! \brief Get the id of the block */
				ir::ControlFlowGraph::BasicBlock::Id id() const;
				/*! \brief Get a pointer to the underlying block */
				ir::ControlFlowGraph::iterator block();
				/*! \brief Get a pointer to the underlying block */
				ir::ControlFlowGraph::const_iterator block() const;
				/*! \brief Get a pointer to the underlying dfg */
				DataflowGraph* dfg();

			public:
				/*! \brief Determine the block that produced a register */
				BlockVector::const_iterator producer( const Register& r ) const;
				/*! \brief Determine the alive registers immediately
					before a given instruction in the block */
				RegisterSet alive( const 
					InstructionVector::const_iterator& i );
		};
		
	public:
		/*! \brief Iterator over phi instructions */
		typedef PhiInstructionVector::iterator phi_iterator;
		
		/*! \brief Iterator over normal instructions */
		typedef InstructionIterator instruction_iterator;
		/*! \brief Value type */
		typedef BlockVector::value_type value_type;
		/*! \brief Size type */
		typedef BlockVector::size_type size_type;
		/*! \brief Register set iterator */
		typedef RegisterSet::const_iterator const_register_iterator;

		/*! \brief A vector of iterators */
		typedef std::vector< iterator >	BlockPointerVector;
		/*! \brief A pointer to an iterator */
		typedef BlockPointerVector::iterator pointer_iterator;		
		/*! \brief A reverse pointer to an iterator */
		typedef BlockPointerVector::reverse_iterator 
			reverse_pointer_iterator;
		/*! \brief A map from cfg blocks to dfg equivalents */
		typedef std::unordered_map< ir::ControlFlowGraph::const_iterator, 
			iterator > IteratorMap;

		typedef std::set<RegisterId> RegisterIdSet;
		typedef std::map<const PhiInstruction*, RegisterIdSet > PhiPredicateMap;

	private:
		BlockVector _blocks;
		ir::ControlFlowGraph* _cfg;
		bool _consistent;
		SsaType _ssa;
		RegisterId _maxRegister;
		PhiPredicateMap _phiPredicateMap;

	public:
		/*! \brief Convert from a PTXInstruction to an Instruction  */
		Instruction convert( ir::PTXInstruction& i );

	public:
		/*! \brief Build from a kernel CFG */
		void analyze( ir::IRKernel& kernel );

	public:
		/*! \brief The constructor */
		DataflowGraph();
		/*! \brief The destructor */
		~DataflowGraph();
		
	public:
		#ifndef _WIN32
		DataflowGraph(const DataflowGraph& ) = delete;
		DataflowGraph& operator=(const DataflowGraph& ) = delete;
		#endif

	public:
		/*! \brief Return an iterator to the program entry point */
		iterator begin();
		/*! \brief Return an iterator to the program entry point */
		const_iterator begin() const;
		/*! \brief Return an iterator just beyond the program exit point */
		iterator end();
		/*! \brief Return an iterator just beyond the program exit point */
		const_iterator end() const;

	public:			
		/*! \brief Is the graph empty? */
		bool empty() const;
		/*! \brief Get the number of blocks */
		size_type size() const;
		/*! \brief Get the max number of blocks */
		size_type max_size() const;

	public:
		/*! \brief Insert a Block between two existing blocks.
			\param predecessor An iterator to the previous block.
			\return An iterator to the inserted block.
			
			Note that this insert splits the fallthrough edge
		*/
		iterator insert( iterator predecessor );
		/*! \brief Insert a Block between two existing blocks.
			\param predecessor An iterator to the previous block.
			\param successor An iterator to the next block.
			\return An iterator to the inserted block
			
			Note that this insert splits the edge 
				between predecessor and successor.
		*/
		iterator insert( iterator predecessor, iterator successor );
		/*! \brief Split a block into two starting at a given instruction,
			the split instruction goes in the first block */
		iterator split( iterator block, unsigned int instruction, 
			bool isFallthrough );
		/*! \brief Split a block into two starting at a given instruction iterator,
			the split instruction goes in the first block */
		iterator split( iterator block, InstructionVector::iterator position, 
			bool isFallthrough );
		/*! \brief Redirect an edge between two blocks to a third */
		void redirect( iterator source, 
			iterator destination, iterator newTarget );
		/*! \brief Set the target of a block */
		void target( iterator block, iterator target,
			bool fallthrough = false );
		/*! \brief disconnect a specified block from the graph,
			removing all edges.  The block remains intact */
		void disconnect( iterator block );
		/*! \brief disconnect a specified block from the graph,
			removing all out-edges.  The block remains intact */
		void disconnectOutEdges( iterator block );
		/*! \brief Delete a block, joining predecessors and successors */
		iterator erase( iterator block );
		/*! \brief Revert back to a single entry and exit block */
		void clear();
	
	public:
		/*! \brief Insert an instruction into a block 
			immediately before the specified index */
		InstructionVector::iterator insert( iterator block,
			const ir::Instruction& instruction, unsigned int index );
		/*! \brief Insert an instruction into a block 
			immediately before the element at specified  position*/
        InstructionVector::iterator insert( iterator block,
                const ir::Instruction& instruction, 
			InstructionVector::iterator position );
		/*! \brief Insert an instruction at the end of a block */
		InstructionVector::iterator insert( iterator block,
			const ir::Instruction& instruction );
		/*! \brief Erase an instruction from a block at the specified
			position */
		InstructionVector::iterator erase( iterator block,
			InstructionVector::iterator position );
		/*! \brief Erase an instruction from a block at the specified
			index */
		void erase( iterator block, unsigned int index );
		/*! \brief Erase an instruction from its block at the specified
			position */
		InstructionVector::iterator erase(
			InstructionVector::iterator position );
		
	public:
		/*! \brief Compute live ranges */
		void compute();
		/*! \brief Determine the max register used in the graph */
		RegisterId maxRegister() const;
		/*! \brief Allocate a new register that is not used elswhere 
			in the graph */
		RegisterId newRegister();
		
	public:
		/*! \brief Convert into ssa form */
		void toSsa(SsaType form = SsaType::Default);
		/*! \brief Convert out of ssa form */
		void fromSsa();
		/*! \brief Is the graph in ssa form? */
		unsigned int ssa() const;
		
		/*! \brief Set the type of SSA to use when the graph is constructed */
		void setPreferredSSAType(SsaType form);
		
		/*! \brief Convert to the specified SSA type if it is not already  */
		void convertToSSAType(SsaType form);
		
	public:
		/*! \brief Get an executable sequence of blocks */
		BlockPointerVector executableSequence();
		/*! \brief Get a map from CFG to DFG iterators */
		IteratorMap getCFGtoDFGMap();

	public:
		const RegisterIdSet& getPhiPredicates(
			PhiInstruction const* phi ) const;
		const PhiPredicateMap& phiPredicateMap() const;
		bool hasPhi( DataflowGraph::PhiInstruction const *phi ) const;

	public:
		void write(std::ostream& out) const;

};

std::ostream& operator<<( std::ostream& out, const DataflowGraph& graph );
std::ostream& operator<<( std::ostream& out,
	const DataflowGraph::PhiInstruction& phi );
std::ostream& operator<<( std::ostream& out,
	const DataflowGraph::Instruction& i );

}

namespace std
{
	template<>
	struct hash< analysis::DataflowGraph::iterator >
	{
		inline size_t operator()(
			const analysis::DataflowGraph::iterator& it ) const
		{
			return ( size_t )it->id();
		}
	};
	
	template<>
	struct hash< analysis::DataflowGraph::const_iterator >
	{
		inline size_t operator()(
			const analysis::DataflowGraph::const_iterator& it ) const
		{
			return ( size_t )it->id();
		}
	};

	template<>
	struct hash< analysis::DataflowGraph::Register >
	{
		inline size_t operator()(
			const analysis::DataflowGraph::Register& r ) const
		{
			return r.id;
		}
	};

	template<>
	struct hash< analysis::DataflowGraph::RegisterPointer >
	{
		inline size_t operator()(
			const analysis::DataflowGraph::RegisterPointer& r ) const
		{
			return *r.pointer;
		}
	};
}

#endif

