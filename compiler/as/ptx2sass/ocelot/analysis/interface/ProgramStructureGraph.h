/*! \file   ProgramStructureGraph.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Monday August 1, 2011
	\brief  The header file for the ProgramStructureGraph class.
*/

#pragma once

// Ocelot Includes
#include <ocelot/ir/interface/ControlFlowGraph.h>

namespace analysis
{

/*! \brief ProgramStructureGraphs are overlays over the ControlFlowGraph that
	capture some structure other than basic blocks.  
	
	Examples of program structures from literature may include Superblocks,
		Hyperblocks, Treegions, or Subkernels.
*/
class ProgramStructureGraph
{
public:
	class Block
	{
	public:
		typedef ir::ControlFlowGraph           CFG;
		typedef ir::BasicBlock                 BB;
		typedef BB::instruction_iterator       instruction_iterator;
		typedef CFG::pointer_iterator          basic_block_iterator;
		typedef BB::const_instruction_iterator const_instruction_iterator;
		typedef CFG::const_pointer_iterator    const_basic_block_iterator;

		// Forward Declarations
		class const_iterator;
		
		/*! \brief An iterator over basic blocks */
		class block_iterator
		{
		public:
			typedef block_iterator self;
			typedef ir::BasicBlock value_type;
			typedef value_type&    reference;
			typedef value_type*    pointer;
		
		public:	      
			block_iterator();
			block_iterator(const block_iterator&);
			explicit block_iterator(const basic_block_iterator& i,
				const basic_block_iterator& begin,
				const basic_block_iterator& end);

		public:
			reference operator*() const;
			pointer operator->() const;
			self& operator++();
			self operator++(int);
			self& operator--();
			self operator--(int);
			
			bool operator==(const self&) const;
			bool operator!=(const self&) const;	
		
			bool begin() const;
			bool end() const;
		
		private:
			basic_block_iterator _iterator;	
			basic_block_iterator _begin;
			basic_block_iterator _end;

			friend class const_block_iterator;
			friend class Block;
		};

		/*! \brief A const iterator over basic blocks */
		class const_block_iterator
		{
		public:
			typedef const_block_iterator self;
			typedef ir::BasicBlock       value_type;
			typedef const value_type&    reference;
			typedef const value_type*    pointer;
		
		public:
			const_block_iterator();
			const_block_iterator(const const_block_iterator&);
			const_block_iterator(const block_iterator&);
			explicit const_block_iterator(const const_basic_block_iterator& i,
				const const_basic_block_iterator& begin,
				const const_basic_block_iterator& end);

		public:
			reference operator*() const;
			pointer operator->() const;
			self& operator++();
			self operator++(int);
			self& operator--();
			self operator--(int);
			
			bool operator==(const self&) const;
			bool operator!=(const self&) const;	
			
			bool begin() const;
			bool end() const;
			
		private:
			const_basic_block_iterator _iterator;	
			const_basic_block_iterator _begin;
			const_basic_block_iterator _end;
		};

		/*! \brief An iterator over the instructions in the
			contained basic blocks */
		class iterator
		{
		public:
			typedef iterator                         self;
			typedef instruction_iterator::value_type value_type;
			typedef value_type&                      reference;
			typedef value_type*                      pointer;

		public:	      
			iterator();
			explicit iterator(const block_iterator&,
				const instruction_iterator&);
		public:
			reference operator*() const;
			pointer operator->() const;
			self& operator++();
			self operator++(int);
			self& operator--();
			self operator--(int);
			
			bool operator==(const self&) const;
			bool operator!=(const self&) const;	

		private:
			instruction_iterator _instruction;
			block_iterator       _basicBlock;
		
			friend class const_iterator;
			friend class Block;

		};
		
		/*! \brief A const iterator */
		class const_iterator
		{
		public:
			typedef const_iterator                   self;
			typedef instruction_iterator::value_type value_type;
			typedef value_type&                      reference;
			typedef value_type*                      pointer;
		
		public:	      
			const_iterator();
			const_iterator(const iterator&);
			const_iterator(const const_iterator&);
			explicit const_iterator(const const_block_iterator&,
				const const_instruction_iterator&);
		public:
			reference operator*() const;
			pointer operator->() const;
			self& operator++();
			self operator++(int);
			self& operator--();
			self operator--(int);
			
			bool operator==(const self&) const;
			bool operator!=(const self&) const;	
		
		private:
			const_instruction_iterator _instruction;
			const_block_iterator       _basicBlock;		
		};
		
		class const_successor_iterator;
		
		/*! \brief An iterator over block successors */
		class successor_iterator
		{
		public:
			typedef successor_iterator self;
			typedef ir::BasicBlock     value_type;
			typedef value_type&        reference;
			typedef value_type*        pointer;
		
		public:	      
			successor_iterator();
			successor_iterator(const successor_iterator&);
			explicit successor_iterator(const block_iterator&,
				const basic_block_iterator&);
		public:
			reference operator*() const;
			pointer operator->() const;
			self& operator++();
			self operator++(int);
			self& operator--();
			self operator--(int);
						
			bool operator==(const self&) const;
			bool operator!=(const self&) const;	
		
		private:
			block_iterator         _block;
			basic_block_iterator _successor;		
		
			friend class const_successor_iterator;
		};
		
		/*! \brief An iterator over block successors */
		class const_successor_iterator
		{
		public:
			typedef const_successor_iterator self;
			typedef ir::BasicBlock           value_type;
			typedef const value_type&        reference;
			typedef const value_type*        pointer;
		
		public:
			const_successor_iterator();
			const_successor_iterator(const const_successor_iterator&);
			const_successor_iterator(const successor_iterator&);
			explicit const_successor_iterator(const const_block_iterator&,
				const const_basic_block_iterator&);

		public:
			reference operator*() const;
			pointer operator->() const;
			self& operator++();
			self operator++(int);
			self& operator--();
			self operator--(int);
						
			bool operator==(const self&) const;
			bool operator!=(const self&) const;	
		
		private:
			const_block_iterator       _block;
			const_basic_block_iterator _successor;		
		};
		
		class const_predecessor_iterator;
		
		/*! \brief An iterator over block predecessors */
		class predecessor_iterator
		{
		public:
			typedef predecessor_iterator     self;
			typedef ir::BasicBlock           value_type;
			typedef value_type&              reference;
			typedef value_type*              pointer;
		
		public:	      
			predecessor_iterator();
			predecessor_iterator(const predecessor_iterator&);
			explicit predecessor_iterator(const block_iterator&,
				const basic_block_iterator&);
		public:
			reference operator*() const;
			pointer operator->() const;
			self& operator++();
			self operator++(int);
			self& operator--();
			self operator--(int);
			
			bool operator==(const self&) const;
			bool operator!=(const self&) const;	
		
		private:
			block_iterator       _block;
			basic_block_iterator _predecessor;	
		
			friend class const_predecessor_iterator;
		};
		
		/*! \brief A const iterator over block predecessors */
		class const_predecessor_iterator
		{
		public:
			typedef const_predecessor_iterator self;
			typedef ir::BasicBlock             value_type;
			typedef const value_type&          reference;
			typedef const value_type*          pointer;
		
		public:	      
			const_predecessor_iterator();
			const_predecessor_iterator(const const_predecessor_iterator&);
			const_predecessor_iterator(const predecessor_iterator&);
			explicit const_predecessor_iterator(const const_block_iterator&,
				const const_basic_block_iterator&);
		public:
			reference operator*() const;
			pointer operator->() const;
			self& operator++();
			self operator++(int);
			self& operator--();
			self operator--(int);
			
			bool operator==(const self&) const;
			bool operator!=(const self&) const;	
		
		private:
			const_block_iterator       _block;
			const_basic_block_iterator _predecessor;	
		};
		
	public:
		/*! \brief Get an iterator to the first instruction in the block */
		iterator begin();
		/*! \brief Get an iterator to the end of the instruction list */
		iterator end();
		
		/*! \brief Get a const iterator to the first instruction in the block */
		const_iterator begin() const;
		/*! \brief Get a const iterator to the end of the instruction list */
		const_iterator end() const;
		
	public:
		/*! \brief Get a block iterator to the first block */
		block_iterator block_begin();
		/*! \brief Get an iterator to the end of the block list */
		block_iterator block_end();
		
		/*! \brief Get a const iterator to the first block */
		const_block_iterator block_begin() const;
		/*! \brief Get a const iterator to the end of the block list */
		const_block_iterator block_end() const;

	public:
		/*! \brief Get a block iterator to the first successor */
		successor_iterator successors_begin();
		/*! \brief Get an iterator to the end of the successor list */
		successor_iterator successors_end();
		
		/*! \brief Get a const iterator to the first successor */
		const_successor_iterator successors_begin() const;
		/*! \brief Get a const iterator to the end of the successor list */
		const_successor_iterator successors_end() const;

	public:
		/*! \brief Get a block iterator to the first predecessor */
		predecessor_iterator predecessors_begin();
		/*! \brief Get an iterator to the end of the predecessor list */
		predecessor_iterator predecessors_end();
		
		/*! \brief Get a const iterator to the first predecessor */
		const_predecessor_iterator predecessors_begin() const;
		/*! \brief Get a const iterator to the end of the predecessor list */
		const_predecessor_iterator predecessors_end() const;

	public:
		/*! \brief insert a new block */
		block_iterator insert(ir::ControlFlowGraph::iterator block,
			block_iterator position);
		/*! \brief insert a new block to the end */
		block_iterator insert(ir::ControlFlowGraph::iterator block);

	public:
		/*! \brief insert a new instruction */
		iterator insert(ir::Instruction* instruction, iterator position);
		/*! \brief insert a new instruction to the end of the last block */
		iterator insert(ir::Instruction* instruction);
		
	public:
		/*! \brief Are there any instructions in the block? */
		bool empty() const;
		/*! \brief Get the number of instructions in the block */
		size_t instructions() const;
		/*! \brief Get the number of basic blocks in the block */
		size_t basicBlocks() const;
		
	private:
		typedef ir::ControlFlowGraph::BlockPointerVector BlockPointerVector;
		
	private:
		BlockPointerVector _blocks;
	};

public:
	typedef std::vector<Block>          BlockVector;
	typedef BlockVector::iterator       iterator;
	typedef BlockVector::const_iterator const_iterator;

public:
	/*! \brief Get an iterator to the first block */
	iterator begin();
	/*! \brief Get an iterator to the end of the block list */
	iterator end();
	
	/*! \brief Get a const iterator to the first block */
	const_iterator begin() const;
	/*! \brief Get a const iterator to the end of the block list */
	const_iterator end() const;

public:
	/*! \brief Get the number of basic blocks in the graph */
	size_t size() const;
	/*! \brief Is the graph empty? */
	bool empty() const;
	
protected:
	BlockVector _blocks;
};

}


