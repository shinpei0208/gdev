/*! \file   ProgramStructureGraph.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Monday August 1, 2011
	\brief  The source file for the ProgramStructureGraph class.
*/

// Ocelot Includes
#include <ocelot/analysis/interface/ProgramStructureGraph.h>

namespace analysis
{

ProgramStructureGraph::Block::block_iterator::block_iterator()
{

}

ProgramStructureGraph::Block::block_iterator::block_iterator(
	const block_iterator& i)
: _iterator(i._iterator), _end(i._end)
{

}

ProgramStructureGraph::Block::block_iterator::block_iterator(
	const basic_block_iterator& i, const basic_block_iterator& b,
	const basic_block_iterator& e)
: _iterator(i), _begin(b), _end(e)
{

}

ProgramStructureGraph::Block::block_iterator::reference
	ProgramStructureGraph::Block::block_iterator::operator*() const
{
	return **_iterator;
}

ProgramStructureGraph::Block::block_iterator::pointer
	ProgramStructureGraph::Block::block_iterator::operator->() const
{
	return &**_iterator;
}

ProgramStructureGraph::Block::block_iterator::self&
	ProgramStructureGraph::Block::block_iterator::operator++()
{
	while(true)
	{
		if(_iterator != _end)
		{
			++_iterator;
			if(!(*_iterator)->instructions.empty())
			{
				break;
			}
		}
		else
		{
			break;
		}
	}
	
	return *this;
}

ProgramStructureGraph::Block::block_iterator::self
	ProgramStructureGraph::Block::block_iterator::operator++(int)
{
	self copy = *this;
	
	++*this;
	
	return copy;
}

ProgramStructureGraph::Block::block_iterator::self&
	ProgramStructureGraph::Block::block_iterator::operator--()
{
	while(true)
	{
		if(_iterator != _begin)
		{
			--_iterator;
			if(!(*_iterator)->instructions.empty())
			{
				break;
			}
		}
		else
		{
			break;
		}
	}
	
	return *this;
}

ProgramStructureGraph::Block::block_iterator::self
	ProgramStructureGraph::Block::block_iterator::operator--(int)
{
	self copy = *this;
	
	--*this;
	
	return copy;
}

bool ProgramStructureGraph::Block::block_iterator::operator==(
	const self& i) const
{
	return i._iterator == _iterator && i._end == _end;
}

bool ProgramStructureGraph::Block::block_iterator::operator!=(
	const self& i) const
{
	return !(i == *this);
}

bool ProgramStructureGraph::Block::block_iterator::begin() const
{
	return _iterator == _begin;
}

bool ProgramStructureGraph::Block::block_iterator::end() const
{
	return _iterator == _end;
}

ProgramStructureGraph::Block::const_block_iterator::const_block_iterator()
{

}

ProgramStructureGraph::Block::const_block_iterator::const_block_iterator(
	const block_iterator& i)
: _iterator(i._iterator), _end(i._end)
{

}

ProgramStructureGraph::Block::const_block_iterator::const_block_iterator(
	const const_block_iterator& i)
: _iterator(i._iterator), _end(i._end)
{

}

ProgramStructureGraph::Block::const_block_iterator::const_block_iterator(
	const const_basic_block_iterator& i, const const_basic_block_iterator& b,
	const const_basic_block_iterator& e)
: _iterator(i), _begin(b), _end(e)
{

}

ProgramStructureGraph::Block::const_block_iterator::reference
	ProgramStructureGraph::Block::const_block_iterator::operator*() const
{
	return **_iterator;
}

ProgramStructureGraph::Block::const_block_iterator::pointer
	ProgramStructureGraph::Block::const_block_iterator::operator->() const
{
	return &**_iterator;
}

ProgramStructureGraph::Block::const_block_iterator::self&
	ProgramStructureGraph::Block::const_block_iterator::operator++()
{
	while(true)
	{
		if(_iterator != _end)
		{
			++_iterator;
			if(!(*_iterator)->instructions.empty())
			{
				break;
			}
		}
		else
		{
			break;
		}
	}
	
	return *this;
}

ProgramStructureGraph::Block::const_block_iterator::self
	ProgramStructureGraph::Block::const_block_iterator::operator++(int)
{
	self copy = *this;
	
	++*this;
	
	return copy;
}

ProgramStructureGraph::Block::const_block_iterator::self&
	ProgramStructureGraph::Block::const_block_iterator::operator--()
{
	while(true)
	{
		if(_iterator != _begin)
		{
			--_iterator;
			if(!(*_iterator)->instructions.empty())
			{
				break;
			}
		}
		else
		{
			break;
		}
	}
	
	return *this;
}

ProgramStructureGraph::Block::const_block_iterator::self
	ProgramStructureGraph::Block::const_block_iterator::operator--(int)
{
	self copy = *this;
	
	--*this;
	
	return copy;
}

bool ProgramStructureGraph::Block::const_block_iterator::operator==(
	const self& i) const
{
	return i._iterator == _iterator && i._end == _end;
}

bool ProgramStructureGraph::Block::const_block_iterator::operator!=(
	const self& i) const
{
	return !(i == *this);
}

bool ProgramStructureGraph::Block::const_block_iterator::begin() const
{
	return _iterator == _begin;
}

bool ProgramStructureGraph::Block::const_block_iterator::end() const
{
	return _iterator == _end;
}

ProgramStructureGraph::Block::iterator::iterator()
{

}

ProgramStructureGraph::Block::iterator::iterator(
	const block_iterator& b, const instruction_iterator& i)
: _instruction(i), _basicBlock(b)
{

}

ProgramStructureGraph::Block::iterator::reference
	ProgramStructureGraph::Block::iterator::operator*() const
{
	return *_instruction;
}

ProgramStructureGraph::Block::iterator::pointer
	ProgramStructureGraph::Block::iterator::operator->() const
{
	return &*_instruction;
}

ProgramStructureGraph::Block::iterator::self&
	ProgramStructureGraph::Block::iterator::operator++()
{
	if(_instruction != _basicBlock->instructions.end())
	{
		++_instruction;
		
		if(_instruction == _basicBlock->instructions.end())
		{
			++_basicBlock;
			_instruction = _basicBlock->instructions.begin();
		}
	}
	
	return *this;
}

ProgramStructureGraph::Block::iterator::self
	ProgramStructureGraph::Block::iterator::operator++(int)
{
	self copy = *this;
	
	++*this;
	
	return copy;
}

ProgramStructureGraph::Block::iterator::self&
	ProgramStructureGraph::Block::iterator::operator--()
{
	if(_instruction != _basicBlock->instructions.begin())
	{
		--_instruction;
	}
	else
	{
		--_basicBlock;
		if(_basicBlock->instructions.empty())
		{
			_instruction = _basicBlock->instructions.begin();
		}
		else
		{
			_instruction = --_basicBlock->instructions.end();
		}
	}
	
	return *this;
}

ProgramStructureGraph::Block::iterator::self
	ProgramStructureGraph::Block::iterator::operator--(int)
{
	self copy = *this;
	
	--*this;
	
	return copy;
}

bool ProgramStructureGraph::Block::iterator::operator==(const self& i) const
{
	return i._instruction == _instruction && i._basicBlock == _basicBlock;
}

bool ProgramStructureGraph::Block::iterator::operator!=(const self& i) const
{
	return !(i == *this);
}
		
ProgramStructureGraph::Block::const_iterator::const_iterator()
{

}

ProgramStructureGraph::Block::const_iterator::const_iterator(const iterator& i)
: _instruction(i._instruction), _basicBlock(i._basicBlock)
{

}

ProgramStructureGraph::Block::const_iterator::const_iterator(
	const const_iterator& i)
: _instruction(i._instruction), _basicBlock(i._basicBlock)
{

}

ProgramStructureGraph::Block::const_iterator::const_iterator(
	const const_block_iterator& b,
	const const_instruction_iterator& i)
: _instruction(i), _basicBlock(b)
{

}

ProgramStructureGraph::Block::const_iterator::reference
	ProgramStructureGraph::Block::const_iterator::operator*() const
{
	return const_cast<reference>(*_instruction);
}

ProgramStructureGraph::Block::const_iterator::pointer
	ProgramStructureGraph::Block::const_iterator::operator->() const
{
	return const_cast<pointer>(&*_instruction);
}

ProgramStructureGraph::Block::const_iterator::self&
	ProgramStructureGraph::Block::const_iterator::operator++()
{
	if(_instruction != _basicBlock->instructions.end())
	{
		++_instruction;
		
		if(_instruction == _basicBlock->instructions.end())
		{
			++_basicBlock;
			_instruction = _basicBlock->instructions.begin();
		}
	}
	
	return *this;
}

ProgramStructureGraph::Block::const_iterator::self 
	ProgramStructureGraph::Block::const_iterator::operator++(int)
{
	self copy = *this;
	
	++*this;
	
	return copy;
}

ProgramStructureGraph::Block::const_iterator::self& 
	ProgramStructureGraph::Block::const_iterator::operator--()
{
	if(_instruction != _basicBlock->instructions.begin())
	{
		--_instruction;
	}
	else
	{
		--_basicBlock;
		if(_basicBlock->instructions.empty())
		{
			_instruction = _basicBlock->instructions.begin();
		}
		else
		{
			_instruction = --_basicBlock->instructions.end();
		}
	}
	
	return *this;
}

ProgramStructureGraph::Block::const_iterator::self 
	ProgramStructureGraph::Block::const_iterator::operator--(int)
{
	self copy = *this;
	
	--*this;
	
	return copy;
}

bool ProgramStructureGraph::Block::const_iterator::operator==(
	const self& i) const
{
	return i._instruction == _instruction && i._basicBlock == _basicBlock;
}

bool ProgramStructureGraph::Block::const_iterator::operator!=(
	const self& i) const
{
	return !(*this == i);
}

ProgramStructureGraph::Block::successor_iterator::successor_iterator()
{
	
}

ProgramStructureGraph::Block::successor_iterator::successor_iterator(
	const successor_iterator& i)
: _block(i._block), _successor(i._successor)
{

}

ProgramStructureGraph::Block::successor_iterator::successor_iterator(
	const block_iterator& block, const basic_block_iterator& successor)
: _block(block), _successor(successor)
{

}

ProgramStructureGraph::Block::successor_iterator::reference 
	ProgramStructureGraph::Block::successor_iterator::operator*() const
{
	return **_successor;
}

ProgramStructureGraph::Block::successor_iterator::pointer 
	ProgramStructureGraph::Block::successor_iterator::operator->() const
{
	return &**_successor;
}

ProgramStructureGraph::Block::successor_iterator::self& 
	ProgramStructureGraph::Block::successor_iterator::operator++()
{
	++_successor;
	
	while(!_block.end() && _successor == _block->successors.end())
	{
		++_block;
		_successor = _block->successors.begin();
	}
	
	return *this;
}

ProgramStructureGraph::Block::successor_iterator::self 
	ProgramStructureGraph::Block::successor_iterator::operator++(int)
{
	self copy = *this;
	
	++*this;

	return copy;
}

ProgramStructureGraph::Block::successor_iterator::self& 
	ProgramStructureGraph::Block::successor_iterator::operator--()
{
	if(_successor != _block->successors.begin())
	{
		--_successor;
	}
	else
	{
		while(!_block.begin() && _block->successors.empty())
		{
			--_block;
		}
		_successor = --_block->successors.end();
	}
	
	return *this;
}

ProgramStructureGraph::Block::successor_iterator::self 
	ProgramStructureGraph::Block::successor_iterator::operator--(int)
{
	self copy = *this;
	
	--*this;
	
	return copy;
}

bool ProgramStructureGraph::Block::successor_iterator::operator==(
	const self& i) const
{
	return i._block == _block && i._successor != _successor;
}

bool ProgramStructureGraph::Block::successor_iterator::operator!=(
	const self& i) const
{
	return !(i == *this);
}
				
ProgramStructureGraph::Block::const_successor_iterator::const_successor_iterator()
{
	
}

ProgramStructureGraph::Block::const_successor_iterator::const_successor_iterator(
	const const_successor_iterator& i)
: _block(i._block), _successor(i._successor)
{

}

ProgramStructureGraph::Block::const_successor_iterator::const_successor_iterator(
	const successor_iterator& i)
: _block(i._block), _successor(i._successor)
{

}

ProgramStructureGraph::Block::const_successor_iterator::const_successor_iterator(
	const const_block_iterator& block,
	const const_basic_block_iterator& successor)
: _block(block), _successor(successor)
{

}

ProgramStructureGraph::Block::const_successor_iterator::reference 
	ProgramStructureGraph::Block::const_successor_iterator::operator*() const
{
	return **_successor;
}

ProgramStructureGraph::Block::const_successor_iterator::pointer 
	ProgramStructureGraph::Block::const_successor_iterator::operator->() const
{
	return &**_successor;
}

ProgramStructureGraph::Block::const_successor_iterator::self& 
	ProgramStructureGraph::Block::const_successor_iterator::operator++()
{
	++_successor;
	
	while(!_block.end() && _successor == _block->successors.end())
	{
		++_block;
		_successor = _block->successors.begin();
	}
	
	return *this;
}

ProgramStructureGraph::Block::const_successor_iterator::self 
	ProgramStructureGraph::Block::const_successor_iterator::operator++(int)
{
	self copy = *this;
	
	++*this;

	return copy;
}

ProgramStructureGraph::Block::const_successor_iterator::self& 
	ProgramStructureGraph::Block::const_successor_iterator::operator--()
{
	if(_successor != _block->successors.begin())
	{
		--_successor;
	}
	else
	{
		while(!_block.begin() && _block->successors.empty())
		{
			--_block;
		}
		_successor = --_block->successors.end();
	}
	
	return *this;
}

ProgramStructureGraph::Block::const_successor_iterator::self 
	ProgramStructureGraph::Block::const_successor_iterator::operator--(int)
{
	self copy = *this;
	
	--*this;
	
	return copy;
}

bool ProgramStructureGraph::Block::const_successor_iterator::operator==(
	const self& i) const
{
	return i._block == _block && i._successor != _successor;
}

bool ProgramStructureGraph::Block::const_successor_iterator::operator!=(
	const self& i) const
{
	return !(i == *this);
}

ProgramStructureGraph::Block::predecessor_iterator::predecessor_iterator()
{

}

ProgramStructureGraph::Block::predecessor_iterator::predecessor_iterator(
	const predecessor_iterator& i)
: _block(i._block), _predecessor(i._predecessor)
{

}

ProgramStructureGraph::Block::predecessor_iterator::predecessor_iterator(
	const block_iterator& b, const basic_block_iterator& p)
: _block(b), _predecessor(p)
{

}

ProgramStructureGraph::Block::predecessor_iterator::reference 
	ProgramStructureGraph::Block::predecessor_iterator::operator*() const
{
	return **_predecessor;
}

ProgramStructureGraph::Block::predecessor_iterator::pointer 
	ProgramStructureGraph::Block::predecessor_iterator::operator->() const
{
	return &**_predecessor;
}

ProgramStructureGraph::Block::predecessor_iterator::self& 
	ProgramStructureGraph::Block::predecessor_iterator::operator++()
{
	++_predecessor;
	
	while(!_block.end() && _predecessor == _block->predecessors.end())
	{
		++_block;
		_predecessor = _block->predecessors.begin();
	}
	
	return *this;
}

ProgramStructureGraph::Block::predecessor_iterator::self 
	ProgramStructureGraph::Block::predecessor_iterator::operator++(int)
{
	self copy = *this;
	
	++*this;
	
	return copy;
}

ProgramStructureGraph::Block::predecessor_iterator::self& 
	ProgramStructureGraph::Block::predecessor_iterator::operator--()
{
	if(_predecessor != _block->predecessors.begin())
	{
		--_predecessor;
	}
	else
	{
		while(!_block.begin() && _block->predecessors.empty())
		{
			--_block;
		}
		_predecessor = --_block->predecessors.end();
	}

	return *this;
}

ProgramStructureGraph::Block::predecessor_iterator::self 
	ProgramStructureGraph::Block::predecessor_iterator::operator--(int)
{
	self copy = *this;
	
	++*this;
	
	return copy;
}

bool ProgramStructureGraph::Block::predecessor_iterator::operator==(
	const self& i) const
{
	return i._block == _block && i._predecessor == _predecessor;
}

bool ProgramStructureGraph::Block::predecessor_iterator::operator!=(
	const self& i) const
{
	return !(i == *this);
}

ProgramStructureGraph::Block::const_predecessor_iterator::const_predecessor_iterator()
{

}

ProgramStructureGraph::Block::const_predecessor_iterator::const_predecessor_iterator(
	const predecessor_iterator& i)
: _block(i._block), _predecessor(i._predecessor)
{

}

ProgramStructureGraph::Block::const_predecessor_iterator::const_predecessor_iterator(
	const const_predecessor_iterator& i)
: _block(i._block), _predecessor(i._predecessor)
{

}

ProgramStructureGraph::Block::const_predecessor_iterator::const_predecessor_iterator(
	const const_block_iterator& b, const const_basic_block_iterator& p)
: _block(b), _predecessor(p)
{

}

ProgramStructureGraph::Block::const_predecessor_iterator::reference 
	ProgramStructureGraph::Block::const_predecessor_iterator::operator*() const
{
	return **_predecessor;
}

ProgramStructureGraph::Block::const_predecessor_iterator::pointer 
	ProgramStructureGraph::Block::const_predecessor_iterator::operator->() const
{
	return &**_predecessor;
}

ProgramStructureGraph::Block::const_predecessor_iterator::self& 
	ProgramStructureGraph::Block::const_predecessor_iterator::operator++()
{
	++_predecessor;
	
	while(!_block.end() && _predecessor == _block->predecessors.end())
	{
		++_block;
		_predecessor = _block->predecessors.begin();
	}
	
	return *this;
}

ProgramStructureGraph::Block::const_predecessor_iterator::self 
	ProgramStructureGraph::Block::const_predecessor_iterator::operator++(int)
{
	self copy = *this;
	
	++*this;
	
	return copy;
}

ProgramStructureGraph::Block::const_predecessor_iterator::self& 
	ProgramStructureGraph::Block::const_predecessor_iterator::operator--()
{
	if(_predecessor != _block->predecessors.begin())
	{
		--_predecessor;
	}
	else
	{
		while(!_block.begin() && _block->predecessors.empty())
		{
			--_block;
		}
		_predecessor = --_block->predecessors.end();
	}

	return *this;
}

ProgramStructureGraph::Block::const_predecessor_iterator::self 
	ProgramStructureGraph::Block::const_predecessor_iterator::operator--(int)
{
	self copy = *this;
	
	++*this;
	
	return copy;
}

bool ProgramStructureGraph::Block::const_predecessor_iterator::operator==(
	const self& i) const
{
	return i._block == _block && i._predecessor == _predecessor;
}

bool ProgramStructureGraph::Block::const_predecessor_iterator::operator!=(
	const self& i) const
{
	return !(i == *this);
}

ProgramStructureGraph::Block::iterator ProgramStructureGraph::Block::begin()
{
	return iterator(block_begin(), block_begin()->instructions.begin());
}

ProgramStructureGraph::Block::iterator ProgramStructureGraph::Block::end()
{
	return iterator(block_end(), block_end()->instructions.begin());
}

ProgramStructureGraph::Block::const_iterator
	ProgramStructureGraph::Block::begin() const
{
	return const_iterator(block_begin(), block_begin()->instructions.begin());
}

ProgramStructureGraph::Block::const_iterator
	ProgramStructureGraph::Block::end() const
{
	return const_iterator(block_end(), block_end()->instructions.begin());
}

ProgramStructureGraph::Block::block_iterator
	ProgramStructureGraph::Block::block_begin()
{
	return block_iterator(_blocks.begin(), _blocks.begin(), _blocks.end());
}

ProgramStructureGraph::Block::block_iterator
	ProgramStructureGraph::Block::block_end()
{
	return block_iterator(_blocks.end(), _blocks.begin(), _blocks.end());
}

ProgramStructureGraph::Block::const_block_iterator
	ProgramStructureGraph::Block::block_begin() const
{
	return const_block_iterator(_blocks.begin(),
		_blocks.begin(), _blocks.end());
}

ProgramStructureGraph::Block::const_block_iterator
	ProgramStructureGraph::Block::block_end() const
{
	return const_block_iterator(_blocks.end(),
		_blocks.begin(), _blocks.end());
}

ProgramStructureGraph::Block::successor_iterator
	ProgramStructureGraph::Block::successors_begin()
{
	return successor_iterator(block_begin(), block_begin()->successors.begin());
}
	
ProgramStructureGraph::Block::successor_iterator
	ProgramStructureGraph::Block::successors_end()
{
	return successor_iterator(block_end(), block_end()->successors.begin());
}

ProgramStructureGraph::Block::const_successor_iterator 
	ProgramStructureGraph::Block::successors_begin() const
{
	return const_successor_iterator(block_begin(),
		block_begin()->successors.begin());
}

ProgramStructureGraph::Block::const_successor_iterator 
	ProgramStructureGraph::Block::successors_end() const
{
	return const_successor_iterator(block_end(),
		block_end()->successors.begin());
}

ProgramStructureGraph::Block::predecessor_iterator 
	ProgramStructureGraph::Block::predecessors_begin()
{
	return predecessor_iterator(block_begin(),
		block_begin()->predecessors.begin());
}

ProgramStructureGraph::Block::predecessor_iterator
	ProgramStructureGraph::Block::predecessors_end()
{
	return predecessor_iterator(block_end(),
		block_end()->predecessors.begin());
}

ProgramStructureGraph::Block::const_predecessor_iterator 
	ProgramStructureGraph::Block::predecessors_begin() const
{
	return const_predecessor_iterator(block_begin(),
		block_begin()->predecessors.begin());
}

ProgramStructureGraph::Block::const_predecessor_iterator 
	ProgramStructureGraph::Block::predecessors_end() const
{
	return const_predecessor_iterator(block_end(),
		block_end()->predecessors.begin());
}

ProgramStructureGraph::Block::block_iterator
	ProgramStructureGraph::Block::insert(ir::ControlFlowGraph::iterator block,
	block_iterator position)
{
	return block_iterator(_blocks.insert(position._iterator, block),
		_blocks.begin(), _blocks.end());
}

ProgramStructureGraph::Block::block_iterator
	ProgramStructureGraph::Block::insert(ir::ControlFlowGraph::iterator block)
{
	return insert(block, block_end());
}

ProgramStructureGraph::Block::iterator ProgramStructureGraph::Block::insert(
	ir::Instruction* instruction, iterator position)
{
	return iterator(position._basicBlock,
		position._basicBlock->instructions.insert(
		position._instruction, instruction));
}

ProgramStructureGraph::Block::iterator ProgramStructureGraph::Block::insert(
	ir::Instruction* instruction)
{
	return insert(instruction, end());
}

bool ProgramStructureGraph::Block::empty() const
{
	return _blocks.empty();
}

size_t ProgramStructureGraph::Block::instructions() const
{
	size_t sum = 0;
	
	for(const_block_iterator i = block_begin(); i != block_end(); ++i)
	{
		sum += i->instructions.size();
	}
	
	return sum;
}

size_t ProgramStructureGraph::Block::basicBlocks() const
{
	return _blocks.size();
}

ProgramStructureGraph::iterator ProgramStructureGraph::begin()
{
	return _blocks.begin();
}

ProgramStructureGraph::iterator ProgramStructureGraph::end()
{
	return _blocks.end();
}

ProgramStructureGraph::const_iterator ProgramStructureGraph::begin() const
{
	return _blocks.begin();
}

ProgramStructureGraph::const_iterator ProgramStructureGraph::end() const
{
	return _blocks.end();
}

size_t ProgramStructureGraph::size() const
{
	return _blocks.size();
}

bool ProgramStructureGraph::empty() const
{
	return _blocks.empty();
}

}


