/*!
	\file BTree.h
	\date Wednesday May 13, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the BTree class
*/

#ifndef BTREE_H_INCLUDED
#define BTREE_H_INCLUDED

#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/macros.h>
#include <hydrazine/interface/ValueCompare.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

#include <stack>
#include <cstring>
#include <utility>

namespace hydrazine
{

	/*!
		\brief A Btree data structure providing the STL map interface.
	*/
	template< typename Key, typename Value, typename Compare = std::less<Key>, 
		typename _Allocator = std::allocator< std::pair< const Key, 
		Value > >, size_t PageSize = 1024 >
	class BTree
	{
		template< typename K, typename V, typename C, typename A, size_t P > 
			friend std::ostream& operator<<( std::ostream&, 
			const BTree< K, V, C, A, P >& );
		public:
			class Iterator;
			class ConstIterator;

		public:
			typedef Key key_type;
			typedef Value mapped_type;
			typedef std::pair< key_type, mapped_type > value_type;
		
		public:
			typedef typename _Allocator::template rebind< value_type >::other
				Allocator;

		public:
			typedef BTree type;	
			typedef Compare key_compare;
			typedef Allocator allocator_type;
			typedef typename Allocator::reference reference;
			typedef typename Allocator::const_reference const_reference;
			typedef Iterator iterator;
			typedef ConstIterator const_iterator;
			typedef typename Allocator::size_type size_type;
			typedef typename Allocator::difference_type difference_type;
			typedef typename Allocator::pointer pointer;
			typedef typename Allocator::const_pointer const_pointer;
			typedef std::reverse_iterator< iterator > reverse_iterator;
			typedef std::reverse_iterator< const_iterator > 
				const_reverse_iterator;
			typedef std::pair< iterator, bool > insertion;
			typedef ValueCompare< Compare, type > value_compare;

		private:
			Allocator _allocator;
			value_compare _compare;
			key_compare _keyCompare;
		
		private:
			class Body;
			class Node;
			class Leaf;
			
		private:
			typedef std::pair< Node*, size_type > StackElement;
			typedef std::stack< StackElement > Stack;
		
		private:
			static const size_type MaxNodes = MAX( 8, 
				PageSize / ( sizeof( key_type ) + sizeof( mapped_type ) ) );
			static const size_type MaxLeafs = MAX( 8, 
				PageSize / ( sizeof( key_type ) + sizeof( Node* ) ) );
			static const size_type MinNodes = MaxNodes / 2;
			static const size_type MinLeafs = MaxLeafs / 2;
	
		private:
			typedef typename _Allocator::template rebind< Body >::other
				BodyAllocator;
			typedef typename _Allocator::template rebind< Leaf >::other
				LeafAllocator;

		private:
			BodyAllocator _bodyAllocator;
			LeafAllocator _leafAllocator;
		
		private:
			/*!
				\brief A base class for an internal node.
			*/
			class Node
			{
				public:
					size_type level;
					size_type size;
					
				public:
					inline void construct( size_type l )
					{
						 level = l;
						 size = 0;
					}
					
					inline bool leaf() const
					{
						return level == 0;
					}
			};

			/*!
				\brief Body Node
			*/
			class Body : public Node
			{
				public:
					key_type keys[ MaxNodes ];
					Node* children[ MaxNodes + 1 ];
										
				public:
					inline void construct( const size_type level )
					{
						Node::construct( level );
					}
					
					inline bool full() const
					{
						return size == MaxNodes;
					}
					
					inline bool deficient() const
					{
						return size < MinNodes;
					}
					
					inline bool borderline() const
					{
						return size <= MinNodes;
					}
			};
			
			/*!
				\brief A leaf node
			*/
			class Leaf : public Node
			{
				public:
					Leaf* previous;
					Leaf* next;
					value_type data[ MaxLeafs ];
				
				public:
					inline void construct()
					{
						Node::construct( 0 );
						previous = 0;
						next = 0;
					}
					
					inline bool full() const
					{
						return size == MaxLeafs;
					}
					
					inline bool deficient() const
					{
						return size < MinLeafs;
					}
					
					inline bool borderline() const
					{
						return size <= MaxLeafs;
					}
			};

		public:
		
			class Iterator
			{
				friend class BTree;
				friend class Node;
				friend class Leaf;
				public:
					typedef Iterator iterator_type;
					typedef std::bidirectional_iterator_tag iterator_category;
					typedef BTree::value_type value_type;
					typedef BTree::pointer pointer;
					typedef BTree::reference reference;
					typedef BTree::difference_type difference_type;

				private:
					Leaf* _leaf;
					size_type _current;

				private:
					inline Iterator( Leaf* l, size_type c ) : 
						_leaf( l ), _current( c )
					{}
					
				public:
					inline Iterator( ) {}
					inline Iterator( const Iterator& i ) : _leaf( i._leaf ), 
						_current( i._current ) {}
					inline Iterator& operator=( const Iterator& i )
					{
						_leaf = i._leaf;
						_current = i._current;
						return *this;
					}
					
					inline reference operator*() const
					{
						assert( _current < _leaf->size );
						return _leaf->data[ _current ];
					}
					
					inline pointer operator->() const
					{
						assert( _current < _leaf->size );
						return &_leaf->data[ _current ];
					}
					
					inline Iterator& operator++()
					{
						if( _current + 1 < _leaf->size )
						{
							++_current;
						}
						else if( 0 != _leaf->next )
						{
							_current = 0;
							_leaf = _leaf->next;
						}
						else
						{
							_current = _leaf->size;
						}
						return *this;
					}
					
					inline Iterator operator++( int )
					{
						Iterator next( *this );
						++next;
						return std::move( next );	
					}
					
					inline Iterator& operator--()
					{
						if( _current != 0 )
						{
							--_current;
						}
						else if( _leaf->previous != 0 )
						{
							_leaf = _leaf->previous;
							_current = _leaf->size - 1;		
						}
						else
						{
							_current = 0;
						}
						return *this;
					}
					
					inline Iterator operator--( int )
					{
						Iterator previous( *this );
						--previous;
						return std::move( previous );	
					}
					
				public:
					bool operator==( const ConstIterator& i )
					{
						return _leaf == i._leaf && _current == i._current;
					}

					bool operator!=( const ConstIterator& i )
					{
						return _leaf != i._leaf || _current != i._current;
					}
				
			};

		public:
		
			class ConstIterator
			{
				friend class BTree;
				friend class Node;
				friend class Leaf;
				public:
					typedef Iterator iterator_type;
					typedef std::bidirectional_iterator_tag iterator_category;
					typedef BTree::value_type value_type;
					typedef BTree::const_pointer pointer;
					typedef BTree::const_reference reference;
					typedef BTree::difference_type difference_type;

				private:
					Leaf* _leaf;
					size_type _current;

				private:
					inline ConstIterator( Leaf* l, size_type c ) : 
						_leaf( l ), _current( c )
					{}
				
				public:
					inline ConstIterator() {}
					inline ConstIterator( const Iterator& i ) : 
						_leaf( i._leaf), _current( i._current ) {}
					inline ConstIterator( const ConstIterator& i ) : 
						_leaf( i._leaf), _current( i._current ) {}
					inline ConstIterator& operator=( const ConstIterator& i )
					{
						_leaf = i._leaf;
						_current = i._current;
						return *this;
					}
					
					inline reference operator*() const
					{
						assert( _current < _leaf->size );
						return _leaf->data[ _current ];
					}
					
					inline pointer operator->() const
					{
						assert( _current < _leaf->size );
						return &_leaf->data[ _current ];
					}
					
					inline ConstIterator& operator++()
					{
						if( _current + 1 < _leaf->size )
						{
							++_current;
						}
						else if( 0 != _leaf->next )
						{
							_current = 0;
							_leaf = _leaf->next;
						}
						else
						{
							_current = _leaf->size;
						}
						return *this;
					}
					
					inline ConstIterator operator++( int )
					{
						Iterator next( *this );
						++next;
						return std::move( next );	
					}
					
					inline ConstIterator& operator--()
					{
						if( _current != 0 )
						{
							--_current;
						}
						else if( _leaf->previous != 0 )
						{
							_leaf = _leaf->previous;
							_current = _leaf->size - 1;		
						}
						else
						{
							_current = 0;
						}
						return *this;
					}
					
					inline ConstIterator operator--( int )
					{
						Iterator previous( *this );
						--previous;
						return std::move( previous );	
					}
					
					const pointer& base() const
					{
						return _current;
					}
				
				public:
					bool operator==( const ConstIterator& i )
					{
						return _leaf == i._leaf && _current == i._current;
					}

					bool operator!=( const ConstIterator& i )
					{
						return _leaf != i._leaf || _current != i._current;
					}

			};
		
		private:
			class Stats
			{
				public:
					size_type elements;
					size_type leafs;
					size_type bodies;
				
				public:
					inline Stats() : elements( 0 ), leafs( 0 ), bodies( 0 ) {}
					inline size_type nodes() const { return leafs + bodies; }
			};
			
			/*!
				\brief Internal Data
			*/
		private:
			Node* _root;
			Leaf* _begin;
			Leaf* _end;
			Stats _stats;

			/*!
				\brief Copy/Construct/Destroy
			*/
		public:
			explicit BTree( const Compare& comp = Compare(), 
				const Allocator& alloc = Allocator() ) : 
				_allocator( alloc ), _compare( comp ),
				_root( 0 ), _begin( 0 ), _end( 0 ) 
			{
			}

			template < typename InputIterator >			
			BTree( InputIterator first, InputIterator last, 
				const Compare& comp = Compare(), 
				const Allocator& alloc = Allocator() ) : 
				_allocator( alloc ), _compare( comp ),
				_root( 0 ), _begin( 0 ), _end( 0 ) 
			{
				insert( first, last );
			}
			
			BTree( const BTree& tree ) : 
				_allocator( tree._allocator ), _compare( tree._compare ),
				_root( 0 ), _begin( 0 ), _end( 0 ) 
			{
				insert( tree.begin(), tree.end() );
			}
			
			~BTree()
			{
				clear();
			}
			
			BTree& operator=( const BTree& tree )
			{
				clear();
				insert( tree.begin(), tree.end() );
				return *this;
			}
			
			/*!
				\brief Iterators
			*/
		public:
			inline iterator begin()
			{
				return iterator( _begin, 0 );
			}
			
			inline const_iterator begin() const
			{
				return const_iterator( _begin, 0 );
			}
			
			inline iterator end()
			{
				return iterator( _end, _end != 0 ? _end->size : 0 );
			}
			
			inline const_iterator end() const
			{
				return const_iterator( _end, _end != 0 ? _end->size : 0 );
			}
			
			inline reverse_iterator rbegin()
			{
				return reverse_iterator( end() );
			}
			
			inline const_reverse_iterator rbegin() const
			{
				return const_reverse_iterator( end() );
			}
			
			inline reverse_iterator rend()
			{
				return reverse_iterator( begin() );
			}
			
			inline const_reverse_iterator rend() const
			{
				return const_reverse_iterator( begin() );
			}
			
			/*!
				\brief Capacity
			*/
		public:
			inline bool empty() const
			{
				return size() == 0;
			}
			
			inline size_type size() const
			{
				return _stats.elements;
			}
			
			inline size_type max_size() const
			{
				return _allocator.max_size();
			}
			
			/*!
				\brief Element access
			*/
		public:
			inline reference operator[]( const key_type& key )
			{
				iterator fi = lower_bound( key );
				
				if( fi == end() )
				{
					fi = _root->insert( fi, 
						std::make_pair( key, mapped_type() ) );					
				}
				else
				{
					if( _compare( *fi, key ) || _compare( key, *fi ) )
					{
						fi = _root->insert( fi, 
							std::make_pair( key, mapped_type() ) );
					}
				}
				
				return fi->second;
			}
		
			/*!
				\brief Modifiers
			*/
		public:
			inline insertion insert( const_reference x )
			{
				report( "Inserting " << x.first << "," << x.second );
				if( _root != 0 )
				{
					Stack stack;
					stack.push( StackElement( _root, 0 ) );
					
					_findInsertLeaf( stack, x );					
					insertion result( _insertLeaf( stack.top().first, x ) );

					if( result.first._leaf->full() )
					{
						assert( result.second );
						report( " Insert caused a split." );
						_split( stack, result.first );
					}
					
					return result;
				}
				else
				{
					report( " Creating the root node." );
					_createRoot();
					assert( _root->leaf() );
					Leaf* leaf = static_cast< Leaf* >( _root );
					leaf->data[ 0 ] = x;
					leaf->size = 1;
					return insertion( begin(), true );
				}
			}

			inline iterator insert( iterator position, const_reference x )
			{
				return insert( x ).first;
			}

			template < typename InputIterator >
			inline void insert( InputIterator first, InputIterator last )
			{
				for( InputIterator fi = first; fi != last; ++fi )
				{
					insert( *fi );
				}
			}
			
			inline void erase( iterator position )
			{
				assert( "Erase not implemented." == 0 );
			}
			
			inline size_type erase( const key_type& x )
			{
				iterator position = find( x );
				if( x != end() )
				{
					erase( position );
					return 1;
				}
				return 0;
			}
			
			inline void erase( iterator first, iterator last )
			{
				assert( "Erase not implemented." == 0 );
			}
			
			inline void swap( BTree& tree )
			{
				std::swap( _stats, tree._stats );
				std::swap( _begin, tree._begin );
				std::swap( _end, tree._end );
				std::swap( _root, tree._root );
			}

			inline void clear()
			{
				if( _root != 0 )
				{
					_clear( _root );
					_free( _root );
					
					_root = 0;
					_begin = 0;
					_end = 0;
					
					_stats = Stats();
				}
			}
		
		private:
			inline void _createRoot()
			{
				assert( _root == 0 );
				Leaf* leaf = _leafAllocator.allocate( 1 );
				leaf->construct();
				_root = leaf;
			}
			
			inline void _findInsertLeaf( Stack& stack, const_reference value )
			{
				for( size_type level = _root->level; level > 0; --level )
				{
					assert( stack.top().first->level == level );
					Body* body = static_cast< Body* >( stack.top().first );
					Key* key = std::lower_bound( body->keys, 
						body->keys + body->size, value, _compare );
					size_type position = key - body->keys;
					stack.push( StackElement( body->children[ position ], 
						position ) );
				}
				assert( stack.top().first->leaf() );
			}
			
			inline insertion _insertLeaf( Node* node, const_reference value )
			{
				Leaf* leaf = static_cast< Leaf* >( node );
				pointer position = std::lower_bound( leaf->data, 
					leaf->data + leaf->size, value, _compare );
				if( position != leaf->data + leaf->size )
				{
					if( !_compare( value, *position ) )
					{
						return insertion( iterator( leaf, 
							position - leaf->data ), false );
					}
				}
				std::copy_backward( position, leaf->data + leaf->size, 
					leaf->data + leaf->size + 1 );
				*position = value;
				++leaf->size;
				return insertion( iterator( leaf, position - leaf->data ), 
					true );
			} 
			
			inline void _split( Stack& stack, iterator& fi )
			{
				assert( stack.top().first->leaf() );
				Leaf* leaf = static_cast< Leaf* >( stack.top().first );
				size_type position = stack.top().second;
				assert( leaf == fi._leaf );
				
				if( !leaf->full() )
				{
					return;
				}
				
				Leaf* right = _splitLeaf( leaf, fi );
				stack.pop();
				
				if( stack.empty() )
				{
					report( "  Splitting the root." );
					_bumpRoot( leaf, right );
					return;
				}
				else
				{
					Body* parent = static_cast< Body* >( stack.top().first );
					_propagate( parent, leaf, right, position );
				}
				
				while( stack.size() > 1 )
				{
					Body* body = static_cast< Body* >( stack.top().first );
					size_type position = stack.top().second;
					stack.pop();
					if( !body->full() )
					{
						return;
					}
					size_type splitKey;
					Body* right = _splitBody( body, splitKey );
					Body* parent = static_cast< Body* >( stack.top().first );
					_propagate( parent, body, right, position, splitKey );
				}
				
				assert( stack.top().first == _root );
				
				Body* body = static_cast< Body* >( stack.top().first );
				stack.pop();
				if( !body->full() )
				{
					return;
				}
				report( "  Splitting the root." );
				size_type splitKey;
				Body* rightBody = _splitBody( body, splitKey );
				_bumpRoot( body, rightBody, splitKey );		
			}
			
			inline Leaf* _splitLeaf( Leaf* leaf, iterator& fi )
			{
				report( "   Splitting leaf node." );
				Leaf* right = _allocateLeaf();
				size_type median = leaf->size / 2;
				right->size = leaf->size - median;
				std::copy( leaf->data + median, leaf->data + leaf->size, 
					right->data );
				leaf->size = median;
				
				if( fi._current >= median )
				{
					fi._leaf = right;
					fi._current -= median;
				}
				
				if( _end == leaf )
				{
					_end = right;
				}
				
				return right;
			}
			
			inline Body* _splitBody( Body* body, size_type& key )
			{
				assert( body->full() );
				report( "   Splitting body node." );
				Body* right = _allocateBody( body->level );
				size_type median = body->size / 2;
				report( "    median index is " << median );
				right->size = body->size - median - 1;
				std::copy( body->keys + median + 1, body->keys + body->size, 
					right->keys );
				std::copy( body->children + median + 1, 
					body->children + body->size + 1, right->children );
				body->size = median;
				key = body->keys[ median ];
				report( "    right size is " << right->size );
				report( "    left size is " << body->size );
				report( "    split on key " << key );
				return right;
			}
			
			inline void _bumpRoot( Body* left, Body* right, size_type key )
			{
				assert( left == _root );
				report( "   Bumped to level " << (left->level + 1) 
					<< " on key " << key );
				Body* root = _allocateBody( left->level + 1 );
				root->size = 1;
				root->keys[0] = key;
				root->children[0] = left;
				root->children[1] = right;
				report( "    left is below " << left->keys[ left->size - 1 ] );	
				report( "    right is above " << key );	
				_root = root;	
			}
			
			inline void _bumpRoot( Leaf* left, Leaf* right )
			{
				assert( left == _root );
				report( "    Bumped to level 1" );
				Body* root = _allocateBody( 1 );
				root->size = 1;
				root->keys[0] = right->data[0].first;
				root->children[0] = left;	
				root->children[1] = right;
				_root = root;	
			}
		
			inline void _propagate( Body* parent, Leaf* left, 
				Leaf* right, size_type position )
			{
				assert( position <= parent->size );
				assert( parent->children[ position ] == left );
				report( "  Propagating the split up the tree at index " 
					<< position << "." );
				std::copy_backward( parent->keys + position, 
					parent->keys + parent->size, 
					parent->keys + parent->size + 1 );
				std::copy_backward( parent->children + position + 1, 
					parent->children + parent->size + 1, 
					parent->children + parent->size + 2 );
				++parent->size;
				parent->keys[ position ] = right->data[0].first;
				parent->children[ position + 1 ] = right;
			}
			
			inline void _propagate( Body* parent, Body* left, 
				Body* right, size_type position, size_type& key )
			{
				assert( position <= parent->size );
				assert( parent->children[ position ] == left );
				report( "  Propagating the split up the tree at index " 
					<< position << "." );
				std::copy_backward( parent->keys + position, 
					parent->keys + parent->size, 
					parent->keys + parent->size + 1 );
				std::copy_backward( parent->children + position + 1, 
					parent->children + parent->size + 1, 
					parent->children + parent->size + 2 );
				++parent->size;
				parent->keys[ position ] = key;
				parent->children[ position + 1 ] = right;
			}
		
			inline void _clear( Node* n )
			{
				if( !n->leaf() )
				{
					Body* body = static_cast< Body* >( n );
					for( size_type i = 0; i < n->size; ++i )
					{
						_clear( body->children[ i ] );
						_free( body->children[ i ] );
					}
				}
			}
		
			inline Body* _allocateBody( size_type level )
			{
				Body* body = _bodyAllocator.allocate( 1 );
				body->construct( level );
				++_stats.bodies;
				return body;
			}
			
			inline Leaf* _allocateLeaf( )
			{
				Leaf* leaf = _leafAllocator.allocate( 1 );
				leaf->construct();
				++_stats.leafs;
				return leaf;
			}
			
			inline void _free( Node* n )
			{
				if( n->leaf() )
				{
					_leafAllocator.deallocate( static_cast< Leaf* >( n ), 1 );
					--_stats.leafs;
				}
				else
				{
					_bodyAllocator.deallocate( static_cast< Body* >( n ), 1 );				
					--_stats.bodies;
				}
			}
			
			inline void _insert( iterator position, const_reference val )
			{
				assert( "_insert not implemented." == 0 );
			}
			
			/*!
				\brief Observers
			*/
		public:
			inline key_compare key_comp() const
			{
				return _compare.compare;
			}
			
			inline value_compare value_comp() const
			{
				return _compare;
			}
			
			/*!
				\brief Map Operations
			*/
		public:
			inline iterator find( const key_type& x )
			{
				iterator result = lower_bound( x );
				report( "Finding key " << x );
				if( result != end() )
				{
					if( !_keyCompare( x, result->first ) )
					{
						report( " Found value " 
							<< result->second );
						return result;
					}
				}
				reportE( result == end(), " Could not find value for key." );
				return end();
			}
			
			inline const_iterator find( const key_type& x ) const
			{
				const_iterator result = lower_bound( x );
				report( "Finding key " << x );
				if( result != end() )
				{
					if( !_compareKey( x, result->first ) )
					{
						report( " Found value " 
							<< result->second );
						return result;
					}
				}
				reportE( result == end(), " Could not find value for key." );
				return end();
			}
			
			inline size_type count( const key_type& x ) const
			{
				return find( x ) != end();
			}
			
			inline iterator lower_bound( const key_type& x )
			{
				if ( _root == 0 ) return end();
				Node* node = _root;
				
				while( !node->leaf() )
				{
					Body* body = static_cast< Body* >( node );
					Key* key = std::lower_bound( body->keys, 
						body->keys + body->size, x, _keyCompare );
					node = body->children[ key - body->keys ];
				}
				
				Leaf* leaf = static_cast< Leaf* >( node );
				value_type* fi = std::lower_bound( leaf->data, 
					leaf->data + leaf->size, x, _compare );
				size_type index = fi - leaf->data;
				if( index == leaf->size )
				{
					return end();
				}
				return iterator( leaf, index );
			}
			
			inline const_iterator lower_bound( const key_type& x ) const
			{
				return lower_bound( x );
			}
			
			inline iterator upper_bound( const key_type& x )
			{
				iterator result = lower_bound( x );
				if( result != end() )
				{
					if( !_compare( x, *result ) )
					{
						++result;
					}
				}
				return result;
			}

			inline const_iterator upper_bound( const key_type& x ) const
			{
				const_iterator result = lower_bound( x );
				if( result != end() )
				{
					if( !_compare( x, *result ) )
					{
						++result;
					}
				}
				return result;
			}

			inline std::pair< iterator, iterator > equal_range( const key_type& x )
			{
				std::pair< iterator, iterator > result;
				result.first = lower_bound( x );
				result.second = result.first;
				if( result.second != end() )
				{
					if( !_compare( x, *result.second ) )
					{
						++result.second;
					}
				}
				return result;
			}
			
			inline std::pair< const_iterator, const_iterator > equal_range( 
				const key_type& x ) const
			{
				std::pair< iterator, iterator > result;
				result.first = lower_bound( x );
				result.second = result.first;
				if( result.second != end() )
				{
					if( !_compare( x, *result.second ) )
					{
						++result.second;
					}
				}
				return result;			
			}

	};
	
	template <typename Key, typename T, typename Compare, typename Allocator, 
		size_t PageSize>
	bool operator==(const BTree< Key, T, Compare, Allocator, PageSize>& x,
		const BTree< Key, T, Compare, Allocator, PageSize >& y)
	{
		if( x.size() != y.size() )
		{
			return false;
		}
		return std::equal( x.begin(), x.end(), y.begin() );
	}

	template < typename Key, typename T, typename Compare, typename Allocator, 
		size_t PageSize >
	bool operator< (const BTree< Key, T, Compare, Allocator, PageSize >& x,
		const BTree< Key, T, Compare, Allocator, PageSize >& y)
	{
		return std::lexicographical_compare( x.begin(), x.end(), y.begin(), 
			y.end(), x.value_comp() );
	}

	template < typename Key, typename T, typename Compare, typename Allocator, 
		size_t PageSize >
	bool operator!=(const BTree< Key, T, Compare, Allocator, PageSize >& x,
		const BTree< Key, T, Compare, Allocator, PageSize >& y)
	{
		return !( x == y );
	}
	
	template < typename Key, typename T, typename Compare, typename Allocator, 
		size_t PageSize >
	bool operator> (const BTree< Key, T, Compare, Allocator, PageSize >& x,
		const BTree< Key, T, Compare, Allocator, PageSize >& y)
	{
		return y < x;
	}
	
	template < typename Key, typename T, typename Compare, typename Allocator, 
		size_t PageSize >
	bool operator>=(const BTree< Key, T, Compare, Allocator, PageSize >& x,
		const BTree< Key, T, Compare, Allocator, PageSize >& y)
	{
		return !( x < y );
	}
	
	template < typename Key, typename T, typename Compare, typename Allocator, 
		size_t PageSize >
	bool operator<=(const BTree< Key, T, Compare, Allocator, PageSize >& x,
		const BTree< Key, T, Compare, Allocator, PageSize >& y )
	{
		return !( x > y );
	}
	
	// specialized algorithms:
	template < typename Key, typename T, typename Compare, typename Allocator, 
		size_t PageSize >
	void swap( BTree< Key, T, Compare, Allocator, PageSize >& x,
		BTree< Key, T, Compare, Allocator, PageSize >& y )
	{
		x.swap( y );
	}
	
	template < typename Key, typename T, typename Compare, typename Allocator, 
		size_t PageSize >
	std::ostream& operator<<( std::ostream& out, 
		const BTree< Key, T, Compare, Allocator, PageSize >& tree )
	{
		typedef BTree< Key, T, Compare, Allocator, PageSize > BTree;
		typedef std::stack< const typename BTree::Node* > NodeStack;

		if( tree._root == 0 )
		{
			return out;
		}
		
		NodeStack stack;

		stack.push( tree._root );

		out << "digraph BTree_" << &tree << " {\n";
		out << "\tnode [ shape = record ];\n\n";
		while( !stack.empty() )
		{
			const typename BTree::Node* node = stack.top();
			stack.pop();
			out << "\tnode_" << node << " [ ";
			if( !node->leaf() )
			{
				out << "color = red, ";					
			} 
			else
			{
				out << "color = black, ";
			}					
			out << "label = \"{";
			if( node->leaf() )
			{
				const typename BTree::Leaf* leaf 
					= static_cast< const typename BTree::Leaf* >( node );
				out << "<head> leaf_" << leaf->data->first 
					<< " (" << leaf->size << ")" << " | { { ";
				for( typename BTree::const_pointer fi = leaf->data; 
					fi != leaf->data + leaf->size; ++fi )
				{
					if( fi != leaf->data )
					{
						out << "| { ";
					}
					out << "<key_" << ( fi - leaf->data ) 
						<< "> " << fi->first << " | " 
						<< fi->second << " } ";
				}
				out << "} }\"];\n";
			}
			else
			{
				const typename BTree::Body* body 
					= static_cast< const typename BTree::Body* >( node );
				out << "<head> node_" << *body->keys << " (" 
					<< body->size << ")" << "(level_" << body->level 
					<< ")" << " | { {";
				out << "<key_0> previous } ";
				for( const typename BTree::key_type* ki = body->keys;
					ki != body->keys + body->size; ++ki )
				{
					out << "| { ";
					out << "<key_" << ( ki - body->keys + 1) 
						<< "> " << *ki << " } ";
				}
				out << "} }\"];\n";

				for( typename BTree::Node* const* ni = &body->children[0];
					ni != body->children + body->size + 1; ++ni )
				{
					out << "\tnode_" << node << ":key_" 
						<< ( ni - body->children ) << " -> node_"
						<< *ni << ":head;\n";
					stack.push( *ni );
				}
				out << "\n";
			}

		}
		out << "}";
		return out;
	}
		
}

#endif

