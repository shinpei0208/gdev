/*!

	\file XmlTree.cpp
	
	\date Saturday September 13, 2008
	
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	
	\brief The source file for the XmlTree class.

*/

#ifndef XML_TREE_CPP_INCLUDED
#define XML_TREE_CPP_INLCLUDED

#include <hydrazine/interface/XmlTree.h>
#include <cassert>

#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace hydrazine
{

	////////////////////////////////////////////////////////////////////////////
	// XmlTree::XmlTreeNode
	XmlTree::XmlTreeNode::Type XmlTree::XmlTreeNode::getType() const
	{
	
		return type;
	
	}
	////////////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////////////
	// XmlTree::iterator
	XmlTree::iterator& XmlTree::iterator::operator=( 
		const XmlTree::iterator& i )
	{
	
		if( &i != this )
		{
		
			node = i.node;
			successor = i.successor;
		
		}
		
		return *this;
	
	}
	
	XmlTree::XmlTreeNode& XmlTree::iterator::operator*() const
	{
	
		return *node;
	
	}
	
	XmlTree::XmlTreeNode* XmlTree::iterator::operator->() const
	{
	
		return &(*node);
	
	}
	
	bool XmlTree::iterator::operator==( const XmlTree::iterator& i ) const
	{
	
		return node == i.node && i.successor == successor;
	
	}
	
	bool XmlTree::iterator::operator!=( const XmlTree::iterator& i ) const
	{
	
		return node != i.node || i.successor != successor;
	
	}
	
	XmlTree::iterator& XmlTree::iterator::operator++()
	{
		
		report( "Incrementing iterator currently at node " 
			<< node->identifier );
		switch( node->type )
		{
	
			case XmlTree::XmlTreeNode::Root:
			{

				report( " Hit the root node with name ( " << 
					node->identifier << " )." );			
				assert( successor.top() == node->successors.begin() );
			
				if( node->successors.empty() )
				{
			
					node = node->parent;
					report( "  Ascending to the end." );
			
				}
				else
				{
			
					node = successor.top()->second;
					successor.push( node->successors.begin() );
					report( "  Descending to node " << node->identifier );
			
				}
			
				break;
			
			}
		
			case XmlTree::XmlTreeNode::Intermediate:
			{
			
				assert( !node->successors.empty() );
				assert( successor.top() != node->successors.end() );
				
				report( " Hit intermediate node " << node->identifier );
				
				//always descend
				node = successor.top()->second;
				successor.push( node->successors.begin() );
				
				report( "  Descending to first successor " 
					<< node->identifier );

				break;
			
			}
		
			case XmlTree::XmlTreeNode::Leaf:
			{
				
				report( " Hit leaf node " << node->identifier );
				
				// we should be at the end since this is a leaf
				assert( successor.top() == node->successors.end() );
				
				assert( !successor.empty() );
				successor.pop();
				report( "  Ascending to node " << 
					node->parent->identifier );					
				node = node->parent;
				assert( successor.top() != node->successors.end() );
				++successor.top();
				
				// ascend
				while( node->successors.end() == successor.top() )
				{
					
					assert( !successor.empty() );
					successor.pop();
					report( "  Ascending to node " << 
						node->parent->identifier );					
					node = node->parent;
					
					if( node->type == XmlTree::XmlTreeNode::End )
					{
					
						break;
					
					}
					
					assert( successor.top() != node->successors.end() );
					++successor.top();
			
				}
			
				// then descend one if we aren't at the end
				if( node->type != XmlTree::XmlTreeNode::End )
				{

					assert( successor.top() != node->successors.end() );		
					node = successor.top()->second;
					report( "   Descending to node " << node->identifier );
					successor.push( node->successors.begin() );
					
				}				
			
				break;
			
			}
		
			case XmlTree::XmlTreeNode::End:
			{
			
				break;
			
			}
			
			case XmlTree::XmlTreeNode::Unspecified:
			{
			
				assert( "Found node with unspecified type" == 0 );
			
			}
	
		}
	
		return *this;
	
	}
	
	XmlTree::iterator XmlTree::iterator::operator++( int )
	{
	
		XmlTree::iterator i = *this;
		return ++i;
	
	}

	void XmlTree::iterator::descend( std::string tag )
	{
		
		report( "At node " << node->identifier << ", descending to " << tag );
		XmlTreeNode::SuccessorIterator next = node->successors.find( tag );
		
		if( next == node->successors.end() )
		{
			
			report( " Node not found, rewinding to end." );
			while( !successor.empty() )
			{
			
				ascend();
			
			}
			
		}
		else
		{
		
			successor.push( next );
			node = successor.top()->second;
		
		}
	
	}
	
	void XmlTree::iterator::ascend(  )
	{
	
		report( "Ascending from node " << node->identifier 
			<< " to " << node->parent->identifier );
		
		if( !successor.empty() )
		{
			successor.pop();
			node = node->parent;
		}
		
	}
	
	const std::string& XmlTree::iterator::leaf() const
	{
	
		report( "Returning leaf from node " << node->identifier );
		assert( node->leaves == 1 );
		bool found = false;
		ConstSuccessorIterator fi = node->successors.begin();
		
		for( ; fi != node->successors.end(); ++fi )
		{
		
			if( fi->second->type == XmlTree::XmlTreeNode::Leaf )
			{
				
				found = true;
				break;
			
			}
		
		}
		
		assert( found );
		return fi->second->identifier;
		
	}
	
	std::map< std::string, std::string > XmlTree::iterator::map() const
	{
	
		std::map< std::string, std::string > _map;
		
		for( ConstSuccessorIterator fi = node->successors.begin(); 
			fi != node->successors.end(); ++fi )
		{
		
			if( fi->second->leaves == 1 )
			{
			
				assert( !fi->second->successors.empty() );
				_map.insert( std::make_pair( fi->second->identifier, 
					fi->second->successors.begin()->second->identifier ) );
			
			}
		
		}
		
		return _map;
	
	}
	////////////////////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////////////////////
	// XmlTree
	void XmlTree::init( const XmlTree::iterator& constBase )
	{
	
		report( "Initializing new XmlTree." );
	
		XmlTree::iterator& base = 
			const_cast< XmlTree::iterator& >( constBase );
		
		std::stack< SuccessorIterator > successor;
		Iterator node = base.node;
		successor = base.successor;
		
		iterator fi = begin();
		
		while( node->type != XmlTree::XmlTreeNode::End )
		{
		
			switch( node->type )
			{
	
				case XmlTree::XmlTreeNode::Root:
				{

					report( " Hit the root node with name ( " << 
						node->identifier << " )." );			
					assert( successor.top() == node->successors.begin() );
					fi.node->identifier = node->identifier;
					
					if( node->successors.empty() )
					{
			
						node = node->parent;
						report( "  Ascending to the end." );
			
					}
					else
					{
			
						node = successor.top()->second;
						successor.push( node->successors.begin() );
						report( "  Descending to node " << node->identifier );
			
					}
			
					break;
			
				}
		
				case XmlTree::XmlTreeNode::Intermediate:
				{
			
					report( " Hit intermediate node " << node->identifier );
				
					fi = insert( node->identifier, fi, 
						XmlTree::XmlTreeNode::Intermediate );	
				
					if( !node->successors.empty() )
					{
					
						//always descend
						node = successor.top()->second;
						successor.push( node->successors.begin() );
					
						report( "  Descending to first successor " 
							<< node->identifier );
							
					}
					else
					{
					
						
					
						// rewind
						while( node->successors.end() == successor.top() )
						{
						
							assert( !successor.empty() );
							successor.pop();
							report( "  Not successors found, " 
								<< "ascending to node " << 
								node->parent->identifier );					
							node = node->parent;
						
							if( node->type == XmlTree::XmlTreeNode::End )
							{
						
								break;
						
							}
						
							assert( successor.top() != node->successors.end() );
							++successor.top();
							fi.ascend();
											
						}
				
						// then descend one if we aren't at the end
						if( node->type != XmlTree::XmlTreeNode::End )
						{

							assert( successor.top() != node->successors.end() );		
							node = successor.top()->second;
							report( "   Descending to node " << node->identifier );
							successor.push( node->successors.begin() );
						
						}	
					
					}
					break;
			
				}
		
				case XmlTree::XmlTreeNode::Leaf:
				{
				
					report( " Hit leaf node " << node->identifier );
				
					fi = insert( node->identifier, fi, 
						XmlTree::XmlTreeNode::Leaf );
				
					// we should be at the end since this is a leaf
					assert( successor.top() == node->successors.end() );
				
					assert( !successor.empty() );
					successor.pop();
					report( "  Ascending to node " << 
						node->parent->identifier );					
					fi.ascend();
					node = node->parent;
					assert( successor.top() != node->successors.end() );
					++successor.top();
				
					// ascend
					while( node->successors.end() == successor.top() )
					{
					
						assert( !successor.empty() );
						successor.pop();
						report( "  Ascending to node " << 
							node->parent->identifier );					
						node = node->parent;
					
						fi.ascend();
					
						if( node->type == XmlTree::XmlTreeNode::End )
						{
					
							break;
					
						}
					
						assert( successor.top() != node->successors.end() );
						++successor.top();
			
					}
			
					// then descend one if we aren't at the end
					if( node->type != XmlTree::XmlTreeNode::End )
					{

						assert( successor.top() != node->successors.end() );		
						node = successor.top()->second;
						report( "   Descending to node " << node->identifier );
						successor.push( node->successors.begin() );
					
					}				
			
					break;
			
				}
		
				case XmlTree::XmlTreeNode::End:
				{
		
					assert( "Hit end node." == 0 );	
					break;
			
				}
			
				case XmlTree::XmlTreeNode::Unspecified:
				{
			
					assert( "Found node with unspecified type" == 0 );
			
				}
	
			}
			
		}
	
	}
	
	XmlTree::XmlTree()
	{
	
		XmlTreeNode Root;
		
		Root.type = XmlTreeNode::Root;
		rootNode = nodes.insert( nodes.end(), Root );
		Root.type = XmlTreeNode::End;
		endNode = nodes.insert( nodes.end(), Root );
		rootNode->parent = endNode;
		endNode->parent = endNode;
		rootNode->leaves = 0;
		endNode->leaves = 0;
		
	}
	
	XmlTree::~XmlTree()
	{
	
	
	}

	XmlTree::XmlTree( const XmlTree& constTree )
	{
	
		XmlTree& tree = const_cast< XmlTree& >( constTree );
		XmlTreeNode Root;
		
		Root.type = XmlTreeNode::Root;
		rootNode = nodes.insert( nodes.end(), Root );
		Root.type = XmlTreeNode::End;
		endNode = nodes.insert( nodes.end(), Root );
		rootNode->parent = endNode;
		endNode->parent = endNode;
		rootNode->leaves = 0;
		endNode->leaves = 0;

		init( tree.begin() );
	
	}
	
	XmlTree& XmlTree::operator=( const XmlTree& constTree )
	{

		XmlTree& tree = const_cast< XmlTree& >( constTree );
	
		if( this != &tree )
		{
		
			clear();
			init( tree.begin() );
		
		}
		
		return *this;
		
	}
	
	XmlTree::iterator XmlTree::begin()
	{
	
		iterator begin;
		begin.node = rootNode;
		begin.successor.push( rootNode->successors.begin() );
		return begin;
		
	}
	
	XmlTree::iterator XmlTree::end()
	{
	
		iterator end;
		end.node = endNode;
		return end;
	
	}
	
	XmlTree::iterator XmlTree::insert( std::string tag )
	{
	
		iterator fi = begin();
		return insert( tag, fi );
	
	}
	
	XmlTree::iterator XmlTree::insert( std::string tag, 
		XmlTree::iterator& base,
		XmlTreeNode::Type type )
	{

		report( "Inserting node " << tag << ", at current node " 
			<< base->identifier );

		assert( type != XmlTreeNode::End );
		assert( type != XmlTreeNode::Root );

		iterator result = base;				
		result.successor.top() = result.node->successors.find( tag );

		XmlTreeNode newNode;
		newNode.identifier = tag;
		
		if( type == XmlTreeNode::Unspecified )
		{
		
			if( result.node->leaves == 0 )
			{
				newNode.type = XmlTreeNode::Leaf;
				++result.node->leaves;
			}
			else
			{
			
				newNode.type = XmlTreeNode::Intermediate;
			
			}
			
		}
		else
		{
		
			newNode.type = type;
			
			if( type == XmlTreeNode::Leaf )
			{
			
				assert( result.node->leaves == 0 );
				++result.node->leaves;
			
			}
		
		}
		
		newNode.parent = result.node;
		newNode.leaves = 0;
		
		if( result.successor.top() == result.node->successors.end() )
		{
		
			report( " Node " << tag << " is new, adding it." );
		
			if( result.node->type == XmlTreeNode::Leaf )
			{
			
				result.node->type = XmlTreeNode::Intermediate;
				assert( result.node->parent->leaves == 1 );
				--result.node->parent->leaves;
				report( " Node " << result.node->identifier 
					<< " was a leaf, converting to an intermediate node." );
			
			}
		
		}
		
		result.successor.top() = result.node->successors.insert( 
			std::make_pair( tag, nodes.insert( endNode, newNode ) ) );

		result.node = result.successor.top()->second;
		result.node->parent = base.node;
		result.successor.push( result.node->successors.begin() );
		
		return result;
	
	}
	
	XmlTree::iterator XmlTree::erase( XmlTree::iterator& i )
	{
	
		assert( i != begin() );
		assert( i != end() );
		assert( i.node->type == XmlTreeNode::Leaf );
		
		XmlTree::iterator result = i;
					
		++result;			
		i.node->parent->successors.erase( i.successor.top() );
		
		if( i.node->parent->successors.empty() )
		{
		
			i.node->parent->type = XmlTreeNode::Leaf;
		
		}
		
		nodes.erase( i.node );
		i = end();
		
		return result;
	
	}
	
	unsigned int XmlTree::size() const
	{
	
		report( "Size of tree is " << nodes.size() );
		return nodes.size();
	
	}
	
	void XmlTree::clear()
	{
	
		XmlTreeNode Root;
		
		nodes.clear();			
		Root.type = XmlTreeNode::Root;
		rootNode = nodes.insert( nodes.end(), Root );
		Root.type = XmlTreeNode::End;
		endNode = nodes.insert( nodes.end(), Root );
		rootNode->parent = endNode;
		endNode->parent = endNode;
	
	}
	
	void XmlTree::toStream( std::ostream& stream )
	{
	
		std::string tab;
		std::stack< std::string > tags;
		std::stack< SuccessorIterator > successor;
		Iterator node = rootNode;
		successor.push( rootNode->successors.begin() );

		report( "Writing XML Tree out to a stream." );
		
		while( node != endNode )
		{
		
			switch( node->type )
			{
		
				case XmlTree::XmlTreeNode::Root:
				{

					report( " Hit the root node with name ( " << 
						node->identifier << " )." );			
					assert( successor.top() == node->successors.begin() );
				
				
					if( node->successors.empty() )
					{
				
						node = node->parent;
						report( "  Ascending to the end." );
				
					}
					else
					{
				
						node = successor.top()->second;
						successor.push( node->successors.begin() );
						report( "  Descending to node " << node->identifier );
				
					}
				
					break;
				
				}
			
				case XmlTree::XmlTreeNode::Intermediate:
				{
				
					report( " Hit intermediate node " << node->identifier );
				
					std::string temp;
					std::string temp1;
				
					stream << tab;
					temp = '<';
					stream << temp;
					stream << node->identifier;
					temp1 = ">\n";
					stream << temp1;
					report( "Writing: " << tab << temp << node->identifier 
						<< temp1 );
					tab.push_back( '\t' );
					tags.push( node->identifier );
				
					if( !node->successors.empty() )
					{
					
						//always descend
						node = successor.top()->second;
						successor.push( node->successors.begin() );
					
						report( "  Descending to first successor " 
							<< node->identifier );
							
					}
					else
					{
					
						// rewind
						// ascend
						while( node->successors.end() == successor.top() )
						{
						
							assert( !successor.empty() );
							successor.pop();
							report( "  Not successors found, " 
								<< "ascending to node " << 
								node->parent->identifier );					
							node = node->parent;
						
							if( node->type == XmlTree::XmlTreeNode::End )
							{
						
								break;
						
							}
						
							assert( successor.top() != node->successors.end() );
							++successor.top();
							assert( !tab.empty() );
							tab.erase( tab.end() - 1, tab.end() );
							std::string temp( "</" );
							std::string temp1( ">\n" );
							stream << tab << temp << tags.top() << temp1;
							report( "Writing: " << tab << temp << tags.top() 
								<< temp1 );
							tags.pop(); 
				
						}
				
						// then descend one if we aren't at the end
						if( node->type != XmlTree::XmlTreeNode::End )
						{

							assert( successor.top() != node->successors.end() );		
							node = successor.top()->second;
							report( "   Descending to node " << node->identifier );
							successor.push( node->successors.begin() );
						
						}	
					
					}
					break;
				
				}
			
				case XmlTree::XmlTreeNode::Leaf:
				{
					
					report( " Hit leaf node " << node->identifier );
					
					{
					
						std::string temp( "\n" );
						stream << tab << node->identifier << temp;
						report( "Writing: " << tab << node->identifier << temp );					
						
					}
					
					// we should be at the end since this is a leaf
					assert( successor.top() == node->successors.end() );
					
					assert( !successor.empty() );
					successor.pop();
					report( "  Ascending to node " << 
						node->parent->identifier );					
					node = node->parent;
					assert( successor.top() != node->successors.end() );
					++successor.top();
					
					// ascend
					while( node->successors.end() == successor.top() )
					{
						
						assert( !successor.empty() );
						successor.pop();
						report( "  Ascending to node " << 
							node->parent->identifier );					
						node = node->parent;
						
						if( node->type == XmlTree::XmlTreeNode::End )
						{
						
							break;
						
						}
						
						assert( successor.top() != node->successors.end() );
						++successor.top();
						assert( !tab.empty() );
						tab.erase( tab.end() - 1, tab.end() );
						std::string temp( "</" );
						std::string temp1( ">\n" );
						stream << tab << temp << tags.top() << temp1;
						report( "Writing: " << tab << temp << tags.top() 
							<< temp1 );
						tags.pop(); 
				
					}
				
					// then descend one if we aren't at the end
					if( node->type != XmlTree::XmlTreeNode::End )
					{

						assert( successor.top() != node->successors.end() );		
						node = successor.top()->second;
						report( "   Descending to node " << node->identifier );
						successor.push( node->successors.begin() );
						
					}				
				
					break;
				
				}
			
				case XmlTree::XmlTreeNode::End:
				{
					assert( "Hit the end node during iteration " == 0 );
					break;
				
				}	
			
				case XmlTree::XmlTreeNode::Unspecified:
				{
			
					assert( "Found node with unspecified type" == 0 );
			
				}	
		
			}
					
		}
		
		report( " Finished writing to stream." );
	
	}
	
	XmlTree XmlTree::subtree( const XmlTree::iterator& base )
	{
	
		XmlTree tree;

		tree.init( base );
					
		return tree;
	
	}
	////////////////////////////////////////////////////////////////////////////
	
}

#endif

