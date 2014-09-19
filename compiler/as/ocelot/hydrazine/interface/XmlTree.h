/*!

	\file XmlTree.h
	
	\date Saturday September 13, 2008
	
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	
	\brief The header file for the XmlTree class.

*/

#ifndef XML_TREE_H_INCLUDED
#define XML_TREE_H_INCLUDED

#include <string>
#include <map>
#include <stack>
#include <list>

namespace hydrazine
{

	class XmlTree
	{
	
		public:
		
			class XmlTreeNode
			{
			
				public:
					
					typedef std::list< XmlTreeNode > List;
					typedef	List::iterator Iterator; 
					typedef std::multimap< std::string, Iterator > Map;
					typedef Map::iterator SuccessorIterator;
					typedef Map::const_iterator ConstSuccessorIterator;
					
					enum Type
					{
						
						Root = 0,
						Leaf = 1,
						Intermediate = 2,
						End = 3,
						Unspecified
											
					};
					
					friend class XmlTree;
				
				private:
				
					Type type;
					Map successors;
					Iterator parent;
					unsigned int leaves;
					
				public:
				
					Type getType() const;
					std::string identifier;
			
			};
			
			class iterator
			{
			
				private:
				
					XmlTreeNode::Iterator node;
					std::stack< XmlTreeNode::SuccessorIterator > successor;
				
					friend class XmlTree;
				
				public:
				
					iterator& operator=( const iterator& i );
					XmlTreeNode& operator*() const;
					XmlTreeNode* operator->() const;
					bool operator==( const iterator& i ) const;
					bool operator!=( const iterator& i ) const;
					iterator& operator++();
					iterator operator++( int );
					
					void descend( std::string tag );
					void ascend( );
					const std::string& leaf() const;
					bool isLeaf() const;
					std::map< std::string, std::string > map() const;
					
			};
			
		private:
		
			typedef XmlTreeNode::Iterator Iterator;
			typedef XmlTreeNode::SuccessorIterator SuccessorIterator;
			typedef XmlTreeNode::ConstSuccessorIterator ConstSuccessorIterator;
			XmlTreeNode::List nodes;
			Iterator rootNode;
			Iterator endNode;
		
		private:
		
			void init( const iterator& fi );
		
		public:
		
			XmlTree();
			~XmlTree();
			
			XmlTree( const XmlTree& tree );
			XmlTree& operator=( const XmlTree& tree );
						
			iterator begin();
			iterator end();
			iterator insert( std::string tag );
			iterator insert( std::string tag, iterator& base, 
				XmlTreeNode::Type type = XmlTreeNode::Unspecified );
			iterator erase( iterator& i );
			unsigned int size() const;
			void clear();
			void toStream( std::ostream& stream );
			
			XmlTree subtree( const iterator& base );
	
	};

}

#endif

