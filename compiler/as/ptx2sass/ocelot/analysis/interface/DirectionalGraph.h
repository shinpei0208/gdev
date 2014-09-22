/*! \file DirectionalGraph.h
	\date Jul 1, 2010
	\author Diogo Sampaio <dnsampaio@gmail.com>
	\file The header file for the DirectionalGraph class.
*/

#ifndef DIRECTIONALGRAPH_H_INCLUDED
#define DIRECTIONALGRAPH_H_INCLUDED

// Standard Library Includes
#include <ostream>
#include <set>
#include <map>

namespace analysis {
class DirectionalGraph
{
	public:
		typedef unsigned int node_type;

		/* A set of nodes of a directional graph */
		typedef std::set<node_type> node_set;
		typedef node_set::iterator node_iterator;
		typedef node_set::const_iterator const_node_iterator;

		/* Map every node to a list[set] of edges[arrows] */
		typedef std::map<node_type, node_set> arrows_map;

		typedef arrows_map::iterator arrow_iterator;

		typedef std::pair<int, node_iterator> node_action_info;

	protected:
		/*!\brief The set with the graph nodes */
		std::set<node_type> nodes;
		/*!\brief For each node, maps to which other nodes
			it has a edge[arrow] going to */
		std::map<node_type, node_set> inArrows;
		/*!\brief For each node, maps from which other nodes
			it has a edge[arrow] coming from */
		std::map<node_type, node_set> outArrows;

	public:
		~DirectionalGraph();
		void clear();
		void insertNode( const node_type &nodeId );
		size_t nodesCount() const;
		const_node_iterator getBeginNode() const;
		const_node_iterator getEndNode() const;
		const_node_iterator findNode( const node_type &nodeId ) const;
		bool hasNode( const node_type nodeId ) const;
		bool eraseNode( const node_type &nodeId );
		bool eraseNode( const node_iterator &node );
		const node_set getOutNodesSet( const node_type& nodeId ) const;
		const node_set getInNodesSet( const node_type& nodeId ) const;
		int insertEdge( const node_type &fromNode,
			const node_type &toNode, const bool createNewNodes = true );
		int eraseEdge( const node_type &fromNode,
			const node_type &toNode, const bool removeIsolatedNodes = false );
		std::ostream& print( std::ostream& out ) const;
};

std::ostream& operator<<( std::ostream& out, const DirectionalGraph& graph );

}

#endif

