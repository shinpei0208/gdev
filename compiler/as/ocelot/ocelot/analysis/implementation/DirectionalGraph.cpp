/*
 * DirectionalGraph.cpp
 *
 *  Created on: Jul 1, 2010
 *      Author: Diogo Sampaio
 */

#include <ocelot/analysis/interface/DirectionalGraph.h>

namespace analysis
{

/*!\brief Clears the graph data */
void DirectionalGraph::clear(){
	outArrows.clear();
	inArrows.clear();
	nodes.clear();
}

/*!\brief Clears the graph data */
DirectionalGraph::~DirectionalGraph(){
	clear();
}

/*!\brief Insert a node with nodeId. A duplicated ids is ignored */
void DirectionalGraph::insertNode( const node_type &nodeId ){
	nodes.insert(nodeId);
}

/*!\brief Number of nodes on the graph */
size_t DirectionalGraph::nodesCount() const{
	return nodes.size();
}

/*!\brief Get the first node */
DirectionalGraph::const_node_iterator DirectionalGraph::getBeginNode() const{
	return nodes.begin();
}

/*!\brief Get the node limit iterator */
DirectionalGraph::const_node_iterator DirectionalGraph::getEndNode() const{
	return nodes.end();
}

/*!\brief Get a node by it's id */
DirectionalGraph::const_node_iterator DirectionalGraph::findNode( const node_type &nodeId ) const{
	return nodes.find(nodeId);
}

/*!\brief Tests if the graph has a node with certain id */
bool DirectionalGraph::hasNode( const node_type nodeId ) const{
	return nodes.find(nodeId) != nodes.end();
}

/*!\brief Remove a node with certain id from the graph
 * 1) Tests if node is present
 * 2) Tests If there are outgoing arrows
 * 2.1) Remove this node from each destination's node id source list
 * 2.2) Remove this node outgoing list
 * 3) Tests If there are incoming arrows
 * 3.1) Remove this node from each sources's node id destination list
 * 3.2) Remove this node sources list
 * 4) Remove the node from the node list
 */
bool DirectionalGraph::eraseNode( const node_type &nodeId ){
	arrow_iterator arrows;
	node_iterator nextNode;

	/* 1) Tests if node is present */
	if( nodes.find(nodeId) == nodes.end() )
		return false;

	/* 2) Tests If there are outgoing arrows */
	arrows = outArrows.find(nodeId);
	if( arrows != outArrows.end() ){
		nextNode = arrows->second.begin();

		/* 2.1) Remove this node from each destination's node id source list */
		for( ; nextNode != arrows->second.end(); nextNode++ ){
			arrow_iterator inArrow = inArrows.find(*nextNode);

			if( inArrow != inArrows.end() ){
				inArrows.find(*nextNode)->second.erase(nodeId);
			}
		}
		/* 2.2) Remove this node outgoing list */
		outArrows.erase(nodeId);
	}
	/* 3) Tests If there are incoming arrows */
	arrows = inArrows.find(nodeId);
	if( arrows != inArrows.end() ){
		nextNode = arrows->second.begin();

		/* 3.1) Remove this node from each sources's node id destination list */
		for( ; nextNode != arrows->second.end(); nextNode++ ){
			arrow_iterator outArrow = outArrows.find(*nextNode);

			if( outArrow != outArrows.end() ){
				outArrows.find(*nextNode)->second.erase(nodeId);
			}
		}
		/* 3.2) Remove this node sources list */
		inArrows.erase(nodeId);
	}
	/* 4) Remove the node from the node list */
	nodes.erase(nodeId);

	return true;
}

/*!\brief Remove a node with the id of the node from a node iterator */
bool DirectionalGraph::eraseNode( const node_iterator &node ){
	if( nodes.find(*node) == nodes.end() )
		return false;

	return eraseNode(*node);
}

/*!\brief Get the outgoing edges[arrows] for a certain node */
const DirectionalGraph::node_set DirectionalGraph::getOutNodesSet(
	const node_type& nodeId ) const{
	if( outArrows.find(nodeId) != outArrows.end() )
		return outArrows.find(nodeId)->second;

	node_set tmp;
	return tmp;
}

/*!\brief Get the incoming edges[arrows] for a certain node */
const DirectionalGraph::node_set DirectionalGraph::getInNodesSet(
	const node_type& nodeId ) const {
	if( inArrows.find(nodeId) != inArrows.end() )
		return inArrows.find(nodeId)->second;

	node_set tmp;
	return tmp;
}
/*!\brief Insert a edge between two nodes. If the nodes doesn't exist they can be created
 * 1) Tests if can create new nodes
 * 1.1) Create ongoing and destination nodes
 * 1.2) Or test if both nodes exist
 * 2) Insert / create / update list of ongoing edges for source node
 * 3) Insert / create / update list of incoming edges for destination node
 */
int DirectionalGraph::insertEdge( const node_type &fromNode,
	const node_type &toNode, const bool createNewNodes ){
	/* 1) Tests if can create new nodes */
	if( createNewNodes ){
		/* 1.1) Create ongoing and destination nodes */
		nodes.insert(fromNode);
		nodes.insert(toNode);
	}else{
		/* 1.2) Or test if both nodes exist */
		if( nodes.find(fromNode) == nodes.end() ){
			return 1;
		}

		if( nodes.find(toNode) == nodes.end() ){
			return 2;
		}
	}
	/* 2) Insert / create / update list of ongoing edges for source node
	 * Insert A->B arrow in outArrows: outArrows[A].insert(B) */
	arrow_iterator arrowIt;
	bool createdOutArrowLine = false;
	arrowIt = outArrows.find(fromNode);

	if( arrowIt == outArrows.end() ){
		node_set a;
		outArrows[fromNode] = a;
		createdOutArrowLine = true;
	}

	arrowIt = outArrows.find(fromNode);

	if( arrowIt == outArrows.end() ){
		return 3;
	}

	arrowIt->second.insert(toNode);

	if( outArrows[fromNode].find(toNode) == outArrows[fromNode].end() ){
		if( createdOutArrowLine ){
			outArrows.erase(arrowIt);
		}
		return 4;
	}

	/* 3) Insert / create / update list of incoming edges for destination node
	 * Insert A->B arrow in inArrows: inArrows[B].insert(A) */
	arrowIt = inArrows.find(toNode);

	bool createdInArrowLine = false;

	if( arrowIt == inArrows.end() ){
		node_set a;
		inArrows[toNode] = a;
		createdInArrowLine = true;
	}

	arrowIt = inArrows.find(toNode);

	if( arrowIt == inArrows.end() ){
		if( createdOutArrowLine ){
			outArrows.erase(fromNode);
		}else{
			outArrows[fromNode].erase(toNode);
		}

		return 5;
	}

	inArrows.find(toNode)->second.insert(fromNode);

	if( inArrows[toNode].find(fromNode) == inArrows[toNode].end() ){
		if( createdOutArrowLine ){
			outArrows.erase(fromNode);
		}else{
			outArrows[fromNode].erase(toNode);
		}

		if( createdInArrowLine ){
			inArrows.erase(toNode);
		}

		return 6;
	}

	return 0;
}

/*!\brief Remove an edge between two nodes. Can remove isolated nodes
 * 1) Test if both nodes of the edge exist
 * 2) Remove edges
 * 3) Remove isolated nodes
 */
int DirectionalGraph::eraseEdge( const node_type &fromNode,
	const node_type &toNode, const bool removeIsolatedNodes ){
	/* 1) Test if both nodes of the edge exist */
	if( nodes.find(fromNode) == nodes.end() ){
		return 1;
	}

	if( nodes.find(toNode) == nodes.end() ){
		return 2;
	}

	/* 2) Remove edges */
	outArrows[fromNode].erase(toNode);
	inArrows[toNode].erase(fromNode);

	/* 3) Remove isolated nodes */
	if( removeIsolatedNodes ){
		if( (outArrows[fromNode].size() == 0)
			&& (inArrows[fromNode].size() == 0) ){
			eraseNode(fromNode);
		}

		if( (outArrows[toNode].size() == 0)
			&& (inArrows[toNode].size() == 0) ){
			eraseNode(toNode);
		}
	}

	return 0;
}
/* Prints graph in dot language */
std::ostream& DirectionalGraph::print( std::ostream& out ) const{
	out << "digraph DataFlowGraph{" << std::endl;

	/* Print nodes */
	const_node_iterator node = getBeginNode();
	const_node_iterator endNode = getEndNode();

	for( ; node != endNode; node++ ){
		out << *node << std::endl;
	}

	/* Print arrows between nodes */
	node = getBeginNode();
	endNode = getEndNode();

	for( ; node != endNode; node++ ){
		const node_set outArrows = getOutNodesSet(*node);
		const_node_iterator nodeOut = outArrows.begin();
		const_node_iterator endNodeOut = outArrows.end();

		for( ; nodeOut != endNodeOut; nodeOut++ ){
			out << *node << "->" << *nodeOut << std::endl;
		}
	}

	out << '}' << std::endl;

	return out;
}

std::ostream& operator<<( std::ostream& out, const DirectionalGraph& graph ){
	return graph.print(out);
}

}
