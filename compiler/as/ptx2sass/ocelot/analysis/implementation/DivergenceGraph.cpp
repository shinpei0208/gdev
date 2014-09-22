/*
 * DivergenceGraph.cpp
 *
 *  Created on: May 19, 2010
 *      Author: Diogo Sampaio
 */

// Ocelot Includes
#include <ocelot/analysis/interface/DivergenceGraph.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE  0
#define REPORT_GRAPH 0

namespace analysis {

/*!\brief Clears the divergence graph */
void DivergenceGraph::clear(){
	DirectionalGraph::clear();
	_divergentNodes.clear();
	_specials.clear();
	_divergenceSources.clear();
	_convergenceSources.clear();
	_upToDate = true;
}

/*!\brief Insert a special register source, possible source of divergence */
void DivergenceGraph::insertSpecialSource( const ir::PTXOperand* tid ){
	_upToDate = false;
	if( _specials.find(tid) == _specials.end() ){
		_specials.insert(std::make_pair(tid, node_set()));
	}
}

/*!\brief Removes a special register source */
void DivergenceGraph::eraseSpecialSource( const ir::PTXOperand* tid ){
	_upToDate = false;
	_specials.erase(tid);
}

/*!\brief Define a node as being divergent,
	not depending on it's predecessors */
void DivergenceGraph::setAsDiv(const node_type &node)
{
	if (_divergenceSources.find(node) == _divergenceSources.end()) {
		_upToDate = false;
		_divergenceSources.insert(node);
	}
	if (nodes.find(node) == nodes.end()) {
		_upToDate = false;
		nodes.insert(node);
	}
}

void DivergenceGraph::forceConvergent(const node_type &node) {
	if (_convergenceSources.find(node) == _convergenceSources.end()) {
		_upToDate = false;
		_convergenceSources.insert(node);
	}
	if (nodes.find(node) == nodes.end()) {
		_upToDate = false;
		nodes.insert(node);
	}
}

/*!\brief Unset a node as being divergent, not depending on it's predecessors */
void DivergenceGraph::unsetAsDiv( const node_type &node ){
	if( _divergenceSources.find(node) != _divergenceSources.end() ){
		_upToDate = false;
		_divergenceSources.erase(node);
	}
}

/*!\brief Removes a node from the divergence graph */
bool DivergenceGraph::eraseNode( const node_type &nodeId ){
	_upToDate = false;
	_divergentNodes.erase(nodeId);
	return DirectionalGraph::eraseNode(nodeId);
}

/*!\brief Removes a node from the divergence graph */
bool DivergenceGraph::eraseNode( const node_iterator &node ){
	if( nodes.find(*node) == nodes.end() ){
		return false;
	}

	_upToDate = false;
	_divergentNodes.erase(*node);

	return DirectionalGraph::eraseNode(*node);
}

/*!\brief Inserts a edge[arrow] between two nodes of the graph
	/ Can create nodes if they don't exist */
int DivergenceGraph::insertEdge( const node_type &fromNode,
	const node_type &toNode, const bool createNewNodes ){
	_upToDate = false;
	return DirectionalGraph::insertEdge(fromNode, toNode, createNewNodes);
}

/*!\brief Inserts a edge[arrow] between two nodes of the graph
	/ Can create nodes if they don't exist */
int DivergenceGraph::insertEdge( const ir::PTXOperand* origin,
	const node_type &toNode, const bool createNewNodes ){
	if( createNewNodes ){
		insertSpecialSource(origin);
	} else if( _specials.find(origin) == _specials.end() ){
		return 1;
	}

	_specials[origin].insert(toNode);
	return 0;
}

/*!\brief Inserts a edge[arrow] between two nodes of the graph
	/ Can remove nodes if they are isolated */
int DivergenceGraph::eraseEdge( const node_type &fromNode,
	const node_type &toNode, const bool removeIsolatedNodes ){
	_upToDate = false;
	if( removeIsolatedNodes ){
		size_t actualNodesCount = nodes.size();
		int result = DirectionalGraph::eraseEdge(
			fromNode, toNode, removeIsolatedNodes);

		if( actualNodesCount == nodes.size() ){
			return result;
		}

		if( nodes.find(fromNode) == nodes.end() ){
			_divergentNodes.erase(fromNode);
		}else{
			_divergentNodes.erase(toNode);
		}

		return result;
	}

	return DirectionalGraph::eraseEdge(fromNode, toNode, removeIsolatedNodes);
}

/*!\brief Gests a list[set] of the divergent nodes */
const DirectionalGraph::node_set DivergenceGraph::getDivNodes() const{
	return _divergentNodes;
}

/*!\brief Tests if a node is divergent */
bool DivergenceGraph::isDivNode( const node_type &node ) const{
	return _divergentNodes.find(node) != _divergentNodes.end();
}

/*!\brief Tests if a node is a divergence source */
bool DivergenceGraph::isDivSource( const node_type &node ) const{
	return _divergenceSources.find(node) != _divergenceSources.end();
}

/*!\brief Tests if a special register is source of divergence */
bool DivergenceGraph::isDivSource( const ir::PTXOperand* srt ) const{
	return ((srt->addressMode == ir::PTXOperand::Special) &&
		( (srt->special == ir::PTXOperand::laneId) ||
		(srt->special == ir::PTXOperand::tid &&
		(srt->vIndex == ir::PTXOperand::ix))));
}

/*!\brief Tests if a special register is present on the graph */
bool DivergenceGraph::hasSpecial( const ir::PTXOperand* special ) const{
	return _specials.find(special) != _specials.end();
}

/*!\brief Gives the number of divergent nodes */
size_t DivergenceGraph::divNodesCount() const{
	return _divergentNodes.size();
}

/*!\brief Computes divergence spread
 * 1) Clear preview divergent nodes list
 * 2) Set all nodes that are directly dependent of a divergent
 	source {tidX, tidY, tidZ and laneid } as new divergent nodes
 * 3) Set all nodes that are explicitly defined as divergence
 	sources as new divergent nodes
 * 4) For each new divergent nodes
 * 4.1) Set all non divergent nodes that depend directly on the
 	divergent node as new divergent nodes
 * 4.1.1) Go to step 4 after step 4.3 until there are new divergent nodes
 * 4.2) Insert the node to the divergent nodes list
 * 4.3) Remove the node from the new divergent list
 */
void DivergenceGraph::computeDivergence(){
	report("Computing divergence");
	
	if( _upToDate ){
		report(" already up to date.");
		return;
	}
	/* 1) Clear preview divergent nodes list */
	_divergentNodes.clear();
	node_set newDivergenceNodes;

	/* 2) Set all nodes that are directly dependent of a divergent
		source {tidX and laneid } as divergent */
	{
		std::map<const ir::PTXOperand*, node_set>::iterator
			divergence = _specials.begin();
		std::map<const ir::PTXOperand*, node_set>::iterator
			divergenceEnd = _specials.end();

		for( ; divergence != divergenceEnd; divergence++ ){
			if( isDivSource(divergence->first) ){
				const_node_iterator node = divergence->second.begin();
				const_node_iterator endNode = divergence->second.end();
				
				report(" Node r" << *node
					<< " is a special divergent register.");

				for( ; node != endNode; node++ ){
					newDivergenceNodes.insert(*node);
				}
			}
		}
	}
	{
		/* 3) Set all nodes that are explicitly defined as divergence
			sources as new divergent nodes */

		node_iterator divergence = _divergenceSources.begin();
		node_iterator divergenceEnd = _divergenceSources.end();

		for( ; divergence != divergenceEnd; divergence++ ){
			newDivergenceNodes.insert(*divergence);
			report(" Node r" << *divergence << " is a new divergent register.");
		}
	}

	/* 4) For each new divergent nodes */
	report(" Propagating divergence");
	while( newDivergenceNodes.size() != 0 ){
		node_type originNode = *newDivergenceNodes.begin();
		node_set newReachedNodes = getOutNodesSet(originNode);
		node_iterator current = newReachedNodes.begin();
		node_iterator last = newReachedNodes.end();

		/* 4.1) Set all non divergent nodes that depend directly on
			the divergent node as new divergent nodes */
		for( ; current != last; current++ ){
			if( !isDivNode(*current) ){
				/* 4.1.1) Go to step 4 after step 4.3 until there are
					new divergent nodes */
				report("  propagated from r" << originNode
					<< " -> r" << *current);
				newDivergenceNodes.insert(*current);
			}
		}

		/* 4.2) Insert the node to the divergent nodes list */
		_divergentNodes.insert(originNode);
		/* 4.3) Remove the node from the new divergent list */
		newDivergenceNodes.erase(originNode);
	}
	
	/* 5) propagate convergence from sources */
	node_set notDivergenceNodes;

	{
		/* 5.1) Set all nodes that are explicitly defined as convergence
			that were divergent as not divergent nodes */

		node_iterator convergence = _convergenceSources.begin();
		node_iterator convergenceEnd = _convergenceSources.end();

		for( ; convergence != convergenceEnd; convergence++ ){
			notDivergenceNodes.insert(*convergence);
			node_iterator divergence = _divergentNodes.find(*convergence);
			
			if (divergence != _divergentNodes.end()) {
				notDivergenceNodes.insert(*convergence);
				report(" Node r" << *convergence <<
					" is a new convergent register.");
			}
		}
	}
	
	report(" Propagating convergence");
	while( notDivergenceNodes.size() != 0 ) {
		node_type originNode = *notDivergenceNodes.begin();
		
		_divergentNodes.erase(originNode);
		notDivergenceNodes.erase(originNode);

		/* 5.2) Propagate forward */
		node_set newReachedNodes = getOutNodesSet(originNode);
		node_iterator current = newReachedNodes.begin();
		node_iterator last = newReachedNodes.end();
		for( ; current != last; current++ ) {
			if( isDivNode(*current) ) {
				
				node_set predecessorNodes = getInNodesSet(*current);
				
				bool allConvergent = true;
				
				node_iterator predecessor = predecessorNodes.begin();
				node_iterator lastPredecessor = predecessorNodes.end();

				for( ; predecessor != lastPredecessor; predecessor++ ) {
					if (isDivNode(*predecessor)) {
						allConvergent = false;
						break;
					}
				}
				
				if (!allConvergent) continue;
				
				report("  propagated forward from r" << originNode
					<< " -> r" << *current);
				notDivergenceNodes.insert(*current);
			}
		}
		
		/* 5.3) Propagate backward */
		newReachedNodes = getInNodesSet(originNode);
		current         = newReachedNodes.begin();
		last            = newReachedNodes.end();
		
		for( ; current != last; current++ ) {
			if( isDivNode(*current) ) {
				
				node_set successorNodes = getOutNodesSet(*current);
				
				bool allConvergent = true;
				
				node_iterator successor     = successorNodes.begin();
				node_iterator lastSuccessor = successorNodes.end();

				for( ; successor != lastSuccessor; successor++ ) {
					if (isDivNode(*successor)) {
						allConvergent = false;
						break;
					}
				}
				
				report("  propagated backward from r" << originNode
					<< " -> r" << *current);
				
				if (!allConvergent) {
					_divergentNodes.erase(*current);
					continue;
				}
				
				notDivergenceNodes.insert(*current);
			}
		}
	}

	_upToDate = true;

	#if REPORT_GRAPH > 0
	std::cout << *this;
	#endif
}

/*!\brief Gives a string as name for a special register */
std::string DivergenceGraph::getSpecialName( const ir::PTXOperand* in ) const{
	assert (in->special < ir::PTXOperand::SpecialRegister_invalid);
	return (ir::PTXOperand::toString(in->special).erase(0, 1)
		+ ir::PTXOperand::toString(in->vIndex));
}

/*!\brief Prints the divergence graph in dot language */
std::ostream& DivergenceGraph::print( std::ostream& out ) const{
	using ir::PTXOperand;
	out << "digraph DivergentVariablesGraph{" << std::endl;

	/* Print divergence sources */
	std::map<const PTXOperand*, node_set>::const_iterator
		divergence = _specials.begin();
	std::map<const PTXOperand*, node_set>::const_iterator
		divergenceEnd = _specials.end();

	out << "//Divergence sources:" << std::endl;
	for( ; divergence != divergenceEnd; divergence++ ){
		if( divergence->second.size() ){
			out << getSpecialName(divergence->first)
				<< "[style=filled, fillcolor = \""
				<< (isDivSource(divergence->first)?"red":"lightblue")
				<< "\"]" << std::endl;
		}
	}

	/* Print nodes */
	out << "//Nodes:" << std::endl;
	const_node_iterator node = getBeginNode();
	const_node_iterator endNode = getEndNode();

	for( ; node != endNode; node++ ){
		out << *node << " [style=filled, fillcolor = \""
			<< (isDivNode(*node)?((isDivSource(*node))?
			"tomato":"yellow"):"white") << "\"]" << std::endl;
	}

	out << std::endl;

	/* Print edges coming out of divergence sources */
	divergence    = _specials.begin();
	divergenceEnd = _specials.end();

	out << "//Divergence out edges:" << std::endl;
	for( ; divergence != divergenceEnd; divergence++ ){
		if( divergence->second.size() ){
			node = divergence->second.begin();
			endNode = divergence->second.end();

			for( ; node != endNode; node++ ){
				out << getSpecialName(divergence->first) << "->"
					<< *node << "[color = \""
					<< (isDivSource(divergence->first)?"red":"blue")
					<< "\"]" << std::endl;
			}
		}
	}

	/* Print arrows between nodes */
	node    = getBeginNode();
	endNode = getEndNode();

	out << "//Nodes edges:" << std::endl;
	for( ; node != endNode; node++ ){
		const node_set outArrows = getOutNodesSet(*node);
		const_node_iterator nodeOut = outArrows.begin();
		const_node_iterator endNodeOut = outArrows.end();

		for( ; nodeOut != endNodeOut; nodeOut++ ){
			out << *node << "->" << *nodeOut << std::endl;
		}
	}

	out << '}';

	return out;
}

std::ostream& operator<<( std::ostream& out, const DivergenceGraph& graph ){
	return graph.print(out);
}

}

