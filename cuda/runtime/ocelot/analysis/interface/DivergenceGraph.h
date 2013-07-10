/*! \file DivergenceGraph.h
	\author Diogo Sampaio <dnsampaio@gmail.com>
	\brief The header file for the DivergenceGraph class.
*/
#ifndef DIVERGENCEGRAPH_H_INCLUDED
#define DIVERGENCEGRAPH_H_INCLUDED

// Ocelot Includes
#include <ocelot/analysis/interface/DirectionalGraph.h>
#include <ocelot/ir/interface/PTXOperand.h>

namespace analysis {

class DivergenceGraph: public DirectionalGraph
{
	private:
		/*!\brief A set of nodes that are divergent because of a
			predecessor divergent block */
		node_set _divergentNodes;
		/*!\brief A set of nodes that are known to be special
			registers, possible sources of divergence */
		std::map<const ir::PTXOperand*, node_set> _specials;
		/*!\brief A set of nodes known to be divergent,
			probably because of creation divergent paths, atomic operands */
		node_set _divergenceSources;
		/*!\brief A set of nodes known to be convergent */
		node_set _convergenceSources;
		/*!\brief Defines if the divergence graph has been
			altered since the last computeDivergence() call */
		bool _upToDate;

	public:
		void clear();

		DivergenceGraph() : DirectionalGraph(), _upToDate(true){};
		~DivergenceGraph() {clear();};
		void insertSpecialSource( const ir::PTXOperand* tid );
		void eraseSpecialSource( const ir::PTXOperand* tid );
		void setAsDiv( const node_type &node );
		void forceConvergent( const node_type &node );
		void unsetAsDiv( const node_type &node );
		bool eraseNode( const node_type &nodeId );
		bool eraseNode( const node_iterator &node );
		int insertEdge( const node_type &fromNode,
			const node_type &toNode, const bool createNewNodes = true );
		int insertEdge( const ir::PTXOperand* origin,
			const node_type &toNode, const bool createNewNodes = true );
		int eraseEdge( const node_type &fromNode,
			const node_type &toNode, const bool removeIsolatedNodes = false );
		int eraseEdge( const ir::PTXOperand* origin,
			const node_type &toNode, const bool removeIsolatedNodes = false );
		const node_set getDivNodes() const;
		bool isDivNode( const node_type &node ) const;
		bool isDivSource( const node_type &node ) const;
		bool isDivSource( const ir::PTXOperand* srt ) const;
		bool hasSpecial( const ir::PTXOperand* special ) const;
		size_t divNodesCount() const;
		inline node_iterator beginDivNodes() const{return _divergentNodes.begin();};
		inline node_iterator endDivNodes() const {return _divergentNodes.end();};
		void computeDivergence();
		std::string getSpecialName( const ir::PTXOperand* in ) const;
		std::ostream& print( std::ostream& out ) const;
};

std::ostream& operator<<( std::ostream& out, const DivergenceGraph& graph );

}

#endif /* DIVERGENCEGRAPH_H_ */

