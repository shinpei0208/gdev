/*
 * KernelDrawerPass.cpp
 *
 *	Created on: Sep 25, 2011
 *			Author: Diogo Sampaio
 */

#include <ocelot/transforms/interface/KernelDrawerPass.h>
#include <ocelot/analysis/interface/DirectionalGraph.h>
#include <ocelot/analysis/interface/AffineAnalysis.h>

// Hydrazine Includes
#include <hydrazine/interface/SystemCompatibility.h>

// Standard Library Includes
#include <fstream>
#include <cassert>

namespace transforms
{

static const std::string BLOCK_OUT             = "aliveOut";
static const std::string FULLGRAPH_PHI         = "phi";
static const std::string FULLGRAPH_INSTRUCTION = "instruction";

/*!\brief Draw edges from the block predecessor to it, and in the (D+C)FG edges
 * inside the block from aliveIn to phis or instructions, and isntructions to
 * aliveOut */
std::string KernelDrawerPass::_instructionColor(
	const analysis::DataflowGraph::InstructionVector::iterator & ins) const
{
	if(_analysis & Analysis::Type::DivergenceAnalysis)
	{
		if(_analysis & Analysis::Type::AffineAnalysis)
		{
			analysis::AffineAbstractState state = _aff->state(ins);
			if(state.undefined()) return "purple";
			
			if(_div->isDivInstruction(*ins))
			{
				if(state.isUniform()) return "green";
				if(state.affine())    return "yellow";
				if(state.divergent()) return "orange";
			}

			if(state.affine())    return "tomato";
			if(state.divergent()) return "red";
		}
		
		if(_div->isDivInstruction(*ins))
		{
			return "tomato";
		}
	}

	if(_analysis & Analysis::Type::AffineAnalysis)
	{
		analysis::AffineAbstractState state = _aff->state(ins);

		if(state.undefined()) return "purple";
		if(state.affine())    return "orange";
		if(state.divergent()) return "tomato";
	}
	return "white";
}

std::string KernelDrawerPass::_blockColor(
	const analysis::DataflowGraph::iterator& bck) const
{
	if(bck->instructions().empty() ||
		!_branchNotUni(bck->instructions().end()--))
	{
		return "white";
	}
	
	return _instructionColor(bck->instructions().end()--);
}

std::string KernelDrawerPass::_edges(
	const analysis::DataflowGraph::iterator &block,
	const bool isFullGraph) const
{
	std::stringstream buff;
	const analysis::DataflowGraph::BlockPointerSet predecessors =
		block->predecessors();
	auto predecessor    = predecessors.begin();
	auto endPredecessor = predecessors.end();

	std::string thisBlockName = _blockName(block->label());

	std::string predecessorName;

	for(; predecessor != endPredecessor; predecessor++)
	{
		predecessorName = _blockName((*predecessor)->label());

		if(isFullGraph)
		{
			if(predecessorName != "entry")
			{
				predecessorName += BLOCK_OUT;
			}
		}

		buff << predecessorName << "->" << thisBlockName;

		if(((*predecessor)->block()->has_fallthrough_edge())
				&& ((*predecessor)->fallthrough()->label() == block->label()))
		{
			buff << " [color = \"blue\"]";
		}

		buff << ';';
	}

	if(isFullGraph && ((thisBlockName != "entry") &&
		(thisBlockName != "exit")))
	{
		if(block->phis().empty())
		{
			buff << thisBlockName << "->" << thisBlockName
				<< FULLGRAPH_INSTRUCTION << ":in;";
		}
		else
		{
			buff << thisBlockName << FULLGRAPH_PHI << ":out->" << thisBlockName
				<< FULLGRAPH_INSTRUCTION << ":in;";
		}
		
		buff << thisBlockName << FULLGRAPH_INSTRUCTION << "->" << thisBlockName
			<< BLOCK_OUT << ';';
	}

	return buff.str();
}

/*!\brief Draw the variables dependency graph */
void KernelDrawerPass::drawVariablesGraph() const
{
	ofstream file(_path + _kernelName + "_DA_dataFlow.dot");

	if(file.is_open() != true)
	{
		cerr << "Coldn't open file " << _path
				<< _kernelName + "_DA_dataFlow.dot for writing." << endl;
		return;
	}

	file << (analysis::DirectionalGraph) (_div->getDivergenceGraph());
	file.close();
}

/*!\brief Draw the divergence spread graph */
void KernelDrawerPass::drawDivergenceGraph() const
{
	ofstream file(_path + _kernelName + "_DA_divergenceFlow.dot");

	if (file.is_open() != true)
	{
		cerr << "Coldn't open file " << _path << _kernelName
				<< "_DA_divergenceFlow.dot for writing." << endl;
		return;
	}

	file << _div->getDivergenceGraph();
	file.close();

	file.open(_path + _kernelName + "_AFF_divergenceFlow.dot");

	if (file.is_open() != true)
	{
		cerr << "Coldn't open file " << _path << _kernelName
				<< "_AFF_divergenceFlow.dot for writing." << endl;
		return;
	}

	_aff->sa()->printPropagationGraphInDot(file);
	file.close();
}

/*!\brief Draw the CFG, can draw comparison statistics to profiling results */
void KernelDrawerPass::drawControlFlowGraph() const
{
	analysis::DataflowGraph::iterator block = _dfg->begin();
	analysis::DataflowGraph::iterator endBlock = _dfg->end();

	ofstream file(_path + _kernelName + "_CFG.dot");

	if (!file.is_open())
	{
		cerr << "Couldn't open file '" << _path
				<< _kernelName + "_CFG.dot' for writing" << endl;
		return;
	}

	file << "digraph ControlGraphDivergenceAnalysis{";
	
	std::stringstream edgesDot;

	for (; block != endBlock; block++)
	{
		string blockLabel = _blockName(block->label());

		if (blockLabel == "entry")
		{
			file << "entry[shape=Mdiamond,style=\"filled\",color=\"white\","
				"fillcolor=\"black\",fontcolor=\"white\",label=\"Entry\"];";
			continue;
		}

		if (blockLabel == "exit")
		{
			file << "exit[shape=Mdiamond,style=\"filled\",color=\"white\","
				"fillcolor=\"black\",fontcolor=\"white\",label=\"Exit\"];";
			edgesDot << _edges(block);
			continue;
		}

		auto instruction = block->instructions().begin();
		auto endInstruction = block->instructions().end();

		file << blockLabel
			<< "[shape=none, margin=0, label=<<TABLE BORDER=\"0\" "
			"CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"1\" "
			"WIDTH=\"600\"><TR><TD BGCOLOR=\""
			<< _blockColor(block) << "\">"
			<< block->label() << "</TD></TR>";
		for (; instruction != endInstruction; instruction++)
		{
			file << "<TR><TD WIDTH=\"600\" BGCOLOR=\""
				<< _instructionColor(instruction) << "\">"
				<< instruction->i->toString() << "</TD></TR>" << endl;
		}

		file << "</TABLE>>];" << endl;
		edgesDot << _edges(block);
	}

	file << edgesDot.str();

	if(_analysis & Analysis::Type::DominatorTreeAnalysis)
	{
		file << _domStr.str();
	}
	
	if(_analysis & Analysis::Type::PostDominatorTreeAnalysis)
	{
		file << _pdomStr.str();;
	}
	
	file << "legend [shape=none, margin=0, label=<<TABLE BORDER=\"0\" "
		"CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"1\">"
		<< "<TR><TD COLSPAN=\"2\">LEGEND</TD></TR>"
		<< "<TR><TD><FONT COLOR=\"blue\">blue edges</FONT>"
			"</TD><TD>fallthrough path</TD></TR>"
		<< "<TR><TD>black edges</TD><TD>branch path</TD></TR>"
		<< "<TR><TD BGCOLOR=\"tomato\">light red instruction</TD>"
			"<TD>divergent instruction</TD></TR>"
		<< "<TR><TD BGCOLOR=\"tomato\">block label</TD><TD>Divergent "
		"block prediction</TD></TR>";
	
	if(_analysis & Analysis::Type::DominatorTreeAnalysis)
	{
		file << "<TR><TD><FONT COLOR=\"green\">green dashed edges</FONT>"
			"</TD><TD>Dominator Block</TD></TR>";
	}
	
	if(_analysis & Analysis::Type::PostDominatorTreeAnalysis)
	{
		file << "<TR><TD><FONT COLOR=\"red\">red dashed edges</FONT></TD>"
			"<TD>PostDominator Block</TD></TR>";
	}
	
	file << "</TABLE>>];};";
	file.close();
}

string KernelDrawerPass::_printAffineTransferFunction(
		const int & id) const
{
	if (Analysis::AffineAnalysis & _analysis)
	{
		auto var = _aff->map().find(id);
		if (var != _aff->map().end())
		{
			stringstream b;
			b << " | " << var->second;
			return b.str();
		}
	}
	
	return "";
}

void KernelDrawerPass::_printFullGraphHeader(ofstream &out) const
{
	out << "digraph fullDFG{toc [shape=none,margin=0,label=<<TABLE BORDER"
		<< "=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"1\">"
		<< "<TR><TD COLSPAN=\"4\">Table of contents</TD></TR>"
		<< "<TR><TD COLSPAN=\"2\"><FONT COLOR=\"blue\">blue edges</FONT>"
			"</TD><TD COLSPAN=\"2\">fallthrough path</TD></TR>"
		<< "<TR><TD COLSPAN=\"2\">black edges</TD><TD COLSPAN=\"2\">"
			"Branch path</TD></TR>";
	
	if(_analysis & Analysis::Type::DominatorTreeAnalysis)
	{
		out << "<TR><TD COLSPAN=\"2\"><FONT COLOR=\"green\">green dashed "
			"edges</FONT></TD><TD COLSPAN=\"2\">Dominator block</TD></TR>";
	}
	
	if(_analysis & Analysis::Type::PostDominatorTreeAnalysis)
	{
		out << "<TR><TD COLSPAN=\"2\"><FONT COLOR=\"red\">red dashed edges"
			"</FONT></TD><TD COLSPAN=\"2\">Post-Dominator b.</TD></TR>";
	}
	
	if(!((_analysis & Analysis::Type::DivergenceAnalysis) ||
			(_analysis & Analysis::Type::AffineAnalysis)))
	{
		out << "</TABLE>>];";
		return;
	}

	if(_analysis & Analysis::Type::DivergenceAnalysis)
	{
		if(_analysis & Analysis::Type::AffineAnalysis)
		{
			out << "<TR><TD>Type and Color</TD><TD>Affine Analysis</TD><TD>Divergence"
				<< "Analysis</TD><TD>Count</TD></TR>"
				<< "<TR><TD BGCOLOR=\"white\">Variable</TD><TD>Constant</TD>"
					"<TD>Constant</TD><TD>"
					<< _variables.aff_div.cstCst << "</TD></TR>"
				<< "<TR><TD BGCOLOR=\"green\">Variable</TD><TD>Constant</TD>"
					"<TD>Divergent</TD><TD>"
					<< _variables.aff_div.cstDiv << "</TD></TR>"

				<< "<TR><TD BGCOLOR=\"light gray\">Variable</TD><TD>Uniform"
					"</TD><TD>Constant</TD><TD>" << _variables.aff_div.uniCst
					<< "</TD></TR>"
				<< "<TR><TD BGCOLOR=\"light green\">Variable</TD><TD>Uniform"
					"</TD><TD>Divergent</TD><TD>"
					<< _variables.aff_div.uniDiv << "</TD></TR>"

				<< "<TR><TD BGCOLOR=\"orange\">Variable</TD><TD>Constant Affine"
					"</TD><TD>Constant</TD><TD>"
					<< _variables.aff_div.knwCst << "</TD></TR>"
				<< "<TR><TD BGCOLOR=\"light yellow\">Variable</TD><TD>"
					"Constant Affine</TD><TD>Divergent</TD><TD>"
					<< _variables.aff_div.knwDiv << "</TD></TR>"

				<< "<TR><TD BGCOLOR=\"tomato\">Variable</TD><TD>Affine"
					"</TD><TD>Constant</TD><TD>"
					<< _variables.aff_div.affCst << "</TD></TR>"
				<< "<TR><TD BGCOLOR=\"yellow\">Variable</TD><TD>Affine"
					"</TD><TD>Divergent</TD><TD>"
					<< _variables.aff_div.affDiv << "</TD></TR>"

				<< "<TR><TD BGCOLOR=\"red\">Variable</TD><TD>Divergent</TD>"
					"<TD>Constant</TD><TD>"
					<< _variables.aff_div.divCst << "</TD></TR>"
				<< "<TR><TD BGCOLOR=\"gray\">Variable</TD><TD>Divergent</TD>"
					"<TD>Divergent</TD><TD>"
					<< _variables.aff_div.divDiv << "</TD></TR>"

				<< "<TR><TD BGCOLOR=\"purple\">Variable</TD><TD>Undefined</TD>"
					"<TD>Constant</TD><TD>" << _variables.aff_div.undCst
					<< "</TD></TR>"
				<< "<TR><TD BGCOLOR=\"purple\">Variable</TD><TD>Undefined</TD>"
					"<TD>Divergent</TD><TD>" << _variables.aff_div.undDiv
					<< "</TD></TR>"

				<< "<TR><TD COLSPAN=\"2\">Total variables</TD>"
					"<TD COLSPAN=\"2\">"
				<< _variables.aff_div.cstCst + _variables.aff_div.cstDiv +
				+ _variables.aff_div.uniCst + _variables.aff_div.uniDiv
				+ _variables.aff_div.knwCst + _variables.aff_div.knwDiv
				+ _variables.aff_div.affCst + _variables.aff_div.affDiv
				+ _variables.aff_div.divCst + _variables.aff_div.divDiv
				+ _variables.aff_div.undCst + _variables.aff_div.undDiv;
		}
		else
		{
			out << "<TR><TD COLSPAN=\"2\">Constant variable</TD>"
					"<TD COLSPAN=\"2\">"
				<< _variables.div.cst << "</TD></TR>"
				<< "<TR><TD COLSPAN=\"2\" BGCOLOR=\"tomato\">Divergent "
					"variable</TD><TD COLSPAN=\"2\">"
				<< _variables.div.div << "</TD></TR>"
				<< "<TR><TD COLSPAN=\"3\">Total variables:><TD>"
				<< _variables.div.cst + _variables.div.div;
		}
	}
	else if(_analysis & Analysis::Type::AffineAnalysis)
	{
		out << "<TR><TD BGCOLOR=\"purple\" COLSPAN=\"2\">T + T + T</TD>"
					"<TD COLSPAN=\"2\">" << _variables.aff.und << "</TD></TR>"
			<< "<TR><TD BGCOLOR=\"white\" COLSPAN=\"2\">0 + 0 + C</TD>"
					"<TD COLSPAN=\"2\">" << _variables.aff.cst << "</TD></TR>"
			<< "<TR><TD BGCOLOR=\"light gray\" COLSPAN=\"2\">0 + 0 + B</TD>"
					"<TD COLSPAN=\"2\">" << _variables.aff.uni << "</TD></TR>"
			<< "<TR><TD COLSPAN=\"2\" BGCOLOR=\"yellow\">0 + C + C</TD>"
					"<TD COLSPAN=\"2\">" << _variables.aff.knw << "</TD></TR>"
			<< "<TR><TD COLSPAN=\"2\" BGCOLOR=\"orange\">0 + B + *</TD>"
					"<TD COLSPAN=\"2\">" << _variables.aff.aff << "</TD></TR>"
			<< "<TR><TD COLSPAN=\"2\" BGCOLOR=\"tomato\">0 + B + *</TD>"
					"<TD COLSPAN=\"2\">" << _variables.aff.div << "</TD></TR>"
			<< "<TR><TD COLSPAN=\"3\">Total variables:</TD><TD>"
			<< _variables.aff.aff + _variables.aff.cst + _variables.aff.div +
			_variables.aff.knw + _variables.aff.und + _variables.aff.uni;
	}
	out << "</TD></TR></TABLE>>];";
}

/*!\brief Draw the (D+C)FG, can draw comparison statistics to profile results */
void KernelDrawerPass::drawFullGraph() const
{
	std::map<analysis::DataflowGraph::RegisterId, string> registersLocations;
	analysis::DataflowGraph::iterator block = _dfg->begin();
	analysis::DataflowGraph::iterator endBlock = _dfg->end();

	std::ofstream file(string(_path + _kernelName + "_FULL.dot").c_str());
	
	if (!file.is_open())
	{
		std::cerr << "Couldn't open file '" + _path + _kernelName
			+ "_FULL.dot' for writing";
		return;
	}

	_printFullGraphHeader(file);
	std::stringstream edgesDot;
	std::string blockName = "";
	for(; block != endBlock; block++)
	{
		string blockLabel = _blockName(block->label());
		if(blockLabel == "entry")
		{
			file << block->label()
				<< "[shape=Mdiamond,style=\"filled\",color=\"white\","
				<< "fillcolor=\"black\",fontcolor=\"white\",label=\"Entry\"];";
			continue;
		}

		if(blockLabel == "exit")
		{

			file << block->label()
				<< "[shape=Mdiamond,style=\"filled\",color=\"white\","
				<< "fillcolor=\"black\",fontcolor=\"white\",label=\"Exit\"];";
			edgesDot << _edges(block, true);
			continue;
		}

		std::stringstream blockLabelCOLSPAN;

		/* Print blocklabel + alive in table */
		if(block->aliveIn().size() > 1)
		{
			blockLabelCOLSPAN << " COLSPAN=\""
				<< block->aliveIn().size() << '"';
		}

		file << "subgraph cluster" << blockLabel << "{" << blockLabel
			<< "[shape=none,margin=0,label=<<TABLE BORDER=\"0\" CELLBO"
			<< "RDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"1\"><TR><TD"
			<< blockLabelCOLSPAN.str() << " BGCOLOR=\""
			<< _blockColor(block) << "\">" << block->label() << "</TD></TR>";
		
		if(block->aliveIn().size() > 0)
		{
			file << "<TR><TD" << blockLabelCOLSPAN.str()
				<< ">AliveIn</TD></TR>";

			std::set<analysis::DataflowGraph::RegisterId> orderedAliveIn;
			
			{
				auto aliveIn    = block->aliveIn().begin();
				auto endAliveIn = block->aliveIn().end();

				for(; aliveIn != endAliveIn; aliveIn++)
				{
					orderedAliveIn.insert(aliveIn->id);
				}
			}

			file << "<TR>";
			
			/* Print alive in in same order of phi sources */
			if(block->phis().size() > 0)
			{
				auto phi    = block->phis().begin();
				auto endPhi = block->phis().end();

				for(; phi != endPhi; phi++)
				{
					auto phiS = phi->s.begin();
					auto endPhiS = phi->s.end();

					for (; phiS != endPhiS; phiS++)
					{
						file << "<TD PORT=\"" << phiS->id << "\" BGCOLOR=\""
							<< _getBGColor(phiS->id) << "\">" << phiS->id
							<< _printAffineTransferFunction(phiS->id)
							<< "</TD>";

						orderedAliveIn.erase(phiS->id);
					}
				}
			}
			
			/* Print the remaining alive in */
			auto aliveIn    = orderedAliveIn.begin();
			auto endAliveIn = orderedAliveIn.end();
			
			for(; aliveIn != endAliveIn; aliveIn++)
			{
				file << "<TD PORT=\"" << *aliveIn << "\" BGCOLOR=\""
					<< _getBGColor(*aliveIn) << "\">" << *aliveIn
					<< _printAffineTransferFunction(*aliveIn) << "</TD>";
			}
			file << "</TR>";
		}

		/* Print alive out table */
		file << "</TABLE>>];" << blockLabel << BLOCK_OUT
				<< "[shape=none,margin=0,label=<<TABLE BORDER=\"0\" CE"
				<<"LLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"1\">";
				
		if (block->aliveOut().size() > 0)
		{

			std::set<analysis::DataflowGraph::RegisterId> orderedAliveOut;
			
			{
				auto aliveOut    = block->aliveOut().begin();
				auto endAliveOut = block->aliveOut().end();

				for(; aliveOut != endAliveOut; aliveOut++)
				{
					orderedAliveOut.insert(aliveOut->id);
				}
			}

			auto aliveOut    = orderedAliveOut.begin();
			auto endAliveOut = orderedAliveOut.end();

			file << "<TR>";
			
			for(; aliveOut != endAliveOut; aliveOut++)
			{
				std::string bgColor = "";
				file << "<TD BGCOLOR=\"" << _getBGColor(*aliveOut)
					<< "\">" << *aliveOut
					<< _printAffineTransferFunction(*aliveOut) << "</TD>";
			}
			file << "</TR>";
		}
		
		file << "<TR><TD";

		if(block->aliveOut().size() > 1)
		{
			file << " COLSPAN=\"" << block->aliveOut().size() << '"';
		}

		file << ">AliveOut</TD></TR></TABLE>>];";

		/* Print instructions */
		auto instruction    = block->instructions().begin();
		auto endInstruction = block->instructions().end();

		file << blockLabel << FULLGRAPH_INSTRUCTION
			<< "[shape=none,margin=0,label="
				"<<TABLE BORDER=\"0\" CELLBORDER=\"1\""
			<< " CELLSPACING=\"0\" CELLPADDING=\"1\"><TR><TD WIDTH=\"60px\">Out"
			<< "<br/>Regs</TD><TD WIDTH=\"580px\" PORT=\"in\">Instruction</TD>"
			<< "<TD WIDTH=\"60px\">In<br/>Regs</TD></TR>";

		for(; instruction != endInstruction; instruction++)
		{
			unsigned totalLines = 1;
			unsigned insColSpan = 1;
			std::stringstream sRowSpan;
			std::stringstream dRowSpan;
			std::stringstream insSpan;

			auto sReg    = instruction->s.begin();
			auto endSReg = instruction->s.end();
			auto dReg    = instruction->d.begin();
			auto endDReg = instruction->d.end();

			if(instruction->d.size() > 1)
			{
				totalLines = instruction->d.size();
			}

			if((instruction->s.size() > 1) &&
				(instruction->s.size() != totalLines))
			{
				totalLines *= instruction->s.size();
			}

			if(dReg != endDReg)
			{
				if ((instruction->d.size() != totalLines) && (totalLines > 1))
				{
					dRowSpan << " ROWSPAN=\""
						<< totalLines / instruction->d.size() << '"';
				}
			}
			else
			{
				insColSpan++;
			}

			if(sReg != endSReg)
			{
				if ((instruction->s.size() != totalLines) && (totalLines > 1))
				{
					sRowSpan << " ROWSPAN=\""
						<< totalLines / instruction->s.size() << '"';
				}
			}
			else
			{
				insColSpan++;
			}

			if(totalLines > 1)
			{
				insSpan << " ROWSPAN=\"" << totalLines << '"';
			}

			if(insColSpan > 1)
			{
				insSpan << " COLSPAN=\"" << insColSpan << '"';
			}

			/* Print the first instruction line, with the instruction label */
			file << "<TR>";
			
			if(dReg != endDReg)
			{
				file << "<TD" << dRowSpan.str() << " BGCOLOR=\""
					<< _getBGColor(*dReg->pointer) << "\">" << *dReg->pointer
					<< _printAffineTransferFunction(*dReg->pointer) << "</TD>";
				dReg++;
			}

			file << "<TD" << insSpan.str() << " BGCOLOR=\""
				<< _instructionColor(instruction)
				<< "\">" << instruction->i->toString() << "</TD>";
			
			if(sReg != endSReg)
			{
				file << "<TD" << sRowSpan.str() << " BGCOLOR=\""
					<< _getBGColor(*sReg->pointer) << "\">" << *sReg->pointer
					<< _printAffineTransferFunction(*sReg->pointer) << "</TD>";
				sReg++;
			}

			file << "</TR>";
			
			/* print lines with destination or source register */
			while((sReg != endSReg) || (dReg != endDReg))
			{
				file << "<TR>";
				
				if(dReg != endDReg)
				{
					file << "<TD" << dRowSpan.str() << " BGCOLOR=\""
						<< _getBGColor(*dReg->pointer)
						<< "\">" << *dReg->pointer
						<< _printAffineTransferFunction(*dReg->pointer)
						<< "</TD>";
					dReg++;
				}

				if(sReg != endSReg)
				{
					file << "<TD" << sRowSpan.str() << "	BGCOLOR=\""
						<< _getBGColor(*sReg->pointer)
						<< "\">" << *sReg->pointer
						<< _printAffineTransferFunction(*sReg->pointer)
						<< "</TD>";
					sReg++;
				}

				file << "</TR>";
			}
		}
		
		file << "</TABLE>>];";

		/* Print phi instructions */
		if(block->phis().size() > 0)
		{
			unsigned totalSources = 0;
			std::stringstream secondLine;

			file << blockLabel << FULLGRAPH_PHI
				<< "[shape=none,margin=0,label=<<TABLE BORDER=\"0\" CELL"
				<< "BORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"1\"><TR>";

			auto phi    = block->phis().begin();
			auto endPhi = block->phis().end();

			for(; phi != endPhi; phi++)
			{
				auto phiS    = phi->s.begin();
				auto endPhiS = phi->s.end();

				for(; phiS != endPhiS; phiS++)
				{
					file << "<TD PORT=\"" << phiS->id << "\"	BGCOLOR=\""
						<< _getBGColor(phiS->id) << "\">" << phiS->id
						<< _printAffineTransferFunction(phiS->id) << "</TD>";

					edgesDot << blockLabel << ":" << phiS->id
						<< "->" << blockLabel << FULLGRAPH_PHI
						<< ":" << phiS->id << "[color="
						<< ((_getBGColor(phiS->id) == "white") ? "black" :
								_getBGColor(phiS->id)) << "];";
				}
				
				secondLine << "<TD COLSPAN=\"" << phi->s.size()
					<< '"' << " BGCOLOR=\""
					<< _getBGColor(phi->d.id) << "\">" << phi->d.id
					<< _printAffineTransferFunction(phi->d.id) << "</TD>";
				
				totalSources += phi->s.size();
			}

			file << "</TR><TR>" << secondLine.str() << "</TR><TR><TD COLSPAN=\""
				<< totalSources << "\" PORT=\"out\">Phis</TD></TR></TABLE>>];";

		}
		
		/*subgraph end*/
		file << '}';

		edgesDot << _edges(block, true);
	}
	
	file << std::endl << edgesDot.str();
	
	if(_analysis & Analysis::Type::DominatorTreeAnalysis)
	{
		file << _domStr.str();
	}
	
	if(_analysis & Analysis::Type::PostDominatorTreeAnalysis)
	{
		file << _pdomStr.str();;
	}

	file << "}";
	file.close();
}

void KernelDrawerPass::computeResults()
{
	if(_analysis & Analysis::Type::DivergenceAnalysis)
	{
		if(_analysis & Analysis::Type::AffineAnalysis)
		{
			auto var    = _div->getDivergenceGraph().getBeginNode();
			auto varEnd = _div->getDivergenceGraph().getEndNode();
			
			for(; var != varEnd; var++)
			{
				const analysis::AffineAbstractState& state =
						_aff->state(*var);
				if(_div->getDivergenceGraph().isDivNode(*var))
				{
					if(state.divergent() || state.hardAffine())
					{
						_variables.aff_div.divDiv++;
					}
					else if(state.constant())
					{
						_variables.aff_div.cstDiv++;
					}
					else if (state.isUniform())
					{
						_variables.aff_div.uniDiv++;
					}
					else if (state.known())
					{
						_variables.aff_div.knwDiv++;
					}
					else if(state.unknown())
					{
						_variables.aff_div.affDiv++;
					}
					else if(state.undefined())
					{
						_variables.aff_div.undDiv++;
					}
					else
					{
						assertM(false, "Untreated state:" << state);
					}
				}
				else
				{
					if(state.divergent() || state.hardAffine())
					{
						_variables.aff_div.divCst++;
					}
					else if(state.constant())
					{
						_variables.aff_div.cstCst++;
					}
					else if (state.isUniform())
					{
						_variables.aff_div.uniCst++;
					}
					else if (state.known())
					{
						_variables.aff_div.knwCst++;
					}
					else if(state.unknown())
					{
						_variables.aff_div.affCst++;
					}
					else if(state.undefined())
					{
						_variables.aff_div.undCst++;
					}
					else
					{
						assertM(false, "Untreated state:" << state);
					}
				}
			}
		}
		else
		{
			bool divResult;
			auto var    = _div->getDivergenceGraph().getBeginNode();
			auto varEnd = _div->getDivergenceGraph().getEndNode();
			for(; var != varEnd; var++)
			{
				divResult = _div->getDivergenceGraph().isDivNode(*var);
				_variables.div.div += (unsigned long) (divResult);
				_variables.div.cst += (unsigned long) (!divResult);
			}
		}
	}
	else if(_analysis & Analysis::Type::AffineAnalysis)
	{
		auto var    = _aff->map().begin();
		auto varEnd = _aff->map().end();
		for(; var != varEnd; var++)
		{
			if(var->second.divergent() || var->second.hardAffine())
			{
				_variables.aff_div.divCst++;
			}
			else if(var->second.constant())
			{
				_variables.aff_div.cstCst++;
			}
			else if (var->second.isUniform())
			{
				_variables.aff_div.uniCst++;
			}
			else if (var->second.known())
			{
				_variables.aff_div.knwCst++;
			}
			else if(var->second.unknown())
			{
				_variables.aff_div.affCst++;
			}
			else if(var->second.undefined())
			{
				_variables.aff_div.undCst++;
			}
			else
			{
				assertM(false, "Untreated state:" << var->second);
			}
		}
	}
	
	if(_analysis & Analysis::Type::DominatorTreeAnalysis)
	{
		_domStr.clear();
	}
	
	if(_analysis & Analysis::Type::PostDominatorTreeAnalysis)
	{
		_pdomStr.clear();
	}

	for(auto b = _dfg->begin(); b != _dfg->end(); b++)
	{
		instructionsCount += b->instructions().size();
		
		if(b->label() == "entry")
		{
			continue;
		}

		if(b->label() == "exit")
		{
			continue;
		}
		
		if(_analysis & Analysis::Type::DominatorTreeAnalysis)
		{
			if(_hasDom(b) && (_domName(b) != "entry"))
			{
				_domStr << _blockName(b->label()) << "->" << _domName(b)
					<< "[color = \"green\" style=\"dashed\"];";
			}
		}
		if(_analysis & Analysis::Type::PostDominatorTreeAnalysis)
		{
			if(_hasPdom(b) && (_pdomName(b) != "exit"))
			{
				_pdomStr << _blockName(b->label()) << "->" << _pdomName(b)
					<< "[color = \"red\" style=\"dashed\"];";
			}
		}
	}
}

void KernelDrawerPass::printResults() const
{
	std::ofstream file(_path + _kernelName + ".csv");
	assertM(file.is_open(),
			"Couldn't open file '" + _path + _kernelName + ".csv' for writing");

	if(_analysis & Analysis::Type::DivergenceAnalysis)
	{
		if(_analysis & Analysis::Type::AffineAnalysis)
		{
			file << "cst/cst;cst/div;uni/cst;uni/div;caff/cst;caff/div;"
				<<	"aff/cst;aff/div;div/cst;div/div;# instructions"
				<< std::endl
				<< _variables.aff_div.cstCst << ';'
				<< _variables.aff_div.cstDiv << ';'
				<< _variables.aff_div.uniCst << ';'
				<< _variables.aff_div.uniDiv << ';'
				<< _variables.aff_div.knwCst << ';'
				<< _variables.aff_div.knwDiv << ';'
				<< _variables.aff_div.affCst << ';'
				<< _variables.aff_div.affDiv << ';'
				<< _variables.aff_div.divCst + _variables.aff_div.undCst << ';'
				<< _variables.aff_div.divDiv + _variables.aff_div.undDiv << ';'
				<< instructionsCount;
		}
		else
		{
			file << "cst;div;# instructions"
				<< std::endl << _variables.div.cst << ';'
				<< _variables.div.div << ';'
				<< instructionsCount;
		}
	}
	else if(_analysis & Analysis::Type::AffineAnalysis)
	{
		file << "T + T;0 + C;0 + B;C + C;C + B;B + B;# instructions"
			<< std::endl
			<< _variables.aff.und << ';'
			<< _variables.aff.cst << ';'
			<< _variables.aff.uni << ';'
			<< _variables.aff.knw << ';'
			<< _variables.aff.aff << ';'
			<< _variables.aff.div << ';'
			<< instructionsCount;
	}

	file.close();
}

/* Constructor that sets everything and make it ready to draw, sets profiling on
 * being able to read the profiling results */
KernelDrawerPass::KernelDrawerPass(const std::string &path,
	unsigned graphs, unsigned analysis) :
	KernelPass(analysis | Analysis::Type::DataflowGraphAnalysis),
	_todo(graphs), _analysis(analysis), instructionsCount(0),
	_path(path), _k(NULL), _dfg(NULL), _div(NULL),
	_aff(NULL), _dom(NULL), _pdom(NULL)
{

}

void KernelDrawerPass::runOnKernel(ir::IRKernel& k)
{
	instructionsCount = 0;
	
	_k = &k;

	_dfg = (analysis::DataflowGraph*)
		(getAnalysis(Analysis::Type::DataflowGraphAnalysis));
	
	variables_t var = {{0}};
	_variables = var;
	
	if(_analysis & Analysis::Type::DivergenceAnalysis)
	{
		_div = (analysis::DivergenceAnalysis*)
			(getAnalysis(Analysis::Type::DivergenceAnalysis));
	}
	
	if(_analysis & Analysis::Type::AffineAnalysis)
	{
		_aff = (analysis::AffineAnalysis*)
			(getAnalysis(Analysis::Type::AffineAnalysis));
	}
	
	if(_analysis & Analysis::Type::DominatorTreeAnalysis)
	{
		_dom = (analysis::DominatorTree*)
			(getAnalysis(Analysis::Type::DominatorTreeAnalysis));
	}
	
	if(_analysis & Analysis::Type::PostDominatorTreeAnalysis)
	{
		_pdom = (analysis::PostdominatorTree*)
			(getAnalysis(Analysis::Type::PostDominatorTreeAnalysis));
	}
	
	std::string kernelName = hydrazine::demangleCXXString(k.name);
	
	//Remove parameter from function name
	_kernelName = kernelName.substr(0, kernelName.find("("));
	
	//Remove data type from templated kernel
	if(_kernelName.find('<') != string::npos)
	{
		_kernelName = _kernelName.substr(0, _kernelName.find("<"));
	}
	
	//Remove function namespace from templated kernel
	if(_kernelName.find(':') != string::npos)
	{
		_kernelName.replace(0, 1 + _kernelName.find_last_of(':'), "");
	}

	//Remove function type from templated kernel
	if(_kernelName.find(' ') != string::npos)
	{
		_kernelName.replace(0, 1 + _kernelName.find_last_of(' '), "");
	}

	if(_analysis & Analysis::Type::DivergenceAnalysis)
	{
		_div = (analysis::DivergenceAnalysis*)
			(getAnalysis(Analysis::DivergenceAnalysis));
	}
	
	if(_analysis & Analysis::Type::AffineAnalysis)
	{
		_aff = (analysis::AffineAnalysis*)
			(getAnalysis(Analysis::AffineAnalysis));
	}

	if(_analysis & (Analysis::Type::DivergenceAnalysis |
		Analysis::Type::AffineAnalysis |
		Analysis::Type::PostDominatorTreeAnalysis |
		Analysis::Type::DominatorTreeAnalysis))
	{
		computeResults();
	}

	if(_todo & TODO::results)
	{
		printResults();
	}
	if(_todo & TODO::varsGraph)
	{
		drawVariablesGraph();
	}
	if(_todo & TODO::divGraph)
	{
		drawDivergenceGraph();
	}
	if(_todo & TODO::cfgGraph)
	{
		drawControlFlowGraph();
	}
	if(_todo & TODO::fullGraph)
	{
		drawFullGraph();
	}
}

void KernelDrawerPass::initialize(const ir::Module& m)
{

}

void KernelDrawerPass::finalize()
{

}

std::string KernelDrawerPass::_getBGColor(const unsigned id) const
{
	if(_analysis & Analysis::Type::DivergenceAnalysis)
	{
		const analysis::DivergenceGraph &divergenceGraph =
			_div->getDivergenceGraph();

		if(_analysis & Analysis::Type::AffineAnalysis)
		{
			const analysis::AffineAbstractState & state = _aff->state(id);
			if(state.undefined())
			{
				return "purple";
			}
			
			if(divergenceGraph.isDivNode(id))
			{
				if(state.isUniform()) return "green";
				if(state.affine())    return "yellow";
				if(state.divergent()) return "orange";
			}
			
			if(state.affine())   return "tomato";
			if(state.divergent())return "red";
		}
		if(divergenceGraph.isDivNode(id))
		{
			return "tomato";
		}
	}
	
	if(_analysis & Analysis::Type::AffineAnalysis)
	{
		const analysis::AffineAbstractState & state = _aff->state(id);
		
		if(state.undefined()) return "purple";
		if(state.affine())    return "orange";
		if(state.divergent()) return "tomato";
	}
	return "white";
}

bool KernelDrawerPass::_hasDom(
	const analysis::DataflowGraph::iterator & bck) const
{
	return _dom->getDominator(bck->block()) != _dom->cfg->end();
}

string KernelDrawerPass::_domName(
	const analysis::DataflowGraph::iterator & bck) const
{
	return _blockName(_dom->getDominator(bck->block())->label());
}

bool KernelDrawerPass::_hasPdom(
	const analysis::DataflowGraph::iterator & bck) const
{
	return _pdom->getPostDominator(bck->block()) != _pdom->cfg->end();
}

string KernelDrawerPass::_pdomName(
	const analysis::DataflowGraph::iterator & bck) const
{
	return _blockName(_pdom->getPostDominator(bck->block())->label());
}

string KernelDrawerPass::_blockName(const string & blockLabel) const
{
	if(blockLabel[0] == '$') return blockLabel.substr(1, blockLabel.size() - 1);

	return blockLabel;
}

bool KernelDrawerPass::_branchNotUni(
	const analysis::DataflowGraph::InstructionVector::iterator & ins) const
{
	auto i = static_cast<const ir::PTXInstruction*>(ins->i);
	return ((i->opcode == ir::PTXInstruction::Opcode::Bra) && (!i->uni));
}

}

