/*! \file PtxToIlTranslator.h
 *  \author Rodrigo Dominguez <rdomingu@ece.neu.edu>
 *  \date August 9, 2010
 *  \brief The header file for the PtxToIlTranslator class.
*/

#ifndef PTX_TO_IL_TRANSLATOR_BIN_H_INCLUDED
#define PTX_TO_IL_TRANSLATOR_BIN_H_INCLUDED

/*! \brief A class used to transform a PTX file into an IL equivalent */
class PtxToIlTranslator
{
	public:
		std::string input;
	
	public:
		void translate();
};

#endif

