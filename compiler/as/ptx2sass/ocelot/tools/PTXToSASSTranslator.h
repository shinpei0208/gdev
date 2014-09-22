/*!
 * \file PTXToSASSTranslator.h
 * \brief header of PTXToSASSTranslator.cpp.
 */

#ifndef __PTX_TO_SASS_TRANSLATOR_BIN_H__
#define __PTX_TO_SASS_TRANSLATOR_BIN_H__

class PTXToSASSTranslator {
	public:
		/*! \brief PTX file name */
		std::string input;
		/*! \brief SASS file name */
		std::string output;
	public:
		/*! \brief translate PTX to SASS */
		void translate();
};

#endif /* __PTX_TO_SASS_TRANSLATOR_BIN_H__ */
