/* 	\file   ELFFile.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Thursday March 17, 2011
	\brief  The header file for the ELFFile class.
*/

#ifndef ELF_FILE_H_INCLUDED
#define ELF_FILE_H_INCLUDED

// Hydrazine Includes
#include <hydrazine/interface/ELF.h>

// Standard Library Includes
#include <vector>
#include <ostream>

namespace hydrazine
{

/*! \brief A class for parsing and interpretting an ELF file */
class ELFFile
{
public:
	/*! \brief Represents the ELF file header */
	class Header
	{
	public:
		/*! \brief Create a new header from a header data struct */
		Header(const void* data = 0,
			const elf::Elf64_Ehdr& header = elf::Elf64_Ehdr());
	
	public:
		/*! \brief Get access to the raw header data */
		const elf::Elf64_Ehdr& header() const;
		
		/*! \brief Does the magic word agree that this is an ELF binary? */
		bool checkMagic() const;
	
	public:
		/*! \brief Easier access to the underlying data (section count) */
		unsigned int sectionHeaders() const;

		/*! \brief Easier access to the underlying data (program count) */
		unsigned int programHeaders() const;
		
	private:
		/*! \brief The raw data */
		elf::Elf64_Ehdr _header;
	};
	
	/*! \brief A header for a symbol table entry */
	class SymbolHeader
	{
	public:
		SymbolHeader(const void* data = 0, long long unsigned int byte = 0,
			const elf::Elf64_Sym& header = elf::Elf64_Sym());
	
	public:
		/*! \brief Get access to the raw header data */
		const elf::Elf64_Sym& header() const;
	
	public:
		/*! \brief Get the type */
		int type() const;
	
	private:
		/*! \brief The raw header data */
		elf::Elf64_Sym _header;
	};
	
	/*! \brief Represents an ELF section header */
	class SectionHeader
	{
	public:
		SectionHeader(const void* data = 0, long long unsigned int byte = 0,
			const elf::Elf64_Shdr& header = elf::Elf64_Shdr());
	
	public:
		/*! \brief Get access to the raw header data */
		const elf::Elf64_Shdr& header() const;
		
	private:
		/*! \brief The raw header data */
		elf::Elf64_Shdr _header;
	};
	
	/*! \brief Represents an ELF section header */
	class ProgramHeader
	{
	public:
		ProgramHeader(const void* data = 0, long long unsigned int byte = 0,
			const elf::Elf64_Phdr& header = elf::Elf64_Phdr());
	
	public:
		/*! \brief Get access to the raw header data */
		const elf::Elf64_Phdr& header() const;
		
	private:
		/*! \brief The raw header data */
		elf::Elf64_Phdr _header;
	
	};

public:
	/*! \brief Create a new ELF, bind it to a mmapped or in-memory file */
	ELFFile(const void* fileData = 0);

public:
	/*! \brief Get a reference to the header */
	const Header& header() const;

public:
	/*! \brief Get the number of sections in the ELF file */
	unsigned int sections() const;
	/*! \brief Get a specific section header */
	SectionHeader& sectionHeader(unsigned int header);

public:
	/*! \brief Get the number of program headers in the ELF file */
	unsigned int programs() const;	
	/*! \brief Get a specific program header */
	ProgramHeader& programHeader(unsigned int header);

public:
	/*! \brief Get end of the elf file */
	const void* endOfFile();

public:
	/*! \brief Get the number of non-null symbols in the ELF file */
	unsigned int symbols();
	/*! \brief Get a specific symbol header (the null symbol is skipped) */
	SymbolHeader& symbolHeader(unsigned int header);

public:
	/*! \brief Get access to a string within the string table (in range) */
	const char* getSectionHeaderStringAtOffset(long long unsigned int offset);
	/*! \brief Get access to a string within the string table (in range) */
	const char* getSymbolStringAtOffset(long long unsigned int offset);

public:
	/*! \brief Write out the elf file in a human-readale format */
	void write(std::ostream& stream); 

public:
	/*! \brief Get a string representation of a program type */
	static std::string programTypeToString(int type);
	/*! \brief Get a string representation of a section type */
	static std::string sectionTypeToString(int type);
	/*! \brief Does this section type have a link? */
	static bool sectionTypeHasLink(int type);
	/*! \brief Get a string representation of a section linkage */
	static std::string sectionLinkToString(int type);
	/*! \brief Get a string representation of a symbol type */
	static std::string symbolTypeToString(int type);

private:
	/*! \brief A vector of section headers */
	typedef std::vector<SectionHeader> SectionHeaderVector;
	/*! \brief A vector of program headers */
	typedef std::vector<ProgramHeader> ProgramHeaderVector;
	/*! \brief A vector of program headers */
	typedef std::vector<SymbolHeader> SymbolHeaderVector;
	/*! \brief A vector of program headers */
	typedef std::vector<bool> BitVector;

private:
	/*! \brief Load a program from the header table at the given index */
	void _loadProgram(unsigned int header);
	/*! \brief Load a section from the header table at the given index */
	void _loadSection(unsigned int header);
	/*! \brief Load a symbol from the symbol table at the given index */
	void _loadSymbol(unsigned int header);
	/*! \brief Find the symbol table, set the value of the offset */
	void _findSymbolTable();

private:
	/*! \brief A reference to the file data for lazy access */
	const void* _elfdata;
	/*! \brief The ELF file header */
	Header _header;
	/*! \brief The set of sections in the file */
	SectionHeaderVector _sections;
	/*! \brief The set of programs in the file */
	ProgramHeaderVector _programs;
	/*! \brief Keeps track of the program headers that have been loaded */
	BitVector _loadedPrograms;
	/*! \brief Keeps track of the program headers that have been loaded */
	BitVector _loadedSections;
	/*! \brief The symbol table */
	SymbolHeaderVector _symbols;
	/*! \brief A vector of symbol headers */
	BitVector _loadedSymbols;
	/*! \brief The symbol table offset or -1 if unknown */
	long long unsigned int _symbolTableOffset;
	/*! \brief The string table for the symbol table */
	long long unsigned int _symbolStringTableOffset;
	
};

}

std::ostream& operator<<(std::ostream& out, hydrazine::ELFFile& elf);

#endif

