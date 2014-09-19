/* 	\file   ELFFile.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Thursday March 17, 2011
	\brief  The source file for the ELFFile class.
*/

#ifndef ELF_FILE_CPP_INCLUDED
#define ELF_FILE_CPP_INCLUDED

// Hydrazine Includes
#include <hydrazine/interface/ELFFile.h>
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <cstring>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace hydrazine
{

ELFFile::Header::Header(const void* d, const elf::Elf64_Ehdr& h)
: _header(h)
{
	if(d != 0)
	{
		std::memcpy(&_header, d, sizeof(elf::Elf64_Ehdr));
	}
	
	assertM(!checkMagic() || _header.e_type == elf::ELFCLASS64,
		"Only 64-bit ELF files are supported for now.");
	assertM(!checkMagic() || _header.e_ident[5] == elf::ELFDATA2LSB,
		"No support for big-endian ELF files." );
}

const elf::Elf64_Ehdr& ELFFile::Header::header() const
{
	return _header;
}

bool ELFFile::Header::checkMagic() const
{
	bool isMagic = (
		   header().e_ident[0] == elf::ElfMagic[0]
		&& header().e_ident[1] == elf::ElfMagic[1]
		&& header().e_ident[2] == elf::ElfMagic[2]
		&& header().e_ident[3] == elf::ElfMagic[3] );

	return isMagic;
}

unsigned int ELFFile::Header::sectionHeaders() const
{
	return header().e_shnum;
}

unsigned int ELFFile::Header::programHeaders() const
{
	return header().e_phnum;
}

ELFFile::SymbolHeader::SymbolHeader(const void* d, long long unsigned int b,
	const elf::Elf64_Sym& h)
: _header(h)
{
	if(d != 0)
	{
		std::memcpy(&_header, (char*)d + b, sizeof(elf::Elf64_Sym));
	}
}

const elf::Elf64_Sym& ELFFile::SymbolHeader::header() const
{
	return _header;
}

int ELFFile::SymbolHeader::type() const
{
	return _header.st_info & 0xf;
}

ELFFile::SectionHeader::SectionHeader(const void* d, long long unsigned int b,
	const elf::Elf64_Shdr& h)
: _header(h)
{
	if(d != 0)
	{
		std::memcpy(&_header, (char*)d + b, sizeof(elf::Elf64_Shdr));
	}
}

const elf::Elf64_Shdr& ELFFile::SectionHeader::header() const
{
	return _header;
}

ELFFile::ProgramHeader::ProgramHeader(const void* d, long long unsigned int b,
	const elf::Elf64_Phdr& h)
: _header(h)
{
	if(d != 0)
	{
		std::memcpy(&_header, (char*)d + b, sizeof(elf::Elf64_Phdr));
	}
}

const elf::Elf64_Phdr& ELFFile::ProgramHeader::header() const
{
	return _header;
}

ELFFile::ELFFile(const void* d)
: _elfdata(d), _header(d), _symbolTableOffset(-1)
{
	_loadedPrograms.resize(programs(), false);
	_programs.resize(programs());

	_loadedSections.resize(sections(), false);
	_sections.resize(sections());

	report("Loading ELF file: \n" << *this);
}

const ELFFile::Header& ELFFile::header() const
{
	return _header;
}

unsigned int ELFFile::sections() const
{
	return header().sectionHeaders();
}

ELFFile::SectionHeader& ELFFile::sectionHeader(unsigned int h)
{
	_loadSection(h);
	
	return _sections[h];
}

unsigned int ELFFile::programs() const
{
	return header().programHeaders();
}

ELFFile::ProgramHeader& ELFFile::programHeader(unsigned int h)
{
	_loadProgram(h);
	
	return _programs[h];
}

const void* ELFFile::endOfFile()
{
	return (const char*) _elfdata + header().header().e_phoff
		+ sizeof(elf::Elf64_Phdr) * (programs() + 1);
}

unsigned int ELFFile::symbols()
{
	_findSymbolTable();
	
	return _symbols.size();
}

ELFFile::SymbolHeader& ELFFile::symbolHeader(unsigned int h)
{
	_loadSymbol(h);
	
	return _symbols[h];
}

const char* ELFFile::getSectionHeaderStringAtOffset(
	long long unsigned int offset)
{
	SectionHeader& stringTableSection = sectionHeader(
		header().header().e_shstrndx);

	return (const char*) _elfdata
		+ stringTableSection.header().sh_offset + offset;
}

const char* ELFFile::getSymbolStringAtOffset(
	long long unsigned int offset)
{
	_findSymbolTable();

	report("Getting symbol string at " << std::hex
		<< (_symbolStringTableOffset + offset) << std::dec);

	return (const char*) _elfdata + _symbolStringTableOffset + offset;
}

void ELFFile::write(std::ostream& out)
{
	out << "ELF Header:\n";
	out << " data sections:               " << sections() << "\n";
	out << " program sections:            " << programs() << "\n";
	out << " symbols:                     " << symbols() << "\n";
	out << " program header table offset: "
		<< std::hex << header().header().e_phoff << std::dec << "\n";
	out << " section header table offset: "
		<< std::hex << header().header().e_shoff << std::dec << "\n";

	out << "\n";
	
	for(unsigned int s = 0; s < sections(); ++s)
	{
		int type = sectionHeader(s).header().sh_type;
		out << "Section " << s << " Header:\n";
		out << " name: '"
			<< getSectionHeaderStringAtOffset(sectionHeader(s).header().sh_name)
			<< "'\n";
		out << " type: '"
			<< sectionTypeToString(type) << "'\n";
		out << " size in memory: "
			<< sectionHeader(s).header().sh_size << "\n";
		out << " offset:         "
			<< std::hex << sectionHeader(s).header().sh_offset
			<< std::dec << "\n";
		if(sectionTypeHasLink(type))
		{
			out << " link:           " << sectionLinkToString(type)
				<< " (section " << sectionHeader(s).header().sh_link << ")"
				<< "\n";
		}
		out << "\n";
	}
	
	for(unsigned int p = 0; p < programs(); ++p)
	{
		out << "Program " << p << " Header:\n";
		out << " type:          '"
			<< programTypeToString(programHeader(p).header().p_type) << "'\n";
		out << " size in memory: " << programHeader(p).header().p_memsz << "\n";
		out << " offset:         "
			<< std::hex << programHeader(p).header().p_offset
			<< std::dec << "\n";
		out << "\n";
	}

	for(unsigned int s = 0; s < symbols(); ++s)
	{
		out << "Symbol " << s << " Header:\n";
		out << " name:               '"
			<< getSymbolStringAtOffset(symbolHeader(s).header().st_name)
			<< "'\n";
		out << " type:               '"
			<< symbolTypeToString(symbolHeader(s).type()) << "'\n";
		out << " size in memory:     "
			<< symbolHeader(s).header().st_size << "\n";
		out << " value:              "
			<< symbolHeader(s).header().st_value << "\n";
		out << " containing section: "
			<< symbolHeader(s).header().st_shndx << "\n";
		
		out << "\n";
	}
}

std::string ELFFile::programTypeToString(int type)
{
	switch(type)
	{
	case elf::PT_NULL:    return "unused entry";
	case elf::PT_LOAD:    return "loadable segment";
	case elf::PT_DYNAMIC: return "dynamic linking information";
	case elf::PT_INTERP:  return "interpreter pathname";
	case elf::PT_NOTE:    return "a note";
	case elf::PT_SHLIB:   return "reserved";
	case elf::PT_PHDR:    return "the program header table";
	case elf::PT_LOOS:    return "low operating system";
	case elf::PT_HIOS:    return "high operating system";
	case elf::PT_LOPROC:  return "lowest processor specific program";
	case elf::PT_HIPROC:  return "highest processor specific program";
	default: break;
	}
	
	return "";
}

std::string ELFFile::sectionTypeToString(int type)
{
	switch(type)
	{
	case elf::SHT_NULL:          return "null section";
	case elf::SHT_PROGBITS:      return "program defined";
	case elf::SHT_SYMTAB:        return "symbol table";
	case elf::SHT_STRTAB:        return "string table";
	case elf::SHT_RELA:          return "relocations (addends)";
	case elf::SHT_HASH:          return "symbol hash table";
	case elf::SHT_DYNAMIC:       return "dynamic linking information";
	case elf::SHT_NOTE:          return "information about the file";
	case elf::SHT_NOBITS:        return "blank data with 0 size";
	case elf::SHT_REL:           return "relocations";
	case elf::SHT_SHLIB:         return "reserved";
	case elf::SHT_DYNSYM:        return "dynamic symbol table";
	case elf::SHT_INIT_ARRAY:    return "pointers to initialization functions";
	case elf::SHT_FINI_ARRAY:    return "pointers to termination functions";
	case elf::SHT_PREINIT_ARRAY: return "pointers to pre-init functions";
	case elf::SHT_GROUP:         return "section group";
	case elf::SHT_SYMTAB_SHNDX:  return "SHNDX entries";
	case elf::SHT_LOOS:          return "lowest OS type";
	case elf::SHT_HIOS:          return "highest OS type";
	case elf::SHT_LOPROC:        return "lowest processor type";
	case elf::SHT_HIPROC:        return "highest processor type";
	case elf::SHT_LOUSER:        return "lowest application type";
	case elf::SHT_HIUSER:        return "highest application type";
	default: break;
	}
	
	return "";
}

std::string ELFFile::sectionLinkToString(int type)
{
	switch(type)
	{
	case elf::SHT_DYNAMIC:       return "string table used by entries";
	case elf::SHT_HASH:          return "symbol table that the has applies to";
	case elf::SHT_REL:           return "symbol table referenced by relocs";
	case elf::SHT_RELA:          return "symbol table referenced by relocs";
	case elf::SHT_SYMTAB:        return "string table used by entries";
	case elf::SHT_DYNSYM:        return "string table used by entries";
	default: break;
	}
	
	return "undefined";
}

bool ELFFile::sectionTypeHasLink(int type)
{
	return type == elf::SHT_DYNAMIC
			|| type == elf::SHT_HASH
			|| type == elf::SHT_REL
			|| type == elf::SHT_RELA
			|| type == elf::SHT_SYMTAB
			|| type == elf::SHT_DYNSYM;
}

std::string ELFFile::symbolTypeToString(int type)
{
	switch(type)
	{
	case elf::STT_NOTYPE:  return "type not specified";
	case elf::STT_OBJECT:  return "data object";
	case elf::STT_FUNC:    return "executable object (code)";
	case elf::STT_SECTION: return "a section";
	case elf::STT_FILE:    return "local absolute reference to a file";
	case elf::STT_LOPROC:  return "lowest processor specific type";
	case elf::STT_HIPROC:  return "highest processor specific type";
	default: break;
	}
	
	return "";
}

void ELFFile::_loadProgram(unsigned int h)
{
	assert(h < programs());
	
	if(!_loadedPrograms[h])
	{
		long long unsigned int offset = header().header().e_phoff
			+ h * header().header().e_phentsize;
		_programs[h] = ProgramHeader(_elfdata, offset);
		_loadedPrograms[h] = true;
	}
}

void ELFFile::_loadSection(unsigned int h)
{
	assert(h < sections());
	
	if(!_loadedSections[h])
	{
		long long unsigned int offset = header().header().e_shoff
			+ h * header().header().e_shentsize;
		_sections[h] = SectionHeader(_elfdata, offset);
		_loadedSections[h] = true;
	}
}

void ELFFile::_loadSymbol(unsigned int h)
{
	_findSymbolTable();
	
	assert(h < symbols());
	
	if(!_loadedSymbols[h])
	{
		long long unsigned int offset = _symbolTableOffset
			+ (h + 1) * sizeof(elf::Elf64_Sym);
		_symbols[h] = SymbolHeader(_elfdata, offset);
		_loadedSymbols[h] = true;
	}
}

void ELFFile::_findSymbolTable()
{
	if(_symbolTableOffset != (long long unsigned int)-1) return;
	
	for(unsigned int s = 0; s < sections(); ++s)
	{
		const elf::Elf64_Shdr& header = sectionHeader(s).header();
		if(header.sh_type == elf::SHT_SYMTAB)
		{
			_symbolTableOffset = header.sh_offset;
			_symbolStringTableOffset = 
				sectionHeader(header.sh_link).header().sh_offset;
			unsigned int symbols =
				(header.sh_size / sizeof(elf::Elf64_Sym)) - 1;
			_symbols.resize(symbols);
			_loadedSymbols.resize(symbols, false);
			break;
		}
	}

	report("Found " << _symbols.size() << " symbols of size "
		<< sizeof(elf::Elf64_Sym));
}

}

std::ostream& operator<<(std::ostream& out, hydrazine::ELFFile& elf)
{
	elf.write(out);
	return out;
}

#endif


