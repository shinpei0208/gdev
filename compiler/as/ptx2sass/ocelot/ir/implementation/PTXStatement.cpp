/*! \file PTXStatement.cpp
	\date Monday January 19, 2009
	\author Andrew Kerr
	\brief The header file for the PTXStatement class
*/

#ifndef IR_PTXSTATEMENT_CPP_INCLUDED
#define IR_PTXSTATEMENT_CPP_INCLUDED

// Ocelot Includes
#include <ocelot/ir/interface/PTXStatement.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/Casts.h>

// Standard Library Includes
#include <cstring>
#include <sstream>
#include <stack>
#include <cstdint>

namespace ir {

	PTXStatement::Symbol::Symbol(const std::string& n, unsigned int o)
	: name(n), offset(o)
	{
	
	}

	static void write(std::ostream &out,
		const ir::PTXStatement::ArrayVector & values,
		ir::PTXOperand::DataType type) {
		ir::PTXStatement::ArrayVector::const_iterator it = values.begin();
		for (int n = 0; it != values.end(); ++it, ++n) {
			out << (n ? ", " : "");
			switch (type) {
				case ir::PTXOperand::s8:  out << it->s8; break;
				case ir::PTXOperand::s16: out << it->s16; break;
				case ir::PTXOperand::s32: out << it->s32; break;
				case ir::PTXOperand::s64: out << it->s64;  break;
				case ir::PTXOperand::u8:  out << it->u8; break;
				case ir::PTXOperand::u16: out << it->u16; break;
				case ir::PTXOperand::u32: out << it->u32; break;
				case ir::PTXOperand::u64: out << it->u64; break;
				case ir::PTXOperand::f32:
				{
					out << "0f" << std::hex
						<< hydrazine::bit_cast<uint32_t>(it->f32)
					    << std::dec;
					break;
				}
				case ir::PTXOperand::f64: out << it->f64; break;
				case ir::PTXOperand::b8:  out << it->b8; break;
				case ir::PTXOperand::b16: out << it->b16; break;
				case ir::PTXOperand::b32: out << it->b32; break;
				case ir::PTXOperand::b64: out << it->b64; break;
				default: break;
			}
		}
	}

	std::string PTXStatement::StaticArray::dimensions() const {
		if( stride.size() == 0 ) {
			return "";
		}
		
		std::stringstream stream;
		for( ArrayStrideVector::const_iterator si = stride.begin(); 
			si != stride.end(); ++si ) {
			
			if( *si == 0 ) {
				stream << "[]";
				continue; 
			}
			
			stream << "[" << *si << "]";
		}
		
		return stream.str();
	}
	
	std::string PTXStatement::StaticArray::initializer( 
		PTXOperand::DataType t ) const {
		std::stringstream stream;
		if( stride.size() == 0 ) {
			assert( values.size() == 1 );
			stream << PTXStatement::toString( values.front(), t );
		}
		std::stack< unsigned int > stack;
		stream << "{ ";
		stack.push( 0 );
		ArrayStrideVector::const_iterator si = stride.begin();
		
		unsigned int index = 0;
		
		while( !stack.empty() ) {	
				
			if( si == --stride.end() ) {
			
				for( ; stack.top() < *si; ++stack.top() ) {
				
					if( vec == PTXOperand::v1 ) {
					
						assert( index < values.size() );
						stream << valueAt( index, t );
						++index;
						
					}
					else if( vec == PTXOperand::v2 ) {
					
						assert( index < values.size() );
						stream << "{ " << valueAt( index, t ) << ", ";
						++index;
						assert( index < values.size() );
						stream << valueAt( index, t ) << " }";
						++index;
						
					}
					else {
					
						assert( index < values.size() );
						stream << "{ " << valueAt( index, t ) << ", ";
						++index;
						assert( index < values.size() );
						stream << valueAt( index, t ) << ", ";
						++index;
						assert( index < values.size() );
						stream << valueAt( index, t ) << ", ";
						++index;
						assert( index < values.size() );
						stream << valueAt( index, t ) << " }";
						++index;				
					}
					
					if( stack.top() != ( *si - 1 ) ) {
						stream << ", ";
					}
					
				}
				
			}
			
			if( stack.top() < *si ) {
			
				if( stack.top() != 0 ) {
					stream << ", ";
				}
				stream << "{ ";
				++stack.top();
				stack.push( 0 );
				++si;
				
			}
			else {
			
				stream << " }";
				stack.pop();
				if( !stack.empty() ) {
					// If stack.empty cannot decrement si as it is at begin()
					// and it is invalid to decrement such an iterator.
					--si;
				}	
			}
			
		}
		
		assert( index == values.size() );
		return stream.str();
		
	}
	
	std::string PTXStatement::StaticArray::valueAt( unsigned int index, 
		PTXOperand::DataType t ) const
	{
		for(auto symbol = symbols.begin(); symbol != symbols.end(); ++symbol)
		{
			if(symbol->offset == index) return symbol->name;
		}
		
		return toString( values[ index ], t );
	}

	std::string PTXStatement::toString( TextureSpace space ) {
		switch ( space ) {
			case GlobalSpace: return "global";
			case ParameterSpace: return "param";
			default: break;
		}
		return "";
	}
	
	std::string PTXStatement::toString( Attribute attribute ) {
		switch( attribute ) {
			case Visible: return "visible"; break;
			case Extern:  return "extern";  break;
			case Weak:    return "weak";    break;
			default: break;
		}
		return "";
	}
	
	std::string PTXStatement::toString( Data d, PTXOperand::DataType t ) {
		std::stringstream stream;
		switch( t ) {
			case PTXOperand::s8:  /*fall through*/
			case PTXOperand::s16: /*fall through*/
			case PTXOperand::s32: /*fall through*/
			case PTXOperand::s64: stream << d.s64; break;
			case PTXOperand::u8:  /*fall through*/
			case PTXOperand::u16: /*fall through*/
			case PTXOperand::u32: /*fall through*/
			case PTXOperand::u64: /*fall through*/
			case PTXOperand::b8:  /*fall through*/
			case PTXOperand::b16: /*fall through*/
			case PTXOperand::b32: /*fall through*/
			case PTXOperand::b64: stream << d.u64; break;
			case PTXOperand::f32: 
			{
				stream << "0f" << std::hex
					<< hydrazine::bit_cast<uint32_t>(d.f32)
				    << std::dec;
				break;
			}
			case PTXOperand::f64: stream << d.f64; break;
			default : assert("Invalid type" == 0);
		}
		return stream.str();
	}
	
	PTXStatement::PTXStatement( Directive d ) {
		directive = d;
		alignment = 1;
		array.vec = PTXOperand::v1;
		attribute = NoAttribute;
		ptrAddressSpace = PTXInstruction::AddressSpace_Invalid;
	
	}
	
	PTXStatement::~PTXStatement() {
	
	}

	unsigned int PTXStatement::elements() const {
		unsigned int result = 1;
		for( ArrayStrideVector::const_iterator stride = array.stride.begin(); 
			stride != array.stride.end(); ++stride ) {
			result *= *stride;
		}
		result *= array.vec;
		return result;
	}

	unsigned int PTXStatement::bytes() const {
		unsigned int result = elements();
		unsigned int elementsize = PTXOperand::bytes(type);
		result *= elementsize;
		return result;
	}

	unsigned int PTXStatement::initializedBytes() const {
		unsigned int result = array.values.size();
		unsigned int elementsize = PTXOperand::bytes(type);
		result *= elementsize;
		return result;
	}

	void PTXStatement::copy(void* dest) const {
		char* d = (char*) dest;
		unsigned int bytes = PTXOperand::bytes(type);
		for (ArrayVector::const_iterator element = array.values.begin(); 
			element != array.values.end(); ++element, d += bytes) {
			memcpy(d, &element->b8, bytes);
		}
	}
	
	unsigned int PTXStatement::accessAlignment() const {
		return std::max( (unsigned int) alignment, 
			PTXOperand::bytes(type) * array.vec );
	}
	
	/*!
	
	*/
	std::string PTXStatement::toString( Directive directive ) {
		switch (directive) {
			case Instr: return "Instr";
			case CallTargets: return "CallTargets";
			case Const: return "Const";
			case Entry: return "Entry";
			case File: return "File";
			case Func: return "Func";
			case FunctionPrototype: return "FunctionPrototype";
			case Global: return "Global";
			case Label: return "Label";
			case Local: return "Local";
			case Loc: return "Loc";
			case Maxnreg: return "Maxnreg";
			case Maxntid: return "Maxntid";
			case Maxnctapersm: return "Maxnctapersm";
			case Minnctapersm: return "Minnctapersm";
			case Param: return "Param";
			case Pragma: return "Pragma";
			case Reg: return "Reg";
			case Reqntid: return "Reqntid";
			case Samplerref: return "Samplerref";
			case Section: return "Section";
			case Shared: return "Shared";
			case Sreg: return "Sreg";
			case Surfref: return "Surfref";
			case Target: return "Target";
			case Texref: return "Texref";
			case Version: return "Version";
			case StartScope: return "StartScope";
			case EndScope: return "EndScope";
			case StartParam: return "StartParam";
			case EndParam: return "EndParam";
			case FunctionName: return "FunctionName";
			case EndFuncDec: return "EndFuncDec";
			case Directive_invalid: 
			default:
				break;
		}
		return "Directive_invalid";
	}
	
	/*!
		
	*/
	std::string PTXStatement::toString() const {
	
		switch( directive ) {
			case Instr: {
				return instruction.toString() + ";" + instruction.metadata;
				break;
			}
			case AddressSize: {
				std::stringstream stream;
				stream << ".address_size " << addressSize;
				break;
			}
			case CallTargets: {
				std::string result = name + ": .calltargets ";
				for( StringVector::const_iterator target = targets.begin(); 
					target != targets.end(); ++target ) {
					if( target != targets.begin() ) {
						result += ", ";
					}
					result += *target;
				}
				return result + ";";
				break;
			}
			case Const: {
				std::stringstream stream;
				if( attribute != NoAttribute ) {
					stream << "." << toString( attribute ) << " ";
				}
				stream << ".const";
				assert( alignment != 0);
				if( alignment != 1 ) {
					stream << " .align " << alignment;
				}
				if( array.vec != PTXOperand::v1 ) {
					stream << " ." << PTXInstruction::toString( array.vec );
				}
				stream << " ." << PTXOperand::toString( type ) << " " << name;
				stream << array.dimensions();
				if( !array.values.empty() ) { 
					stream << " = " << array.initializer( type );
				}
				stream << ";";
				return stream.str();
				break;
			}
			case Entry: {
				return ".entry " + name;
				break;
			}
			case File: {
				std::stringstream stream;
				stream << ".file " << sourceFile << " \"" << name << "\"";
				return stream.str();
				break;
			}
			case Func: {
				std::string result;
				if( attribute != NoAttribute ) {
					result += "." + toString( attribute ) + " ";
				}
				return result + ".func";
				break;
			}
			case FunctionPrototype: {
				std::string result = name + ": .callprototype ";
				
				if(!returnTypes.empty()) {
					result += "(";
					for(TypeVector::const_iterator type = returnTypes.begin(); 
						type != returnTypes.end(); ++type) {
						if( type != returnTypes.begin() ) result += ", ";
						result += ".param ." +
							PTXOperand::toString( *type ) + " _";
					}
					result += ") ";
				}
				
				result += name + " (";
				for(TypeVector::const_iterator type = argumentTypes.begin(); 
					type != argumentTypes.end(); ++type) {
					if( type != argumentTypes.begin() ) result += ", ";
					result += ".param ." + PTXOperand::toString( *type ) + " _";
				}
				return result + ");";
			}
			case Global: {
				std::stringstream stream;
				if( attribute != NoAttribute ) {
					stream << "." << toString( attribute ) << " ";
				}
				stream << ".global";
				assert( alignment != 0);
				if( alignment != 1 ) {
					stream << " .align " << alignment;
				}
				if( array.vec != PTXOperand::v1 ) {
					stream << " ." << PTXInstruction::toString( array.vec );
				}
				stream << " ." << PTXOperand::toString( type ) << " " << name;
				stream << array.dimensions();
				if( !array.values.empty() ) { 
					stream << " = " << array.initializer( type );
				}
				stream << ";";
				return stream.str();
				break;
			}
			case Label: {
				return name + ":" + instruction.metadata;
				break;
			}
			case Local: {
				std::stringstream stream;
				if( attribute != NoAttribute ) {
					stream << "." << toString( attribute ) << " ";
				}
				stream << ".local";
				assert( alignment != 0);
				if( alignment != 1 ) {
					stream << " .align " << alignment;
				}
				if( array.vec != PTXOperand::v1 ) {
					stream << " ." << PTXInstruction::toString( array.vec );
				}
				stream << " ." << PTXOperand::toString( type ) << " " << name;
				stream << array.dimensions();
				stream << ";";
				return stream.str();
				break;
			}
			case Loc: {
				std::stringstream stream;
				stream << ".loc " << sourceFile << " " 
					<< sourceLine << " " << sourceColumn;
				return stream.str();
				break;
			}
			
			case Maxnreg: {
				std::stringstream ss;
				ss << ".maxnreg ";
				write(ss, array.values, ir::PTXOperand::u32);
				ss << ";";
				return ss.str();
				break;
			}
			case Maxntid: {
				std::stringstream ss;
				ss << ".maxntid ";
				write(ss, array.values, ir::PTXOperand::u32);
				ss << ";";
				return ss.str();
			}
			case Maxnctapersm: {
				std::stringstream ss;
				ss << ".maxnctapersm ";
				write(ss, array.values, ir::PTXOperand::u32);
				ss << ";";
				return ss.str();
			}
			case Minnctapersm: {
				std::stringstream ss;
				ss << ".minnctapersm ";
				write(ss, array.values, ir::PTXOperand::u32);
				ss << ";";
				return ss.str();
			}
			case Param: {
				assert( array.values.empty() );
				std::stringstream stream;
				if( attribute != NoAttribute ) {
					stream << "." << toString( attribute ) << " ";
				}
				stream << ".param ";
	
				stream << " ." << PTXOperand::toString( type );
	
				if (ptrAddressSpace != PTXInstruction::AddressSpace_Invalid) {
					if (ptrAddressSpace == PTXInstruction::Generic) {
						stream << " .ptr";
					}
					else {
						stream << " .ptr ." << PTXInstruction::toString(ptrAddressSpace);
					}
				}
				if( alignment != 1 ) {
					stream << " .align " << alignment;
				}
				if( array.vec != PTXOperand::v1 ) {
					stream << " ." << PTXInstruction::toString( array.vec );
				}
	
				stream << " " << name;
				stream << array.dimensions();
				return stream.str();
				break;
			}
			case Pragma: {
				return ".pragma \"" + name + "\"";
				break;
			}
			case Reg: {
				std::stringstream stream;
				if( attribute != NoAttribute ) {
					stream << "." << toString( attribute ) << " ";
				}
				stream << ".reg";
				if( array.vec != PTXOperand::v1 ) {
					stream << " ." << PTXInstruction::toString( array.vec );
				}
				stream << " ." << PTXOperand::toString( type ) << " " << name;
				assert( array.stride.size() == 1 );
				if( array.stride[0] != 0 )
				{
					stream << "<" << array.stride[0] << ">";
				}
				stream << ";";
				return stream.str();
				break;
			}
			case Reqntid: {
				std::stringstream ss;
				ss << ".reqntid ";
				write(ss, array.values, ir::PTXOperand::u32);
				ss << ";";
				return ss.str();
			}
			case Samplerref:
			{
				return "." + toString(space) + " .samplerref " + name + ";";
				break;
			}
			case Section:
				return ".section " + section_type + ", " + section_name;
				break;
			case Shared: {
				std::stringstream stream;
				if( attribute != NoAttribute ) {
					stream << "." << toString( attribute ) << " ";
				}
				stream << ".shared";
				assert( alignment != 0);
				stream << " .align " << alignment;
				if( array.vec != PTXOperand::v1 ) {
					stream << " ." << PTXInstruction::toString( array.vec );
				}
				stream << " ." << PTXOperand::toString( type ) << " " << name;
				stream << array.dimensions();
				stream << ";";
				return stream.str();
				break;
			}
			case Sreg:
				return ".sreg " + PTXInstruction::toString( array.vec ) + 
					" ." + PTXOperand::toString( type ) + " " + name + ";";
				break;
			case Target: {
				return ".target " + hydrazine::toString( targets.begin(), 
					targets.end(), ", " );
				break;
			}
			case Surfref: {
				return "." + toString(space) + " .surfref " + name + ";";
				break;
			}
			case Texref: {
				return "." + toString( space ) + " .texref " + name + ";";
				break;
			}
			case Version: {
			  std::stringstream stream;
			  stream << ".version " << major << '.' << minor;
				return stream.str();
				break;
			}
			case StartScope:
				return "{";
				break;		
			case EndScope:
				return "}";
				break;			
			case StartParam:
				return "(";
				break;		
			case EndParam:
				return ")";
				break;
			case FunctionName:
				return name;
				break;
			case EndFuncDec:
				return "";
				break;
			case Directive_invalid:
				return "";
				break;
			default:
				break;
		}
		return "";
	
	}
	
}

#endif

