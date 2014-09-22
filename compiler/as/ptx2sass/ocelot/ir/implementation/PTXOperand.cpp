/*! \file PTXOperand.cpp
	\author Andrew Kerr <arkerr@gatech.edu>
	\date Jan 15, 2009
	\brief internal representation of a PTX operand
*/

#include <ocelot/ir/interface/PTXOperand.h>

#include <hydrazine/interface/debug.h>

#include <cassert>
#include <sstream>
#include <iomanip>
#include <iostream>

std::string ir::PTXOperand::toString( VectorIndex index ) {
	switch( index ) {
		case ix:   return "x"; break;
		case iy:   return "y"; break;
		case iz:   return "z"; break;
		case iw:   return "w"; break;
		case iAll: return "";  break;
	}
	return "";
}


std::string ir::PTXOperand::toString(Vec index) {
	switch (index) {
		case v1: return "v1"; break;
		case v2: return "v2"; break;
		case v4: return "v4"; break;
	}
	return "";
}

std::string ir::PTXOperand::toString( DataType type ) {
	switch( type ) {
		case s8:   return "s8";   break;
		case s16:  return "s16";  break;
		case s32:  return "s32";  break;
		case s64:  return "s64";  break;
		case u8:   return "u8";   break;
		case u16:  return "u16";  break;
		case u32:  return "u32";  break;
		case u64:  return "u64";  break;
		case b8:   return "b8";   break;
		case b16:  return "b16";  break;
		case b32:  return "b32";  break;
		case b64:  return "b64";  break;
		case f16:  return "f16";  break;
		case f32:  return "f32";  break;
		case f64:  return "f64";  break;
		case pred: return "pred"; break;
		default: break;
	}
	return "Invalid";
}

std::string ir::PTXOperand::toString( SpecialRegister reg ) {
	switch( reg ) {
		case tid:          return "%tid";          break;
		case ntid:         return "%ntid";         break;
		case laneId:       return "%laneid";       break;
		case warpId:       return "%warpid";       break;
		case nwarpId:      return "%nwarpid";      break;
		case warpSize:     return "WARP_SZ";       break;
		case ctaId:        return "%ctaid";        break;
		case nctaId:       return "%nctaid";       break;
		case smId:         return "%smid";         break;
		case nsmId:        return "%nsmid";        break;
		case gridId:       return "%gridid";       break;
		case clock:        return "%clock";        break;
		case clock64:      return "%clock64";      break;
		case lanemask_eq:  return "%lanemask_eq";  break;
		case lanemask_le:  return "%lanemask_le";  break;
		case lanemask_lt:  return "%lanemask_lt";  break;
		case lanemask_ge:  return "%lanemask_ge";  break;
		case lanemask_gt:  return "%lanemask_gt";  break;
		case pm0:          return "%pm0";          break;
		case pm1:          return "%pm1";          break;
		case pm2:          return "%pm2";          break;
		case pm3:          return "%pm3";          break;
		case envreg0:      return "%envreg0";      break;
		case envreg1:      return "%envreg1";      break;
		case envreg2:      return "%envreg2";      break;
		case envreg3:      return "%envreg3";      break;
		case envreg4:      return "%envreg4";      break;
		case envreg5:      return "%envreg5";      break;
		case envreg6:      return "%envreg6";      break;
		case envreg7:      return "%envreg7";      break;
		case envreg8:      return "%envreg8";      break;
		case envreg9:      return "%envreg9";      break;
		case envreg10:     return "%envreg10";     break;
		case envreg11:     return "%envreg11";     break;
		case envreg12:     return "%envreg12";     break;
		case envreg13:     return "%envreg13";     break;
		case envreg14:     return "%envreg14";     break;
		case envreg15:     return "%envreg15";     break;
		case envreg16:     return "%envreg16";     break;
		case envreg17:     return "%envreg17";     break;
		case envreg18:     return "%envreg18";     break;
		case envreg19:     return "%envreg19";     break;
		case envreg20:     return "%envreg20";     break;
		case envreg21:     return "%envreg21";     break;
		case envreg22:     return "%envreg22";     break;
		case envreg23:     return "%envreg23";     break;
		case envreg24:     return "%envreg24";     break;
		case envreg25:     return "%envreg25";     break;
		case envreg26:     return "%envreg26";     break;
		case envreg27:     return "%envreg27";     break;
		case envreg28:     return "%envreg28";     break;
		case envreg29:     return "%envreg29";     break;
		case envreg30:     return "%envreg30";     break;
		case envreg31:     return "%envreg31";     break;
		default: break;
	}
	return "SpecialRegister_invalid";
}

std::string ir::PTXOperand::toString( AddressMode mode ) {
	switch( mode ) {
		case Register:     return "Register";     break;
		case Indirect:     return "Indirect";     break;
		case Immediate:    return "Immediate";    break;
		case Address:      return "Address";      break;
		case Label:        return "Label";        break;
		case Special:      return "Special";      break;
		case BitBucket:    return "BitBucket";    break;
		case ArgumentList: return "ArgumentList"; break;
		case FunctionName: return "FunctionName"; break;
		default: break;
	}
	return "Invalid";
}

std::string ir::PTXOperand::toString( DataType type, RegisterType reg ) {
	std::stringstream stream;
	if( type == pred ) {
		stream << "%p" << reg;
	}
	else {
		stream << "%r_" << toString( type ) << "_" << reg;
	}
	return stream.str();
}

bool ir::PTXOperand::isFloat( DataType type ) {
	bool result = false;
	switch( type ) {
		case f16: /* fall through */
		case f32: /* fall through */
		case f64: result = true;
		default: break;
	}
	return result;
}

bool ir::PTXOperand::isInt( DataType type ) {
	bool result = false;
	switch( type ) {
		case s8:  /* fall through */
		case s16: /* fall through */
		case s32: /* fall through */
		case s64: /* fall through */
		case u8:  /* fall through */
		case u16: /* fall through */
		case u32: /* fall through */
		case u64: result = true; break;
		default: break;
	}
	return result;
}

bool ir::PTXOperand::isSigned( DataType type ) {
	bool result = false;
	switch( type ) {
		case s8:  /* fall through */
		case s16: /* fall through */
		case s32: /* fall through */
		case s64: result = true; break;
		default: break;
	}
	return result;
}

unsigned int ir::PTXOperand::bytes( DataType type ) {
	assert( type != TypeSpecifier_invalid );
	switch( type ) {
		case pred: /* fall through */
		case b8:   /* fall through */
		case u8:   /* fall through */
		case s8:   return 1; break;
		case u16:  /* fall through */
		case f16:  /* fall through */
		case b16:  /* fall through */
		case s16:  return 2; break;
		case u32:  /* fall through */
		case b32:  /* fall through */
		case f32:  /* fall through */
		case s32:  return 4; break;
		case f64:  /* fall through */
		case u64:  /* fall through */
		case b64:  /* fall through */
		case s64:  return 8; break;
		default:   return 0; break;
	}
	return 0;	
}

bool ir::PTXOperand::valid( DataType destination, DataType source ) {
	switch( destination ) {
		case b64: {
			switch( source ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case f64: /* fall through */
				case b64: return true; break;
				default: break;
			}
			break;
		}
		case b32: {
			switch( source ) {
				case s32: /* fall through */
				case u32: /* fall through */
				case f32: /* fall through */
				case b32: return true; break;
				default: break;
			}
			break;
		}
		case b16: {
			switch( source ) {
				case s16: /* fall through */
				case u16: /* fall through */
				case f16: /* fall through */
				case b16: return true; break;
				default: break;
			}
			break;
		}
		case b8: {
			switch( source ) {
				case s8: /* fall through */
				case u8: /* fall through */
				case b8: return true; break;
				default: break;
			}
			break;
		}
		case u64: {
			switch( source ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case b64: return true; break;
				default: break;
			}
			break;
		}
		case u32: {
			switch( source ) {
				case s32: /* fall through */
				case u32: /* fall through */
				case b32: return true; break;
				default: break;
			}
			break;
		}
		case u16: {
			switch( source ) {
				case s16: /* fall through */
				case u16: /* fall through */
				case b16: return true; break;
				default: break;
			}
			break;
		}
		case u8: {
			switch( source ) {
				case s8: /* fall through */
				case u8: /* fall through */
				case b8: return true; break;
				default: break;
			}
			break;
		}
		case s64: {
			switch( source ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case b64: return true; break;
				default: break;
			}
			break;
		}
		case s32: {
			switch( source ) {
				case s32: /* fall through */
				case u32: /* fall through */
				case b32: return true; break;
				default: break;
			}
			break;
		}
		case s16: {
			switch( source ) {
				case s16: /* fall through */
				case u16: /* fall through */
				case b16: return true; break;
				default: break;
			}
			break;
		}
		case s8: {
			switch( source ) {
				case s8: /* fall through */
				case u8: /* fall through */
				case b8: return true; break;
				default: break;
			}
			break;
		}
		case f64: {
			switch( source ) {
				case b64: /* fall through */
				case f64: return true; break;
				default: break;
			}
			break;
		}
		case f32: {
			switch( source ) {
				case b32: /* fall through */
				case f32: return true; break;
				default: break;
			}
			break;
		}
		case f16: {
			switch( source ) {
				case b16: /* fall through */
				case f16: return true; break;
				default: break;
			}
			break;
		}
		case pred: {
			return source == pred;
			break;
		}
		default: break;
		
	}
	return false;
}

bool ir::PTXOperand::relaxedValid( DataType instructionType, 
	DataType operand ) {
	switch( instructionType ) {
		case b64: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case f64: /* fall through */
				case b64: return true; break;
				default: break;
			}
			break;
		}
		case b32: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case f64: /* fall through */
				case b64: /* fall through */
				case s32: /* fall through */
				case u32: /* fall through */
				case f32: /* fall through */
				case b32: return true; break;
				default: break;
			}
			break;
		}
		case b16: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case f64: /* fall through */
				case b64: /* fall through */
				case s32: /* fall through */
				case u32: /* fall through */
				case f32: /* fall through */
				case b32: /* fall through */
				case s16: /* fall through */
				case u16: /* fall through */
				case f16: /* fall through */
				case b16: return true; break;
				default: break;
			}
			break;
		}
		case b8: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case f64: /* fall through */
				case b64: /* fall through */
				case s32: /* fall through */
				case u32: /* fall through */
				case f32: /* fall through */
				case b32: /* fall through */
				case s16: /* fall through */
				case u16: /* fall through */
				case f16: /* fall through */
				case b16: /* fall through */
				case s8: /* fall through */
				case u8: /* fall through */
				case b8: return true; break;
				default: break;
			}
			break;
		}
		case u64: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case b64: return true; break;
				default: break;
			}
			break;
		}
		case u32: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case b64: /* fall through */
				case s32: /* fall through */
				case u32: /* fall through */
				case b32: return true; break;
				default: break;
			}
			break;
		}
		case u16: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case b64: /* fall through */
				case s32: /* fall through */
				case u32: /* fall through */
				case b32: /* fall through */
				case s16: /* fall through */
				case u16: /* fall through */
				case b16: return true; break;
				default: break;
			}
			break;
		}
		case u8: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case b64: /* fall through */
				case s32: /* fall through */
				case u32: /* fall through */
				case b32: /* fall through */
				case s16: /* fall through */
				case u16: /* fall through */
				case b16: /* fall through */
				case s8: /* fall through */
				case u8: /* fall through */
				case b8: return true; break;
				default: break;
			}
			break;
		}
		case s64: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case b64: return true; break;
				default: break;
			}
			break;
		}
		case s32: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case b64: /* fall through */
				case s32: /* fall through */
				case u32: /* fall through */
				case b32: return true; break;
				default: break;
			}
			break;
		}
		case s16: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case b64: /* fall through */
				case s32: /* fall through */
				case u32: /* fall through */
				case b32: /* fall through */
				case s16: /* fall through */
				case u16: /* fall through */
				case b16: return true; break;
				default: break;
			}
			break;
		}
		case s8: {
			switch( operand ) {
				case s64: /* fall through */
				case u64: /* fall through */
				case b64: /* fall through */
				case s32: /* fall through */
				case u32: /* fall through */
				case b32: /* fall through */
				case s16: /* fall through */
				case u16: /* fall through */
				case b16: /* fall through */
				case s8: /* fall through */
				case u8: /* fall through */
				case b8: return true; break;
				default: break;
			}
			break;
		}
		case f64: {
			switch( operand ) {
				case b64: /* fall through */
				case f64: return true; break;
				default: break;
			}
			break;
		}
		case f32: {
			switch( operand ) {
				case b32: /* fall through */
				case f32: return true; break;
				default: break;
			}
			break;
		}
		case f16: {
			switch( operand ) {
				case b16: /* fall through */
				case f16: return true; break;
				default: break;
			}
			break;
		}
		case pred: {
			return operand == pred;
			break;
		}
		default: break;
		
	}
	return false;
}


ir::PTXOperand::PTXOperand() {
	identifier = "";
	addressMode = Invalid;
	type = PTXOperand::s32;
	relaxedType = TypeSpecifier_invalid;
	offset = 0;
	imm_int = 0;
	reg = 0;
	vec = v1;
}

ir::PTXOperand::PTXOperand(SpecialRegister r, VectorIndex i, DataType t) : 
	addressMode(Special), type(t), relaxedType(TypeSpecifier_invalid),
	vIndex(i), special(r), 
	reg(0), vec(i == iAll ? v4 : v1) {
	std::stringstream name;
	name << toString(r);
	if( i != iAll )
	{
		name << "." << toString(i);
	}
	else
	{
		array.push_back( PTXOperand( r, ix, t ) );
		array.push_back( PTXOperand( r, iy, t ) );
		array.push_back( PTXOperand( r, iz, t ) );
		array.push_back( PTXOperand( r, iw, t ) );
	}
	identifier = name.str();
}

ir::PTXOperand::PTXOperand(const std::string& l) : identifier(l), 
	addressMode(Label), type(TypeSpecifier_invalid),
	relaxedType(TypeSpecifier_invalid),
	offset(0), condition(Pred), reg(0), vec(v1) {
}

ir::PTXOperand::PTXOperand(AddressMode m, DataType t, RegisterType r, 
	int o, Vec v) : addressMode(m), type(t), relaxedType(TypeSpecifier_invalid),
	offset(o), condition(Pred), reg(r), vec(v) {
}

ir::PTXOperand::PTXOperand(AddressMode m, DataType t, 
	const std::string& i, int o, Vec v) : identifier(i), 
	addressMode(m), type(t), relaxedType(TypeSpecifier_invalid), offset(o),
	condition(Pred), vec(v) {
}

ir::PTXOperand::PTXOperand(AddressMode m, const std::string& i) : identifier(i),
	addressMode(m), type(TypeSpecifier_invalid),
	relaxedType(TypeSpecifier_invalid), offset(0), condition(Pred),
	reg(0), vec(v1) {
}

ir::PTXOperand::PTXOperand(PredicateCondition c) : 
	addressMode(Register), type(pred),
	relaxedType(TypeSpecifier_invalid), offset(0), condition(c),
	reg(0), vec(v1) {
}

ir::PTXOperand::~PTXOperand() {

}

/*!
	Displays a binary represetation of a 32-bit floating-point value
*/
static std::ostream & write(std::ostream &stream, float value) {
	union {
		unsigned int imm_uint;
		float value;
	} float_union;
	float_union.value = value;
	stream << "0f" << std::setw(8) << std::setfill('0') 
		<< std::hex << float_union.imm_uint << std::dec;
	return stream;
}

/*!
	Displays a binary represetation of a 64-bit floating-point value
*/
static std::ostream & write(std::ostream &stream, double value) {
	union {
		long long unsigned int imm_uint;
		double value;
	} double_union;
	double_union.value = value;
	stream << "0d" << std::setw(16) << std::setfill('0') << std::hex 
		<< double_union.imm_uint;
	return stream;
}

std::string ir::PTXOperand::toString() const {
	
	if( addressMode == BitBucket ) {
		return "_";
	} else if( addressMode == Indirect ) {
		std::stringstream stream;

		if ( identifier != "" ) {
			stream << identifier;
		}
		else {
			stream << "%r" << reg;
		}
	
		if( offset < 0 ) {
			// The NVIDIA driver does not support 
			//   '- offset' it needs '+ -offset'
			stream << " + " << ( offset );
		} else if ( offset > 0 ) {
			stream << " + " << ( offset );
		}

		return stream.str();
	} else if( addressMode == Address ) {
		std::stringstream stream;
		if( offset == 0 ) {
			return identifier;
		}
		else if( offset < 0 ) {
			stream << ( offset );
			return identifier + " + " + stream.str();
		} else {
			stream << ( offset );
			return identifier + " + " + stream.str();
		}
	} else if( addressMode == Immediate ) {
		std::stringstream stream;
		switch( type ) {
			case pred: /* fall through */
			case s8:  /* fall through */
			case s16: /* fall through */
			case s32: /* fall through */
			case s64: stream << imm_int; break;
			case u8:  /* fall through */
			case u16: /* fall through */
			case u32: /* fall through */
			case u64: /* fall through */
			case b8:  /* fall through */
			case b16: /* fall through */
			case b32: /* fall through */
			case b64: stream << imm_int; break;
			case f16: /* fall through */
			case f32: {
				write(stream, imm_single);
			} break;
			case f64: {
				write(stream, imm_float);
			} break;
			default: 
				assertM( false, "Invalid immediate type " 
				+ PTXOperand::toString( type ) ); break;
		}
		return stream.str();
	} else if( addressMode == Special ) {
		bool isScalar = true;
		switch (special) {
		case tid: // fall through
		case ntid: // fall through
		case ctaId: // fall through
		case nctaId:  // fall through
		case smId:  // fall through
		case nsmId:  // fall through
		case gridId:  // fall through
			isScalar = false;
			break;
		default:
			isScalar = true;
		}
		if( vec != v1 || isScalar) {
			return toString( special );
		}
		else {
			assert( vec == v1 );
			assert( array.empty() );
			return toString( special ) + "." + toString( vIndex );
		}
	} else if( addressMode == ArgumentList ) {
		std::string result = "(";
		for( Array::const_iterator fi = array.begin(); 
			fi != array.end(); ++fi ) {
			result += fi->toString();
			if( std::next(fi) != array.end() ) {
				result += ", ";
			}
		}
		return result + ")";
	} else if( type == pred ) {
		switch( condition ) {
			case PT: return "%pt"; break;
			case nPT: return "%pt"; break;
			default:
			{
				if( !identifier.empty() ) {
					return identifier;
				}
				else {
					std::stringstream stream;
					if( condition == InvPred ) stream << "!";
					stream << "%p" << reg;
					return stream.str();
				}
				break;
			}
		}
	} 
	else if( vec != v1 ) {
		if( !array.empty() ) {
			assert( ( vec == v2 && array.size() == 2 ) 
				|| ( vec == v4 && array.size() == 4 ) );
			std::string result = "{";
			for( Array::const_iterator fi = array.begin(); 
				fi != array.end(); ++fi ) {
				result += fi->toString();
				if( fi != --array.end() ) {
					result += ", ";
				}
			}
			return result + "}";
		}
		else {
			assert( vIndex != iAll );
			std::stringstream stream;
			if( !identifier.empty() ) {
				stream << identifier;
			}
			else {
				stream << "%r" << reg;
			}
			stream << "." << toString(vIndex);
			return stream.str();
		}
	}
	
	if (addressMode == Register && array.size()) {
		std::stringstream stream;
		stream << "{";
		for (size_t n = 0; n < array.size(); n++) {
			stream << (n ? ", " : "") << array[n].toString();
		}
		stream << "}";
		return stream.str();
	}
	else {
		if( !identifier.empty() ) {
			return identifier;
		}
		else {
			std::stringstream stream;
			stream << "%r" << reg;
			return stream.str();
		}
	}
}

std::string ir::PTXOperand::registerName() const {
	assertM( addressMode == Indirect || addressMode == Register
		|| addressMode == BitBucket, "invalid register address mode "
	    << toString(addressMode) );
	
	if (addressMode == BitBucket) return "_";
	
	if( !identifier.empty() ) {
		return identifier;
	}
	else {
		std::stringstream stream;
		if(type == pred) {
			switch( condition ) {
				case PT: return "%pt"; break;
				case nPT: return "%pt"; break;
				default:
				{
					std::stringstream stream;
					stream << "%p" << reg;
					return stream.str();
					break;
				}
			}
		}
		else {
			stream << "%r" << reg;
		}
		return stream.str();
	}
}

unsigned int ir::PTXOperand::bytes() const {
	return bytes( type ) * vec;
}

bool ir::PTXOperand::isRegister() const {
	return addressMode == Register || addressMode == Indirect
		|| addressMode == BitBucket;
}

bool ir::PTXOperand::isVector() const {
	return isRegister() && vec != v1;
}


