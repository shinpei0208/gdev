/*!
	\file LLVMInstruction.cpp
	\date Tuesday July 21, 2009
	\author Gregroy Diamos <gregory.diamos@gatech.edu>
	\brief The source file for the LLVMInstruction class.
*/

#ifndef LLVM_INSTRUCTION_CPP_INCLUDED
#define LLVM_INSTRUCTION_CPP_INCLUDED

#include <ocelot/ir/interface/LLVMInstruction.h>
#include <hydrazine/interface/debug.h>

namespace ir
{
	LLVMInstruction::Type::Type( DataType t, Category c, LLVMI32 v ) 
		: type( t ), category( c ), vector( v )
	{
	
	}
	
	std::string LLVMInstruction::Type::toString() const
	{
		switch( category )
		{
			case Element:
			{
				return LLVMInstruction::toString( type );
				break;
			}
			case Array:
			{
				std::stringstream stream;
				stream << "[ " << vector << " x " 
					<< LLVMInstruction::toString( type ) << " ]";
				return stream.str();
				break;
			}
			case Function:
			{
				assertM( false, 
					"Basic LLVM Type does not support functions." );
				break;
			}
			case Structure:
			{
				if( !label.empty() ) return label;
				std::string result = "{ ";
				for( TypeVector::const_iterator fi = members.begin(); 
					fi != members.end(); ++fi )
				{
					if( fi != members.begin() ) result += ", ";
					result += fi->toString();
				}
				result += " }";
				return result;
				break;
			}
			case PackedStructure:
			{
				return label;
				break;
			}
			case Pointer:
			{
				if( members.empty() ) 
				{
					if( type != InvalidDataType )
					{
						if( vector > 1 )
						{
							std::stringstream stream;
							stream << "< " << vector << " x " 
								<< LLVMInstruction::toString( type ) << " >*";
							return stream.str();
						}
						else
						{
							return LLVMInstruction::toString( type ) + "*";
						}
					}
					else
					{
						return label + "*";
					}
				}
				else
				{
					assertM( members.size() == 1, 
						"Pointers can only contain one subtype." );
					return members[0].toString() + "*";
				}
				break;
			}
			case Vector:
			{
				std::stringstream stream;
				stream << "< " << vector << " x " 
					<< LLVMInstruction::toString( type ) << " >";
				return stream.str();
				break;
			}
			case Opaque:
			{
				return label;
				break;
			}
			case InvalidCategory:
			{
				return "INVALID_CATEGORY";
				break;
			}
		}
		return "";
	}
		
	LLVMInstruction::Operand::Operand( const std::string& n, const Type& t ) :
		name( n ), constant( false ), type( t )
	{
	
	}

	LLVMInstruction::Operand::Operand( LLVMI64 value ) :
		constant( true ), type( Type( I64, Type::Element ) )
	{
		i64 = value;
	}

	LLVMInstruction::Operand::Operand( LLVMI32 value ) :
		constant( true ), type( Type( I32, Type::Element ) )
	{
		i32 = value;
	}
	
	LLVMInstruction::Operand::Operand( LLVMF32 value ) :
		constant( true ), type( Type( F32, Type::Element ) )
	{
		f32 = value;
	}
	
	LLVMInstruction::Operand::Operand( LLVMF64 value ) :
		constant( true ), type( Type( F64, Type::Element ) )
	{
		f64 = value;
	}
	
	LLVMInstruction::Operand::Operand( LLVMI1 value ) :
		constant( true ), type( Type( I1, Type::Element ) )
	{
		i1 = value;
	}
	
	bool LLVMInstruction::Operand::valid() const
	{
		return type.category != Type::InvalidCategory;
	}

	std::string LLVMInstruction::Operand::toString() const
	{
		if( constant )
		{
			switch( type.category )
			{
				case Type::Element:
				{
					std::stringstream stream;
					switch( type.type )
					{
						case I1: stream << (((int) i1) & 0x1); break;
						case I8: stream << (~((int) (~i8 + 1)) + 1); break;
						case I16: stream << i16; break;
						case I32: stream << i32; break;
						case I64: stream << i64; break;
						case I128: stream << i128; break;
						case F32:
						{
							union
							{
								LLVMF64 f;
								LLVMI64 i;
							} convert;
							
							convert.f = f32;
							stream << "0x" << std::hex << convert.i;
							break;
						}
						case F64: stream << "0x" << std::hex << i64; break;
						case F128: stream << f128; break;
						case InvalidDataType: break;
					}
					return stream.str();
					break;
				}
				case Type::Array:
				{
					assertM( false, "Array constant not implemented." );
					break;
				}
				case Type::Function:
				{
					assertM( false, "Function constant not implemented." );
					break;
				}
				case Type::Structure:
				{
					assertM( false, "Structure constant not implemented." );
					break;
				}
				case Type::PackedStructure:
				{
					assertM( false, 
						"Packed Structure constant not implemented." );
					break;
				}
				case Type::Pointer:
				{
					std::stringstream stream;
					stream << i64;
					return stream.str();
					break;
				}
				case Type::Vector:
				{
					std::stringstream stream;
					stream << "< ";
					for( ValueVector::const_iterator fi = values.begin(); 
						fi != values.end(); ++fi )
					{
						if( fi != values.begin() ) stream << ", ";
						stream << LLVMInstruction::toString( type.type ) << " ";
						switch( type.type )
						{
							case I1: stream << (((int) fi->i1) & 0x1); break;
							case I8: stream << (~((int) (~fi->i8 + 1)) + 1); 
								break;
							case I16: stream << fi->i16; break;
							case I32: stream << fi->i32; break;
							case I64: stream << fi->i64; break;
							case I128: stream << fi->i128; break;
							case F32:
							{
								union
								{
									LLVMF64 f;
									LLVMI64 i;
								} convert;
							
								convert.f = fi->f32;
								stream << "0x" << std::hex << convert.i;
								break;
							}
							case F64: stream << "0x" << std::hex << fi->i64; 
								break;
							case F128: stream << f128; break;
							case InvalidDataType: break;
						}
					}
					stream << " >";
					return stream.str();
					break;
				}
				case Type::Opaque:
				{
					return "Opaque constant not implemented.";
					break;
				}
				case Type::InvalidCategory:
				{
					return "INVALID_CATEGORY";
					break;
				}
			}
		}
		else
		{
			return name;
		}
		return "";
	}
	
	LLVMInstruction::Parameter::Parameter() 
		: attribute( InvalidParameterAttribute )
	{
	
	}

	LLVMInstruction::Parameter::Parameter( const Operand& op ) 
		: Operand(op), attribute( InvalidParameterAttribute )
	{
	
	}

	const LLVMInstruction::Parameter& LLVMInstruction::Parameter::operator=( 
		const Operand& op )
	{
		Operand::operator=( op );
		return *this;
	}

	const LLVMInstruction::Parameter& LLVMInstruction::Parameter::operator=( 
		const Parameter& p )
	{
		attribute = p.attribute;
		Operand::operator=( p );
		return *this;
	}


	std::string LLVMInstruction::toString( Opcode code )
	{
		switch( code )
		{
			case Add: return "add"; break;
			case Alloca: return "alloca"; break;
			case And: return "and"; break;
			case Ashr: return "ashr"; break;
			case Atomicrmw: return "atomicrmw"; break;
			case Bitcast: return "bitcast"; break;
			case Br: return "br"; break;
			case Call: return "call"; break;
			case Cmpxchg: return "cmpxchg"; break;
			case Extractelement: return "extractelement"; break;
			case Extractvalue: return "extractvalue"; break;
			case Fadd: return "fadd"; break;
			case Fcmp: return "fcmp"; break;
			case Fdiv: return "fdiv"; break;
			case Fmul: return "fmul"; break;
			case Fpext: return "fpext"; break;
			case Fptosi: return "fptosi"; break;
			case Fptoui: return "fptoui"; break;
			case Fptrunc: return "fptrunc"; break;
			case Free: return "free"; break;
			case Frem: return "frem"; break;
			case Fsub: return "fsub"; break;
			case Getelementptr: return "getelementptr"; break;
			case Icmp: return "icmp"; break;
			case Insertelement: return "insertelement"; break;
			case Insertvalue: return "insertvalue"; break;
			case Inttoptr: return "inttoptr"; break;
			case Invoke: return "invoke"; break;
			case Load: return "load"; break;
			case Lshr: return "lshr"; break;
			case Malloc: return "malloc"; break;
			case Mul: return "mul"; break;
			case Or: return "or"; break;
			case Phi: return "phi"; break;
			case Ptrtoint: return "ptrtoint"; break;
			case Ret: return "ret"; break;
			case Sdiv: return "sdiv"; break;
			case Select: return "select"; break;
			case Sext: return "sext"; break;
			case Shl: return "shl"; break;
			case Shufflevector: return "shufflevector"; break;
			case Sitofp: return "sitofp"; break;
			case Srem: return "srem"; break;
			case Store: return "store"; break;
			case Sub: return "sub"; break;
			case Switch: return "switch"; break;
			case Trunc: return "trunc"; break;
			case Udiv: return "udiv"; break;
			case Uitofp: return "uitofp"; break;
			case Unreachable: return "unreachable"; break;
			case Unwind: return "unwind"; break;
			case Urem: return "urem"; break;
			case VaArg: return "va_arg"; break;
			case Xor: return "xor"; break;
			case Zext: return "zext"; break;
			case InvalidOpcode: return "INVALID_OPCODE"; break;
		}
		return "";
	}

	std::string LLVMInstruction::toString( DataType d )
	{
		switch( d )
		{
			case I1:   return "i1"; break;
			case I8:   return "i8"; break;
			case I16:  return "i16"; break;
			case I32:  return "i32"; break;
			case I64:  return "i64"; break;
			case I128: return "i128"; break;
			case F32:  return "float"; break;
			case F64:  return "double"; break;
			case F128: return "fp128"; break;
			default: break;
		}
		return "";
	}

	std::string LLVMInstruction::toString( CallingConvention cc )
	{
		switch( cc )
		{
			case CCallingConvention:       return "ccc"; break;
			case FastCallingConvention:    return "fastcc"; break;
			case ColdCallingConvention:    return "coldcc"; break;
			case DefaultCallingConvention: return ""; break;
			case InvalidCallingConvention: return "INVALID_CC"; break;
		}
		return "";
	}
	
	std::string LLVMInstruction::toString( ParameterAttribute attribute )
	{
		switch( attribute )
		{
			case ZeroExtend:                return "zeroext"; break;
			case SignExtend:                return "signext"; break;
			case InRegister:                return "inreg"; break;
			case ByValue:                   return "byval"; break;
			case StructureReturn:           return "sret"; break;
			case NoAlias:                   return "noalias"; break;
			case NoCapture:                 return "nocapture"; break;
			case Nested:                    return "nest"; break;
			case InvalidParameterAttribute: return ""; break;
		}
		return "";
	}
	
	std::string LLVMInstruction::toString( AtomicOperation operation )
	{
		switch( operation )
		{
			case AtomicXchg:             return "xchg"; break;
			case AtomicAdd:              return "add"; break;
			case AtomicSub:              return "sub"; break;
			case AtomicAnd:              return "and"; break;
			case AtomicNand:             return "nand"; break;
			case AtomicOr:               return "or"; break;
			case AtomicXor:              return "xor"; break;
			case AtomicMax:              return "max"; break;
			case AtomicMin:              return "min"; break;
			case AtomicUmax:             return "umax"; break;
			case AtomicUmin:             return "umin"; break;
			case InvalidAtomicOperation: return ""; break;
		}
		return "";
	}

	std::string LLVMInstruction::toString( Comparison comp )
	{
		switch( comp )
		{
			case False: return "false"; break;
			case Oeq:   return "oeq"; break;
			case Ogt:   return "ogt"; break;
			case Oge:   return "oge"; break;
			case Olt:   return "olt"; break;
			case Ole:   return "ole"; break;
			case One:   return "one"; break;
			case Ord:   return "ord"; break;
			case Ueq:   return "ueq"; break;
			case Ugt:   return "ugt"; break;
			case Uge:   return "uge"; break;
			case Ult:   return "ult"; break;
			case Ule:   return "ule"; break;
			case Une:   return "une"; break;
			case Uno:   return "uno"; break;
			case True:  return "true"; break;
			case Eq:    return "eq"; break;
			case Ne:    return "ne"; break;
			case Sgt:   return "sgt"; break;
			case Sge:   return "sge"; break;
			case Slt:   return "slt"; break;
			case Sle:   return "sle"; break;
		}
		return "";
	}
	
	std::string LLVMInstruction::functionAttributesToString( int attributes )
	{
		std::string result;
		if( attributes & AlwaysInline )
		{
			result += "alwaysinline";
		}
		if( attributes & NoInline )
		{
			if( !result.empty() ) result += " ";
			result += "noinline";
		}
		if( attributes & OptimizeSize )
		{
			if( !result.empty() ) result += " ";
			result += "optsize";
		}
		if( attributes & NoReturn )
		{
			if( !result.empty() ) result += " ";
			result += "noreturn";
		}
		if( attributes & NoUnwind )
		{
			if( !result.empty() ) result += " ";
			result += "nounwind";
		}
		if( attributes & ReadNone )
		{
			if( !result.empty() ) result += " ";
			result += "readnone";
		}
		if( attributes & ReadOnly )
		{
			if( !result.empty() ) result += " ";
			result += "readonly";
		}
		if( attributes & StackSmashingProtector )
		{
			if( !result.empty() ) result += " ";
			result += "ssp";
		}
		if( attributes & StackSmashingProtectorRequired )
		{
			if( !result.empty() ) result += " ";
			result += "sspreq";
		}
		if( attributes & NoRedZone )
		{
			if( !result.empty() ) result += " ";
			result += "noredzone";
		}
		if( attributes & NoImplicitFloat )
		{
			if( !result.empty() ) result += " ";
			result += "noimplicitfloat";
		}
		if( attributes & Naked )
		{
			if( !result.empty() ) result += " ";
			result += "naked";
		}
		return result;
	}

	bool LLVMInstruction::isInt( DataType d )
	{
		switch( d )
		{
			case I1: /* fall through */
			case I8: /* fall through */
			case I16: /* fall through */
			case I32: /* fall through */
			case I64: /* fall through */
			case I128: return true; break;
			case F32: /* fall through */
			case F64: /* fall through */
			case F128: /* fall through */
			default: break;
		}
		return false;
	}
	
	LLVMInstruction::DataType LLVMInstruction::getIntOfSize( unsigned int bits )
	{
		switch( bits )
		{
			case 1:   return I1;   break;
			case 8:   return I8;   break;
			case 16:  return I16;  break;
			case 32:  return I32;  break;
			case 64:  return I64;  break;
			case 128: return I128; break;
			default:               break;
		}
		return InvalidDataType;
	}
	
	unsigned int LLVMInstruction::bits( DataType type )
	{
		switch( type )
		{
			case I1:   return 1;   break;
			case I8:   return 8;   break;
			case I16:  return 16;  break;
			case F32:  /* fall through */
			case I32:  return 32;  break;
			case F64:  /* fall through */
			case I64:  return 64;  break;
			case I128: /* fall through */
			case F128: return 128; break;
			default:               break;
		}
		return 0;
	}

	LLVMInstruction::LLVMInstruction( Opcode op ) : opcode( op )
	{
		
	}

	LLVMInstruction::LLVMInstruction( const LLVMInstruction& i ) 
		: opcode( i.opcode )
	{
	
	}
	
	const LLVMInstruction& LLVMInstruction::operator=( 
		const LLVMInstruction& i )
	{
		/* intentionally empty */
		assert( i.opcode == opcode );
		return *this;
	}

	LLVMUnaryInstruction::LLVMUnaryInstruction( Opcode op, 
		const Operand& _d, const Operand& _a ) : LLVMInstruction( op ), 
		d( _d ), a( _a )
	{
	
	}
	
	std::string LLVMUnaryInstruction::toString() const
	{
		return d.toString() + " = " + LLVMInstruction::toString( opcode ) + " " 
			+ a.type.toString() + " " + a.toString();
	}
	
	std::string LLVMUnaryInstruction::valid() const
	{
		if( a.type.toString() != d.type.toString() )
		{
			return "Source operand type " + a.type.toString() 
				+ " does not equal destination operand type " 
				+ d.type.toString();
		}
		return "";
	}

	LLVMBinaryInstruction::LLVMBinaryInstruction( Opcode op, 
		const Operand& _d, const Operand& _a, const Operand& _b )
		: LLVMInstruction( op ), d( _d ), a( _a ), b( _b )
	{
	
	}

	std::string LLVMBinaryInstruction::toString() const
	{
		return d.toString() + " = " + LLVMInstruction::toString( opcode ) + " " 
			+ a.type.toString() + " " + a.toString() + ", " + b.toString();
	}

	std::string LLVMBinaryInstruction::valid() const
	{
		if( a.type.toString() != d.type.toString() )
		{
			return "First source operand type " + a.type.toString() 
				+ " does not equal destination operand type " 
				+ d.type.toString();
		}
		if( b.type.toString() != d.type.toString() )
		{
			return "Second source operand type " + b.type.toString() 
				+ " does not equal destination operand type " 
				+ d.type.toString();
		}
		return "";
	}
	
	LLVMConversionInstruction::LLVMConversionInstruction( Opcode op, 
		const Operand& d, const Operand& a ) : LLVMUnaryInstruction( op, d, a )
	{
		
	}

	std::string LLVMConversionInstruction::toString() const
	{
		return d.toString() + " = " + LLVMInstruction::toString( opcode ) + " " 
			+ a.type.toString() + " " + a.toString() + " to " 
			+ d.type.toString();
	}

	std::string LLVMConversionInstruction::valid() const
	{
		return "";
	}

	LLVMComparisonInstruction::LLVMComparisonInstruction( Opcode op )
		: LLVMBinaryInstruction( op ), comparison( False )
	{
	
	}
	
	std::string LLVMComparisonInstruction::toString() const
	{
		return d.toString() + " = " + LLVMInstruction::toString( opcode ) + " " 
			+ LLVMInstruction::toString( comparison ) + " " 
			+ a.type.toString() + " " + a.toString() + ", " + b.toString();
	}
	
	std::string LLVMComparisonInstruction::valid() const
	{
		if( a.type.toString() != b.type.toString() )
		{
			return "First source operand type " + a.type.toString() 
				+ " does not equal second source operand type " 
				+ d.type.toString();
		}
		return "";	
	}

	LLVMAdd::LLVMAdd( const Operand& d, const Operand& a, const Operand& b, 
		LLVMI1 nuw, LLVMI1 nsw ) : LLVMBinaryInstruction( Add, d, a, b ), 
		noUnsignedWrap( nuw ), noSignedWrap( nsw )
	{
	
	}
	
	std::string LLVMAdd::toString() const
	{
		std::string result = d.toString() + " = " 
			+ LLVMInstruction::toString( opcode ) + " ";
		if( noUnsignedWrap ) result += "nuw ";
		if( noSignedWrap ) result += "nsw ";
		result += a.type.toString() + " " + a.toString() + ", " + b.toString();		
		return result;
	}
	
	std::string LLVMAdd::valid() const
	{
		return LLVMBinaryInstruction::valid();
	}

	Instruction* LLVMAdd::clone(bool copy) const
	{
		return new LLVMAdd( *this );
	}

	LLVMAlloca::LLVMAlloca( LLVMI32 e, LLVMI32 a ) : LLVMInstruction( Alloca ), 
		elements( e ), alignment( a )
	{
		
	}
	
	std::string LLVMAlloca::toString() const
	{
		std::stringstream stream;
		stream << d.toString() << " = " 
			<< LLVMInstruction::toString( opcode ) << " " 
			<< LLVMInstruction::toString( d.type.type );
		if( elements != 1 ) stream << ", i32 " << elements;
		if( alignment != 1 ) stream << ", align " << alignment;
		return stream.str();		
	}
	
	std::string LLVMAlloca::valid() const
	{
		if( alignment == 0 )
		{
			return "Alignment cannot be 0";
		}
		if( d.type.category != Type::Pointer )
		{
			return "Destination must be a pointer to a type";
		}
		return "";
	}

	Instruction* LLVMAlloca::clone(bool copy) const
	{
		return new LLVMAlloca( *this );
	}

	LLVMAnd::LLVMAnd() : LLVMBinaryInstruction( And )
	{
	
	}

	Instruction* LLVMAnd::clone(bool copy) const
	{
		return new LLVMAnd( *this );
	}
	
	LLVMAshr::LLVMAshr() : LLVMBinaryInstruction( Ashr )
	{
	
	}
	
	std::string LLVMAshr::valid() const
	{
		if( a.type.toString() != d.type.toString() )
		{
			return "First source operand type " + a.type.toString() 
				+ " does not equal destination operand type " 
				+ d.type.toString();
		}
		if( !isInt( b.type.type ) )
		{
			return "Second source operand type " + b.type.toString() 
				+ " is not an integer type";
		}
		return "";
	}

	Instruction* LLVMAshr::clone(bool copy) const
	{
		return new LLVMAshr( *this );
	}
	
	LLVMAtomicrmw::LLVMAtomicrmw() : LLVMBinaryInstruction( Atomicrmw )
	{
	
	}
	
	std::string LLVMAtomicrmw::valid() const
	{
		if( a.type.category != Type::Pointer )
		{
			return "Address must be a pointer";
		}
		if( a.type.toString() != ( d.type.toString() + "*" ) )
		{
			return "Source " + a.type.toString() 
				+ " is not a pointer to destination type " + d.type.toString();
		}
		
		return "";
	}
	
	std::string LLVMAtomicrmw::toString() const
	{
		return d.toString() + " = " + LLVMInstruction::toString( opcode ) + " " 
			+ LLVMInstruction::toString( operation ) + " " +
			a.type.toString() + " " + a.toString() + ", " + b.type.toString() +
			" " + b.toString() + " monotonic";
	}

	Instruction* LLVMAtomicrmw::clone(bool copy) const
	{
		return new LLVMAtomicrmw( *this );
	}

	LLVMBitcast::LLVMBitcast() : LLVMConversionInstruction( Bitcast )
	{
	
	}

	Instruction* LLVMBitcast::clone(bool copy) const
	{
		return new LLVMBitcast( *this );
	}

	LLVMBr::LLVMBr() : LLVMInstruction( Br )
	{
	
	}
	
	std::string LLVMBr::toString() const
	{
		std::string result = LLVMInstruction::toString( opcode ) + " ";
		if( condition.valid() ) result += condition.type.toString() + " " 
			+ condition.toString() + ", ";
		result += "label " + iftrue;
		if( condition.valid() ) result += ", label " + iffalse;
		return result;		
	}
	
	std::string LLVMBr::valid() const
	{
		if( iftrue.empty() ) return "Target label must not be empty";
		
		if( condition.valid() )
		{
			if( iffalse.empty() ) 
			{
				return "For conditional branch, if-false \
					label must not be empty";
			}
			if( condition.type.type != I1 )
			{
				return "Condition type " 
					+ LLVMInstruction::toString( condition.type.type ) 
					+ " not valid for Br, only i1 is valid.";
			}
		}
		return "";
	}

	Instruction* LLVMBr::clone(bool copy) const
	{
		return new LLVMBr( *this );
	}

	LLVMCall::LLVMCall() : LLVMInstruction( Call ), tail( false ), 
		convention( DefaultCallingConvention ),
		functionAttributes( 0 )
	{
	
	}
	
	std::string LLVMCall::toString() const
	{
		std::string result;
		if( d.valid() ) result += d.name + " = ";
		if( tail ) result += "tail ";
		result += LLVMInstruction::toString( opcode ) + " ";
		std::string cc = LLVMInstruction::toString( convention );
		if( !cc.empty() ) result += cc + " ";
		std::string retats = LLVMInstruction::toString( d.attribute );
		if( !retats.empty() ) result += retats + " ";
		if( d.valid() )
		{
			result += d.type.toString() + " ";
		}
		else if( signature.empty() )
		{
			result += "void ";
		}
		if( !signature.empty() ) result += signature + " ";
		result += name + "(";
		for( ParameterVector::const_iterator fi = parameters.begin(); 
			fi != parameters.end(); ++fi )
		{
			if( fi != parameters.begin() ) result += ", ";
			result += fi->type.toString() + " " + fi->toString();
			std::string atts = LLVMInstruction::toString( fi->attribute );
			if( !atts.empty() ) result += " " + atts;
		}
		result += ")";
		std::string funats = LLVMInstruction::functionAttributesToString( 
			functionAttributes );
		if( !funats.empty() ) result += " " + funats;
		return result;
	}

	std::string LLVMCall::valid() const
	{
		if( name.find("@") == std::string::npos )
		{
			return "Function names must begin with global identifier '@'";
		}
		switch( d.attribute )
		{
			case ZeroExtend: /* fall through */
			case SignExtend: /* fall through */
			case InRegister: /* fall through */
			case InvalidParameterAttribute: break;
			case ByValue: /* fall through */
			case StructureReturn: /* fall through */
			case NoAlias: /* fall through */
			case NoCapture: /* fall through */
			case Nested: return LLVMInstruction::toString( d.attribute ) 
				+ " not allowed in call";
		}
		
		for( ParameterVector::const_iterator fi = parameters.begin(); 
			fi != parameters.end(); ++fi )
		{
			switch( fi->attribute )
			{
				case ZeroExtend: /* fall through */
				case SignExtend: /* fall through */
				case InRegister: /* fall through */
				case InvalidParameterAttribute: break;
				case ByValue: /* fall through */
				case StructureReturn: /* fall through */
				case NoAlias: /* fall through */
				case NoCapture: /* fall through */
				case Nested: return LLVMInstruction::toString( fi->attribute ) 
					+ " not allowed in call";
			}
		}
		
		// Only 'noreturn', 'nounwind', 'readonly' and 'readnone' attributes 
		// are valid here.
		if( functionAttributes & AlwaysInline )
		{
			return "alwaysinline not allowed in call";
		}
		if( functionAttributes & NoInline )
		{
			return "noinline not allowed in call";
		}
		if( functionAttributes & OptimizeSize  )
		{
			return "optsize not allowed in call";
		}
		if( functionAttributes & StackSmashingProtector )
		{
			return "ssp not allowed in call";
		}
		if( functionAttributes & StackSmashingProtectorRequired )
		{
			return "sspreq not allowed in call";
		}
		if( functionAttributes & NoRedZone )
		{
			return "noredzone not allowed in call";
		}
		if( functionAttributes & NoImplicitFloat )
		{
			return "noimplicitfloat not allowed in call";
		}
		return "";
	}

	Instruction* LLVMCall::clone(bool copy) const
	{
		return new LLVMCall( *this );
	}
	
	LLVMCmpxchg::LLVMCmpxchg() : LLVMBinaryInstruction( Cmpxchg )
	{
	
	}
	
	std::string LLVMCmpxchg::valid() const
	{
		if( a.type.category != Type::Pointer )
		{
			return "Address must be a pointer";
		}
		if( a.type.toString() != ( d.type.toString() + "*" ) )
		{
			return "Source " + a.type.toString() 
				+ " is not a pointer to destination type " + d.type.toString();
		}
		
		if( b.type.toString() != c.type.toString() )
		{
			return "Source " + a.type.toString() 
				+ " is not a pointer to destination type " + d.type.toString();
		}
		
		return "";
	}
	
	std::string LLVMCmpxchg::toString() const
	{
		return d.toString() + " = " + LLVMInstruction::toString( opcode ) +
			" " + a.type.toString() + " " + a.toString() + ", "  + b.type.toString() +
			" " + b.toString() + ", "  + b.type.toString() + " " +
			b.toString() + " monotonic";
	}

	Instruction* LLVMCmpxchg::clone(bool copy) const
	{
		return new LLVMCmpxchg( *this );
	}

	LLVMExtractelement::LLVMExtractelement() 
		: LLVMBinaryInstruction( Extractelement )
	{
	
	}
	
	std::string LLVMExtractelement::toString() const
	{
		return d.toString() + " = " + LLVMInstruction::toString( opcode ) + " " 
			+ a.type.toString() + " " + a.toString() + ", " + b.type.toString() 
			+ " " + b.toString();
	}
	
	std::string LLVMExtractelement::valid() const
	{
		if( a.type.category != Type::Vector )
		{
			return "First operand must be a vector";
		}
		if( b.type.type != I32 )
		{
			return "Index must be i32 type";
		}
		if( a.type.type != d.type.type )
		{
			return "Destination primitive datatype " 
				+ LLVMInstruction::toString( d.type.type ) 
				+ " does not match source datatype " 
				+ LLVMInstruction::toString( a.type.type );
		}
		return "";
	}

	Instruction* LLVMExtractelement::clone(bool copy) const
	{
		return new LLVMExtractelement( *this );
	}

	LLVMExtractvalue::LLVMExtractvalue() : LLVMInstruction( Extractvalue )
	{
		
	}
	
	std::string LLVMExtractvalue::toString() const
	{
		std::stringstream stream;
		stream << d.toString() << " = " 
			<< LLVMInstruction::toString( opcode ) << " " << a.type.toString() 
			<< " " << a.toString();
		for( IndexVector::const_iterator fi = indices.begin(); 
			fi != indices.end(); ++fi )
		{
			stream << ", " << *fi; 
		}
		return stream.str();
	}
	
	std::string LLVMExtractvalue::valid() const
	{
		if( a.type.category != Type::Structure )
		{
			return "Source operand must be a structure";
		}
		const Type* current = &a.type;
		for( IndexVector::const_iterator fi = indices.begin(); 
			fi != indices.end(); ++fi )
		{
			if( (unsigned int) *fi >= current->members.size() )
			{
				std::stringstream stream;
				LLVMI32 index = std::distance( indices.begin(), fi );
				stream << "At index " << index << " specified element " << *fi 
					<< " in a type with only " << current->members.size() 
					<< " elements.";
				return stream.str();
			}
			
			current = &current->members[ *fi ];
		}
		
		if( current->toString() != d.type.toString() )
		{
			return "Extracted type " + current->toString() 
				+ " does not match destination type " + d.type.toString();
		}
		
		// TODO: add a check that the type at the index matches the destination
		return "";
	}

	Instruction* LLVMExtractvalue::clone(bool copy) const
	{
		return new LLVMExtractvalue( *this );
	}

	LLVMFadd::LLVMFadd() : LLVMBinaryInstruction( Fadd )
	{
		
	}

	Instruction* LLVMFadd::clone(bool copy) const
	{
		return new LLVMFadd( *this );
	}
	
	LLVMFcmp::LLVMFcmp() : LLVMComparisonInstruction( Fcmp )
	{
	
	}

	Instruction* LLVMFcmp::clone(bool copy) const
	{
		return new LLVMFcmp( *this );
	}
	
	LLVMFdiv::LLVMFdiv() : LLVMBinaryInstruction( Fdiv )
	{
	
	}

	Instruction* LLVMFdiv::clone(bool copy) const
	{
		return new LLVMFdiv( *this );
	}
	
	LLVMFmul::LLVMFmul() : LLVMBinaryInstruction( Fmul )
	{
	
	}

	Instruction* LLVMFmul::clone(bool copy) const
	{
		return new LLVMFmul( *this );
	}
	
	LLVMFpext::LLVMFpext() : LLVMConversionInstruction( Fpext )
	{
	
	}

	Instruction* LLVMFpext::clone(bool copy) const
	{
		return new LLVMFpext( *this );
	}

	LLVMFptosi::LLVMFptosi() : LLVMConversionInstruction( Fptosi )
	{
	
	}

	Instruction* LLVMFptosi::clone(bool copy) const
	{
		return new LLVMFptosi( *this );
	}

	LLVMFptoui::LLVMFptoui() : LLVMConversionInstruction( Fptoui )
	{
	
	}

	Instruction* LLVMFptoui::clone(bool copy) const
	{
		return new LLVMFptoui( *this );
	}

	LLVMFptrunc::LLVMFptrunc() : LLVMConversionInstruction( Fptrunc )
	{
	
	}

	Instruction* LLVMFptrunc::clone(bool copy) const
	{
		return new LLVMFptrunc( *this );
	}
	
	LLVMFree::LLVMFree() : LLVMInstruction( Free )
	{
	
	}
	
	std::string LLVMFree::toString() const
	{
		return LLVMInstruction::toString( opcode ) + " " + a.type.toString() 
			+ " " + a.toString();
	}
	
	std::string LLVMFree::valid() const
	{
		if( a.type.category != Type::Pointer )
		{
			return "Cannot free non-pointer type " + a.type.toString();
		}
		return "";
	}

	Instruction* LLVMFree::clone(bool copy) const
	{
		return new LLVMFree( *this );
	}
	
	LLVMFrem::LLVMFrem() : LLVMBinaryInstruction( Frem )
	{
	
	}

	Instruction* LLVMFrem::clone(bool copy) const
	{
		return new LLVMFrem( *this );
	}

	LLVMFsub::LLVMFsub() : LLVMBinaryInstruction( Fsub )
	{
	
	}

	Instruction* LLVMFsub::clone(bool copy) const
	{
		return new LLVMFsub( *this );
	}

	LLVMGetelementptr::LLVMGetelementptr() : LLVMInstruction( Getelementptr )
	{
		
	}
	
	std::string LLVMGetelementptr::toString() const
	{
		std::stringstream stream;
		stream << d.toString() << " = " 
			<< LLVMInstruction::toString( opcode ) << " " 
			<< a.type.toString() << " " << a.toString();
		for( IndexVector::const_iterator fi = indices.begin(); 
			fi != indices.end(); ++fi )
		{
			stream << ", i32 " << *fi; 
		}
		return stream.str();		
	}
	
	std::string LLVMGetelementptr::valid() const
	{
		if( a.type.category != Type::Structure 
			&& a.type.category != Type::Pointer
			&& a.type.category != Type::Vector
			&& a.type.category != Type::Array )
		{
			return "Source operand must be a structure/pointer/vector/array";
		}
		// TODO: add a check that the type at the index matches the destination
		return "";
	}

	Instruction* LLVMGetelementptr::clone(bool copy) const
	{
		return new LLVMGetelementptr( *this );
	}

	LLVMIcmp::LLVMIcmp() : LLVMComparisonInstruction( Icmp )
	{
	
	}

	Instruction* LLVMIcmp::clone(bool copy) const
	{
		return new LLVMIcmp( *this );
	}

	LLVMInsertelement::LLVMInsertelement() 
		: LLVMBinaryInstruction( Insertelement )
	{
	
	}
	
	std::string LLVMInsertelement::toString() const
	{
		return d.toString() + " = " + LLVMInstruction::toString( opcode ) + " " 
			+ a.type.toString() + " " + a.toString() + ", " + b.type.toString() 
			+ " " + b.toString() + ", " + c.type.toString() + " " 
			+ c.toString();
	}
	
	std::string LLVMInsertelement::valid() const
	{
		if( d.type.type != b.type.type )
		{
			return "Destination base type " 
				+ LLVMInstruction::toString( d.type.type ) 
				+ " does not equal source element type " 
				+ LLVMInstruction::toString( b.type.type );
		}
		if( d.type.category != Type::Vector )
		{
			return "Destination is not a vector";
		}
		if( a.type.category != Type::Vector )
		{
			return "Source is not a vector";
		}
		if( a.type.toString() != d.type.toString() )
		{
			return "Destination type " + d.type.toString() 
				+ " does not match source type " + a.type.toString();
		}
		if( c.type.toString() != "i32" )
		{
			return "Index type " + c.type.toString() + " is not i32";
		}
		return "";
	}

	Instruction* LLVMInsertelement::clone(bool copy) const
	{
		return new LLVMInsertelement( *this );
	}

	LLVMInsertvalue::LLVMInsertvalue() : LLVMInstruction( Insertvalue )
	{
	
	}
	
	std::string LLVMInsertvalue::toString() const
	{
		std::string result = d.toString() + " = " 
			+ LLVMInstruction::toString( opcode ) + " " + a.type.toString() 
			+ " " + a.toString() + ", " + b.type.toString() + " " 
			+ b.toString();
		std::stringstream stream;
		for( IndexVector::const_iterator fi = indices.begin(); 
			fi != indices.end(); ++fi )
		{
			stream << ", " << *fi;
		}
		return result + stream.str();
	}
	
	std::string LLVMInsertvalue::valid() const
	{
		if( d.type.category != Type::Structure )
		{
			return "Destination is not a structure";
		}
		if( a.type.category != Type::Structure )
		{
			return "Source is not a structure";
		}
		if( d.type.toString() != a.type.toString() )
		{
			return "Source type " + a.type.toString() 
				+ " does not match destination type " + d.type.toString();
		}
		// TODO: Add a check to make sure that the inserted type matches 
		//	the desintation's type
		return "";
	}

	Instruction* LLVMInsertvalue::clone(bool copy) const
	{
		return new LLVMInsertvalue( *this );
	}

	LLVMInttoptr::LLVMInttoptr() : LLVMConversionInstruction( Inttoptr )
	{
	
	}

	Instruction* LLVMInttoptr::clone(bool copy) const
	{
		return new LLVMInttoptr( *this );
	}

	LLVMInvoke::LLVMInvoke() : LLVMInstruction( Invoke ), 
		convention( DefaultCallingConvention ), 
		returnAttributes( InvalidParameterAttribute ), functionAttributes( 0 )
	{
	
	}
	
	std::string LLVMInvoke::toString() const
	{
		std::string result;
		if( d.valid() ) result += d.name + " = ";
		result += LLVMInstruction::toString( opcode ) + " ";
		std::string cc = LLVMInstruction::toString( convention );
		if( !cc.empty() ) result += cc + " ";
		std::string retats = LLVMInstruction::toString( d.attribute );
		if( !retats.empty() ) result += retats + " ";
		if( d.valid() )
		{
			result += d.type.toString() + " ";
		}
		else if( signature.empty() )
		{
			result += "void ";
		}
		if( !signature.empty() ) result += signature + " ";
		result += name + "(";
		for( ParameterVector::const_iterator fi = parameters.begin(); 
			fi != parameters.end(); ++fi )
		{
			if( fi != parameters.begin() ) result += ", ";
			result += fi->type.toString() + " " + fi->toString();
			std::string atts = LLVMInstruction::toString( fi->attribute );
			if( !atts.empty() ) result += " " + atts;
		}
		result += ")";
		std::string funats 
			= LLVMInstruction::functionAttributesToString( functionAttributes );
		if( !funats.empty() ) result += " " + funats;
		result += " to label " + tolabel + " unwind label " + unwindlabel;
		return result;
	}
	
	std::string LLVMInvoke::valid() const
	{
		switch( returnAttributes )
		{
			case ZeroExtend: /* fall through */
			case SignExtend: /* fall through */
			case InRegister: /* fall through */
			case InvalidParameterAttribute: break;
			case ByValue: /* fall through */
			case StructureReturn: /* fall through */
			case NoAlias: /* fall through */
			case NoCapture: /* fall through */
			case Nested: return LLVMInstruction::toString( returnAttributes ) 
				+ " not allowed in invoke";
		}
		
		// Only 'noreturn', 'nounwind', 'readonly' and 'readnone' attributes 
		// are valid here.
		if( functionAttributes & AlwaysInline )
		{
			return "alwaysinline not allowed in invoke";
		}
		if( functionAttributes & NoInline )
		{
			return "noinline not allowed in invoke";
		}
		if( functionAttributes & OptimizeSize  )
		{
			return "optsize not allowed in invoke";
		}
		if( functionAttributes & StackSmashingProtector )
		{
			return "ssp not allowed in invoke";
		}
		if( functionAttributes & StackSmashingProtectorRequired )
		{
			return "sspreq not allowed in invoke";
		}
		if( functionAttributes & NoRedZone )
		{
			return "noredzone not allowed in invoke";
		}
		if( functionAttributes & NoImplicitFloat )
		{
			return "noimplicitfloat not allowed in invoke";
		}
		
		if( tolabel.empty() ) return "invoke requires a to label";
		if( unwindlabel.empty() ) return "invoke requires an unwind label";
		
		return "";
	}

	Instruction* LLVMInvoke::clone(bool copy) const
	{
		return new LLVMInvoke( *this );
	}

	LLVMLoad::LLVMLoad() : LLVMUnaryInstruction( Load ), isVolatile( false ), 
		alignment( 1 )
	{
	
	}
	
	std::string LLVMLoad::toString() const
	{
		std::stringstream stream;
		stream << d.toString() << " = ";
		stream << LLVMInstruction::toString( opcode ) << " " 	
			<< (isVolatile ? "volatile " : "")
			<< a.type.toString() << " " << a.toString();
		if( alignment != 1 ) stream << ", align " << alignment;
		return stream.str();
	}
	
	std::string LLVMLoad::valid() const
	{
		if( alignment == 0 ) return "Alignment must be greater than 0";
		if( a.type.category != Type::Pointer )
		{
			return "Address must be a pointer";
		}
		if( a.type.toString() != ( d.type.toString() + "*" ) )
		{
			return "Source " + a.type.toString() 
				+ " is not a pointer to destination type " + d.type.toString();
		}
		return "";
	}

	Instruction* LLVMLoad::clone(bool copy) const
	{
		return new LLVMLoad( *this );
	}

	LLVMLshr::LLVMLshr() : LLVMBinaryInstruction( Lshr )
	{
	
	}

	std::string LLVMLshr::valid() const
	{
		if( a.type.toString() != d.type.toString() )
		{
			return "First source operand type " + a.type.toString() 
				+ " does not equal destination operand type " 
				+ d.type.toString();
		}
		if( !isInt( b.type.type ) )
		{
			return "Second source operand type " + b.type.toString() 
				+ " is not an integer type";
		}
		return "";
	}

	Instruction* LLVMLshr::clone(bool copy) const
	{
		return new LLVMLshr( *this );
	}
	
	LLVMMalloc::LLVMMalloc() : LLVMInstruction( Malloc ), alignment( 1 )
	{
		
	}

	std::string LLVMMalloc::toString() const
	{
		std::stringstream stream;
		stream << d.toString() << " = " 
			<< LLVMInstruction::toString( opcode ) << " " << d.type.toString();
		if( !( elements.constant && elements.i8 == 1 ) )
		{
			stream << ", " << elements.type.toString() << " " 
				<< elements.toString();
		}
		if( alignment != 1 ) stream << ", align " << alignment;
		return stream.str();
	}
	
	std::string LLVMMalloc::valid() const
	{
		if( alignment == 0 )
		{
			return "Alignment cannot be 0";
		}
		return "";	
	}

	Instruction* LLVMMalloc::clone(bool copy) const
	{
		return new LLVMMalloc( *this );
	}

	LLVMMul::LLVMMul() : LLVMBinaryInstruction( Mul ), noUnsignedWrap( false ),
		noSignedWrap( false )
	{
	
	}
	
	std::string LLVMMul::toString() const
	{
		std::string result = d.toString() + " = " 
			+ LLVMInstruction::toString( opcode ) + " ";
		if( noUnsignedWrap ) result += "nuw ";
		if( noSignedWrap ) result += "nsw ";
		result += a.type.toString() + " " + a.toString() + ", " + b.toString();		
		return result;	
	}

	Instruction* LLVMMul::clone(bool copy) const
	{
		return new LLVMMul( *this );
	}
	
	LLVMOr::LLVMOr() : LLVMBinaryInstruction( Or )
	{
	
	}

	Instruction* LLVMOr::clone(bool copy) const
	{
		return new LLVMOr( *this );
	}
	
	LLVMPhi::LLVMPhi() : LLVMInstruction( Phi )
	{
	
	} 
	
	std::string LLVMPhi::toString() const
	{
		std::string result = d.toString() + " = " 
			+ LLVMInstruction::toString( opcode ) + " " 
			+ d.type.toString() + " ";
		for( NodeVector::const_iterator ni = nodes.begin(); 
			ni != nodes.end(); ++ni )
		{
			if( ni != nodes.begin() )
			{
				result += ", ";
			}
			result += "[ " + ni->operand.toString() + ", " + ni->label + " ]";
		}
		return result;
	}
	
	std::string LLVMPhi::valid() const
	{
		for( NodeVector::const_iterator ni = nodes.begin(); 
			ni != nodes.end(); ++ni )
		{
			if( ni->operand.type.toString() != d.type.toString() )
			{
				return "Phi node from block " + ni->label + " type " 
					+ ni->operand.type.toString() 
					+ " does not match destination type " + d.type.toString();
			}
		}
		if( nodes.empty() ) return "No phi nodes";
		return "";
	}

	Instruction* LLVMPhi::clone(bool copy) const
	{
		return new LLVMPhi( *this );
	}

	LLVMPtrtoint::LLVMPtrtoint() : LLVMConversionInstruction( Ptrtoint )
	{
	
	}

	Instruction* LLVMPtrtoint::clone(bool copy) const
	{
		return new LLVMPtrtoint( *this );
	}

	LLVMRet::LLVMRet() : LLVMInstruction( Ret )
	{
	
	}
	
	std::string LLVMRet::toString() const
	{
		if( d.valid() )
		{
			return LLVMInstruction::toString( opcode ) + " " 
				+ d.type.toString() + " " + d.toString();
		}
		else
		{
			return LLVMInstruction::toString( opcode ) + " void";
		}
	}
	
	std::string LLVMRet::valid() const
	{
		return "";
	}

	Instruction* LLVMRet::clone(bool copy) const
	{
		return new LLVMRet( *this );
	}

	LLVMSdiv::LLVMSdiv() : LLVMBinaryInstruction( Sdiv ), exact( false )
	{
	
	}
	
	std::string LLVMSdiv::toString() const
	{
		std::string result = d.toString() + " = " 
			+ LLVMInstruction::toString( opcode ) + " ";
		if( exact ) result += "exact ";
		result += d.type.toString() + " " + a.toString() + ", " + b.toString();
		return result;
	}

	Instruction* LLVMSdiv::clone(bool copy) const
	{
		return new LLVMSdiv( *this );
	}

	LLVMSelect::LLVMSelect() : LLVMBinaryInstruction( Select )
	{
	
	}
	
	std::string LLVMSelect::toString() const
	{
		std::string result = d.toString() + " = " 
			+ LLVMInstruction::toString( opcode ) + " " 
			+ condition.type.toString() + " " + condition.toString() + ", " 
			+ a.type.toString() + " " + a.toString() + ", " + b.type.toString() 
			+ " " + b.toString();
		return result;
	}
	
	std::string LLVMSelect::valid() const
	{
		if( condition.type.type != I1 ) return "Condition base type must be i1";
		return LLVMBinaryInstruction::valid();
	}

	Instruction* LLVMSelect::clone(bool copy) const
	{
		return new LLVMSelect( *this );
	}

	LLVMSext::LLVMSext() : LLVMConversionInstruction( Sext )
	{
	
	}

	Instruction* LLVMSext::clone(bool copy) const
	{
		return new LLVMSext( *this );
	}

	LLVMShl::LLVMShl() : LLVMBinaryInstruction( Shl )
	{
	
	}

	std::string LLVMShl::valid() const
	{
		if( a.type.toString() != d.type.toString() )
		{
			return "First source operand type " + a.type.toString() 
				+ " does not equal destination operand type " 
				+ d.type.toString();
		}
		if( !isInt( b.type.type ) )
		{
			return "Second source operand type " + b.type.toString() 
				+ " is not an integer type";
		}
		return "";
	}

	Instruction* LLVMShl::clone(bool copy) const
	{
		return new LLVMShl( *this );
	}

	LLVMShufflevector::LLVMShufflevector() 
		: LLVMBinaryInstruction( Shufflevector )
	{
	
	}
	
	std::string LLVMShufflevector::toString() const
	{
		std::stringstream stream;
		stream << d.toString() << " = " << LLVMInstruction::toString( opcode ) 
			<< " " << a.type.toString() << " " << a.toString() << ", " 
			<< b.type.toString() << " " << b.toString() << ", < " 
			<< mask.size() << " x i32 > < ";
		for( Mask::const_iterator fi = mask.begin(); fi != mask.end(); ++fi )
		{
			if( fi != mask.begin() ) stream << ", ";
			stream << "i32 " << *fi;
		}
		stream << " >";
		return stream.str();
	}
	
	std::string LLVMShufflevector::valid() const
	{
		if( d.type.category != Type::Vector )
		{
			return "Destination must be a vector";
		}
		if( a.type.category != Type::Vector && a.valid() )
		{
			return "Source A must be a vector or undef";
		}
		if( b.type.category != Type::Vector && b.valid() )
		{
			return "Source B must be a vector or undef";
		}
		if( mask.size() != ( unsigned int ) 2 * d.type.vector )
		{
			std::stringstream stream;
			stream << "Mask size " << mask.size() 
				<< " is not twice of vector size " << d.type.vector;
			return stream.str();
		}
		for( Mask::const_iterator fi = mask.begin(); fi != mask.end(); ++fi )
		{
			if( *fi >= 2 * d.type.vector )
			{
				std::stringstream stream;
				stream << "For mask element " 
					<< std::distance( mask.begin(), fi ) << ", index " 
					<< *fi << " is beyond maximum vector index " 
					<< ( 2 * d.type.vector );
				return stream.str();
			}
		}
		return LLVMBinaryInstruction::valid();
	}

	Instruction* LLVMShufflevector::clone(bool copy) const
	{
		return new LLVMShufflevector( *this );
	}

	LLVMSitofp::LLVMSitofp() : LLVMConversionInstruction( Sitofp )
	{
	
	}

	Instruction* LLVMSitofp::clone(bool copy) const
	{
		return new LLVMSitofp( *this );
	}
	
	LLVMSrem::LLVMSrem() : LLVMBinaryInstruction( Srem )
	{
	
	}

	Instruction* LLVMSrem::clone(bool copy) const
	{
		return new LLVMSrem( *this );
	}
	
	LLVMStore::LLVMStore() : LLVMUnaryInstruction( Store ), isVolatile( false ),
		alignment( 1 )
	{
	
	}
	
	std::string LLVMStore::toString() const
	{
		std::stringstream stream;
		stream << LLVMInstruction::toString( opcode ) << " "
			<< (isVolatile ? "volatile " : "")
			<< a.type.toString() << " " << a.toString() << ", " 
			<< d.type.toString() << " " << d.toString();
		if( alignment != 1 ) stream << ", align " << alignment;
		return stream.str();
	}
	
	std::string LLVMStore::valid() const
	{
		if( alignment == 0 ) return "Alignment must be greater than 0";
		if( d.type.category != Type::Pointer )
		{
			return "Address must be a pointer";
		}
		if( d.type.members.empty() )
		{
			if( a.type.type != d.type.type )
			{
				return "Destination " + d.type.toString() 
					+ " is not a pointer to source type " + a.type.toString();
			}
		}
		else
		{
			if( a.type.type != d.type.members[0].type )
			{
				return "Destination " + d.type.toString() 
					+ " is not a pointer to source type " + a.type.toString();
			}
		}
		return "";
	}

	Instruction* LLVMStore::clone(bool copy) const
	{
		return new LLVMStore( *this );
	}

	LLVMSub::LLVMSub() : LLVMBinaryInstruction( Sub ), noUnsignedWrap( false ),
		noSignedWrap( false )
	{
	
	}
	
	std::string LLVMSub::toString() const
	{
		std::string result = d.toString() + " = " 
			+ LLVMInstruction::toString( opcode ) + " ";
		if( noUnsignedWrap ) result += "nuw ";
		if( noSignedWrap ) result += "nsw ";
		result += a.type.toString() + " " + a.toString() + ", " + b.toString();		
		return result;		
	}

	Instruction* LLVMSub::clone(bool copy) const
	{
		return new LLVMSub( *this );
	}

	LLVMSwitch::LLVMSwitch() : LLVMInstruction( Switch )
	{
	
	}
	
	std::string LLVMSwitch::toString() const
	{
		std::string result = LLVMInstruction::toString( opcode ) + " " 
			+ comparison.type.toString() + " " + comparison.toString() 
			+ ", label " + defaultTarget + " [ ";
		for( NodeVector::const_iterator fi = targets.begin(); 
			fi != targets.end(); ++fi )
		{
			result += fi->operand.type.toString() + " " + fi->operand.toString() 
				+ ", label " + fi->label + " ";
		}
		result += "]";
		return result;
	}
	
	std::string LLVMSwitch::valid() const
	{
		if( comparison.type.category != Type::Element )
		{
			return "Comparison value must be a basic element type";
		}
		if( !isInt( comparison.type.type ) )
		{
			return "Comparison value must be an int type";
		}
		if( defaultTarget.empty() ) return "Default target must not be empty";
		for( NodeVector::const_iterator fi = targets.begin(); 
			fi != targets.end(); ++fi )
		{
			if( fi->operand.type.category != Type::Element )
			{
				std::stringstream stream;
				stream << "Target " << std::distance( targets.begin(), fi ) 
					<< " is not a basic element type";
				return stream.str();
			}
			if( !isInt( fi->operand.type.type ) )
			{
				std::stringstream stream;
				stream << "Target " << std::distance( targets.begin(), fi ) 
					<< " is not an int type";
				return stream.str();
			}
			if( fi->label.empty() )
			{
				std::stringstream stream;
				stream << "Target " << std::distance( targets.begin(), fi ) 
					<< " has an empty target label";
				return stream.str();
			}
		}
		return "";
	}

	Instruction* LLVMSwitch::clone(bool copy) const
	{
		return new LLVMSwitch( *this );
	}

	LLVMTrunc::LLVMTrunc() : LLVMConversionInstruction( Trunc )
	{
	
	}

	Instruction* LLVMTrunc::clone(bool copy) const
	{
		return new LLVMTrunc( *this );
	}

	LLVMUdiv::LLVMUdiv() : LLVMBinaryInstruction( Udiv )
	{
	
	}

	Instruction* LLVMUdiv::clone(bool copy) const
	{
		return new LLVMUdiv( *this );
	}

	LLVMUitofp::LLVMUitofp() : LLVMConversionInstruction( Uitofp )
	{
	
	}

	Instruction* LLVMUitofp::clone(bool copy) const
	{
		return new LLVMUitofp( *this );
	}

	LLVMUnreachable::LLVMUnreachable() : LLVMInstruction( Unreachable )
	{
	
	}
	
	std::string LLVMUnreachable::toString() const
	{
		return LLVMInstruction::toString( opcode );
	}
	
	std::string LLVMUnreachable::valid() const
	{
		return "";
	}

	Instruction* LLVMUnreachable::clone(bool copy) const
	{
		return new LLVMUnreachable( *this );
	}

	LLVMUnwind::LLVMUnwind() : LLVMInstruction( Unwind )
	{
	
	}
	
	std::string LLVMUnwind::toString() const
	{
		return LLVMInstruction::toString( opcode );
	}
	
	std::string LLVMUnwind::valid() const
	{
		return "";
	}

	Instruction* LLVMUnwind::clone(bool copy) const
	{
		return new LLVMUnwind( *this );
	}

	LLVMUrem::LLVMUrem() : LLVMBinaryInstruction( Urem )
	{
	
	}

	Instruction* LLVMUrem::clone(bool copy) const
	{
		return new LLVMUrem( *this );
	}

	LLVMVaArg::LLVMVaArg() : LLVMUnaryInstruction( VaArg )
	{
	
	}
	
	std::string LLVMVaArg::toString() const
	{
		return d.toString() + " = " + LLVMInstruction::toString( opcode ) 
			+ " " + a.type.toString() + " " + a.toString() + ", " 
			+ d.type.toString();
	}
	
	std::string LLVMVaArg::valid() const
	{
		/* TODO Extra testing is needed here to make sure that any operands 
			are really valid. */
		return "";
	}

	Instruction* LLVMVaArg::clone(bool copy) const
	{
		return new LLVMVaArg( *this );
	}

	LLVMXor::LLVMXor() : LLVMBinaryInstruction( Xor )
	{
	
	}

	Instruction* LLVMXor::clone(bool copy) const
	{
		return new LLVMXor( *this );
	}

	LLVMZext::LLVMZext() : LLVMConversionInstruction( Zext )
	{
	
	}

	Instruction* LLVMZext::clone(bool copy) const
	{
		return new LLVMZext( *this );
	}
	
}

#endif

