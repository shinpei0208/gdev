/*!
	\file PTXInstruction.cpp
	\author Andrew Kerr <arkerr@gatech.edu>
	\date Jan 15, 2009
	\brief base class for all instructions
*/

#include <ocelot/ir/interface/PTXInstruction.h>
#include <hydrazine/interface/debug.h>
#include <sstream>

std::string ir::PTXInstruction::toString( Level l ) {
	switch( l ) {
		case CtaLevel:    return "cta"; break;
		case GlobalLevel: return "gl";  break;
		case SystemLevel: return "sys";  break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString(CacheLevel cache) {
	switch( cache ) {
		case L1: return "L1";
		case L2: return "L2";
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toStringLoad(CacheOperation operation) {
	switch( operation ) {
		case Ca: return "ca";
		case Cg: return "cg";
		case Cs: return "cs";
		case Cv: return "cv";
		case Nc: return "nc";
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toStringStore(CacheOperation operation) {
	switch( operation ) {
		case Cg: return "cg";
		case Cs: return "cs";
		case Wb: return "wb";
		case Wt: return "wt";
		default: break;
	}
	return "";
}


std::string ir::PTXInstruction::toString( PermuteMode m ) {
	switch( m ) {
		case ForwardFourExtract:  return "f4e"; break;
		case BackwardFourExtract: return "b4e"; break;
		case ReplicateEight:      return "rc8"; break;
		case EdgeClampLeft:       return "ecl"; break;
		case EdgeClampRight:      return "ecr"; break;
		case ReplicateSixteen:    return "rc16"; break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( FloatingPointMode m ) {
	switch( m ) {
		case Finite:     return "finite"; break;
		case Infinite:   return "infinite"; break;
		case Number:     return "number"; break;
		case NotANumber: return "notanumber"; break;
		case Normal:     return "normal"; break;
		case SubNormal:  return "subnormal"; break;
		default: break;
	}
	return "";
}
std::string ir::PTXInstruction::toString( Vec v ) {
	switch( v ) {
		case PTXOperand::v1: return "v1"; break;
		case PTXOperand::v2: return "v2"; break;
		case PTXOperand::v4: return "v4"; break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( AddressSpace space ) {
	switch( space ) {
		case Const:    return "const";   break;
		case Global:   return "global";  break;
		case Local:    return "local";   break;
		case Param:    return "param";   break;
		case Shared:   return "shared";  break;
		case Texture:  return "tex";     break;
		case Generic:  return "generic"; break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( AtomicOperation operation ) {
	switch( operation ) {
		case AtomicAnd:  return "and";  break;
		case AtomicOr:   return "or";   break;
		case AtomicXor:  return "xor";  break;
		case AtomicCas:  return "cas";  break;
		case AtomicExch: return "exch"; break;
		case AtomicAdd:  return "add";  break;
		case AtomicInc:  return "inc";  break;
		case AtomicDec:  return "dec";  break;
		case AtomicMin:  return "min";  break;
		case AtomicMax:  return "max";  break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( ReductionOperation operation ) {
	switch( operation ) {
		case ReductionAnd: return "and"; break;
		case ReductionOr:  return "or";  break;
		case ReductionXor: return "xor"; break;
		case ReductionAdd: return "add"; break;
		case ReductionInc: return "inc"; break;
		case ReductionDec: return "dec"; break;
		case ReductionMin: return "min"; break;
		case ReductionMax: return "max"; break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( SurfaceQuery query ) {
	switch (query) {
		case Width:                  return "width";
		case Height:                 return "height";
		case Depth:                  return "depth";
		case ChannelDataType:        return "channel_data_type";
		case ChannelOrder:           return "channel_order";
		case NormalizedCoordinates:  return "normalized_coords";
		case SamplerFilterMode:      return "filter_mode";
		case SamplerAddrMode0:       return "addr_mode_0";
		case SamplerAddrMode1:       return "addr_mode_1";
		case SamplerAddrMode2:       return "addr_mode_2";
		default:                     break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( FormatMode mode ) {
	switch (mode) {
		case Unformatted:        return ".b";
		case Formatted:          return ".p";
		case FormatMode_Invalid: break;
		default:                 break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( ClampOperation clamp ) {
	switch (clamp) {
		case TrapOOB:                return ".trap";
		case Clamp:                  return ".clamp";
		case Zero:                   return ".zero";
		case Mirror:                 return ".mirror";
		case ClampOperation_Invalid: break;
		default:                     break;
	}
	return "";
}

std::string ir::PTXInstruction::roundingMode( Modifier modifier ) {
	switch( modifier ) {
		case rn: return "rn"; break;
		case rz: return "rz"; break;
		case rm: return "rm"; break;
		case rp: return "rp"; break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::modifierString( unsigned int modifier, 
	CarryFlag carry ) {
	std::string result;
	if( modifier & approx ) {
		result += "approx.";
	}
	else if( modifier & wide ) {
		result += "wide.";
	}
	else if( modifier & hi ) {
		result += "hi.";
	}
	else if( modifier & lo ) {
		result += "lo.";
	}
	else if( modifier & rn ) {
		result += "rn.";
	}
	else if( modifier & rz ) {
		result += "rz.";
	}
	else if( modifier & rm ) {
		result += "rm.";
	}
	else if( modifier & rp ) {
		result += "rp.";
	}

	if( modifier & ftz ) {
		result += "ftz.";
	}
	if( modifier & sat ) {
		result += "sat.";
	}
	if( carry == CC ) {
		result += "cc.";
	}
	return result;
}

std::string ir::PTXInstruction::toString( Modifier modifier ) {
	switch( modifier ) {
		case hi:     return "hi";     break;
		case lo:     return "lo";     break;
		case wide:   return "wide";   break;
		case sat:    return "sat";    break;
		case rn:     return "rn";     break;
		case rz:     return "rz";     break;
		case rm:     return "rm";     break;
		case rp:     return "rp";     break;
		case approx: return "approx"; break;
		case ftz:    return "ftz";    break;
		default: break;
	}
	return "";	
}


std::string ir::PTXInstruction::toString( BarrierOperation operation) {
	switch (operation) {
		case BarSync:      return "sync";
		case BarArrive:    return "arrive";
		case BarReduction: return "red";
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( CmpOp operation ) {
	switch( operation ) {
		case Eq: return "eq";   break;
		case Ne: return "ne";   break;
		case Lt: return "lt";   break;
		case Le: return "le";   break;
		case Gt: return "gt";   break;
		case Ge: return "ge";   break;
		case Lo: return "lo";   break;
		case Ls: return "ls";   break;
		case Hi: return "hi";   break;
		case Hs: return "hs";   break;
		case Equ: return "equ"; break;
		case Neu: return "neu"; break;
		case Ltu: return "ltu"; break;
		case Leu: return "leu"; break;
		case Gtu: return "gtu"; break;
		case Geu: return "geu"; break;
		case Num: return "num"; break;
		case Nan: return "nan"; break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( BoolOp operation ) {
	switch( operation ) {
		case BoolAnd: return "and"; break;
		case BoolOr:  return "or";  break;
		case BoolXor: return "xor"; break;	
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( Geometry geometry ) {
	switch( geometry ) {
		case _1d: return "1d"; break;
		case _2d: return "2d"; break;
		case _3d: return "3d"; break;
		case _a1d: return "a1d"; break;
		case _a2d: return "a2d"; break;
		case _cube: return "cube"; break;
		case _acube: return "acube"; break;		
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( VoteMode mode ) {
	switch( mode ) {
		case All:    return "all";    break;
		case Any:    return "any";    break;
		case Uni:    return "uni";    break;
		case Ballot: return "ballot"; break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( ShuffleMode mode ) {
	switch( mode ) {
		case Up:   return "up";   break;
		case Down: return "down"; break;
		case Bfly: return "bfly"; break;
		case Idx:  return "idx";  break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( ColorComponent color ) {
	switch( color ) {
		case red:   return "r"; break;
		case green: return "g"; break;
		case blue:  return "b"; break;
		case alpha: return "a"; break;
		default: break;
	}
	return "";
}

std::string ir::PTXInstruction::toString( Opcode opcode ) {
	switch( opcode ) {
		case Abs:        return "abs";        break;
		case Add:        return "add";        break;
		case AddC:       return "addc";       break;
		case And:        return "and";        break;
		case Atom:       return "atom";       break;
		case Bar:        return "bar";        break;
		case Bfe:        return "bfe";        break;
		case Bfi:        return "bfi";        break;
		case Bfind:      return "bfind";      break;
		case Bra:        return "bra";        break;
		case Brev:       return "brev";       break;
		case Brkpt:      return "brkpt";      break;
		case Call:       return "call";       break;
		case Clz:        return "clz";        break;
		case CNot:       return "cnot";       break;
		case CopySign:   return "copysign";   break;
		case Cos:        return "cos";        break;
		case Cvt:        return "cvt";        break;
		case Cvta:       return "cvta";       break;
		case Div:        return "div";        break;
		case Ex2:        return "ex2";        break;
		case Exit:       return "exit";       break;
		case Fma:        return "fma";        break;
		case Isspacep:   return "isspacep";   break;
		case Ld:         return "ld";         break;
		case Ldu:        return "ldu";        break;
		case Lg2:        return "lg2";        break;
		case Mad24:      return "mad24";      break;
		case Mad:        return "mad";        break;
		case MadC:       return "madc";        break;
		case Max:        return "max";        break;
		case Membar:     return "membar";     break;
		case Min:        return "min";        break;
		case Mov:        return "mov";        break;
		case Mul24:      return "mul24";      break;
		case Mul:        return "mul";        break;
		case Neg:        return "neg";        break;
		case Not:        return "not";        break;
		case Or:         return "or";         break;
		case Pmevent:    return "pmevent";    break;
		case Popc:       return "popc";       break;
		case Prefetch:   return "prefetch";   break;
		case Prefetchu:  return "prefetchu";  break;
		case Prmt:       return "prmt";       break;
		case Rcp:        return "rcp";        break;
		case Red:        return "red";        break;
		case Rem:        return "rem";        break;
		case Ret:        return "ret";        break;
		case Rsqrt:      return "rsqrt";      break;
		case Sad:        return "sad";        break;
		case SelP:       return "selp";       break;
		case Set:        return "set";        break;
		case SetP:       return "setp";       break;
		case Shfl:       return "shfl";       break;
		case Shl:        return "shl";        break;
		case Shr:        return "shr";        break;
		case Sin:        return "sin";        break;
		case SlCt:       return "slct";       break;
		case Sqrt:       return "sqrt";       break;
		case St:         return "st";         break;
		case Sub:        return "sub";        break;
		case SubC:       return "subc";       break;
		case Suld:       return "suld";       break;
		case Sured:      return "sured";      break;
		case Sust:       return "sust";       break;
		case Suq:        return "suq";        break;
		case TestP:      return "testp";      break;
		case Tex:        return "tex";        break;
		case Tld4:       return "tld4";       break;
		case Txq:        return "txq";        break;
		case Trap:       return "trap";       break;
		case Vabsdiff:   return "vabsdiff";   break;
		case Vadd:       return "vadd";       break;
		case Vmad:       return "vmad";       break;
		case Vmax:       return "vmax";       break;
		case Vmin:       return "vmin";       break;
		case Vset:       return "vset";       break;
		case Vshl:       return "vshl";       break;
		case Vshr:       return "vshr";       break;
		case Vsub:       return "vsub";       break;
		case Vote:       return "vote";       break;
		case Xor:        return "xor";        break;
		case Reconverge: return "reconverge"; break;
		case Phi:        return "phi";        break;
		case Nop:        return "nop";        break;
		case Invalid_Opcode: break;
	}
	return "INVALID";
}

bool ir::PTXInstruction::isPt( const PTXOperand& op )
{
	return op.toString() == "%pt";
}

ir::PTXInstruction::PTXInstruction( Opcode op, const PTXOperand& _d, 
	const PTXOperand& _a, const PTXOperand& _b, const PTXOperand& _c ) 
	: opcode(op), d(_d), a(_a), b(_b), c(_c) {
	ISA = Instruction::PTX;
	type = PTXOperand::s32;
	modifier = 0;
	reconvergeInstruction = 0;
	branchTargetInstruction = 0;
	vec = PTXOperand::v1;
	pg.condition = PTXOperand::PT;
	pg.type = PTXOperand::pred;
	barrierOperation = BarSync;
	carry = None;
	statementIndex = -1;
	geometry = Geometry_Invalid;
	cc = 0;
	addressSpace = AddressSpace_Invalid;
	tailCall = false;
}

ir::PTXInstruction::~PTXInstruction() {

}

bool ir::PTXInstruction::operator==( const PTXInstruction& i ) const {
	return opcode == i.opcode;
}

std::string ir::PTXInstruction::valid() const {
	switch (opcode) {
		case Abs: {
			if ( !( type == PTXOperand::s16 || type == PTXOperand::s32 || 
				type == PTXOperand::s64 || type == PTXOperand::f32 || 
				type == PTXOperand::f64 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );	
			}			
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}		
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case Add: {
			if ( !( type != PTXOperand::s8 && type != PTXOperand::u8 && 
				type != PTXOperand::b8 && type != PTXOperand::f16 
				&& type != PTXOperand::pred ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( carry == CC ) {
				if( ( modifier & sat ) ) {
					return "saturate not supported with carry out";
				}
				if( !( type == PTXOperand::u32 || type == PTXOperand::s32 ) ) {
					return "invalid instruction type " 
						+ PTXOperand::toString( type );
				}
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case AddC: {
			if( !( type == PTXOperand::u32 || type == PTXOperand::s32 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case And: {
			if( !( type == PTXOperand::b16 || type == PTXOperand::b32 || 
				type == PTXOperand::b64 || type == PTXOperand::pred ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;	
		}
		case Atom: {
			if( !PTXOperand::valid( PTXOperand::b32, type ) 
				&& !PTXOperand::valid( PTXOperand::b64, type ) 
				&& ( atomicOperation == AtomicAnd || atomicOperation == AtomicOr 
				|| atomicOperation == AtomicXor || atomicOperation == AtomicCas 
				|| atomicOperation == AtomicExch ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type ) + " for atomic " 
					+ toString( atomicOperation );
			}
				
			if( !PTXOperand::valid( PTXOperand::u32, type ) 
				&& !PTXOperand::valid( PTXOperand::u64, type ) 
				&& !PTXOperand::valid( PTXOperand::s32, type ) 
				&& ( atomicOperation == AtomicInc 
				|| atomicOperation == AtomicDec ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type ) + " for atomic " 
					+ toString( atomicOperation );
			}
			if( !PTXOperand::valid( PTXOperand::f32, type ) 
				&& !PTXOperand::valid( PTXOperand::u32, type ) 
				&& !PTXOperand::valid( PTXOperand::u64, type ) 
				&& !PTXOperand::valid( PTXOperand::s32, type ) 
				&& ( atomicOperation == AtomicAdd 
				|| atomicOperation == AtomicMin 
				|| atomicOperation == AtomicMax ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type ) + " for atomic " 
					+ toString( atomicOperation );
			}
			if( !( addressSpace == Shared || addressSpace == Global ) ) {
				return "invalid adress space " + toString( addressSpace );
			}
			break;
		}
		case Bar: {
			if( d.addressMode != PTXOperand::Immediate ) {
				return "only support Immediate targets";
			}
			break;
		}
        case Bfe: {
            if( !( type == PTXOperand::u32 
                   || type == PTXOperand::s32 
                   || type == PTXOperand::u64
                   || type == PTXOperand::s64 ) ){
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
            }
            if( !PTXOperand::valid( type, d.type ) ) {
                return "operand D type " + PTXOperand::toString( d.type )
                    + " cannot be assigned to " + PTXOperand::toString( type );
            }
            if( !PTXOperand::valid( type, a.type )
                && a.addressMode != PTXOperand::Immediate ) {
                return "operand 1 type " + PTXOperand::toString( a.type )
                    + " cannot be assigned to " + PTXOperand::toString( type );
            }
            if( !PTXOperand::valid( PTXOperand::u32, b.type )
                && a.addressMode != PTXOperand::Immediate ) {
                return "operand 1 type " + PTXOperand::toString( b.type )
                    + " cannot be assigned to " + PTXOperand::toString( PTXOperand::u32 );
            }
            if( !PTXOperand::valid( PTXOperand::u32, c.type )
                && a.addressMode != PTXOperand::Immediate ) {
                return "operand 1 type " + PTXOperand::toString( c.type )
                    + " cannot be assigned to " + PTXOperand::toString( PTXOperand::u32 );
            }

            break;
        }
		case Bfi: {
			if( !( type == PTXOperand::b32 || type == PTXOperand::b64 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, pq.type ) 
				&& pq.addressMode != PTXOperand::Immediate ) {
				return "operand 1 type " + PTXOperand::toString( pq.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand 2 type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( PTXOperand::u32, b.type )  ) {
				return "operand 3 type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " 
					+ PTXOperand::toString( PTXOperand::u32 );
			}
			if( !PTXOperand::valid( PTXOperand::u32, b.type )  ) {
				return "operand 4 type " + PTXOperand::toString( c.type ) 
					+ " cannot be assigned to " 
					+ PTXOperand::toString( PTXOperand::u32 );
			}
			break;
		}
		case Bfind: {
			if( !( type == PTXOperand::u32 || type == PTXOperand::u64 
				|| type == PTXOperand::s32 || type == PTXOperand::s64 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( PTXOperand::u32, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " 
					+ PTXOperand::toString( PTXOperand::u32 );
			}
			break;
		}
		case Bra: {
			if( !( d.addressMode == PTXOperand::Label 
				|| d.addressMode == PTXOperand::Register ) ) {
				return "no support for types other than Label and Register";
			}
			break;
		}
		case Brev: {
			if( !( type == PTXOperand::b32 || type == PTXOperand::b64 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Brkpt: {
			break;
		}
		case Call: {
			if( a.addressMode != PTXOperand::Register
				&& a.addressMode != PTXOperand::FunctionName ) {
				return "operand A must be a function name or register.";
			}
			if( d.addressMode != PTXOperand::ArgumentList 
				&& d.addressMode != PTXOperand::Register 
				&& d.addressMode != PTXOperand::Invalid ) {
				return "operand D must be an argument/register "
					"list if it is specified.";
			}
			if( b.addressMode != PTXOperand::ArgumentList
				&& b.addressMode != PTXOperand::Register 
				&& b.addressMode != PTXOperand::Invalid ) {
				return "operand B must be an argument/register "
					"list if it is specified.";
			}
			if( a.addressMode == PTXOperand::Register
				&& c.addressMode != PTXOperand::FunctionName ) {
				return "operand C must be function name if A is a register.";
			}
			
			for( auto operand = d.array.begin();
				operand != d.array.end(); ++operand )
			{
				if (operand->addressMode != PTXOperand::Register &&
					operand->addressMode != PTXOperand::BitBucket &&
					operand->addressMode != PTXOperand::Address &&
					operand->addressMode != PTXOperand::Immediate) {
					return "return arguments must be registers, parameters, "
						"or immediates.";
				}
			}
			
			for( auto operand = b.array.begin();
				operand != b.array.end(); ++operand )
			{
				if (operand->addressMode != PTXOperand::Register &&
					operand->addressMode != PTXOperand::BitBucket &&
					operand->addressMode != PTXOperand::Address &&
					operand->addressMode != PTXOperand::Immediate) {
					return "function arguments must be registers"
						", parameters, or immediates.";
				}
			}
			
			break;
		}
		case Clz: {
			if( !( type == PTXOperand::b32 || type == PTXOperand::b64 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( PTXOperand::u32, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " 
					+ PTXOperand::toString( PTXOperand::u32 );
			}
			break;
		}
		case CNot: {
			if( !( type == PTXOperand::b16 || type == PTXOperand::b32 || 
				type == PTXOperand::b64 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case CopySign : {
			if( type != PTXOperand::f32 && type != PTXOperand::f64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, a.type ) ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, b.type ) ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Cos: {
			if( !( type == PTXOperand::f32 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type ) ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case Cvt: {
			if( type == PTXOperand::pred ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}					
			if( d.bytes() < PTXOperand::bytes( type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned from " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( !(PTXOperand::isFloat( type ) || PTXOperand::isFloat(a.type))) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			if( vec == PTXOperand::v1
				&& PTXOperand::BitBucket != d.addressMode 
				&& !PTXOperand::relaxedValid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type ) 
					+ ", not even for relaxed typed instructions.";
			}
			break;
		}
		case Cvta: {
			if (!(type == PTXOperand::u32 || type == PTXOperand::u64)) {
				return "invalid instruction type " + PTXOperand::toString(type);
			}
			if (!(addressSpace == Global || addressSpace == Local
				|| addressSpace == Shared || addressSpace == Const)) {
				return "invalid address space " + toString(addressSpace);
			}
			break;
		}
		case Div: {
			if( ( modifier & sat ) ) {
				return "no support for saturating divide.";
			}
			if( ( modifier & approx ) ) {
				if( type != PTXOperand::f32 ) {
					return "only f32 supported for approximate";
				}
			}
			if( type == PTXOperand::f64 ) {
				if( !( modifier & rn ) && !( modifier & rz ) 
					&& !( modifier & rm ) && !( modifier & rp ) ) {
					return "requires a rounding modifier";
				}
				if( !( modifier & rn ) ) {
					return "only nearest rounding supported";
				}
			}
			if( !( type == PTXOperand::u16 || type == PTXOperand::u32 
				|| type == PTXOperand::u64 || type == PTXOperand::s16 
				|| type == PTXOperand::s32 || type == PTXOperand::s64 
				|| type == PTXOperand::f32 || type == PTXOperand::f64 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case Ex2: {
			if( !( type == PTXOperand::f32 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case Exit: {
			break;
		}
		case Fma: {
			if (!(type == ir::PTXOperand::f32 || type == ir::PTXOperand::f64)) {
				return "invalid instruction type " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Isspacep: {
			if (!(addressSpace == PTXInstruction::Global
				|| addressSpace == PTXInstruction::Shared
				|| addressSpace == PTXInstruction::Local)) {
				return "invalid address space " + toString(addressSpace);
			}
			if (!(d.addressMode == PTXOperand::Register
				&& a.addressMode == PTXOperand::Register)) {
				return "invalid address mode for operands";
			}
			break;
		}
		case Ldu: // fall through
		case Ld: {
			if( !( a.addressMode == PTXOperand::Register 
				|| a.addressMode == PTXOperand::Address 
				|| a.addressMode == PTXOperand::Indirect 
				|| a.addressMode == PTXOperand::Immediate ) ) {
				return "invalid address mode " 
					+ PTXOperand::toString( a.addressMode ) 
					+ " for operand A ";
			}
			if( addressSpace == AddressSpace_Invalid ) {
				return "invalid address space";
			}
			if( addressSpace != Global && addressSpace != Shared 
				&& volatility == Volatile && addressSpace != Generic ) {
				return "only shared and global address spaces supported " 
					"for volatile loads";
			}
			if( d.addressMode != PTXOperand::Register ) {
				return "operand D must be a register not a " 
					+ PTXOperand::toString( d.addressMode );
			}
			if( vec == PTXOperand::v1
				&& PTXOperand::BitBucket != d.addressMode 
				&& !PTXOperand::relaxedValid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type ) 
					+ ", not even for relaxed typed instructions: -\n" 
					+ toString() + "\nd.addressMode: "
					+ PTXOperand::toString(d.addressMode);
			}
			break;
		}
		case Lg2: {
			if( !( type == PTXOperand::f32 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type ) ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case Mad24: {
			if( !( type == PTXOperand::u32 || type == PTXOperand::s32 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, c.type )  ) {
				return "operand C type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Mad: {
			if( !( type != PTXOperand::s8 && type != PTXOperand::u8 && 
				type != PTXOperand::b8 && type != PTXOperand::f16 
				&& type != PTXOperand::pred ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( type == PTXOperand::f64 ) {
				if( !( modifier & rn ) && !( modifier & rz ) 
					&& !( modifier & rm ) && !( modifier & rp ) ) {
					return "requires a rounding modifier";
				}
			}
			if( a.type != b.type ) {
				return "type of operand A " + PTXOperand::toString( a.type ) 
					+ " does not equal type of operand B " 
					+ PTXOperand::toString( b.type );
			}
			if( !( c.bytes() == d.bytes() ) ) {
				std::stringstream stream;
				stream << "size of operand C " << c.bytes() 
					<< " does not equal size of operand D " << d.bytes();
				return stream.str();
			}
			if( ( ( modifier & hi ) || ( modifier & lo ) ) 
				&& ( a.bytes() != d.bytes() ) ) {
				std::stringstream stream;
				stream << "not wide and size of operand A " << a.bytes() 
					<< " does not equal size of operand D " << d.bytes();
				return stream.str();
			}
			if( ( modifier & wide ) && ( 2 * a.bytes() != d.bytes() ) ) {
				std::stringstream stream;
				stream << "wide and size of operand A " << a.bytes() 
					<< " does not equal half the size of operand D " 
					<< d.bytes();
				return stream.str();
			}
			if( ( modifier & sat ) && ( type != PTXOperand::s32
				&& type != PTXOperand::f32 ) ) {
				return "saturate only valid for s32/f32";
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case MadC: {
			if( !( type == PTXOperand::u32 || type == PTXOperand::s32 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, c.type )  ) {
				return "operand C type " + PTXOperand::toString( c.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !( ( modifier & hi ) || ( modifier & lo ) ) ) {
				return "modifier must be hi or lo";
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Max: {
			if( !( type != PTXOperand::s8 && type != PTXOperand::u8 && 
				type != PTXOperand::b8 && type != PTXOperand::f16 
				&& type != PTXOperand::pred ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type ) ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type ) ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case Membar: {
			break;
		}
		case Min: {
			if( !( type != PTXOperand::s8 && type != PTXOperand::u8 && 
				type != PTXOperand::b8 && type != PTXOperand::f16 
				&& type != PTXOperand::pred ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type ) ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type ) ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case Mov: {
			if ( ( a.type == PTXOperand::f16 ) &&
				a.addressMode != PTXOperand::Address &&
				a.addressMode != PTXOperand::Immediate ) {
				return "invalid type for operand A " 
					+ PTXOperand::toString( a.type );
			}
			if ( !( d.type != PTXOperand::s8 && d.type != PTXOperand::u8 
				&& d.type != PTXOperand::b8 && d.type != PTXOperand::f16 ) ) {
				return "invalid type for operand D " 
					+ PTXOperand::toString( d.type );
			}
			if( ( a.vec != PTXOperand::v1 || d.vec != PTXOperand::v1 ) 
				&& ( a.bytes() != d.bytes() ) ) {
				std::stringstream stream;
				stream << "at least one vector operand and A size " << a.bytes()
					<< " does not equal D size " << d.bytes();
				return stream.str();
			}
			break;
		}
		case Mul24: {
			if( type != PTXOperand::u32 && type != PTXOperand::s32 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !( modifier & lo ) && !( modifier & hi ) ) {
				return "must be either lo or hi";
			}
			break;
		}
		case Mul: {
			if( type == PTXOperand::s8 || type == PTXOperand::u8 
				|| type == PTXOperand::b8 || type == PTXOperand::f16 
				|| type == PTXOperand::pred ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( PTXOperand::isInt( type ) && ( !( modifier & lo ) 
				&& !( modifier & hi ) && !( modifier & wide ) ) ) {
				return "int operations must be hi, lo, or wide";
			}
			if( ( ( modifier & lo ) || ( modifier & hi ) ) 
				&& a.bytes() != d.bytes() ) {
				std::stringstream stream;
				stream << "for lo and hi, size of operand A " << a.bytes() 
					<< " does not equal size of operand D " << d.bytes();
				return stream.str();
			}
			if( ( modifier & wide ) && ( 2 * a.bytes() != d.bytes() ) ) {
				std::stringstream stream;
				stream << "wide, size of operand A " << a.bytes() 
					<< " does not equal half the size of operand D " 
					<< d.bytes();
				return stream.str();
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case Neg: {
			if( type != PTXOperand::s16 && type != PTXOperand::s32 && 
				type != PTXOperand::s64 && type != PTXOperand::f32 && 
				type != PTXOperand::f64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case Nop: {
			return "NOP is not a valid instruction.";
			break;
		}
		case Not: {
			if( !( type == PTXOperand::b16 || type == PTXOperand::b32 || 
				type == PTXOperand::b64 || type == PTXOperand::pred ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Pmevent: {
			if( d.addressMode != PTXOperand::Immediate ) {
				return "only support Immediate targets";
			}
			break;
		}
		case Popc: {
			if( !( type == PTXOperand::b32 || type == PTXOperand::b64 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( PTXOperand::u32, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " 
					+ PTXOperand::toString( PTXOperand::u32 );
			}
			break;
		}
		case Prefetch: {
			if (!(cacheLevel == L1 || cacheLevel == L2)) {
				return "cache level must be L1 or L2";
			}
			if (!(addressSpace == Local || addressSpace == Global)) {
				return "address space must be .local or .global, not " + toString(addressSpace);
			}
			if (!(d.addressMode == PTXOperand::Indirect || d.addressMode == PTXOperand::Address ||
				d.addressMode == PTXOperand::Immediate)) {
				
				return "address mode of destination operand must be Indirect, Address, or Immediate. Not " +
					PTXOperand::toString(d.addressMode);
			}
		}
		break;
		case Prefetchu: {
		
			if (!(cacheLevel == L1)) {
				return "cache level must be L1, not " + toString(cacheLevel);
			}
			if (!(d.addressMode == PTXOperand::Indirect || d.addressMode == PTXOperand::Address ||
				d.addressMode == PTXOperand::Immediate)) {
				
				return "address mode of destination operand must be Indirect, Address, or Immediate. Not " +
					PTXOperand::toString(d.addressMode);
			}
		}
		break;
		case Prmt: {
			if( type != PTXOperand::b32 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, c.type )  ) {
				return "operand C type " + PTXOperand::toString( c.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Or: {
			if( !( type == PTXOperand::b16 || type == PTXOperand::b32 || 
				type == PTXOperand::b64 || type == PTXOperand::pred ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Rcp: {
			if( type != PTXOperand::f32 && type != PTXOperand::f64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( type == PTXOperand::f64 ) {
				if( modifier & ftz ) {
					if( !( modifier & approx ) ) {
						return "requires .approx.ftz for f64";
					}
				}
				else if ( !( modifier & rn ) && !( modifier & rz ) 
					&& !( modifier & rm ) && !( modifier & rp ) ) {
					return "rounding mode required";
				}
			
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case Red: {
			if( ( reductionOperation == ReductionAnd 
				|| reductionOperation == ReductionOr
				|| reductionOperation == ReductionXor ) 
				&& type != PTXOperand::b32 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type ) + " for reduction " 
					+ toString( reductionOperation );
			}
			if( reductionOperation == ReductionAdd
				&& ( type != PTXOperand::u32 && type != PTXOperand::s32 
				&& type != PTXOperand::f32 && type != PTXOperand::u64 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type ) + " for reduction " 
					+ toString( reductionOperation );
			}
			if( ( reductionOperation == ReductionInc
				|| reductionOperation == ReductionDec )
				&& type != PTXOperand::u32 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type ) + " for reduction " 
					+ toString( reductionOperation );
			}
			if( ( reductionOperation == ReductionMin
				|| reductionOperation == ReductionMax )
				&& ( type != PTXOperand::u32 && type != PTXOperand::s32 
				&& type != PTXOperand::f32 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type ) + " for reduction " 
					+ toString( reductionOperation );
			}
			if( a.addressMode != PTXOperand::Address 
				&& a.addressMode != PTXOperand::Register 
				&& a.addressMode != PTXOperand::Indirect
				&& a.addressMode != PTXOperand::Immediate ) {
				return "operand A must be an address";
			}
			if( addressSpace != Shared && addressSpace != Global ) {
				return "address space much be either shared or global";
			}
			break;
		}
		case Rem: {
			if( type != PTXOperand::s16 && type != PTXOperand::s32 
				&& type != PTXOperand::s64 && type != PTXOperand::u16 
				&& type != PTXOperand::u32 && type != PTXOperand::u64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;		
		}
		case Ret: {
			break;
		}
		case Rsqrt: {
			if( type != PTXOperand::f32 && type != PTXOperand::f64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;			
		}
		case Sad: {
			if( type != PTXOperand::s16 && type != PTXOperand::s32 
				&& type != PTXOperand::s64 && type != PTXOperand::u16 
				&& type != PTXOperand::u32 && type != PTXOperand::u64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, c.type )  ) {
				return "operand C type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;		
		}
		case SelP: {
			if( type != PTXOperand::s16 && type != PTXOperand::s32 
				&& type != PTXOperand::s64 && type != PTXOperand::u16 
				&& type != PTXOperand::u32 && type != PTXOperand::u64
				&& type != PTXOperand::b16 && type != PTXOperand::b32 
				&& type != PTXOperand::b64 && type != PTXOperand::f32
				&& type != PTXOperand::f64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( c.type != PTXOperand::pred ) {
				return "operand C type " + PTXOperand::toString( c.type ) 
					+ " must be a predicate.";
			}
			break;
		}
		case Set: {
			if( type != PTXOperand::s16 && type != PTXOperand::s32 
				&& type != PTXOperand::s64 && type != PTXOperand::u16 
				&& type != PTXOperand::u32 && type != PTXOperand::u64
				&& type != PTXOperand::b16 && type != PTXOperand::b32 
				&& type != PTXOperand::b64 && type != PTXOperand::f32
				&& type != PTXOperand::f64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( d.type != PTXOperand::s32 && d.type != PTXOperand::f32 
				&& d.type != PTXOperand::u32 ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " invalid (must be u32, s32, or f32)";
			}
			if( c.type != PTXOperand::pred && 
				c.addressMode != PTXOperand::Invalid ) {
				return "operand C type " + PTXOperand::toString( c.type ) 
					+ " must be a predicate.";
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( a.type ) ) {
					return " .ftz only valid when source is .f32.";
				}
			}
			break;
		}
		case SetP: {
			if( d.type != PTXOperand::pred ) {
				return "destination must be a predicate";
			}
			if( pq.type != PTXOperand::pred 
				&& pq.addressMode != PTXOperand::Invalid ) {
				return "Pq must be a predicate";
			}
			if( c.type != PTXOperand::pred 
				&& c.addressMode != PTXOperand::Invalid ) {
				return "operand C type " + PTXOperand::toString( c.type ) 
					+ " must be a predicate.";
			}
			if( type != PTXOperand::s16 && type != PTXOperand::s32 
				&& type != PTXOperand::s64 && type != PTXOperand::u16 
				&& type != PTXOperand::u32 && type != PTXOperand::u64
				&& type != PTXOperand::b16 && type != PTXOperand::b32 
				&& type != PTXOperand::b64 && type != PTXOperand::f32
				&& type != PTXOperand::f64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;			
		}
		case Shfl: {
			if( type != PTXOperand::b32 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( d.bytes() != a.bytes() 
				&& a.addressMode != PTXOperand::Immediate ) {
				std::stringstream stream;
				stream << "size of operand A " << a.bytes() 
					<< " does not match size of operand D " << d.bytes();
				return stream.str(); 
			}
			if( b.bytes() != 4 && b.addressMode != PTXOperand::Immediate ) {
				std::stringstream stream;
				stream << "size of operand B " << b.bytes() 
					<< " must be 4 bytes";
				return stream.str();
			}
			if( c.bytes() != 4 && c.addressMode != PTXOperand::Immediate ) {
				std::stringstream stream;
				stream << "size of operand C " << b.bytes() 
					<< " must be 4 bytes";
				return stream.str();
			}
			break;
		}
		case Shl: {
			if( type != PTXOperand::b16 && type != PTXOperand::b32 
				&& type != PTXOperand::b64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( d.bytes() != a.bytes() 
				&& a.addressMode != PTXOperand::Immediate ) {
				std::stringstream stream;
				stream << "size of operand A " << a.bytes() 
					<< " does not match size of operand D " << d.bytes();
				return stream.str(); 
			}
			if( b.bytes() != 4 && b.addressMode != PTXOperand::Immediate ) {
				std::stringstream stream;
				stream << "size of operand B " << a.bytes() 
					<< " must be 4 bytes";
				return stream.str();
			}
			break;
		}
		case Shr: {
			if( type != PTXOperand::b16 && type != PTXOperand::b32 
				&& type != PTXOperand::b64 && type != PTXOperand::s16 
				&& type != PTXOperand::s32 && type != PTXOperand::s64 
				&& type != PTXOperand::u16 && type != PTXOperand::u32 
				&& type != PTXOperand::u64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( d.bytes() != a.bytes() ) {
				std::stringstream stream;
				stream << "size of operand A " << a.bytes() 
					<< " does not match size of operand D " << d.bytes();
				return stream.str(); 
			}
			if( b.bytes() != 4 && b.addressMode != PTXOperand::Immediate ) {
				std::stringstream stream;
				stream << "size of operand B " << a.bytes() 
					<< " must be 4 bytes";
				return stream.str();
			}
			break;
		}
		case Sin: {
			if( !( type == PTXOperand::f32 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, a.type ) ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case SlCt: {
			if( type != PTXOperand::s16 && type != PTXOperand::s32 
				&& type != PTXOperand::s64 && type != PTXOperand::u16 
				&& type != PTXOperand::u32 && type != PTXOperand::u64
				&& type != PTXOperand::b16 && type != PTXOperand::b32 
				&& type != PTXOperand::b64 && type != PTXOperand::f32
				&& type != PTXOperand::f64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			unsigned int bytes = PTXOperand::bytes( type );
			if( bytes != a.bytes() ) {
				std::stringstream stream;
				stream << "size of operand A " << a.bytes() 
					<< " does not match size of instruction " << bytes;
				return stream.str(); 	
			}
			if( bytes != b.bytes() ) {
				std::stringstream stream;
				stream << "size of operand B " << b.bytes() 
					<< " does not match size of instruction " << bytes;
				return stream.str(); 	
			}
			if( bytes != d.bytes() ) {
				std::stringstream stream;
				stream << "size of operand D " << d.bytes() 
					<< " does not match size of instruction " << bytes;
				return stream.str(); 	
			}
			if( !PTXOperand::valid( PTXOperand::f32, c.type ) 
				&& !PTXOperand::valid( PTXOperand::s32, c.type ) ) {
				return "operand C must be either s32 or f32 assignable";
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
			
		}
		case Sqrt: {
			if( ( modifier & approx ) ) {
				if( type != PTXOperand::f32 ) {
					return "only f32 supported for approximate";
				}
			}
			if( type != PTXOperand::f32 && type != PTXOperand::f64 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( type == PTXOperand::f64 ) {
				if( !( modifier & rn ) && !( modifier & rz ) 
					&& !( modifier & rm ) && !( modifier & rp ) ) {
					return "requires a rounding modifier";
				}
				if( !( modifier & rn ) ) {
					return "only nearest rounding supported";
				}
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;			
		}
		case St: {
			if( d.addressMode != PTXOperand::Register 
				&& d.addressMode != PTXOperand::Address 
				&& d.addressMode != PTXOperand::Indirect 
				&& d.addressMode != PTXOperand::Immediate ) {
				return "invalid address mode " 
					+ PTXOperand::toString( d.addressMode ) 
					+ " for operand D ";
			}
			if( addressSpace == AddressSpace_Invalid ) {
				return "invalid address space";
			}
			if( ( addressSpace != Global && addressSpace != Shared
				&& addressSpace != Generic ) 
				&& volatility == Volatile ) {
				return "only shared and global address spaces supported " 
					"for volatile stores";
			}
			if( a.addressMode != PTXOperand::Register
				&& a.addressMode != PTXOperand::Immediate ) {
				return "operand A must be a register or immediate";
			}
			if( vec == PTXOperand::v1
				&& PTXOperand::BitBucket != a.addressMode 
				&& !PTXOperand::relaxedValid( type, a.type ) ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type ) 
					+ ", not even for relaxed typed instructions.";
			}
			break;
		}
		case Sub: {
			if ( !( type != PTXOperand::s8 && type != PTXOperand::u8 && 
				type != PTXOperand::b8 && type != PTXOperand::f16 
				&& type != PTXOperand::pred ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( carry == CC ) {
				if( ( modifier & sat ) ) {
					return "saturate not supported with carry out";
				}
				if( !( type == PTXOperand::u32 || type == PTXOperand::s32 ) ) {
					return "invalid instruction type " 
						+ PTXOperand::toString( type );
				}
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}			
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( modifier & ftz ) {
				if( PTXOperand::isInt( type ) ) {
					return toString( ftz ) 
						+ " only valid for float point instructions.";
				}
			}
			break;
		}
		case SubC: {
			if( !( type == PTXOperand::u32 || type == PTXOperand::s32 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type ) ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case TestP: {
			if( !( type == PTXOperand::f32 || type == PTXOperand::f64 ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( d.type != ir::PTXOperand::pred ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " should be a predicate instead.";
			}
			break;
		}
		case Tex: {
			if( geometry == _2d && c.vec == PTXOperand::v1 ) {
				return "for 2d tectures, C must be a at least a 2d vector";
			}
			if( geometry == _3d && c.vec != PTXOperand::v2 
				&& c.vec != PTXOperand::v4  ) {
				return "for 3d textures, C must be at least a 4d vector";
			}
			if( !PTXOperand::valid( PTXOperand::s32, c.type ) 
				&& !PTXOperand::valid( PTXOperand::f32, c.type ) ) {
				return "operand C must be assignable to f32 or s32";
			}
			if( !PTXOperand::valid( PTXOperand::s32, type ) 
				&& !PTXOperand::valid( PTXOperand::f32, type )
				&& !PTXOperand::valid( PTXOperand::s32, type ) ) {
				return "instruction must be be assignable to u32, f32, or s32";
			}
			if( geometry != _a2d && !PTXOperand::valid( type, c.type )  ) {
				return "operand C type " + PTXOperand::toString( c.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( geometry == _a2d && !PTXOperand::valid( type, d.type )  ) {//SI_FIX: temp, for SimpleLayeredTexture
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		case Tld4: {
			if( type != PTXOperand::f32 && type != PTXOperand::s32
				&& type != PTXOperand::u32 ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( c.type != PTXOperand::f32 ) {
				return "source C type must be f32 " 
					+ PTXOperand::toString( c.type );
			}
			if( c.vec != PTXOperand::v2 ) {
				return "operand C must be a 2-component vector ";
			}
			break;
		}
		case Txq: {
			if (type != ir::PTXOperand::b32) {
				return "data type must be .b32";
			}
			break;
		}
		case Suld: {
			if (formatMode == Formatted && !(type == ir::PTXOperand::b32
				|| type == ir::PTXOperand::u32 
				|| type == ir::PTXOperand::s32
				|| type == ir::PTXOperand::f32)) {
				return "sust.p - data type must be .b32, .u32, .s32, or .f32";
			}
			else if (formatMode == Unformatted && !(type == ir::PTXOperand::b8
				|| type == ir::PTXOperand::b16 
				|| type == ir::PTXOperand::b32
				|| type == ir::PTXOperand::b64)) {
				return "sust.b - data type must be .b8, .b16, .b32, or .b64";
			}
			break;
		}
		case Suq: {
			if (!(surfaceQuery == Width || surfaceQuery == Height
				|| surfaceQuery == Depth)) {
				return "surface query must be .width, .height, or .depth";
			}
			if (type != ir::PTXOperand::b32) {
				return "data type must be .b32";
			}
			break;
		}
		case Sured: {
			if (!(reductionOperation == ReductionAdd
				|| reductionOperation == ReductionMin 
				|| reductionOperation == ReductionMax
				|| reductionOperation == ReductionAnd
				|| reductionOperation == ReductionOr)) {
				return "reduction operation must be .add, .min, .max, .and, or .or";
			}
			if (!(type == ir::PTXOperand::u32 || type == ir::PTXOperand::u64 
				|| type == ir::PTXOperand::s32 || type == ir::PTXOperand::b32)) {
				return "data type must be .u32, .u64, .s32, or .b32";
			}
			break;
		}
		case Sust: {
			if (formatMode == Formatted && !(type == ir::PTXOperand::b32
				|| type == ir::PTXOperand::u32 
				|| type == ir::PTXOperand::s32
				|| type == ir::PTXOperand::f32)) {
				return "sust.p - data type must be .b32, .u32, .s32, or .f32";
			}
			else if (formatMode == Unformatted && !(type == ir::PTXOperand::b8
				|| type == ir::PTXOperand::b16 
				|| type == ir::PTXOperand::b32
				|| type == ir::PTXOperand::b64)) {
				return "sust.b - data type must be .b8, .b16, .b32, or .b64";
			}
			break;
		}
		case Trap: {
			break;
		}
		case Vote: {
			if( vote != Ballot && d.type != PTXOperand::pred ) {
				return "destination must be a predicate";
			}
			else if( vote == Ballot &&
				!PTXOperand::valid(d.type, PTXOperand::b32) ) {
				return "destination must be assignable to b32 for .ballot";
			}
			if( a.type != PTXOperand::pred ) {
				return "operand A must be a predicate";
			}
			break;
		}
		case Xor: {
			if( !( type == PTXOperand::b16 || type == PTXOperand::b32 || 
				type == PTXOperand::b64 || type == PTXOperand::pred ) ) {
				return "invalid instruction type " 
					+ PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, a.type )  ) {
				return "operand A type " + PTXOperand::toString( a.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, d.type )  ) {
				return "operand D type " + PTXOperand::toString( d.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			if( !PTXOperand::valid( type, b.type )  ) {
				return "operand B type " + PTXOperand::toString( b.type ) 
					+ " cannot be assigned to " + PTXOperand::toString( type );
			}
			break;
		}
		default: return "check not implemented for " + toString(opcode); break;
	}
	return "";
}

std::string ir::PTXInstruction::guard() const {
	switch( pg.condition ) {
		case PTXOperand::PT: return ""; break;
		case PTXOperand::nPT: return "@!%pt "; break;
		case PTXOperand::InvPred: // return "@!" + pg.toString() + " "; break;
		case PTXOperand::Pred: return "@" + pg.toString() + " " ; break;
	}
	return "";
}

std::string ir::PTXInstruction::toString() const {
	switch (opcode) {
		case Abs: {
			return guard() + "abs." + modifierString(modifier, carry)
				+ PTXOperand::toString( type ) + " " 
				+ d.toString() + ", " + a.toString();
		}
		case Add: {
			std::string result = guard() + "add.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " 
					+ d.toString() + ", " + a.toString() + ", " 
					+ b.toString();
			return result;
		}
		case AddC: {
			std::string result = guard() + "addc.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " 
					+ d.toString() + ", " + a.toString() + ", " 
					+ b.toString();
			return result;
		}
		case And: {
			return guard() + "and." + PTXOperand::toString( type ) + " " 
					+ d.toString() + ", " + a.toString() + ", " 
					+ b.toString();
		}
		case Atom: {
			std::string result = guard() + "atom." + toString( addressSpace ) 
				+ "." + toString( atomicOperation ) + "." 
				+ PTXOperand::toString( type ) + " "
				+ d.toString() + ", [" + a.toString() + "], " 
				+ b.toString();
			if( c.addressMode != PTXOperand::Invalid ) {
				result += ", " + c.toString();
			}
			return result;
		}
		case Bar: {
			std::string result = guard() + "bar." + toString(barrierOperation);
			ir::PTXOperand ir::PTXInstruction::* instrMembers [] = { 
				&ir::PTXInstruction::d, 
				&ir::PTXInstruction::a, 
				&ir::PTXInstruction::b, 
				&ir::PTXInstruction::c,
				&ir::PTXInstruction::pq
			};
			switch (barrierOperation) {
				case BarReduction: 
				{
					result += toString(reductionOperation)
						+ PTXOperand::toString(type);
				}
				break;
				default: break;
			};
			for (int i = 0; instrMembers[i] != &ir::PTXInstruction::pq; i++) {
				if ((this->*instrMembers[i]).addressMode != ir::PTXOperand::Invalid) {
					result += " " + (this->*instrMembers[i]).toString();
				}
			}
			return result;
		}
        case Bfe: {
			return guard() + "bfe." + PTXOperand::toString( type ) + " " 
				+ d.toString() + ", " + a.toString()
				+ ", " + b.toString() + ", " + c.toString();
		}
		case Bfi: {
			return guard() + "bfi." + PTXOperand::toString( type ) + " " 
				+ d.toString() + ", " + pq.toString() + ", " + a.toString()
				+ ", " + b.toString() + ", " + c.toString();
		}
		case Bfind: {
			std::string result = guard() + "bfind.";
			if( shiftAmount ) result += "shiftamt.";
			result += PTXOperand::toString( type ) + " " 
				+ d.toString() + ", " + a.toString();
			return result;
		}
		case Bra: {
			std::string result = guard() + "bra";
			if( uni ) {
				result += ".uni";
			}
			return result + " " + d.toString();
		}
		case Brev: {
			return guard() + "brev." + PTXOperand::toString( type ) + " " 
				+ d.toString() + ", " + a.toString();
		}
		case Brkpt: {
			return guard() + "brkpt";
		}
		case Call: {
			std::string result = guard() + "call";
			if( uni ) {
				result += ".uni";
			}
			else if( tailCall ) {
				result += ".tail";
			}
			result += " ";
			if( d.addressMode != PTXOperand::Invalid ) {
				result += d.toString() + ", ";
			}
			result += a.toString();
			if( b.addressMode != PTXOperand::Invalid ) {
				result += ", " + b.toString();
			}
			if( a.addressMode == PTXOperand::Register ) {
				result += ", " + c.toString();
			}
			return result;
		}
		case Clz: {
			return guard() + "clz." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString();
		}
		case CNot: {
			return guard() + "cnot." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString();
		}
		case CopySign: {
			return guard() + "copysign." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString();
		}
		case Cos: {
			return guard() + "cos." + modifierString( modifier, carry )
				+ PTXOperand::toString( type ) + " " + d.toString()
				+ ", " + a.toString();
		}
		case Cvt: {
			std::string result = guard() + "cvt.";
			if( PTXOperand::isFloat( d.type )) {
				if ((d.type == PTXOperand::f32 && a.type == PTXOperand::f64) 
					|| PTXOperand::isInt(a.type)) {
					result += modifierString( modifier, carry );
				}
			}
			else {
				if( modifier & rn ) {
					result += "rn.";
				} else if( modifier & rz ) {
					result += "rz.";
				} else if( modifier & rm ) {
					result += "rm.";
				} else if( modifier & rp ) {
					result += "rp.";
				} else if( modifier & rni ) {
          result += "rni.";
        } else if( modifier & rzi ) {
          result += "rzi.";
        } else if( modifier & rmi ) {
          result += "rmi.";
        } else if( modifier & rpi ) {
          result += "rpi.";
        }

				if( modifier & ftz ) {
					result += "ftz.";
				}
				if( modifier & sat ) {
					result += "sat.";
				}
			}
			
			PTXOperand::DataType sourceType = a.type;
	
			if( a.relaxedType != PTXOperand::TypeSpecifier_invalid ) {
				sourceType = a.relaxedType;
			}
			
			result += PTXOperand::toString( type ) + "." 
				+ PTXOperand::toString( sourceType ) + " " + d.toString() + ", " 
				+ a.toString();
			return result;
		}
		case Cvta: {
			std::string result = guard() + "cvta.";
			
			if (toAddrSpace) {
				result += "to.";
			}
			
			result += toString(addressSpace)
				+ "." + PTXOperand::toString(type) + " " + d.toString() + ", "
				+ a.toString();
			return result;
		}
		case Div: {
			std::string result = guard() + "div.";
			if( divideFull ) {
				result += "full.";
			}
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " + d.toString() + ", " 
				+ a.toString() + ", " + b.toString();
			return result;
		}
		case Ex2: {
			std::string result = guard() + "ex2.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " + d.toString() 
				+ ", " + a.toString();
			return result;
		}
		case Exit: {
			return "exit";
		}
		case Fma: {
			std::string result = guard() + "fma."
				+ modifierString(modifier, carry)
				+ PTXOperand::toString(type) + " " + d.toString() + ", "
				+ a.toString() + ", " + b.toString() + ", " + c.toString();
			return result;
		}
		case Isspacep: {
			std::string result = guard() + "isspacep." + toString(addressSpace) 
				+ " " + d.toString() + ", " + a.toString();
			return result;
		}
		case Ld: {
			std::string result = guard() + "ld.";
			if( volatility == Volatile ) {
				result += "volatile.";
			}
			if( cacheOperation != Ca ) {
				result += toStringLoad(cacheOperation) + ".";
			}
			if( addressSpace != Generic ) {
				result += toString(addressSpace) + ".";
			}
			if( d.vec != PTXOperand::v1 ) {
				result += toString( d.vec ) + ".";
			}
			result += PTXOperand::toString( type ) + " " + d.toString() + ", [" 
				+ a.toString() + "]";
			return result;
		}
		case Ldu: {
			std::string result = guard() + "ldu.";
			if( volatility == Volatile ) {
				result += "volatile.";
			}
			if( cacheOperation != Ca ) {
				result += toStringLoad(cacheOperation) + ".";
			}
			if( addressSpace != Generic ) {
				result += toString(addressSpace) + ".";
			}
			if( d.vec != PTXOperand::v1 ) {
				result += toString( d.vec ) + ".";
			}
			result += PTXOperand::toString( type ) + " " + d.toString() + ", [" 
				+ a.toString() + "]";
			return result;
		}
		case Lg2: {
			std::string result = guard() + "lg2."; 
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " + d.toString() 
				+ ", " + a.toString();
			return result;
		}
		case Mad24: {
			std::string result = guard() + "mad24.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " + d.toString() + ", " 
				+ a.toString() + ", " + b.toString() + ", " + c.toString();
			return result;
		}
		case Mad: {
			std::string result = guard() + "mad.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " + d.toString() + ", " 
				+ a.toString() + ", " + b.toString() + ", " + c.toString();
			return result;
		}
		case MadC: {
			std::string result = guard() + "madc.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " + d.toString() + ", " 
				+ a.toString() + ", " + b.toString() + ", " + c.toString();
			return result;
		}
		case Max: {
			return guard() + "max." + modifierString(modifier, carry)
				+ PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString();
		}
		case Membar: {
			return guard() + "membar." + toString( level );
		}
		case Min: {
			return guard() + "min." + modifierString(modifier, carry)
				+ PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString();
		}
		case Mov: {
			return guard() + "mov." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString();
		}
		case Mul24: {
			std::string result = guard() + "mul24.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " + d.toString() + ", " 
				+ a.toString() + ", " + b.toString();
			return result;
		}
		case Mul: {
			std::string result = guard() + "mul.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " + d.toString() + ", " 
				+ a.toString() + ", " + b.toString();
			return result;
		}
		case Neg: {
			return guard() + "neg." + modifierString(modifier, carry)
				+ PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString();
		}
		case Not: {
			return guard() + "not." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString();
		}
		case Pmevent: {
			return guard() + "pmevent." + toString( level );
		}
		case Popc: {
			return guard() + "popc." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString();
		}
		case Prefetch: {
			return guard() + "prefetch." + toString(addressSpace) + "." + 
				PTXInstruction::toString(cacheLevel) + " [" + d.toString() + "]";
		}
		case Prefetchu: {
			return guard() + "prefetchu.L1 [" + d.toString() + "]";
		}
		case Prmt: {
			std::string result = guard() + "prmt." 
				+ PTXOperand::toString( type );
			if( permuteMode != DefaultPermute )
			{
				result += "." + toString( permuteMode );
			}
			result += " " + d.toString() + ", " + a.toString() + ", " 
				+ b.toString() + ", " + c.toString();
			return result;
		}
		case Or: {
			return guard() + "or." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString();
		}
		case Rcp: {
			std::string result = guard() + "rcp.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString();
			return result;
		}
		case Red: {
			return guard() + "red." + toString( addressSpace ) + "." 
				+ toString( reductionOperation ) + "." 
				+ PTXOperand::toString( type ) + " " + d.toString() + ", " 
				+ a.toString();
		}
		case Rem: {
			return guard() + "rem." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString();
		}
		case Ret: {
			std::string result = guard() + "ret";
			if( uni ) {
				result += ".uni";
			}
			return result;
		}
		case Rsqrt: {
			std::string result = guard() + "rsqrt.";
			result += modifierString( modifier, carry );			
			result += PTXOperand::toString( type ) + " " + d.toString() 
				+ ", " + a.toString();			
			return result;
		}
		case Sad: {
			return guard() + "sad." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString() 
				+ ", " + c.toString();
		}
		case SelP: {
			return guard() + "selp." + PTXOperand::toString( type ) 
				+ " " + d.toString() + ", " + a.toString() + ", " + b.toString()
				+ ", " + c.toString();
		}
		case Set: {
			std::string result = guard() + "set." 
				+ toString( comparisonOperator ) + ".";
			if( c.addressMode != PTXOperand::Invalid ) {
				result += toString( booleanOperator ) + ".";
			}
			if( ftz & modifier ) result += "ftz.";
			result += PTXOperand::toString( type ) + "." 
				+ PTXOperand::toString( a.type ) + " " + d.toString() 
				+ ", " + a.toString() + ", " + b.toString();
			if( c.addressMode != PTXOperand::Invalid ) {
				result += ", " + c.toString();
			}
			return result;				
		}
		case SetP: {
			std::string result = guard() + "setp." 
				+ toString( comparisonOperator ) + ".";
			if( c.addressMode != PTXOperand::Invalid ) {
				result += toString( booleanOperator ) + ".";
			}
			result += PTXOperand::toString( type ) + " " + d.toString();
			if( pq.addressMode != PTXOperand::Invalid && !isPt( pq ) ) {
				result += "|" + pq.toString();
			}
			result += ", " + a.toString() + ", " + b.toString();
			if( c.addressMode != PTXOperand::Invalid ) {
				result += ", " + c.toString();
			}
			return result;
		}
		case Shfl: {
			std::string result = guard() + "shl." + toString( shuffleMode )
				+ "." +	PTXOperand::toString( type ) + " " + d.toString();

			if( pq.addressMode != PTXOperand::Invalid && !isPt( pq ) ) {
				result += "|" + pq.toString();
			}
				
			result += ", " + a.toString() + ", " + b.toString() + ", " +
				c.toString();
		
			return result;
		}
		case Shl: {
			return guard() + "shl." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString();
		}
		case Shr: {
			return guard() + "shr." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString();
		}
		case Sin: {
			std::string result = guard() + "sin.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " + d.toString() 
				+ ", " + a.toString();
			return result;
		}
		case SlCt: {
			return guard() + "slct." + modifierString( modifier, carry )
				+ PTXOperand::toString( type ) + "." 
				+ PTXOperand::toString( c.type ) + " " + d.toString() + ", " 
				+ a.toString() + ", " + b.toString() + ", " + c.toString();
		}
		case Sqrt: {
			std::string result = guard() + "sqrt.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " " + d.toString() 
				+ ", " + a.toString();
			return result;
		}
		case St: {
			std::string result = guard() + "st.";
			if( volatility == Volatile ) {
				result += "volatile.";
			}
			if( cacheOperation != Wb ) {
				result += toStringStore(cacheOperation) + ".";
			}
			if( addressSpace != Generic ) {
				result += toString(addressSpace) + ".";
			}
			if( a.vec != PTXOperand::v1 ) {
				result += toString( a.vec ) + ".";
			}
			result += PTXOperand::toString( type ) + " [" + d.toString() + "], "
				+ a.toString();
			return result;
		}
		case Sub: {
			std::string result = guard() + "sub.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString();
			return result;
		}
		case SubC: {
			std::string result = guard() + "subc.";
			result += modifierString( modifier, carry );
			result += PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString();
			return result;
		}
		case Suld: {
			return guard() + "suld" + toString(formatMode) + "." 
				+ toString(geometry) + "." + toString(vec) + "."
				+ PTXOperand::toString(type) + toString(clamp) +
				" " + d.toString() + ", [" + a.toString() + ", " 
				+ b.toString() + "]";
		}
		case Suq: {
			return guard() + "suq." + toString( surfaceQuery ) 
				+ "." + PTXOperand::toString(type) + " " + d.toString() 
				+ ", [" + a.toString() + "]";
		}
		case Sured: {
			return guard() + "sured" + toString(formatMode) + "." 
				+ toString(reductionOperation) + "." +
				toString(geometry) + "." + PTXOperand::toString(type) 
				+  toString(clamp) + " [" + d.toString() + ", " + a.toString()
				+ "], " + b.toString();
		}
		case Sust: {			
			return guard() + "sust" + toString(formatMode) + "." 
				+ toString(geometry) + "." 
				+ ((vec != Vec::v1)?toString(vec) + ".":"") + PTXOperand::toString(type) 
				+ toString(clamp) + " [" + d.toString() +", " + a.toString() 
				+ "], " + b.toString();
		}
		case TestP: {
			return guard() + "testp." + toString( floatingPointMode ) 
				+ "." + PTXOperand::toString( type ) + " " + d.toString() + ", " 
				+ a.toString();
		}
		case Tex: {			
			return guard() + "tex." + toString( geometry ) + ".v4." 
				+ PTXOperand::toString( d.type ) + "." 
				+ PTXOperand::toString( type ) + " " + d.toString() + ", [" 
				+ a.toString() + ", " + c.toString() + "]"; 
		}
		case Tld4: {
			return guard() + "tld4." + toString( colorComponent ) + ".2d.v4." 
				+ PTXOperand::toString( d.type ) + "." 
				+ PTXOperand::toString( c.type ) + " " + d.toString() + ", [" 
				+ a.toString() + ", " + c.toString() + "]"; 
		}
		case Trap: {
			return guard() + "trap";
		}
		case Txq: {
			return guard() + "txq." + toString( surfaceQuery ) 
				+ "." + PTXOperand::toString(type) + " " + d.toString()
				+ ", [" + a.toString() + "]";
		}
		case Vote: {
			return guard() + "vote." + toString( vote ) + "."
				+ PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString();
		}
		case Xor: {
			return guard() + "xor." + PTXOperand::toString( type ) + " "
				+ d.toString() + ", " + a.toString() + ", " + b.toString();
		}
		case Reconverge: {
			return "reconverge";
		}
		default: break;
	}
	assertM(false, "Instruction opcode " << toString(opcode) 
		<< " not implemented.");
		
	return "";
}

ir::Instruction* ir::PTXInstruction::clone(bool copy) const {
	if (copy) {
		return new PTXInstruction(*this);
	}
	else {
		return new PTXInstruction;
	}
}

bool ir::PTXInstruction::isBranch() const {
	return opcode == Bra || opcode == Call;
}

bool ir::PTXInstruction::isCall() const {
	return opcode == Call;
}

bool ir::PTXInstruction::isLoad() const {
	return opcode == Ld || opcode == Ldu;
}

bool ir::PTXInstruction::isStore() const {
	return opcode == St;
}

bool ir::PTXInstruction::mayHaveAddressableOperand() const {
	return opcode == Mov || opcode == Ld || opcode == St || opcode == Cvta
		|| opcode == Atom || opcode == Ldu;
}

bool ir::PTXInstruction::mayHaveRelaxedTypeDestination() const {
	return opcode == Ld || opcode == Cvt
		|| opcode == Ldu || opcode == SelP;
}

bool ir::PTXInstruction::hasSideEffects() const {
	return opcode == St || opcode == Atom || opcode == Bar
		|| opcode == Bra || opcode == Call || opcode == Exit || opcode == Ldu
		|| opcode == Membar || opcode == Tex || opcode == Tld4
		|| opcode == Prefetch || opcode == Sust || opcode == Suq
		|| opcode == Trap || opcode == Reconverge || opcode == Ret;
}

bool ir::PTXInstruction::canObserveSideEffects() const {
	if (opcode == Atom) return true;

	if (opcode == Ld) {
		if (volatility == Volatile) {
			return true;
		}
		
		if (cacheOperation == Cv || cacheOperation == Cg) {
			return true;
		}
		
		return false;
	}
	
	return false;
}

bool ir::PTXInstruction::isMemoryInstruction() const {
	return opcode == St || opcode == Atom
		|| opcode == Ldu
		|| opcode == Tex || opcode == Tld4
		|| opcode == Prefetch || opcode == Sust || opcode == Ld;
}

bool ir::PTXInstruction::isExit() const {
	return opcode == Exit || opcode == Ret;
}


