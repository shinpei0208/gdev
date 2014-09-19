/*! \file PTXToILTranslator.cpp
 *  \author Rodrigo Dominguez <rdomingu@ece.neu.edu>
 *  \date April 19, 2010
 *  \brief The implementation file for the PTX to IL Translator class.
 */

// Ocelot includes
#include <ocelot/translator/interface/PTXToILTranslator.h>
#include <ocelot/ir/interface/PTXKernel.h>

// Hydrazine includes
#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace translator
{
	PTXToILTranslator::PTXToILTranslator(OptimizationLevel l)
		:
			Translator(ir::Instruction::PTX, ir::Instruction::CAL, l),
			_ilKernel(0),
			_temps(_maxRegs)
	{
	}

	ir::Kernel* PTXToILTranslator::translate(const ir::Kernel* k)
	{
		assertM(0, "Translator needs the kernel to be an executable kernel");
		return 0;
	}

	ir::Kernel* PTXToILTranslator::translate(const ATIExecutableKernel* k)
	{
		report("Translating kernel " << k->name);

		assertM(k->ISA == ir::Instruction::PTX, "Kernel must be in PTX");

		_ilKernel = new ir::ILKernel(*k);

		_translateInstructions();
		_addKernelPrefix(k);

 		return _ilKernel;
  	}

	void PTXToILTranslator::_translateInstructions()
	{
		// translate instructions iterating thru the control tree
		
		analysis::ControlTree controlTree;
		
		controlTree.analyze(*_ilKernel);
		
		_translate(controlTree.get_root_node());
	}

	void PTXToILTranslator::_translate(const CT::Node* node)
	{
		report("Translating " << node->label());

		switch (node->rtype())
		{
			case CT::Inst:
			{
				_translate(static_cast<const CT::InstNode*>(node)); 
				break;
			}
			case CT::Block:
			{
				_translate(static_cast<const CT::BlockNode*>(node)); 
				break;
			}
			case CT::IfThen:
			{
				_translate(static_cast<const CT::IfThenNode*>(node)); 
				break;
			}
			case CT::Natural: 
			{
				_translate(static_cast<const CT::NaturalNode*>(node)); 
				break;
			}
			default: assertM(false, "Invalid region type " << node->rtype());
		}
	}

	void PTXToILTranslator::_translate(const CT::InstNode* node)
	{
		for (CT::InstructionList::const_iterator
				ins = node->bb()->instructions.begin(), 
				end = node->bb()->instructions.end() ;
				ins != end ; ins++)
		{
			_translate(static_cast<ir::PTXInstruction &>(**ins));
		}
	}

	void PTXToILTranslator::_translate(const CT::BlockNode* node)
	{
		for (CT::NodeList::const_iterator child = node->children().begin(),
				end = node->children().end() ; child != end ; child++)
		{
			_translate(*child);
		}	
	}

	ir::PTXInstruction* getLastIns(const CT::Node* node)
	{
		switch (node->rtype())
		{
			case CT::Inst:
			{
				const CT::InstNode* inode = 
					static_cast<const CT::InstNode*>(node);

				return static_cast<ir::PTXInstruction*>(
						inode->bb()->instructions.back());
			}
			case CT::Block:
			{
				const CT::BlockNode* bnode =
					static_cast<const CT::BlockNode*>(node);

				return getLastIns(bnode->children().back());
			}
			case CT::IfThen:
			{
				const CT::IfThenNode* ifnode =
					static_cast<const CT::IfThenNode*>(node);

				return getLastIns(ifnode->ifTrue());

			}
			default: assertM(false, "Invalid region type " << node->rtype());
		}
		
		return 0;
	}

	void PTXToILTranslator::_translate(const CT::IfThenNode* node)
	{
		// translate condition
		assertM(node->cond()->rtype() == CT::Inst || 
				node->cond()->rtype() == CT::Block,
				"Invalid condition node " << node->cond()->rtype());
		_translate(node->cond());

		// translate branch
		ir::PTXInstruction* bra = getLastIns(node->cond());
		assertM(bra->opcode == ir::PTXInstruction::Bra, "Invalid instruction");
		_translateBra(*bra);
		
		// translate then
		_translate(node->ifTrue());

		// translate else (if necessary)
		if (node->ifFalse() != NULL)
		{
			_add(ir::ILElse());
			_translate(node->ifFalse());
		}

		// done!
		_add(ir::ILEndIf());
	}

	void PTXToILTranslator::_translate(const CT::NaturalNode* node)
	{
		// translate while
		_add(ir::ILWhileLoop());

		// translate body (except last block)
		CT::NodeList::const_iterator last = (--node->children().end());
		for (CT::NodeList::const_iterator child = node->children().begin() ; 
				child != last ; child++)
		{
			// the fall-through edge should be the next node in the loop
			CT::NodeList::const_iterator next(child); next++;
			assertM((*child)->fallthrough() == *next, "Invalid Natural loop");
			_translate(*child);

			// translate optional side exit (invert logic)
			ir::PTXInstruction* bra = getLastIns(*child);
			if (bra->opcode == ir::PTXInstruction::Bra &&
					(bra->pg.condition == ir::PTXOperand::Pred ||
					 bra->pg.condition == ir::PTXOperand::InvPred))
			{
				bra->pg.condition = ir::PTXOperand::InvPred;
				_translateBra(*bra);
				_add(ir::ILBreak());
				_add(ir::ILEndIf());
			}
		}

		// the fall-through edge should be the head node of the loop
		assert((*last)->fallthrough() != *(node->children().begin()));

		// translate last block
		_translate(*last);

		// translate optional side exit 
		ir::PTXInstruction* bra = getLastIns(*last);
		if (bra->opcode == ir::PTXInstruction::Bra &&
				(bra->pg.condition == ir::PTXOperand::Pred ||
				 bra->pg.condition == ir::PTXOperand::InvPred))
		{
			_translateBra(*bra);
			_add(ir::ILBreak());
			_add(ir::ILEndIf());
		}

		// done!
		_add(ir::ILEndLoop());
	}

	void PTXToILTranslator::_translate(const ir::PTXInstruction &i)
	{
		report("Translating: " << i.toString());
		switch (i.opcode) 
		{
 			case ir::PTXInstruction::Abs:    _translateAbs(i);    break;
 			case ir::PTXInstruction::Add:    _translateAdd(i);    break;
			case ir::PTXInstruction::And:    _translateAnd(i);    break;
			case ir::PTXInstruction::Atom:   _translateAtom(i);   break;
			case ir::PTXInstruction::Bar:    _translateBar(i);    break;
			case ir::PTXInstruction::Bra:    /* control tree */   break;
			case ir::PTXInstruction::Clz:    _translateClz(i);    break;
			case ir::PTXInstruction::Cos:    _translateCos(i);    break;
 			case ir::PTXInstruction::Cvt:    _translateCvt(i);    break;
			case ir::PTXInstruction::Div:    _translateDiv(i);    break;
			case ir::PTXInstruction::Ex2:    _translateEx2(i);    break;
 			case ir::PTXInstruction::Exit:   _translateExit(i);   break;
			case ir::PTXInstruction::Fma:    _translateFma(i);    break;
 			case ir::PTXInstruction::Ld:     _translateLd(i);     break;
 			case ir::PTXInstruction::Ldu:    _translateLdu(i);    break;
			case ir::PTXInstruction::Lg2:    _translateLg2(i);    break;
			case ir::PTXInstruction::Mad:    _translateMad(i);    break;
			case ir::PTXInstruction::Max:    _translateMax(i);    break;
			case ir::PTXInstruction::Membar: _translateMembar(i); break;
			case ir::PTXInstruction::Min:    _translateMin(i);    break;
			case ir::PTXInstruction::Mov:    _translateMov(i);    break;
 			case ir::PTXInstruction::Mul:    _translateMul(i);    break;
			case ir::PTXInstruction::Mul24:  _translateMul24(i);  break;
			case ir::PTXInstruction::Neg:    _translateNeg(i);    break;
			case ir::PTXInstruction::Not:    _translateNot(i);    break;
			case ir::PTXInstruction::Or:     _translateOr(i);     break;
			case ir::PTXInstruction::Popc:   _translatePopc(i);   break;
			case ir::PTXInstruction::Rcp:    _translateRcp(i);    break;
			case ir::PTXInstruction::Rem:    _translateRem(i);    break;
			case ir::PTXInstruction::Rsqrt:  _translateRsqrt(i);  break;
			case ir::PTXInstruction::SelP:   _translateSelP(i);   break;
			case ir::PTXInstruction::Set:    _translateSet(i);    break;
			case ir::PTXInstruction::SetP:   _translateSetP(i);   break;
			case ir::PTXInstruction::Sin:    _translateSin(i);    break;
			case ir::PTXInstruction::Shl:    _translateShl(i);    break;
			case ir::PTXInstruction::Shr:    _translateShr(i);    break;
			case ir::PTXInstruction::SlCt:   _translateSlct(i);   break;
			case ir::PTXInstruction::Sqrt:   _translateSqrt(i);   break;
 			case ir::PTXInstruction::St:     _translateSt(i);     break;
			case ir::PTXInstruction::Sub:    _translateSub(i);    break;
			case ir::PTXInstruction::Vote:   _translateVote(i);   break;
			case ir::PTXInstruction::Xor:    _translateXor(i);    break;
			default:
			{
				assertM(0, "Opcode \"" << i.toString() << "\" not supported");
			}
		}
	}

	ir::ILOperand PTXToILTranslator::_translate(const ir::PTXOperand &o)
	{
		ir::ILOperand op;

		switch (o.addressMode)
		{
			case ir::PTXOperand::Register:
			case ir::PTXOperand::Indirect:
			case ir::PTXOperand::BitBucket:
			{
				assertM(o.reg < _maxRegs, "Max number of registers exceeded");

				// look-up register table
				const RegisterMap::const_iterator it = _registers.find(o.reg);
				if (it != _registers.end()) return it->second;

				// create new il register
				ir::ILOperand op(ir::ILOperand::RegType_Temp);
				op.num = _registers.size();
				op = op.x();
				_registers.insert(std::make_pair(o.reg, op));

				return op;
			}
			case ir::PTXOperand::Immediate:
			{
				switch (o.type)
				{
					case ir::PTXOperand::s8:
					case ir::PTXOperand::s32:
					case ir::PTXOperand::u16:
					case ir::PTXOperand::u32:
					case ir::PTXOperand::u64:
					case ir::PTXOperand::b8:
					{
						//TODO Need to check for out-of-range literals
						op = _translateLiteral(o.imm_uint);
						break;
					}
					case ir::PTXOperand::f32:
					case ir::PTXOperand::f64:
					{
						union { float f; int i; } convert;
						convert.f = o.imm_float;
						op = _translateLiteral(convert.i);
						break;
					}
					default:
					{
						assertM(false, "Immediate operand type \""
							   << ir::PTXOperand::toString(o.type) 
							   << "\" not supported");
					}
				}
				break;
			}
			case ir::PTXOperand::Special:
			{
				op = _translate(o.special, o.vIndex);
				break;
			}
			default:
			{
				assertM(false, "Address Mode " 
						<< o.toString(o.addressMode) << " not supported");
			}
		}

		return op;
	}

	ir::ILOperand PTXToILTranslator::_translateArrayDst(
			const ir::PTXOperand::Array a)
	{
		// create new il register with 2/4 components
		ir::ILOperand op(ir::ILOperand::RegType_Temp);
		op.num = _registers.size();

		switch (a.size())
		{
			case 2: 
			{
				_registers.insert(std::make_pair(a[0].reg, op.x()));
				_registers.insert(std::make_pair(a[1].reg, op.y()));
				op = op.xy();
				break;
			}
			case 4:
			{
				_registers.insert(std::make_pair(a[0].reg, op.x()));
				_registers.insert(std::make_pair(a[1].reg, op.y()));
				_registers.insert(std::make_pair(a[2].reg, op.z()));
				_registers.insert(std::make_pair(a[3].reg, op.w()));
				break;
			}
			default:
			{
				assertM(false, "Invalid array size " << a.size());
				break;
			}
		}

		return op;
	}

	ir::ILOperand PTXToILTranslator::_translateArraySrc(
			const ir::PTXOperand::Array a)
	{
		// create new il register with 2/4 components
		ir::ILOperand op(ir::ILOperand::RegType_Temp);
		op.num = _registers.size();

		switch (a.size())
		{
			case 2:
			{
				// look-up register table
				const RegisterMap::const_iterator a0 = _registers.find(a[0].reg);
				const RegisterMap::const_iterator a1 = _registers.find(a[1].reg);

				assert(a0 != _registers.end() && a1 != _registers.end());

				if (a0->second.num == a1->second.num)
				{
					op.num = a0->second.num;
					return op.xy();
				}

				// x component
				{
					ir::ILMov mov;
					mov.d = op.x();
					mov.a = a0->second;
					_add(mov);
				}

				// y component
				{
					ir::ILMov mov;
					mov.d = op.y();
					mov.a = a1->second;
					_add(mov);
				}

				op = op.xy();

				break;
			}
			case 4:
			{
				// look-up register table
				const RegisterMap::const_iterator a0 = _registers.find(a[0].reg);
				const RegisterMap::const_iterator a1 = _registers.find(a[1].reg);
				const RegisterMap::const_iterator a2 = _registers.find(a[2].reg);
				const RegisterMap::const_iterator a3 = _registers.find(a[3].reg);

				assert(a0 != _registers.end() && a1 != _registers.end() &&
						a2 != _registers.end() && a3 != _registers.end());

				if (a0->second.num == a1->second.num && 
						a2->second.num == a3->second.num &&
						a0->second.num == a2->second.num)
				{
					op.num = a0->second.num;
					return op;
				}

				// x component
				{
					ir::ILMov mov;
					mov.d = op.x();
					mov.a = a0->second;
					_add(mov);
				}

				// y component
				{
					ir::ILMov mov;
					mov.d = op.y();
					mov.a = a1->second;
					_add(mov);
				}

				// z component
				{
					ir::ILMov mov;
					mov.d = op.z();
					mov.a = a2->second;
					_add(mov);
				}

				// w component
				{
					ir::ILMov mov;
					mov.d = op.w();
					mov.a = a3->second;
					_add(mov);
				}
				break;
			}
			default:
			{
				assertM(false, "Invalid array size " << a.size());
				break;
			}
		}

		return op;
	}

	ir::ILOperand PTXToILTranslator::_translateMemMask(unsigned int i)
	{
		ir::ILOperand op(ir::ILOperand::RegType_Generic_Mem);
		op.num = 0;

		switch(i)
		{
			case 1: return op.x();
			case 2: return op.xy();
			case 4: return op;
			default:
			{
				assertM(false, "Invalid memory mask");
				break;
			}
		}
	}

	ir::ILInstruction::DataType PTXToILTranslator::_translate(
			const ir::PTXOperand::DataType d)
	{
		switch(ir::PTXOperand::bytes(d))
		{
			case 1: return ir::ILInstruction::Byte;
			case 2: return ir::ILInstruction::Short;
			case 8: return ir::ILInstruction::Dword;
			default:
			{
				assertM(false, "DataType " << d << " not supported");
			}
		}
		
		return ir::ILInstruction::Byte;
	}

	ir::ILOperand PTXToILTranslator::_tempRegister()
	{
		ir::ILOperand op(ir::ILOperand::RegType_Temp);
		op.num = _temps++;
		return op.x();
	}

	ir::ILOperand PTXToILTranslator::_translate(
			const ir::PTXOperand::SpecialRegister &s,
			const ir::PTXOperand::VectorIndex& d)
	{
		typedef ir::PTXOperand PTXOperand;
		typedef ir::ILOperand ILOperand;

		ir::ILOperand op;

		switch (s)
		{
			case PTXOperand::tid:
			{
				op.type = ILOperand::RegType_Thread_Id_In_Group;
				break;
			}
			case PTXOperand::ntid:
			{
				// Translate to cb0[0]
				op.num = 0;
				op.type = ILOperand::RegType_Const_Buf;
				op.immediate_present = true;
				op.imm = 0;
				break;
			}
			case PTXOperand::ctaId:
			{
				op.type = ILOperand::RegType_Thread_Group_Id;
				break;
			}
			case PTXOperand::nctaId:
			{
				// Translate to cb0[1]
				op.num = 0;
				op.type = ILOperand::RegType_Const_Buf;
				op.immediate_present = true;
				op.imm = 1;
				break;
			}
			default: assertM(false, "Special Register " << s
				<< " not supported");
		}

		// Translate modifier
		switch (d)
		{
			case PTXOperand::ix: op = op.x(); break;
			case PTXOperand::iy: op = op.y(); break;
			case PTXOperand::iz: op = op.z(); break;
			default: assertM(false, "Invalid vector index " << d);
		}

		return op;
	}

	void PTXToILTranslator::_translateAbs(const ir::PTXInstruction &i)
	{
		switch (i.type)
		{
			case ir::PTXOperand::s32:
			{
				// There's no abs IL instruction for integers so we need
				// to use logical operations
				// mov r0x, i.a
				// mov r1x, 31
				// ishr r1x, r0x, r1x
				// iadd r0x, r0x, r1x
				// ixor r0x, r0x, r1x
				// mov i.d, r0x

				ir::ILOperand r0x = _tempRegister();
				ir::ILOperand r1x = _tempRegister();

				// mov r0x, i.a
				{
					ir::ILMov mov;
					mov.d = r0x; mov.a = _translate(i.a);
					_add(mov);
				}

				// mov r1x, 31
				{
					ir::ILMov mov;
					mov.d = r1x; mov.a = _translateLiteral(31);
					_add(mov);
				}

				// ishr r1x, r0x, r1x
				{
					ir::ILIshr ishr;
					ishr.d = r1x; ishr.a = r0x; ishr.b = r1x;
					_add(ishr);
				}

				// iadd r0x, r0x, r1x
				{
					ir::ILIadd iadd;
					iadd.d = r0x; iadd.a = r0x; iadd.b = r1x;
					_add(iadd);
				}

				// ixor r0x, r0x, r1x
				{
					ir::ILIxor ixor;
					ixor.d = r0x; ixor.a = r0x; ixor.b = r1x;
					_add(ixor);
				}

				// mov i.d, r0x
				{
					ir::ILMov mov;
					mov.d = _translate(i.d); mov.a = r0x;
					_add(mov);
				}

				break;
			}
			case ir::PTXOperand::f32:
			{
				ir::ILAbs abs;

				abs.d = _translate(i.d);
				abs.a = _translate(i.a);

				_add(abs);

				break;
			}
			default:
			{
				assertM(false, "Type "
						<< ir::PTXOperand::toString(i.type)
						<< " not supported in "
						<< i.toString());
			}
		}
	}

	void PTXToILTranslator::_translateAdd(const ir::PTXInstruction &i)
	{
		switch (i.type)
		{
			case ir::PTXOperand::s32:
			case ir::PTXOperand::s64:
			case ir::PTXOperand::u16:
			case ir::PTXOperand::u32:
			case ir::PTXOperand::u64:
			{
				ir::ILIadd iadd;

				iadd.a = _translate(i.a);
				iadd.b = _translate(i.b);
				iadd.d = _translate(i.d);

				_add(iadd);

				break;
			}
			case ir::PTXOperand::f32:
			{
				ir::ILAdd add;

				add.a = _translate(i.a);
				add.b = _translate(i.b);
				add.d = _translate(i.d);

				_add(add);

				break;
			}
			default:
			{
				assertM(false, "Type "
						<< ir::PTXOperand::toString(i.type)
						<< " not supported in "
						<< i.toString());
			}
		}
	}

	void PTXToILTranslator::_translateAnd(const ir::PTXInstruction &i)
	{
		ir::ILIand iand;

		iand.a = _translate(i.a);
		iand.b = _translate(i.b);
		iand.d = _translate(i.d);

		_add(iand);

	}

	void PTXToILTranslator::_translateAtom(const ir::PTXInstruction& i)
	{
		switch(i.addressSpace)
		{
			case ir::PTXInstruction::Global:
			{
				switch(i.atomicOperation)
				{
					case ir::PTXInstruction::AtomicAdd:
					{
						assertM(i.a.offset == 0, 
								"Atomic Add from offset not supported");

						ir::ILUav_Read_Add_Id uav_read_add_id;

						uav_read_add_id.d = _translate(i.d);
						uav_read_add_id.a = _translate(i.a);
						uav_read_add_id.b = _translate(i.b);

						_add(uav_read_add_id);

						break;
					}
					case ir::PTXInstruction::AtomicExch:
					{
						assertM(i.a.offset == 0, 
								"Atomic Xchg from offset not supported");

						ir::ILUav_Read_Xchg_Id uav_read_xchg_id;

						uav_read_xchg_id.d = _translate(i.d);
						uav_read_xchg_id.a = _translate(i.a);
						uav_read_xchg_id.b = _translate(i.b);

						_add(uav_read_xchg_id);

						break;
					}
					case ir::PTXInstruction::AtomicMax:
					{
						assertM(i.a.offset == 0, 
								"Atomic Max from offset not supported");

						ir::ILUav_Read_Max_Id uav_read_max_id;

						uav_read_max_id.d = _translate(i.d);
						uav_read_max_id.a = _translate(i.a);
						uav_read_max_id.b = _translate(i.b);

						_add(uav_read_max_id);

						break;
					}
					case ir::PTXInstruction::AtomicMin:
					{
						assertM(i.a.offset == 0, 
								"Atomic Min from offset not supported");

						ir::ILUav_Read_Min_Id uav_read_min_id;

						uav_read_min_id.d = _translate(i.d);
						uav_read_min_id.a = _translate(i.a);
						uav_read_min_id.b = _translate(i.b);

						_add(uav_read_min_id);

						break;
					}
					default:
					{
						assertM(false, "Atomic operation \"" 
								<< i.toString(i.atomicOperation) 
								<< "\" not supported in "
								<< i.toString());
					}
				}
				break;
			}
			case ir::PTXInstruction::Shared:
			{
				switch(i.atomicOperation)
				{
					case ir::PTXInstruction::AtomicAdd:
					{
						assertM(i.a.offset == 0, 
								"Atomic Add from offset not supported");

						ir::ILLds_Read_Add_Id lds_read_add_id;

						lds_read_add_id.d = _translate(i.d).x();
						lds_read_add_id.a = _translate(i.a).x();
						lds_read_add_id.b = _translate(i.b).x();

						_add(lds_read_add_id);

						break;
					}
					default:
					{
						assertM(false, "Atomic operation \"" 
								<< i.atomicOperation 
								<< "\" not supported in "
								<< i.toString());
					}
				}
				break;
			}
			default:
			{
				assertM(false, "Address Space \"" 
						<< ir::PTXInstruction::toString(i.addressSpace)
						<< "\" not supported in "
						<< i.toString());
			}
		}
	}

	void PTXToILTranslator::_translateBar(const ir::PTXInstruction &i)
	{
		ir::ILFence fence;
		fence.threads();
		_add(fence);
	}

	void PTXToILTranslator::_translateBra(const ir::PTXInstruction &i)
	{
		switch (i.pg.condition)
		{
			case ir::PTXOperand::Pred:
			{
				ir::ILIfLogicalZ if_logicalz;
				if_logicalz.a = _translate(i.pg);
				_add(if_logicalz);
				break;
			}
			case ir::PTXOperand::InvPred:
			{
				ir::ILIfLogicalNZ if_logicalnz;
				if_logicalnz.a = _translate(i.pg);
				_add(if_logicalnz);
				break;
			}
			default: assertM(false, "Invalid predicate condition");
		}
	}

	void PTXToILTranslator::_convertSrc(const ir::PTXInstruction &i,
			ir::ILOperand& a)
	{
		// TODO Implement relaxed type-checking rules for source operands
		a = _translate(i.a);
	}

	void PTXToILTranslator::_convert(const ir::PTXInstruction &i, 
			const ir::ILOperand a, ir::ILOperand& d)
	{
		d = _tempRegister();

		switch (i.a.type)
		{
			case ir::PTXOperand::s32:
			{
				switch (i.type)
				{
					case ir::PTXOperand::s8:
					{
						// chop
						ir::ILIand iand;
						iand.d = d;
						iand.a = a;
						iand.b = _translateLiteral(0x000000FF);
						_add(iand);
                        return;
					}
					case ir::PTXOperand::s64:
					case ir::PTXOperand::u64:
					{
						// sext (but there are no 64-bit registers in IL)
						ir::ILMov mov;
						mov.d = d;
						mov.a = a;
						_add(mov);
						return;
					}
					case ir::PTXOperand::f32:
					case ir::PTXOperand::f64:
					{
						// s2f
						if(i.modifier & ir::PTXInstruction::rn)
						{
							ir::ILItoF itof;
							itof.d = d;
							itof.a = a;
							_add(itof);
							return;
						}
						break;
					}
					default: break;
				}
				break;
			}
            case ir::PTXOperand::s64:
            {
                switch (i.type)
                {
                    case ir::PTXOperand::s32:
                    {
						// chop (but there are no 64-bit registers in IL)
						ir::ILMov mov;
						mov.d = d;
						mov.a = a;
						_add(mov);
						return;
                    }
                    default: break;
                }
            }
			case ir::PTXOperand::u8:
			{
				switch (i.type)
				{
					case ir::PTXOperand::u32:
					{
						// zext
						ir::ILMov mov;
						mov.d = d;
						mov.a = a;
						_add(mov);
						return;
					}
					default: break;
				}
				break;
			}
			case ir::PTXOperand::u16:
			{
				switch (i.type)
				{
					case ir::PTXOperand::s32:
					case ir::PTXOperand::u32:
					case ir::PTXOperand::u64:
					{
						// zext
						ir::ILMov mov;
						mov.d = d;
						mov.a = a;
						_add(mov);
						return;
					}
					default: break;
				}
				break;
			}
			case ir::PTXOperand::u32:
			{
				switch (i.type)
				{
					case ir::PTXOperand::s32:
					{
						// do nothing
						ir::ILMov mov;
						mov.d = d;
						mov.a = a;
						_add(mov);
						return;
                    }
					case ir::PTXOperand::s64:
					{
						// zext
						ir::ILMov mov;
						mov.d = d;
						mov.a = a;
						_add(mov);
						return;
					}
					case ir::PTXOperand::u8:
					{
						// chop
						ir::ILIand iand;
						iand.d = d;
						iand.a = a;
						iand.b = _translateLiteral(0x000000FF);
						_add(iand);
						return;
					}
					case ir::PTXOperand::u16:
					{
						// chop
						ir::ILIand iand;
						iand.d = d;
						iand.a = a;
						iand.b = _translateLiteral(0x0000FFFF);
						_add(iand);
						return;
					}
					case ir::PTXOperand::u64:
					{
						// zext
						ir::ILMov mov;
						mov.d = d;
						mov.a = a;
						_add(mov);
						return;
					}
					case ir::PTXOperand::f32:
					{
						// u2f
						if(i.modifier & ir::PTXInstruction::rn
								|| i.modifier & ir::PTXInstruction::rz)
						{
							ir::ILUtoF utof;
							utof.d = d;
							utof.a = a;
							_add(utof);
							return;
						}
						break;
					}
					default: break;
				}
				break;
			}
			case ir::PTXOperand::u64:
			{
				switch (i.type)
				{
					case ir::PTXOperand::s32:
					case ir::PTXOperand::u32:
					{
						// chop (but there are no 64-bit registers in IL)
						ir::ILMov mov;
						mov.d = d;
						mov.a = a;
						_add(mov);
						return;
					}
					default: break;
				}
				break;
			}
			case ir::PTXOperand::f32:
			{
				switch (i.type)
				{
					case ir::PTXOperand::u32:
					{
						// f2u
						ir::ILFtoU ftou;
						ftou.d = d;
						ftou.a = a;
						_add(ftou);
						return;
					}
					case ir::PTXOperand::s32:
					{
						// f2i
						ir::ILFtoI ftoi;
						ftoi.d = d;
						ftoi.a = a;
						_add(ftoi);
						return;
					}
					case ir::PTXOperand::f32: 
					{
						if (i.modifier & ir::PTXInstruction::sat) 
						{
							ir::ILMov mov;
							mov.d = d.clamp();
							mov.a = a;
							_add(mov);
							return;
						}

						if(i.modifier & ir::PTXInstruction::rz) 
						{
              // REVISE - is this ::rz or ::rzi?
							ir::ILMov mov;
							mov.d = d;
							mov.a = a;
							_add(mov);
							return;
						}

						if (i.modifier & ir::PTXInstruction::rn)
						{
              // REVISE - is this ::rn or ::rni?
							ir::ILRound_Nearest round_nearest;
							round_nearest.d = d;
							round_nearest.a = a;
							_add(round_nearest);
							return;
						}
						if (i.modifier & ir::PTXInstruction::rm)
						{
              // REVISE - is this ::rm or ::rmi?
							ir::ILRound_Neginf round_neginf;
							round_neginf.d = d;
							round_neginf.a = a;
							_add(round_neginf);
							return;
						}
						break;
					}
					case ir::PTXOperand::f64: 
					{
						ir::ILMov mov;
						mov.d = d;
						mov.a = a;
						_add(mov);
						return;
					}
					default: break;
				}
				break;
			}
			default: break;
		}

		assertM(false, "Opcode \"" << i.toString() << "\" not supported");
	}

	void PTXToILTranslator::_convertDst(const ir::PTXInstruction &i,
			const ir::ILOperand d)
	{
		switch (i.type)
		{
			case ir::PTXOperand::s8:
			{
				switch (i.d.type)
				{
					case ir::PTXOperand::u32:
					{
						// sext
						ir::ILMov mov;
						mov.d = _translate(i.d);
						mov.a = d;
						_add(mov);
						return;
					}
					default: break;
				}
				break;
			}
			case ir::PTXOperand::s32:
			{
				switch (i.d.type)
				{
					case ir::PTXOperand::u32:
					{
						// allowed, no conversion needed
						ir::ILMov mov;
						mov.d = _translate(i.d);
						mov.a = d;
						_add(mov);
						return;
					}
                    case ir::PTXOperand::u64:
                    {
						// sext (but there are no 64-bit registers in IL)
						ir::ILMov mov;
						mov.d = _translate(i.d);
						mov.a = d;
						_add(mov);
						return;
                    }
					default: break;
				}
				break;
			}
			case ir::PTXOperand::s64:
			{
				switch (i.d.type)
				{
					case ir::PTXOperand::u64: 
					{
						// allowed, no conversion needed
						ir::ILMov mov;
						mov.d = _translate(i.d);
						mov.a = d;
						_add(mov);
						return;
					}
					default: break;
				}
				break;
			}
			case ir::PTXOperand::u8:
			{
				switch (i.d.type)
				{
					case ir::PTXOperand::u32:
					{
						// zext
						ir::ILMov mov;
						mov.d = _translate(i.d);
						mov.a = d;
						_add(mov);
						return;
					}
					default: break;
				}
				break;
			}
			case ir::PTXOperand::u16:
			{
				switch (i.d.type)
				{
					case ir::PTXOperand::u16:
					{
						// allowed, no conversion needed
						ir::ILMov mov;
						mov.d = _translate(i.d);
						mov.a = d;
						_add(mov);
						return;
					}
					case ir::PTXOperand::u32:
					{
						// zext
						ir::ILMov mov;
						mov.d = _translate(i.d);
						mov.a = d;
						_add(mov);
						return;
					}
					default: break;
				}
				break;
			}
			case ir::PTXOperand::u32:
			{
				switch (i.d.type)
				{
					case ir::PTXOperand::u32:
					{
						// allowed, no conversion needed
						ir::ILMov mov;
						mov.d = _translate(i.d);
						mov.a = d;
						_add(mov);
						return;
					}
					default: break;
				}
				break;
			}
			case ir::PTXOperand::u64:
			{
				switch (i.d.type)
				{
					case ir::PTXOperand::u64:
					{
						// allowed, no conversion needed
						ir::ILMov mov;
						mov.d = _translate(i.d);
						mov.a = d;
						_add(mov);
						return;
					}
					default: break;
				}
				break;
			}
			case ir::PTXOperand::f32:
			{
				switch (i.d.type)
				{
					case ir::PTXOperand::f32:
					{
						// allowed, no conversion needed
						ir::ILMov mov;
						mov.d = _translate(i.d);
						mov.a = d;
						_add(mov);
						return;
					}
					default: break;
				}
				break;
			}
			case ir::PTXOperand::f64:
			{
				switch (i.d.type)
				{
					case ir::PTXOperand::f64:
					{
						// allowed, no conversion needed
						ir::ILMov mov;
						mov.d = _translate(i.d);
						mov.a = d;
						_add(mov);
						return;
					}
					default: break;
				}
				break;
			}
			default: break;
		}

		assertM(false,
				"Destination operand conversion from \""
				<< ir::PTXOperand::toString(i.type)
				<< "\" to \""
				<< ir::PTXOperand::toString(i.d.type)
				<< "\" not supported");
	}

	void PTXToILTranslator::_translateClz(const ir::PTXInstruction &i)
	{
		// clz returns [0, 32]
		// ffb_hi returns [0, 31] and -1
		//
		// ffb_hi r0, i.a
		// ushr r1, r0, 31
		// cmov_logical i.d, r1, 32, r0
		
		ir::ILOperand r0 = _tempRegister();
		ir::ILOperand r1 = _tempRegister();

		// ffb_hi r0, i.a
		{
			ir::ILFfb_Hi ffb_hi;
			ffb_hi.d = r0; ffb_hi.a = _translate(i.a);
			_add(ffb_hi);
		}

		// ushr r1, r0, 31
		{
			ir::ILUshr ushr;
			ushr.d = r1; ushr.a = r0; ushr.b = _translateLiteral(31);
			_add(ushr);
		}

		// cmov_logical i.d, r1, 32, r0
		{
			ir::ILCmov_Logical cmov_logical;
			cmov_logical.d = _translate(i.d); 
			cmov_logical.a = r1; 
			cmov_logical.b = _translateLiteral(32);
			cmov_logical.c = r0;
			_add(cmov_logical);
		}
	}

	void PTXToILTranslator::_translateCos(const ir::PTXInstruction &i)
	{
		ir::ILCos_Vec cos_vec;
		cos_vec.d = _translate(i.d);
		cos_vec.a = _translate(i.a);
		_add(cos_vec);
	}

	void PTXToILTranslator::_translateCvt(const ir::PTXInstruction &i)
	{
		ir::ILOperand a, d;

		_convertSrc(i, a);
		_convert(i, a, d);
		_convertDst(i, d);
	}

	void PTXToILTranslator::_translateDiv(const ir::PTXInstruction &i)
	{
		switch (i.type)
		{
			case ir::PTXOperand::s32: _translateIDiv(i); break;
			case ir::PTXOperand::u32: _translateUDiv(i); break;
			case ir::PTXOperand::f32: _translateFDiv(i); break;
			default:
			{
				assertM(false, "Opcode \"" << i.toString()
						<< "\" not supported");
			}
		}
	}

	void PTXToILTranslator::_translateIDiv(const ir::PTXInstruction &i)
	{
		// There is no signed integer division in IL so we need to use this 
		// macro based on unsigned integer division:
		//
		// out0 = in0 / in1
		// mdef(222)_out(1)_in(2)
		// mov r0, i.a
		// mov r1, i.b
		// mov r0._y__, r1.x
		// ilt r1.xy, r0, 0
		// iadd r0.xy, r0, r1
		// ixor r0.xy, r0, r1
		// udiv r0.x, r0.x, r0.y
		// ixor r1.x, r1.x, r1.y
		// iadd r0.x, r0.x, r1.x
		// ixor r0.x, r0.x, r1.x
		// mov i.d, r0
		// mend

		ir::ILOperand r0 = _tempRegister();
		ir::ILOperand r1 = _tempRegister();

		// mov r0, i.a
		{
			ir::ILMov mov;
			mov.d = r0; mov.a = _translate(i.a);
			_add(mov);
		}

		// mov r1, i.b
		{
			ir::ILMov mov;
			mov.d = r1; mov.a = _translate(i.b);
			_add(mov);
		}

		// mov r0._y__, r1.x
		{
			ir::ILMov mov;
			mov.d = r0.y(); mov.a = r1.x();
			_add(mov);
		}

		// ilt r1.xy, r0, 0
		{
			ir::ILIlt ilt;
			ilt.d = r1.xy(); ilt.a = r0; ilt.b = _translateLiteral(0);
			_add(ilt);
		}

		// iadd r0.xy, r0, r1
		{
			ir::ILIadd iadd;
			iadd.d = r0.xy(); iadd.a = r0; iadd.b = r1;
			_add(iadd);
		}

		// ixor r0.xy, r0, r1
		{
			ir::ILIxor ixor;
			ixor.d = r0.xy(); ixor.a = r0; ixor.b = r1;
			_add(ixor);
		}

		// udiv r0.x, r0.x, r0.y
		{
			ir::ILUdiv udiv;
			udiv.d = r0.x(); udiv.a = r0.x(); udiv.b = r0.y();
			_add(udiv);
		}

		// ixor r1.x, r1.x, r1.y
		{
			ir::ILIxor ixor;
			ixor.d = r1.x(); ixor.a = r1.x(); ixor.b = r1.y();
			_add(ixor);
		}

		// iadd r0.x, r0.x, r1.x
		{
			ir::ILIadd iadd;
			iadd.d = r0.x(); iadd.a = r0.x(); iadd.b = r1.x();
			_add(iadd);
		}

		// ixor r0.x, r0.x, r1.x
		{
			ir::ILIxor ixor;
			ixor.d = r0.x(); ixor.a = r0.x(); ixor.b = r1.x();
			_add(ixor);
		}

		// mov i.d, r0
		{
			ir::ILMov mov;
			mov.d = _translate(i.d); mov.a = r0;
			_add(mov);
		}
	}

	void PTXToILTranslator::_translateUDiv(const ir::PTXInstruction &i)
	{
		ir::ILUdiv udiv;
		udiv.d = _translate(i.d);
		udiv.a = _translate(i.a);
		udiv.b = _translate(i.b);
		_add(udiv);
	}

	void PTXToILTranslator::_translateFDiv(const ir::PTXInstruction &i)
	{
		ir::ILDiv div;

		div.d = _translate(i.d);
		div.a = _translate(i.a);
		div.b = _translate(i.b);

		_add(div);
	}

	void PTXToILTranslator::_translateEx2(const ir::PTXInstruction &i)
	{
		ir::ILExp_Vec exp_vec;
		exp_vec.d = _translate(i.d);
		exp_vec.a = _translate(i.a);
		_add(exp_vec);
	}

	void PTXToILTranslator::_translateExit(const ir::PTXInstruction &i)
	{
		ir::ILEnd end;

		_add(end);
	}

	void PTXToILTranslator::_translateFma(const ir::PTXInstruction &i)
	{
		ir::ILFma fma;
		fma.d = _translate(i.d);
		fma.a = _translate(i.a);
		fma.b = _translate(i.b);
		fma.c = _translate(i.c);
		_add(fma);
	}

	void PTXToILTranslator::_translateLdParam(const ir::PTXInstruction &i)
	{
		int bytes  = ir::PTXOperand::bytes(i.type);
		int bytesA = ir::PTXOperand::bytes(i.a.type);

		// IL registers are 32 bits
		bytes  = (bytes > 4 ? 4 : bytes);
		bytesA = (bytesA > 4 ? 4 : bytesA);

		assertM(bytes >= bytesA,
				"Type mismatch: " << ir::PTXOperand::toString(i.type) << 
				" != " << ir::PTXOperand::toString(i.a.type));

		assertM(bytes % bytesA == 0,
				"Type mismatch: " << ir::PTXOperand::toString(i.type) << 
				" != " << ir::PTXOperand::toString(i.a.type));

		ir::ILOperand temp1 = _tempRegister();
		ir::ILOperand d = _translate(i.d);
		for (int b = 0 ; b < bytes / bytesA ; ++b)
		{
			if (b == 0)
			{
				ir::ILMov mov;
				mov.a = _translateConstantBuffer(i.a, 0);
				mov.d = d;
				_add(mov);
			}
			else
			{
				ir::ILMov mov;
				mov.a = _translateConstantBuffer(i.a, b);
				mov.d = temp1;
				_add(mov);

				ir::ILIshl ishl;
				ishl.d = temp1;
				ishl.a = temp1;
				ishl.b = _translateLiteral(b * bytesA * 8);
				_add(ishl);

				ir::ILIadd iadd;
				iadd.d = d;
				iadd.a = d;
				iadd.b = temp1;
				_add(iadd);
			}
		}
	}

	void PTXToILTranslator::_translateLd(const ir::PTXInstruction &i)
	{
		switch (i.addressSpace)
		{
			case ir::PTXInstruction::Param: _translateLdParam(i); break;
			case ir::PTXInstruction::Global:
			case ir::PTXInstruction::Const:
			{
				switch (i.vec)
				{
					case ir::PTXOperand::v1:
					{
						if (i.a.offset == 0)
						{
							if (ir::PTXOperand::bytes(i.type) == 4)
							{
								ir::ILUav_Raw_Load_Id uav_raw_load_id;
								uav_raw_load_id.a = _translate(i.a);
								uav_raw_load_id.d = _translate(i.d);
								_add(uav_raw_load_id);
							} else
							{
								ir::ILUav_Arena_Load_Id uav_arena_load_id;
								uav_arena_load_id.a = _translate(i.a);
								uav_arena_load_id.d = _translate(i.d);
								uav_arena_load_id.type = _translate(i.type);
								_add(uav_arena_load_id);
							}
						} else
						{
							ir::ILOperand temp = _tempRegister();

							ir::ILIadd iadd;
							iadd.a = _translate(i.a);
							iadd.b = _translateLiteral(i.a.offset);
							iadd.d = temp;
							_add(iadd);

							if (ir::PTXOperand::bytes(i.type) == 4)
							{
								ir::ILUav_Raw_Load_Id uav_raw_load_id;
								uav_raw_load_id.a = temp;
								uav_raw_load_id.d = _translate(i.d);
								_add(uav_raw_load_id);
							}
							else
							{
								ir::ILUav_Arena_Load_Id uav_arena_load_id;
								uav_arena_load_id.a = temp;
								uav_arena_load_id.d = _translate(i.d);
								uav_arena_load_id.type = _translate(i.type);
								_add(uav_arena_load_id);
							}
						}

						break;
					}
					case ir::PTXOperand::v2:
					case ir::PTXOperand::v4:
					{
						if (ir::PTXOperand::bytes(i.type) == 4)
						{
							ir::ILOperand temp;

							// translate base + offset addressing
							if (i.a.offset == 0)
							{
								temp = _translate(i.a);
							} else
							{
								temp = _tempRegister();

								ir::ILIadd iadd;
								iadd.a = _translate(i.a);
								iadd.b = _translateLiteral(i.a.offset);
								iadd.d = temp;
								_add(iadd);
							}

							ir::ILUav_Raw_Load_Id uav_raw_load_id;
							uav_raw_load_id.a = temp;
							uav_raw_load_id.d = _translateArrayDst(i.d.array);
							_add(uav_raw_load_id);
						} else
						{
							ir::ILOperand temp = _tempRegister();
							ir::PTXOperand::Array::const_iterator dst;
							int offset = i.a.offset;
							for (dst = i.d.array.begin() ;
									dst != i.d.array.end() ; dst++)
							{
								ir::ILIadd iadd;
								iadd.a = _translate(i.a);
								iadd.b = _translateLiteral(offset);
								iadd.d = temp;
								_add(iadd);

								ir::ILUav_Arena_Load_Id uav_arena_load_id;
								uav_arena_load_id.a = temp;
								uav_arena_load_id.d = _translate(*dst);
								uav_arena_load_id.type = _translate(i.type);
								_add(uav_arena_load_id);

								offset += ir::PTXOperand::bytes(i.type);
							}
						}

						break;
					}
					default:
					{
						assertM(false, "Vector operation " << i.vec 
								<< " not supported");
					}
				}
				break;
			}
			case ir::PTXInstruction::Shared:
			{
				switch (i.vec)
				{
					case ir::PTXOperand::v1:
					{
						switch (ir::PTXOperand::bytes(i.type))
						{
							case 1: _translateLdSharedByte(i);  break;
							case 2: _translateLdSharedWord(i); break;
							case 4: _translateLdSharedDword(i); break;
							default:
							{
								assertM(false, "Less-than-32-bits memory "
										" operation not supported");
							}
						}
						break;
					}
					case ir::PTXOperand::v2:
					case ir::PTXOperand::v4:
					{
						assertM(ir::PTXOperand::bytes(i.type) == 4,
								"Less-than-32-bits memory operation "
								"not supported");

						ir::ILOperand temp = _tempRegister();
						ir::PTXOperand::Array::const_iterator dst;
						int offset = i.a.offset;
						for (dst = i.d.array.begin() ;
								dst != i.d.array.end() ; dst++)
						{
							ir::ILIadd iadd;
							iadd.a = _translate(i.a);
							iadd.b = _translateLiteral(offset);
							iadd.d = temp;
							_add(iadd);

							ir::ILLds_Load_Id lds_load_id;
							lds_load_id.a = temp;
							lds_load_id.d = _translate(*dst).x();
							_add(lds_load_id);

							offset += ir::PTXOperand::bytes(i.type);
						}

						break;
					}
					default:
					{
						assertM(false, "Vector operation " 
								<< i.vec 
								<< " not supported");
					}
				}
				break;
			}
			default:
			{
				assertM(false, "Address Space not supported in "
						<< i.toString());
			}
		}
	}

	void PTXToILTranslator::_translateLdu(const ir::PTXInstruction &i)
	{
		_translateLd(i);
	}

	void PTXToILTranslator::_translateLdSharedByte(const ir::PTXInstruction &i)
	{
		// LDS is byte-addressable and the result of a load is a dword. 
		// However, the two least significant bits of the address must be set to 
		// zero. Therefore, we need to extract the correct byte from the dword:
		//
		// iadd temp3, i.a, i.a.offset
		// iand temp1, temp3, 3
		// imul temp1, temp1, 8
		// iand temp2, temp3, 0xFFFFFFFC
		// lds_load_id(1) i.d, temp2
		// ishr i.d, i.d, temp1
		// ishl i.d, i.d, 24
		// ishr i.d, i.d, 24
		
		ir::ILOperand temp1 = _tempRegister();
		ir::ILOperand temp2 = _tempRegister();
		ir::ILOperand temp3 = _tempRegister();

		// add the address and the offset
		{
			if (i.a.offset == 0)
			{
				ir::ILMov mov;
				mov.d = temp3;
				mov.a = _translate(i.a);
				_add(mov);
			} else
			{
				ir::ILIadd iadd;
				iadd.d = temp3;
				iadd.a = _translate(i.a);
				iadd.b = _translateLiteral(i.a.offset);
				_add(iadd);
			}
		}

		// get the two lsb's of the address.
		{
			ir::ILIand iand;
			iand.d = temp1;
			iand.a = temp3;
			iand.b = _translateLiteral(3);
			_add(iand);
		}

		// calculate the offset inside the dword
		{
			ir::ILImul imul;
			imul.d = temp1;
			imul.a = temp1;
			imul.b = _translateLiteral(8);
			_add(imul);
		}

		// set the two lsb's of the address to zero
		{
			ir::ILIand iand;
			iand.d = temp2;
			iand.a = temp3;
			iand.b = _translateLiteral(0xFFFFFFFC);
			_add(iand);
		}

		// load dword
		{
			ir::ILLds_Load_Id lds_load_id;
			lds_load_id.d = _translate(i.d).x();
			lds_load_id.a = temp2;
			_add(lds_load_id);
		}

		// extract the correct byte from the dword
		{
			ir::ILIshr ishr;
			ishr.d = _translate(i.d);
			ishr.a = _translate(i.d);
			ishr.b = temp1;
			_add(ishr);
		}

		{
			ir::ILIshl ishl;
			ishl.d = _translate(i.d);
			ishl.a = _translate(i.d);
			ishl.b = _translateLiteral(24);
			_add(ishl);
		}

		{
			ir::ILIshr ishr;
			ishr.d = _translate(i.d);
			ishr.a = _translate(i.d);
			ishr.b = _translateLiteral(24);
			_add(ishr);
		}
	}

	void PTXToILTranslator::_translateLdSharedWord(const ir::PTXInstruction &i)
	{
		// LDS is byte-addressable and the result of a load is a dword. 
		// However, the two least significant bits of the address must be set to 
		// zero. Therefore, we need to extract the correct word from the dword:
		//
		// iadd temp3, i.a, i.a.offset
		// iand temp1, temp3, 3
		// imul temp1, temp1, 8
		// iand temp2, temp3, 0xFFFFFFFC
		// lds_load_id(1) i.d, temp2
		// ishr i.d, i.d, temp1
		// ishl i.d, i.d, 16
		// ishr i.d, i.d, 16

		ir::ILOperand temp1 = _tempRegister();
		ir::ILOperand temp2 = _tempRegister();
		ir::ILOperand temp3 = _tempRegister();

		// iadd temp3, i.a, i.a.offset
		{
			if (i.a.offset == 0)
			{
				ir::ILMov mov;
				mov.d = temp3;
				mov.a = _translate(i.a);
				_add(mov);
			} else
			{
				ir::ILIadd iadd;
				iadd.d = temp3;
				iadd.a = _translate(i.a);
				iadd.b = _translateLiteral(i.a.offset);
				_add(iadd);
			}
		}

		// iand temp1, temp3, 3
		{
			ir::ILIand iand;

			iand.d = temp1;
			iand.a = temp3;
			iand.b = _translateLiteral(3);
			_add(iand);
		}

		// imul temp1, temp1, 8
		{
			ir::ILImul imul;
			imul.d = temp1;
			imul.a = temp1;
			imul.b = _translateLiteral(8);
			_add(imul);
		}

		// iand temp2, temp3, 0xFFFFFFFC
		{
			ir::ILIand iand;
			iand.d = temp2;
			iand.a = temp3;
			iand.b = _translateLiteral(0xFFFFFFFC);
			_add(iand);
		}

		// lds_load_id(1) i.d, temp2
		{
			ir::ILLds_Load_Id lds_load_id;
			lds_load_id.d = _translate(i.d).x();
			lds_load_id.a = temp2;
			_add(lds_load_id);
		}

		// ishr i.d, i.d, temp1
		{
			ir::ILIshr ishr;
			ishr.d = _translate(i.d);
			ishr.a = _translate(i.d);
			ishr.b = temp1;
			_add(ishr);
		}

		// ishl i.d, i.d, 16
		{
			ir::ILIshl ishl;
			ishl.d = _translate(i.d);
			ishl.a = _translate(i.d);
			ishl.b = _translateLiteral(16);
			_add(ishl);
		}

		// ishr i.d, i.d, 16
		{
			ir::ILIshr ishr;
			ishr.d = _translate(i.d);
			ishr.a = _translate(i.d);
			ishr.b = _translateLiteral(16);
			_add(ishr);
		}
	}

	void PTXToILTranslator::_translateLdSharedDword(const ir::PTXInstruction &i)
	{
		if (i.a.offset == 0)
		{
			ir::ILLds_Load_Id lds_load_id;
			lds_load_id.a = _translate(i.a);
			lds_load_id.d = _translate(i.d).x();
			_add(lds_load_id);
		} else
		{
			ir::ILOperand temp = _tempRegister();

			ir::ILIadd iadd;
			iadd.a = _translate(i.a);
			iadd.b = _translateLiteral(i.a.offset);
			iadd.d = temp;
			_add(iadd);

			ir::ILLds_Load_Id lds_load_id;
			lds_load_id.a = temp;
			lds_load_id.d = _translate(i.d).x();
			_add(lds_load_id);
		}
	}

	void PTXToILTranslator::_translateLg2(const ir::PTXInstruction &i)
	{
		switch (i.type)
		{
			case ir::PTXOperand::f32:
			{
				ir::ILLog_Vec log_vec;

				log_vec.d = _translate(i.d);
				log_vec.a = _translate(i.a);

				_add(log_vec);

				break;
			}
			default:
			{
				assertM(false, "Type "
						<< ir::PTXOperand::toString(i.type)
						<< " not supported in "
						<< i.toString());
			}
		}
	}

	void PTXToILTranslator::_translateMad(const ir::PTXInstruction &i)
	{
		ir::ILMad mad;

		mad.a = _translate(i.a);
		mad.b = _translate(i.b);
		mad.c = _translate(i.c);
		mad.d = _translate(i.d);

		_add(mad);
	}

	void PTXToILTranslator::_translateMax(const ir::PTXInstruction &i)
	{
		ir::ILImax imax;

		imax.d = _translate(i.d);
		imax.a = _translate(i.a);
		imax.b = _translate(i.b);

		_add(imax);
	}

	void PTXToILTranslator::_translateMembar(const ir::PTXInstruction &i)
	{
		assertM(i.level == ir::PTXInstruction::GlobalLevel,
				"Membar instruction '" << i.toString() << "' not supported");

		ir::ILFence fence;
		fence.memory();
		_add(fence);
	}

	void PTXToILTranslator::_translateMin(const ir::PTXInstruction &i)
	{
		ir::ILImin imin;

		imin.d = _translate(i.d);
		imin.a = _translate(i.a);
		imin.b = _translate(i.b);

		_add(imin);
	}

	void PTXToILTranslator::_translateMov(const ir::PTXInstruction &i)
	{
		ir::ILMov mov;

		mov.a = _translate(i.a);
		mov.d = _translate(i.d);

		_add(mov);
	}

	inline bool _isPowerOf2(unsigned int n)
	{
		return (n && ((n & (n-1)) == 0));
	}

	/* returns the log base 2 of a power of 2 number */
	inline int _Log2(unsigned int n)
	{
		int r = 0;
		while (n >>= 1) r++;
		return r;
	}

	void PTXToILTranslator::_translateMul(const ir::PTXInstruction &i)
	{
		switch (i.type)
		{
			case ir::PTXOperand::s32:
			{
				if (i.a.addressMode == ir::PTXOperand::Immediate &&
						_isPowerOf2(i.a.imm_uint))
				{
					ir::ILIshl ishl;
					ishl.d = _translate(i.d);
					ishl.a = _translateLiteral(_Log2(i.a.imm_uint));
					ishl.b = _translate(i.b);
					_add(ishl);
					break;
				}
				if (i.b.addressMode == ir::PTXOperand::Immediate &&
						_isPowerOf2(i.b.imm_uint))
				{
					ir::ILIshl ishl;
					ishl.d = _translate(i.d);
					ishl.a = _translate(i.a);
					ishl.b = _translateLiteral(_Log2(i.b.imm_uint));
					_add(ishl);
					break;
				}

				ir::ILImul imul;
				imul.d = _translate(i.d);
				imul.a = _translate(i.a);
				imul.b = _translate(i.b);
				_add(imul);
				break;
			}
			case ir::PTXOperand::u16:
			case ir::PTXOperand::u32:
			case ir::PTXOperand::u64:
			{
				if (i.a.addressMode == ir::PTXOperand::Immediate &&
						_isPowerOf2(i.a.imm_uint))
				{
					ir::ILIshl ishl;
					ishl.d = _translate(i.d);
					ishl.a = _translateLiteral(_Log2(i.a.imm_uint));
					ishl.b = _translate(i.b);
					_add(ishl);
					break;
				}
				if (i.b.addressMode == ir::PTXOperand::Immediate &&
						_isPowerOf2(i.b.imm_uint))
				{
					ir::ILIshl ishl;
					ishl.d = _translate(i.d);
					ishl.a = _translate(i.a);
					ishl.b = _translateLiteral(_Log2(i.b.imm_uint));
					_add(ishl);
					break;
				}

				ir::ILUmul umul;
				umul.d = _translate(i.d);
				umul.a = _translate(i.a);
				umul.b = _translate(i.b);
				_add(umul);
				break;
			}
			case ir::PTXOperand::f32:
			{
				ir::ILMul mul;

				mul.a = _translate(i.a);
				mul.b = _translate(i.b);
				mul.d = _translate(i.d);

				_add(mul);

				break;
			}
			default:
			{
				assertM(false, "Type "
						<< ir::PTXOperand::toString(i.type)
						<< " not supported in "
						<< i.toString());
			}
		}
	}

	void PTXToILTranslator::_translateMul24(const ir::PTXInstruction &i)
	{
		switch (i.type)
		{
			case ir::PTXOperand::s32:
			{
				// there is no imul24
				_translateMul(i);
				break;
			}
			case ir::PTXOperand::u32:
			{
				ir::ILUmul24 umul24;
				umul24.d = _translate(i.d);
				umul24.a = _translate(i.a);
				umul24.b = _translate(i.b);
				_add(umul24);
				break;
			}
			default:
			{
				assertM(false, "Type "
						<< ir::PTXOperand::toString(i.type)
						<< " not supported in "
						<< i.toString());
			}
		}
	}

	void PTXToILTranslator::_translateNeg(const ir::PTXInstruction &i)
	{
		switch (i.type)
		{
			case ir::PTXOperand::s32:
			{
				ir::ILInegate inegate;

				inegate.a = _translate(i.a);
				inegate.d = _translate(i.d);

				_add(inegate);

				break;
			}
			case ir::PTXOperand::f32:
			{
				ir::ILMov mov;

				mov.d = _translate(i.d);
				mov.a = _translate(i.a).neg();

				_add(mov);

				break;
			}
			default:
			{
				assertM(false, "Type "
						<< ir::PTXOperand::toString(i.type)
						<< " not supported in "
						<< i.toString());
			}
		}
	}

	void PTXToILTranslator::_translateNot(const ir::PTXInstruction &i)
	{
		switch (i.type)
		{
			case ir::PTXOperand::b32:
			{
				ir::ILInot inot;

				inot.d = _translate(i.d);
				inot.a = _translate(i.a);

				_add(inot);

				break;
			}
			default:
			{
				assertM(false, "Type "
						<< ir::PTXOperand::toString(i.type)
						<< " not supported in "
						<< i.toString());
			}
		}
	}

	void PTXToILTranslator::_translateOr(const ir::PTXInstruction &i)
	{
		ir::ILIor ior;

		ior.a = _translate(i.a);
		ior.b = _translate(i.b);
		ior.d = _translate(i.d);

		_add(ior);
	}

	void PTXToILTranslator::_translatePopc(const ir::PTXInstruction &i)
    {
        ir::ILIcbits icbits;
        icbits.d = _translate(i.d);
        icbits.a = _translate(i.a);
        _add(icbits);
    }

	void PTXToILTranslator::_translateRcp(const ir::PTXInstruction &i)
	{
		// rcp operates on the fourth (w) component of a
		ir::ILRcp rcp;

		rcp.d = _translate(i.d);
		rcp.a = _translate(i.a).x();

		_add(rcp);
	}

	void PTXToILTranslator::_translateIRem(const ir::PTXInstruction &i)
	{
		// out0 = in0 % in1
		// mdef(285)_out(1)_in(2)
		// mov r0, in0
		// mov r1, in1
		// dcl_literal l25, 0, 0, 0, 0
		// mov r0._y__, r1.x
		// ilt r1.xy, r0, l25
		// iadd r0.xy, r0, r1
		// ixor r0.xy, r0, r1
		// udiv r2.x, r0.x, r0.y
		// umul r2.x, r2.x, r0.y
		// iadd r0.x, r0.x, r2.x_neg(xyzw)
		// iadd r0.x, r0.x, r1.x
		// ixor r0.x, r0.x, r1.x
		// mov out0, r0
		// mend
		
		ir::ILOperand r0 = _tempRegister();
		ir::ILOperand r1 = _tempRegister();
		ir::ILOperand r2 = _tempRegister();

		// mov r0, in0
		{
			ir::ILMov mov;
			mov.d = r0; mov.a = _translate(i.a);
			_add(mov);
		}

		// mov r1, in1
		{
			ir::ILMov mov;
			mov.d = r1; mov.a = _translate(i.b);
			_add(mov);
		}

		// dcl_literal l25, 0, 0, 0, 0
		ir::ILOperand l25 = _translateLiteral(0);

		// mov r0._y__, r1.x
		{
			ir::ILMov mov;
			mov.d = r0.y(); mov.a = r1.x();
			_add(mov);
		}

		// ilt r1.xy, r0, l25
		{
			ir::ILIlt ilt;
			ilt.d = r1.xy(); ilt.a = r0; ilt.b = l25;
			_add(ilt);
		}

		// iadd r0.xy, r0, r1
		{
			ir::ILIadd iadd;
			iadd.d = r0.xy(); iadd.a = r0; iadd.b = r1;
			_add(iadd);
		}

		// ixor r0.xy, r0, r1
		{
			ir::ILIxor ixor;
			ixor.d = r0.xy(); ixor.a = r0; ixor.b = r1;
			_add(ixor);
		}

		// udiv r2.x, r0.x, r0.y
		{
			ir::ILUdiv udiv;
			udiv.d = r2.x(); udiv.a = r0.x(); udiv.b = r0.y();
			_add(udiv);
		}

		// umul r2.x, r2.x, r0.y
		{
			ir::ILUmul umul;
			umul.d = r2.x(); umul.a = r2.x(); umul.b = r0.y();
			_add(umul);
		}

		// iadd r0.x, r0.x, r2.x_neg(xyzw)
		{
			ir::ILIadd iadd;
			iadd.d = r0.x(); iadd.a = r0.x(); iadd.b = r2.x().neg();
			_add(iadd);
		}

		// iadd r0.x, r0.x, r1.x
		{
			ir::ILIadd iadd;
			iadd.d = r0.x(); iadd.a = r0.x(); iadd.b = r1.x();
			_add(iadd);
		}

		// ixor r0.x, r0.x, r1.x
		{
			ir::ILIxor ixor;
			ixor.d = r0.x(); ixor.a = r0.x(); ixor.b = r1.x();
			_add(ixor);
		}

		// mov out0, r0
		{
			ir::ILMov mov;
			mov.d = _translate(i.d); mov.a = r0;
			_add(mov);
		}
	}

	void PTXToILTranslator::_translateURem(const ir::PTXInstruction &i)
	{
		// out0 = in0 % in1
		// mdef(98)_out(1)_in(2)
		// mov r0, in0
		// mov r1, in1
		// udiv r2.x, r0.x, r1.x
		// umul r2.x, r2.x, r1.x
		// iadd r0.x, r0.x, r2.x_neg(xyzw)
		// mov out0, r0
		// mend

		ir::ILOperand r0 = _tempRegister();
		ir::ILOperand r1 = _tempRegister();
		ir::ILOperand r2 = _tempRegister();

		// mov r0, in0
		{
			ir::ILMov mov;
			mov.d = r0; mov.a = _translate(i.a);
			_add(mov);
		}

		// mov r1, in1
		{
			ir::ILMov mov;
			mov.d = r1; mov.a = _translate(i.b);
			_add(mov);
		}

		// udiv r2.x, r0.x, r1.x
		{
			ir::ILUdiv udiv;
			udiv.d = r2.x(); udiv.a = r0.x(); udiv.b = r1.x();
			_add(udiv);
		}

		// umul r2.x, r2.x, r1.x
		{
			ir::ILUmul umul;
			umul.d = r2.x(); umul.a = r2.x(); umul.b = r1.x();
			_add(umul);
		}

		// iadd r0.x, r0.x, r2.x_neg(xyzw)
		{
			ir::ILIadd iadd;
			iadd.d = r0.x(); iadd.a = r0.x(); iadd.b = r2.x().neg();
			_add(iadd);
		}

		// mov out0, r0
		{
			ir::ILMov mov;
			mov.d = _translate(i.d); mov.a = r0;
			_add(mov);
		}
	}

	void PTXToILTranslator::_translateRem(const ir::PTXInstruction &i)
	{
		switch (i.type)
		{
			case ir::PTXOperand::s32: _translateIRem(i); break;
			case ir::PTXOperand::u32: _translateURem(i); break;
			default: assertM(false, "Opcode \"" << i.toString() 
							 << "\" not supported");
		}
	}

	void PTXToILTranslator::_translateRsqrt(const ir::PTXInstruction &i)
	{
		// IEEE 754-compliant
		// mdef(331)_out(1)_in(1)
		// mov r0, in0
		// 
		// dcl_literal l0, 0x00000000, 0x7FFFFFFF, 0x7F800000, 0x00000000
		// and r0._yz_, r0.x, l0
		// 
		// dcl_literal l1, 0x00000000, 0x00000000, 0x00000000, 0x00000000
		// ieq r0.__z_, r0.z, l1
		// 
		// dcl_literal l2, 0x00000000, 0x00000000, 0x00000000, 0x00000000
		// ine r0.___w, r0.y, l2
		// and r0.__z_, r0.z, r0.w
		// 
		// dcl_literal l3, 0x00000000, 0x00000000, 0x00000000, 0x00000000
		// ilt r1.x___, r0.x, l3
		// and r0.___w, r0.w, r1.x
		// ior r0.__z_, r0.z, r0.w
		// 
		// dcl_literal l4, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000
		// ilt r0._y__, l4, r0.y
		// ior r0.__z_, r0.z, r0.y
		// if_logicalnz r0.z
		//     
		//     dcl_literal l5, 0x007FFFFF, 0x007FFFFF, 0x007FFFFF, 0x007FFFFF
		//     and r0.__z_, r0.x, l5
		//     itof r0.__z_, r0.z
		//     
		//     dcl_literal l6, 0x7F800000, 0x007FFFFF, 0x00000000, 0x00000000
		//     and r1.xy__, r0.z, l6
		//     
		//     dcl_literal l7, 0x00000017, 0x00000017, 0x00000017, 0x00000017
		//     ishr r0.__z_, r1.x, l7
		//     
		//     dcl_literal l8, 0x00000018, 0x00000018, 0x00000018, 0x00000018
		//     iadd r0.__z_, r0.z, l8
		//     
		//     dcl_literal l9, 0x00800000, 0x00800000, 0x00800000, 0x00800000
		//     ior r1.x___, r1.y, l9
		//     
		//     dcl_literal l10, 0x00000096, 0x00000096, 0x00000096, 0x00000096
		//     iadd r0.__z_, l10, r0.z_neg(xyzw)
		//     
		//     dcl_literal l11, 0x00000017, 0x00000017, 0x00000017, 0x00000017
		//     ilt r1._y__, l11, r0.z
		//     
		//     dcl_literal l12, 0x00000018, 0x00000018, 0x00000018, 0x00000018
		//     cmov_logical r0.__z_, r1.y, l12, r0.z
		//     
		//     dcl_literal l13, 0x00000000, 0x00000000, 0x00000000, 0x00000000
		//     ilt r1._y__, l13, r0.z
		//     ishr r1.__z_, r1.x, r0.z
		//     inegate r0.__z_, r0.z
		//     
		//     dcl_literal l14, 0x00000017, 0x00000017, 0x00000017, 0x00000017
		//     ishl r0.__z_, r0.z, l14
		//     iadd r0.__z_, r1.x, r0.z
		//     cmov_logical r0.__z_, r1.y, r1.z, r0.z
		//     rsq_vec r0.__z_, r0.z
		//     
		//     dcl_literal l15, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000
		//     and r1.x___, r0.z, l15
		//     if_logicalz r1.x
		//         
		//         dcl_literal l16, 0x007FFFFF, 0x007FFFFF, 0x007FFFFF, 0x007FFFFF
		//         and r1._y__, r0.z, l16
		//         itof r1._y__, r1.y
		//         
		//         dcl_literal l17, 0x00000000, 0x7F800000, 0x007FFFFF, 0x00000000
		//         and r1._yz_, r1.y, l17
		//         
		//         dcl_literal l18, 0x00000017, 0x00000017, 0x00000017, 0x00000017
		//         ishr r1._y__, r1.y, l18
		//         
		//         dcl_literal l19, 0x0000000C, 0x0000000C, 0x0000000C, 0x0000000C
		//         iadd r1._y__, r1.y, l19
		//         
		//         dcl_literal l20, 0x00800000, 0x00800000, 0x00800000, 0x00800000
		//         ior r1.__z_, r1.z, l20
		//         
		//         dcl_literal l21, 0x00000096, 0x00000096, 0x00000096, 0x00000096
		//         iadd r1._y__, l21, r1.y_neg(xyzw)
		//         
		//         dcl_literal l22, 0x00000017, 0x00000017, 0x00000017, 0x00000017
		//         ilt r1.___w, l22, r1.y
		//         
		//         dcl_literal l23, 0x00000018, 0x00000018, 0x00000018, 0x00000018
		//         cmov_logical r1._y__, r1.w, l23, r1.y
		//         
		//         dcl_literal l24, 0x00000000, 0x00000000, 0x00000000, 0x00000000
		//         ilt r1.___w, l24, r1.y
		//         ishr r2.x___, r1.z, r1.y
		//         inegate r1._y__, r1.y
		//         
		//         dcl_literal l25, 0x00000017, 0x00000017, 0x00000017, 0x00000017
		//         ishl r1._y__, r1.y, l25
		//         iadd r1._y__, r1.z, r1.y
		//         cmov_logical r1._y__, r1.w, r2.x, r1.y
		//     else
		//         
		//         dcl_literal l26, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF
		//         and r0.__z_, r0.z, l26
		//         
		//         dcl_literal l27, 0x00000017, 0x00000017, 0x00000017, 0x00000017
		//         ishr r1.x___, r1.x, l27
		//         
		//         dcl_literal l28, 0x06000000, 0x06000000, 0x06000000, 0x06000000
		//         iadd r0.__z_, r0.z, l28
		//         
		//         dcl_literal l29, 0xFFFFFF8D, 0xFFFFFF8D, 0xFFFFFF8D, 0xFFFFFF8D
		//         iadd r1.x___, r1.x, l29
		//         
		//         dcl_literal l30, 0x0000007F, 0x0000007F, 0x0000007F, 0x0000007F
		//         ilt r1.x___, l30, r1.x
		//         
		//         dcl_literal l31, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000
		//         cmov_logical r1._y__, r1.x, l31, r0.z
		//     endif
		//     
		//     dcl_literal l32, 0xFFC00000, 0xFFC00000, 0xFFC00000, 0xFFC00000
		//     cmov_logical r0.__z_, r0.w, l32, r1.y
		//     
		//     dcl_literal l33, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000
		//     ior r0.___w, r0.x, l33
		//     cmov_logical r0.x___, r0.y, r0.w, r0.z
		// else
		//     rsq_vec r0.x___, r0.x
		// endif
		// mov out0, r0
		// mend

		ir::ILOperand r0 = _tempRegister();
		ir::ILOperand r1 = _tempRegister();
		ir::ILOperand r2 = _tempRegister();

		// mov r0, in0
		{
			ir::ILMov mov;
			mov.d = r0;
			mov.a = _translate(i.a);
			_add(mov);
		}

		// dcl_literal l0, 0x00000000, 0x7FFFFFFF, 0x7F800000, 0x00000000
		// and r0._yz_, r0.x, l0
		{
			// TODO Implement multi-valued literals. Otherwise, we need to use
			// two and's.
			ir::ILAnd and1;
			and1.d = r0.y(); 
			and1.a = r0.x(); 
			and1.b = _translateLiteral(0x7FFFFFFF);
			_add(and1);

			ir::ILAnd and2;
			and2.d = r0.z();
			and2.a = r0.x();
			and2.b = _translateLiteral(0x7F800000);
			_add(and2);
		}

		// dcl_literal l1, 0x00000000, 0x00000000, 0x00000000, 0x00000000
		// ieq r0.__z_, r0.z, l1
		{
			ir::ILIeq ieq;
			ieq.d = r0.z();
			ieq.a = r0.z();
			ieq.b = _translateLiteral(0);
			_add(ieq);
		}

		// dcl_literal l2, 0x00000000, 0x00000000, 0x00000000, 0x00000000
		// ine r0.___w, r0.y, l2
		// and r0.__z_, r0.z, r0.w
		{
			ir::ILIne ine;
			ine.d = r0.w();
			ine.a = r0.y();
			ine.b = _translateLiteral(0);
			_add(ine);

			ir::ILAnd and1;
			and1.d = r0.z();
			and1.a = r0.z();
			and1.b = r0.w();
			_add(and1);
		}

		// dcl_literal l3, 0x00000000, 0x00000000, 0x00000000, 0x00000000
		// ilt r1.x___, r0.x, l3
		// and r0.___w, r0.w, r1.x
		// ior r0.__z_, r0.z, r0.w
		{
			ir::ILIlt ilt;
			ilt.d = r1.x();
			ilt.a = r0.x();
			ilt.b = _translateLiteral(0);
			_add(ilt);

			ir::ILAnd and1;
			and1.d = r0.w();
			and1.a = r0.w();
			and1.b = r1.x();
			_add(and1);

			ir::ILIor ior;
			ior.d = r0.z();
			ior.a = r0.z();
			ior.b = r0.w();
			_add(ior);
		}

		// dcl_literal l4, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000
		// ilt r0._y__, l4, r0.y
		// ior r0.__z_, r0.z, r0.y
		{
			ir::ILIlt ilt;
			ilt.d = r0.y();
			ilt.a = _translateLiteral(0x7F800000);
			ilt.b = r0.y();
			_add(ilt);

			ir::ILIor ior;
			ior.d = r0.z();
			ior.a = r0.z();
			ior.b = r0.y();
			_add(ior);
		}

		// if_logicalnz r0.z
		{
			ir::ILIfLogicalNZ if_logicalnz;
			if_logicalnz.a = r0.z();
			_add(if_logicalnz);
		}
		
		//     dcl_literal l5, 0x007FFFFF, 0x007FFFFF, 0x007FFFFF, 0x007FFFFF
		//     and r0.__z_, r0.x, l5
		//     itof r0.__z_, r0.z
		{
			ir::ILAnd and1;
			and1.d = r0.z();
			and1.a = r0.x();
			and1.b = _translateLiteral(0x007FFFFF);
			_add(and1);

			ir::ILItoF itof;
			itof.d = r0.z();
			itof.a = r0.z();
			_add(itof);
		}

		//     dcl_literal l6, 0x7F800000, 0x007FFFFF, 0x00000000, 0x00000000
		//     and r1.xy__, r0.z, l6
		{
			// TODO Implement multi-valued literals.
			ir::ILAnd and1;
			and1.d = r1.x();
			and1.a = r0.z();
			and1.b = _translateLiteral(0x7F800000);
			_add(and1);

			ir::ILAnd and2;
			and2.d = r1.y();
			and2.a = r0.z();
			and2.b = _translateLiteral(0x007FFFFF);
			_add(and2);
		}

		//     dcl_literal l7, 0x00000017, 0x00000017, 0x00000017, 0x00000017
		//     ishr r0.__z_, r1.x, l7
		{
			ir::ILIshr ishr;
			ishr.d = r0.z();
			ishr.a = r1.x();
			ishr.b = _translateLiteral(0x00000017);
			_add(ishr);
		}

		//     dcl_literal l8, 0x00000018, 0x00000018, 0x00000018, 0x00000018
		//     iadd r0.__z_, r0.z, l8
		{
			ir::ILIadd iadd;
			iadd.d = r0.z();
			iadd.a = r0.z();
			iadd.b = _translateLiteral(0x00000018);
			_add(iadd);
		}

		//     dcl_literal l9, 0x00800000, 0x00800000, 0x00800000, 0x00800000
		//     ior r1.x___, r1.y, l9
		{
			ir::ILIor ior;
			ior.d = r1.x();
			ior.a = r1.y();
			ior.b = _translateLiteral(0x00800000);
			_add(ior);
		}

		//     dcl_literal l10, 0x00000096, 0x00000096, 0x00000096, 0x00000096
		//     iadd r0.__z_, l10, r0.z_neg(xyzw)
		{
			ir::ILIadd iadd;
			iadd.d = r0.z();
			iadd.a = _translateLiteral(0x00000096);
			iadd.b = r0.z().neg();
			_add(iadd);
		}

		//     dcl_literal l11, 0x00000017, 0x00000017, 0x00000017, 0x00000017
		//     ilt r1._y__, l11, r0.z
		{
			ir::ILIlt ilt;
			ilt.d = r1.y();
			ilt.a = _translateLiteral(0x00000017);
			ilt.b = r0.z();
			_add(ilt);
		}

		//     dcl_literal l12, 0x00000018, 0x00000018, 0x00000018, 0x00000018
		//     cmov_logical r0.__z_, r1.y, l12, r0.z
		{
			ir::ILCmov_Logical cmov_logical;
			cmov_logical.d = r0.z();
			cmov_logical.a = r1.y();
			cmov_logical.b = _translateLiteral(0x00000018);
			cmov_logical.c = r0.z();
			_add(cmov_logical);
		}

		//     dcl_literal l13, 0x00000000, 0x00000000, 0x00000000, 0x00000000
		//     ilt r1._y__, l13, r0.z
		//     ishr r1.__z_, r1.x, r0.z
		//     inegate r0.__z_, r0.z
		{
			ir::ILIlt ilt;
			ilt.d = r1.y();
			ilt.a = _translateLiteral(0);
			ilt.b = r0.z();
			_add(ilt);

			ir::ILIshr ishr;
			ishr.d = r1.z();
			ishr.a = r1.x();
			ishr.b = r0.z();
			_add(ishr);

			ir::ILInegate inegate;
			inegate.d = r0.z();
			inegate.a = r0.x();
			_add(inegate);
		}

		//     dcl_literal l14, 0x00000017, 0x00000017, 0x00000017, 0x00000017
		//     ishl r0.__z_, r0.z, l14
		//     iadd r0.__z_, r1.x, r0.z
		//     cmov_logical r0.__z_, r1.y, r1.z, r0.z
		//     rsq_vec r0.__z_, r0.z
		{
			ir::ILIshl ishl;
			ishl.d = r0.z();
			ishl.a = r0.z();
			ishl.b = _translateLiteral(0x00000017);
			_add(ishl);

			ir::ILIadd iadd;
			iadd.d = r0.z();
			iadd.a = r1.x();
			iadd.b = r0.z();
			_add(iadd);

			ir::ILCmov_Logical cmov_logical;
			cmov_logical.d = r0.z();
			cmov_logical.a = r1.y();
			cmov_logical.b = r1.z();
			cmov_logical.c = r0.z();
			_add(cmov_logical);

			ir::ILRsq_Vec rsq_vec;
			rsq_vec.d = r0.z();
			rsq_vec.a = r0.z();
			_add(rsq_vec);
		}

		//     dcl_literal l15, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000
		//     and r1.x___, r0.z, l15
		//     if_logicalz r1.x
		{
			ir::ILAnd and1;
			and1.d = r1.x();
			and1.a = r0.z();
			and1.b = _translateLiteral(0x7F800000);
			_add(and1);

			ir::ILIfLogicalZ if_logicalz;
			if_logicalz.a = r1.x();
			_add(if_logicalz);
		}
		
		//         dcl_literal l16, 0x007FFFFF, 0x007FFFFF, 0x007FFFFF, 0x007FFFFF
		//         and r1._y__, r0.z, l16
		//         itof r1._y__, r1.y
		{
			ir::ILAnd and1;
			and1.d = r1.y();
			and1.a = r0.z();
			and1.b = _translateLiteral(0x007FFFFF);
			_add(and1);

			ir::ILItoF itof;
			itof.d = r1.y();
			itof.a = r1.y();
			_add(itof);
		}

		//         dcl_literal l17, 0x00000000, 0x7F800000, 0x007FFFFF, 0x00000000
		//         and r1._yz_, r1.y, l17
		{
			// TODO Implement multi-valued literals.
			ir::ILAnd and1;
			and1.d = r1.y();
			and1.a = r1.y();
			and1.b = _translateLiteral(0x7F800000);
			_add(and1);

			ir::ILAnd and2;
			and2.d = r1.z();
			and2.a = r1.y();
			and2.b = _translateLiteral(0x007FFFFF);
			_add(and2);
		}

		//         dcl_literal l18, 0x00000017, 0x00000017, 0x00000017, 0x00000017
		//         ishr r1._y__, r1.y, l18
		{
			ir::ILIshr ishr;
			ishr.d = r1.y();
			ishr.a = r1.y();
			ishr.b = _translateLiteral(0x00000017);
			_add(ishr);
		}

		//         dcl_literal l19, 0x0000000C, 0x0000000C, 0x0000000C, 0x0000000C
		//         iadd r1._y__, r1.y, l19
		{
			ir::ILIadd iadd;
			iadd.d = r1.y();
			iadd.a = r1.y();
			iadd.b = _translateLiteral(0x0000000C);
			_add(iadd);
		}

		//         dcl_literal l20, 0x00800000, 0x00800000, 0x00800000, 0x00800000
		//         ior r1.__z_, r1.z, l20
		{
			ir::ILIor ior;
			ior.d = r1.z();
			ior.a = r1.z();
			ior.b = _translateLiteral(0x00800000);
			_add(ior);
		}

		//         dcl_literal l21, 0x00000096, 0x00000096, 0x00000096, 0x00000096
		//         iadd r1._y__, l21, r1.y_neg(xyzw)
		{
			ir::ILIadd iadd;
			iadd.d = r1.y();
			iadd.a = _translateLiteral(0x00000096);
			iadd.b = r1.y().neg();
			_add(iadd);
		}

		//         dcl_literal l22, 0x00000017, 0x00000017, 0x00000017, 0x00000017
		//         ilt r1.___w, l22, r1.y
		{
			ir::ILIlt ilt;
			ilt.d = r1.w();
			ilt.a = _translateLiteral(0x00000017);
			ilt.b = r1.y();
			_add(ilt);
		}

		//         dcl_literal l23, 0x00000018, 0x00000018, 0x00000018, 0x00000018
		//         cmov_logical r1._y__, r1.w, l23, r1.y
		{
			ir::ILCmov_Logical cmov_logical;
			cmov_logical.d = r1.y();
			cmov_logical.a = r1.w();
			cmov_logical.b = _translateLiteral(0x00000018);
			cmov_logical.c = r1.y();
			_add(cmov_logical);
		}

		//         dcl_literal l24, 0x00000000, 0x00000000, 0x00000000, 0x00000000
		//         ilt r1.___w, l24, r1.y
		//         ishr r2.x___, r1.z, r1.y
		//         inegate r1._y__, r1.y
		{
			ir::ILIlt ilt;
			ilt.d = r1.w();
			ilt.a = _translateLiteral(0);
			ilt.b = r1.y();
			_add(ilt);

			ir::ILIshr ishr;
			ishr.d = r2.x();
			ishr.a = r1.z();
			ishr.b = r1.y();
			_add(ishr);

			ir::ILInegate inegate;
			inegate.d = r1.y();
			inegate.a = r1.y();
			_add(inegate);
		}

		//         dcl_literal l25, 0x00000017, 0x00000017, 0x00000017, 0x00000017
		//         ishl r1._y__, r1.y, l25
		//         iadd r1._y__, r1.z, r1.y
		//         cmov_logical r1._y__, r1.w, r2.x, r1.y
		{
			ir::ILIshl ishl;
			ishl.d = r1.y();
			ishl.a = r1.y();
			ishl.b = _translateLiteral(0x00000017);
			_add(ishl);

			ir::ILIadd iadd;
			iadd.d = r1.y();
			iadd.a = r1.z();
			iadd.b = r1.y();
			_add(iadd);

			ir::ILCmov_Logical cmov_logical;
			cmov_logical.d = r1.y();
			cmov_logical.a = r1.w();
			cmov_logical.b = r2.x();
			cmov_logical.c = r1.y();
			_add(cmov_logical);
		}

		//     else
		{
			_add(ir::ILElse());
		}
		
		//         dcl_literal l26, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF
		//         and r0.__z_, r0.z, l26
		{
			ir::ILAnd and1;
			and1.d = r0.z();
			and1.a = r0.z();
			and1.b = _translateLiteral(0x7FFFFFFF);
			_add(and1);
		}

		//         dcl_literal l27, 0x00000017, 0x00000017, 0x00000017, 0x00000017
		//         ishr r1.x___, r1.x, l27
		{
			ir::ILIshr ishr;
			ishr.d = r1.x();
			ishr.a = r1.x();
			ishr.b = _translateLiteral(0x00000017);
			_add(ishr);
		}

		//         dcl_literal l28, 0x06000000, 0x06000000, 0x06000000, 0x06000000
		//         iadd r0.__z_, r0.z, l28
		{
			ir::ILIadd iadd;
			iadd.d = r0.z();
			iadd.a = r0.z();
			iadd.b = _translateLiteral(0x06000000);
			_add(iadd);
		}

		//         dcl_literal l29, 0xFFFFFF8D, 0xFFFFFF8D, 0xFFFFFF8D, 0xFFFFFF8D
		//         iadd r1.x___, r1.x, l29
		{
			ir::ILIadd iadd;
			iadd.d = r1.x();
			iadd.a = r1.x();
			iadd.b = _translateLiteral(0xFFFFFF8D);
			_add(iadd);
		}

		//         dcl_literal l30, 0x0000007F, 0x0000007F, 0x0000007F, 0x0000007F
		//         ilt r1.x___, l30, r1.x
		{
			ir::ILIlt ilt;
			ilt.d = r1.x();
			ilt.a = _translateLiteral(0x0000007F);
			ilt.b = r1.x();
			_add(ilt);
		}

		//         dcl_literal l31, 0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000
		//         cmov_logical r1._y__, r1.x, l31, r0.z
		{
			ir::ILCmov_Logical cmov_logical;
			cmov_logical.d = r1.y();
			cmov_logical.a = r1.x();
			cmov_logical.b = _translateLiteral(0x7F800000);
			cmov_logical.c = r0.z();
			_add(cmov_logical);
		}

		//     endif
		{
			_add(ir::ILEndIf());
		}

		//     dcl_literal l32, 0xFFC00000, 0xFFC00000, 0xFFC00000, 0xFFC00000
		//     cmov_logical r0.__z_, r0.w, l32, r1.y
		{
			ir::ILCmov_Logical cmov_logical;
			cmov_logical.d = r0.z();
			cmov_logical.a = r0.w();
			cmov_logical.b = _translateLiteral(0xFFC00000);
			cmov_logical.c = r1.y();
			_add(cmov_logical);
		}

		//     dcl_literal l33, 0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000
		//     ior r0.___w, r0.x, l33
		//     cmov_logical r0.x___, r0.y, r0.w, r0.z
		{
			ir::ILIor ior;
			ior.d = r0.w();
			ior.a = r0.x();
			ior.b = _translateLiteral(0x7FC00000);
			_add(ior);

			ir::ILCmov_Logical cmov_logical;
			cmov_logical.d = r0.x();
			cmov_logical.a = r0.y();
			cmov_logical.b = r0.w();
			cmov_logical.c = r0.z();
			_add(cmov_logical);
		}

		// else
		{
			_add(ir::ILElse());
		}

		//     rsq_vec r0.x___, r0.x
		{
			ir::ILRsq_Vec rsq_vec;
			rsq_vec.d = r0.x();
			rsq_vec.a = r0.x();
			_add(rsq_vec);
		}

		// endif
		{
			_add(ir::ILEndIf());
		}

		// mov out0, r0
		{
			ir::ILMov mov;
			mov.d = _translate(i.d);
			mov.a = r0;
			_add(mov);
		}

		// mend
	}

	void PTXToILTranslator::_translateSelP(const ir::PTXInstruction &i)
	{
		// Note that IL semantic is cmov_logical dest, pred, iftrue, iffalse
		// while PTX is selp dest, iftrue, iffalse, pred.
		ir::ILCmov_Logical cmov_logical;

		cmov_logical.d = _translate(i.d);
		cmov_logical.a = _translate(i.c);
		cmov_logical.b = _translate(i.a);
		cmov_logical.c = _translate(i.b);

		_add(cmov_logical);
	}

	void PTXToILTranslator::_translateSet(const ir::PTXInstruction &i)
	{
		// In IL there's no difference between predicate and normal registers
		_translateSetP(i);
	}

	void PTXToILTranslator::_translateSetP(const ir::PTXInstruction &i)
	{
		switch (i.type)
		{
			case ir::PTXOperand::s32:
			case ir::PTXOperand::u32:
			case ir::PTXOperand::u64: _translateISetP(i); break;
			case ir::PTXOperand::f32:
			case ir::PTXOperand::f64: _translateFSetP(i); break;
			default:
			{
				assertM(false, "Opcode \"" << i.toString()
						<< "\" not supported");
			}
		}
	}

	void PTXToILTranslator::_translateISetP(const ir::PTXInstruction &i)
	{
		switch (i.comparisonOperator)
		{
			case ir::PTXInstruction::Eq:
			{
				ir::ILIeq ieq;

				ieq.a = _translate(i.a);
				ieq.b = _translate(i.b);
				ieq.d = _translate(i.d);

				_add(ieq);

				break;
			}
			case ir::PTXInstruction::Le:
			{
				// IL doesn't have le but it has ge so switch a & b operands
				ir::ILIge ige;

				ige.a = _translate(i.b);
				ige.b = _translate(i.a);
				ige.d = _translate(i.d);

				_add(ige);

				break;
			}
			case ir::PTXInstruction::Lt:
			{
				ir::ILIlt ilt;

				ilt.a = _translate(i.a);
				ilt.b = _translate(i.b);
				ilt.d = _translate(i.d);

				_add(ilt);

				break;
			}
			case ir::PTXInstruction::Ge:
			{
				ir::ILIge ige;

				ige.a = _translate(i.a);
				ige.b = _translate(i.b);
				ige.d = _translate(i.d);

				_add(ige);

				break;
			}
			case ir::PTXInstruction::Ne:
			{
				ir::ILIne ine;

				ine.a = _translate(i.a);
				ine.b = _translate(i.b);
				ine.d = _translate(i.d);

				_add(ine);

				break;
			}
			case ir::PTXInstruction::Gt:
			{
				// IL doesn't have gt but it has lt so switch a & b operands
				ir::ILIlt ilt;

				ilt.a = _translate(i.b);
				ilt.b = _translate(i.a);
				ilt.d = _translate(i.d);

				_add(ilt);

				break;
			}
			default:
			{
				assertM(false, "comparisonOperator "
						<< ir::PTXInstruction::toString(i.comparisonOperator)
						<< " not supported");
			}
		}
	}

	void PTXToILTranslator::_translateFSetP(const ir::PTXInstruction &i)
	{
		switch (i.comparisonOperator)
		{
			case ir::PTXInstruction::Eq:
			{
				ir::ILEq eq;
				eq.d = _translate(i.d);
				eq.a = _translate(i.a);
				eq.b = _translate(i.b);
				_add(eq);
				break;
			}
			case ir::PTXInstruction::Le:
			{
				// IL doesn't have le but it has ge so switch a & b operands
				ir::ILGe ge;
				ge.d = _translate(i.d);
				ge.a = _translate(i.b);
				ge.b = _translate(i.a);
				_add(ge);
				break;
			}
			case ir::PTXInstruction::Lt:
			{
				ir::ILLt lt;

				lt.a = _translate(i.a);
				lt.b = _translate(i.b);
				lt.d = _translate(i.d);

				_add(lt);

				break;
			}
			case ir::PTXInstruction::Ge:
			{
				ir::ILGe ge;
				ge.d = _translate(i.d);
				ge.a = _translate(i.a);
				ge.b = _translate(i.b);
				_add(ge);
				break;
			}
			case ir::PTXInstruction::Gt:
			{
				// IL doesn't have gt but it has lt so switch a & b operands
				ir::ILLt lt;

				lt.a = _translate(i.b);
				lt.b = _translate(i.a);
				lt.d = _translate(i.d);

				_add(lt);

				break;
			}
			case ir::PTXInstruction::Neu:
			{
				ir::ILNe ne;

				ne.a = _translate(i.a);
				ne.b = _translate(i.b);
				ne.d = _translate(i.d);

				_add(ne);

				break;
			}
			default:
			{
				assertM(false, "comparisonOperator "
						<< ir::PTXInstruction::toString(i.comparisonOperator)
						<< " not supported");
			}
		}
	}

	void PTXToILTranslator::_translateSin(const ir::PTXInstruction &i)
	{
		ir::ILSin_Vec sin_vec;
		sin_vec.d = _translate(i.d);
		sin_vec.a = _translate(i.a);
		_add(sin_vec);
	}

	void PTXToILTranslator::_translateShl(const ir::PTXInstruction &i)
	{
		assertM(i.type == ir::PTXOperand::b32, 
				"Type " << ir::PTXOperand::toString(i.type) << " not supported");

		ir::ILIshl ishl;

		ishl.a = _translate(i.a);
		ishl.b = _translate(i.b);
		ishl.d = _translate(i.d);

		_add(ishl);
	}

	void PTXToILTranslator::_translateShr(const ir::PTXInstruction &i)
	{
		switch (i.type)
		{
			case ir::PTXOperand::s32:
			{
				ir::ILIshr ishr;

				ishr.a = _translate(i.a);
				ishr.b = _translate(i.b);
				ishr.d = _translate(i.d);

				_add(ishr);

				break;
			}
			case ir::PTXOperand::u32:
			{
				ir::ILUshr ushr;

				ushr.a = _translate(i.a);
				ushr.b = _translate(i.b);
				ushr.d = _translate(i.d);

				_add(ushr);

				break;
			}
			default:
			{
				assertM(false, "Type "
						<< ir::PTXOperand::toString(i.type)
						<< " not supported in "
						<< i.toString());
			}
		}
	}

	void PTXToILTranslator::_translateSlct(const ir::PTXInstruction& i)
	{
		ir::ILOperand temp = _tempRegister();

		ir::ILGe ge;
		ge.d = temp;
		ge.a = _translate(i.c);
		ge.b = _translateLiteral(0);
		_add(ge);

		ir::ILCmov_Logical cmov_logical;
		cmov_logical.d = _translate(i.d);
		cmov_logical.a = temp;
		cmov_logical.b = _translate(i.a);
		cmov_logical.c = _translate(i.b);
		_add(cmov_logical);
	}

	void PTXToILTranslator::_translateSqrt(const ir::PTXInstruction &i)
	{
		ir::ILSqrt_Vec sqrt_vec;
		sqrt_vec.d = _translate(i.d);
		sqrt_vec.a = _translate(i.a);
		_add(sqrt_vec);
	}

	void PTXToILTranslator::_translateSt(const ir::PTXInstruction &i)
	{
		switch (i.addressSpace)
		{
			case ir::PTXInstruction::Global:
			{
				assertM(i.volatility != ir::PTXInstruction::Volatile,
						"Volatile global store operation not supported yet in " 
						<< i.toString());

				// TODO uav0 accesses should be aligned to 4
				switch (i.vec)
				{
					case ir::PTXOperand::v1:
					{
						if (i.d.offset == 0)
						{
							if (ir::PTXOperand::bytes(i.type) == 4)
							{
								ir::ILUav_Raw_Store_Id uav_raw_store_id;
								uav_raw_store_id.d = _translateMemMask(1);
								uav_raw_store_id.a = _translate(i.d);
								uav_raw_store_id.b = _translate(i.a);
								_add(uav_raw_store_id);
							} else
							{
								ir::ILUav_Arena_Store_Id uav_arena_store_id;
								uav_arena_store_id.a = _translate(i.d);
								uav_arena_store_id.b = _translate(i.a);
								uav_arena_store_id.type = _translate(i.type);
								_add(uav_arena_store_id);
							}
						} else
						{
							ir::ILOperand temp = _tempRegister();

							ir::ILIadd iadd;
							iadd.a = _translate(i.d);
							iadd.b = _translateLiteral(i.d.offset);
							iadd.d = temp;
							_add(iadd);

							if (ir::PTXOperand::bytes(i.type) == 4)
							{
								ir::ILUav_Raw_Store_Id uav_raw_store_id;
								uav_raw_store_id.d = _translateMemMask(1);
								uav_raw_store_id.a = temp;
								uav_raw_store_id.b = _translate(i.a);
								_add(uav_raw_store_id);
							} else
							{
								ir::ILUav_Arena_Store_Id uav_arena_store_id;
								uav_arena_store_id.a = temp;
								uav_arena_store_id.b = _translate(i.a);
								uav_arena_store_id.type = _translate(i.type);
								_add(uav_arena_store_id);
							}
						}

						break;
					}
					case ir::PTXOperand::v2:
					case ir::PTXOperand::v4:
					{
						if (ir::PTXOperand::bytes(i.type) == 4)
						{
							ir::ILOperand temp;

							// translate base + offset addressing
							if (i.d.offset == 0)
							{
								temp = _translate(i.d);
							} else
							{
								temp = _tempRegister();

								ir::ILIadd iadd;
								iadd.a = _translate(i.d);
								iadd.b = _translateLiteral(i.d.offset);
								iadd.d = temp;
								_add(iadd);
							}

							ir::ILUav_Raw_Store_Id uav_raw_store_id;
							uav_raw_store_id.d = _translateMemMask(i.a.array.size());
							uav_raw_store_id.a = temp;
							uav_raw_store_id.b = _translateArraySrc(i.a.array);
							_add(uav_raw_store_id);
						} else
						{
							ir::ILOperand temp = _tempRegister();
							ir::PTXOperand::Array::const_iterator src;
							int offset = i.d.offset;
							for (src = i.a.array.begin() ;
									src != i.a.array.end() ; src++)
							{
								ir::ILIadd iadd;
								iadd.a = _translate(i.d);
								iadd.b = _translateLiteral(offset);
								iadd.d = temp;
								_add(iadd);

								ir::ILUav_Arena_Store_Id uav_arena_store_id;
								uav_arena_store_id.a = temp;
								uav_arena_store_id.b = _translate(*src);
								uav_arena_store_id.type = _translate(i.type);
								_add(uav_arena_store_id);

								offset += ir::PTXOperand::bytes(i.type);
							}
						}
						break;
					}
					default:
					{
						assertM(false, "Vector operation " << i.vec 
								<< " not supported");
					}
				}
				break;
			}
			case ir::PTXInstruction::Shared:
			{
				switch (i.vec)
				{
					case ir::PTXOperand::v1:
					{
						switch(ir::PTXOperand::bytes(i.type))
						{
							case 1: _translateStSharedByte(i);  break;
							case 2: _translateStSharedWord(i); break;
							case 4: _translateStSharedDword(i); break;
							default:
							{
								assertM(false, "Less-than-32-bits memory "
										"operation not supported");
							}
						}
						break;
					}
					case ir::PTXOperand::v2:
					case ir::PTXOperand::v4:
					{
						assertM(ir::PTXOperand::bytes(i.type) == 4,
								"Less-than-32-bits memory operation "
								"not supported");

						ir::ILOperand temp = _tempRegister();
						ir::PTXOperand::Array::const_iterator src;
						int offset = i.d.offset;
						for (src = i.a.array.begin() ;
								src != i.a.array.end() ; src++)
						{
							ir::ILIadd iadd;
							iadd.a = _translate(i.d);
							iadd.b = _translateLiteral(offset);
							iadd.d = temp;
							_add(iadd);

							ir::ILLds_Store_Id lds_store_id;
							lds_store_id.a = temp;
							lds_store_id.b = _translate(*src).x();
							_add(lds_store_id);

							offset += ir::PTXOperand::bytes(i.type);
						}
						break;
					}
					default:
					{
						assertM(false, "Vector operation " << i.vec 
								<< " not supported");
					}
				}

				// if volatile add fence after the store
				if (i.volatility == ir::PTXInstruction::Volatile)
				{
					ir::ILFence fence;
					fence.threads(false);
					fence.lds(true);
					_add(fence);
				}

				break;
			}
			default:
			{
				assertM(false, "Address Space " << i.addressSpace 
						<< " not supported");
			}
		}
	}

	void PTXToILTranslator::_translateStSharedByte(const ir::PTXInstruction& i)
	{
		// LDS is byte-addressable but the result of a store is a dword. 
		// Therefore, we need to merge the byte with the dword:
		//
		// and temp1, i.a, 0x000000FF
		// and temp2, i.d, 3
		// iadd temp3X, temp2, 0
		// iadd temp3Y, temp2, -1
		// iadd temp3Z, temp2, -2
		// iadd temp3W, temp2, -3
		// imul temp4, temp2, 8
		// cmov_logical temp5, temp3W, 0xFFFFFF00, 0x00FFFFFF
		// cmov_logical temp5, temp3Z, temp5, 0xFF00FFFF
		// cmov_logical temp5, temp3Y, temp5, 0xFFFF00FF
		// ishl temp2, temp1, temp4
		// lds_and_id(1) i.d, temp5
		// lds_or_id(1) i.d, temp2

		assertM(i.a.offset == 0, "St Shared Byte from offset not supported");

		ir::ILOperand temp1  = _tempRegister();
		ir::ILOperand temp2  = _tempRegister();
		ir::ILOperand temp3X = _tempRegister();
		ir::ILOperand temp3Y = _tempRegister();
		ir::ILOperand temp3Z = _tempRegister();
		ir::ILOperand temp3W = _tempRegister();
		ir::ILOperand temp4  = _tempRegister();
		ir::ILOperand temp5  = _tempRegister();

		// set the value to a byte
		ir::ILAnd iland1;
		iland1.d = temp1;
		iland1.a = _translate(i.a);
		iland1.b = _translateLiteral(0x000000FF);
		_add(iland1);

		// get the two lsb's of the address.
		ir::ILAnd iland2;
		iland2.d = temp2;
		iland2.a = _translate(i.d);
		iland2.b = _translateLiteral(3);
		_add(iland2);

		// calculate the mask to merge with the dword
		ir::ILIadd iaddX,iaddY, iaddZ, iaddW;
		iaddX.d = temp3X; iaddY.d = temp3Y; iaddZ.d = temp3Z; iaddW.d = temp3W;
		iaddX.a = iaddY.a = iaddZ.a = iaddW.a = temp2;
		iaddX.b = _translateLiteral(0);
		iaddY.b = _translateLiteral(-1);
		iaddZ.b = _translateLiteral(-2);
		iaddW.b = _translateLiteral(-3);
		_add(iaddX); _add(iaddY); _add(iaddZ); _add(iaddW);

		// calculate the offset inside the dword
		ir::ILImul imul;
		imul.d = temp4;
		imul.a = temp2;
		imul.b = _translateLiteral(8);
		_add(imul);

		ir::ILCmov_Logical cmov_logical1;
		cmov_logical1.d = temp5;
		cmov_logical1.a = temp3W;
		cmov_logical1.b = _translateLiteral(0xFFFFFF00);
		cmov_logical1.c = _translateLiteral(0x00FFFFFF);
		_add(cmov_logical1);

		ir::ILCmov_Logical cmov_logical2;
		cmov_logical2.d = temp5;
		cmov_logical2.a = temp3Z;
		cmov_logical2.b = temp5;
		cmov_logical2.c = _translateLiteral(0xFF00FFFF);
		_add(cmov_logical2);

		ir::ILCmov_Logical cmov_logical3;
		cmov_logical3.d = temp5;
		cmov_logical3.a = temp3Y;
		cmov_logical3.b = temp5;
		cmov_logical3.c = _translateLiteral(0xFFFF00FF);
		_add(cmov_logical3);

		ir::ILIshl ishl;
		ishl.d = temp2;
		ishl.a = temp1;
		ishl.b = temp4;
		_add(ishl);

		ir::ILLds_And_Id lds_and_id;
		lds_and_id.a = _translate(i.d).x();
		lds_and_id.b = temp5.x();
		_add(lds_and_id);

		ir::ILLds_Or_Id lds_or_id;
		lds_or_id.a = _translate(i.d).x();
		lds_or_id.b = temp2.x();
		_add(lds_or_id);
	}

	void PTXToILTranslator::_translateStSharedWord(const ir::PTXInstruction& i)
	{
		// LDS is byte-addressable but the result of a store is a dword. 
		// Therefore, we need to merge the word with the dword:
		//
		// and temp1, i.a, 0x0000FFFF
		// and temp2, i.d, 3
		// ishr temp2, temp2, 1
		// cmov_logical temp3, temp2, 0x0000FFFF, 0xFFFF0000
		// cmov_logical temp2, temp2, 16, 0
		// ishl temp1, temp1, temp2
		// lds_and_id(1) i.d, temp3
		// lds_or_ir(1) i.d, temp1

		assertM(i.a.offset == 0, "St Shared Byte from offset not supported");

		ir::ILOperand temp1  = _tempRegister();
		ir::ILOperand temp2  = _tempRegister();
		ir::ILOperand temp3  = _tempRegister();
		
		// and temp1, i.a, 0x0000FFFF
		{
			ir::ILAnd and1;
			and1.d = temp1;
			and1.a = _translate(i.a);
			and1.b = _translateLiteral(0x0000FFFF);
			_add(and1);
		}

		// and temp2, i.d, 3
		{
			ir::ILAnd and2;
			and2.d = temp2;
			and2.a = _translate(i.d);
			and2.b = _translateLiteral(3);
			_add(and2);
		}

		// ishr temp2, temp2, 1
		{
			ir::ILIshr ishr;
			ishr.d = temp2;
			ishr.a = temp2;
			ishr.b = _translateLiteral(1);
			_add(ishr);
		}

		// cmov_logical temp3, temp2, 0x0000FFFF, 0xFFFF0000
		{
			ir::ILCmov_Logical cmov_logical;
			cmov_logical.d = temp3;
			cmov_logical.a = temp2;
			cmov_logical.b = _translateLiteral(0x0000FFFF);
			cmov_logical.c = _translateLiteral(0xFFFF0000);
			_add(cmov_logical);
		}

		// cmov_logical temp2, temp2, 16, 0
		{
			ir::ILCmov_Logical cmov_logical;
			cmov_logical.d = temp2;
			cmov_logical.a = temp2;
			cmov_logical.b = _translateLiteral(16);
			cmov_logical.c = _translateLiteral(0);
			_add(cmov_logical);
		}

		// ishl temp1, temp1, temp2
		{
			ir::ILIshl ishl;
			ishl.d = temp1;
			ishl.a = temp1;
			ishl.b = temp2;
			_add(ishl);
		}

		// lds_and_id(1) i.d, temp3
		{
			ir::ILLds_And_Id lds_and_id;
			lds_and_id.a = _translate(i.d).x();
			lds_and_id.b = temp3.x();
			_add(lds_and_id);
		}

		// lds_or_ir(1) i.d, temp1
		{
			ir::ILLds_Or_Id lds_or_id;
			lds_or_id.a = _translate(i.d).x();
			lds_or_id.b = temp1.x();
			_add(lds_or_id);
		}
	}

	void PTXToILTranslator::_translateStSharedDword(const ir::PTXInstruction& i)
	{
		if (i.d.offset == 0)
		{
			ir::ILLds_Store_Id lds_store_id;
			lds_store_id.a = _translate(i.d);
			lds_store_id.b = _translate(i.a).x();
			_add(lds_store_id);
		} else
		{
			ir::ILOperand temp = _tempRegister();

			ir::ILIadd iadd;
			iadd.a = _translate(i.d);
			iadd.b = _translateLiteral(i.d.offset);
			iadd.d = temp;
			_add(iadd);

			ir::ILLds_Store_Id lds_store_id;
			lds_store_id.a = temp;
			lds_store_id.b = _translate(i.a).x();
			_add(lds_store_id);
		}
	}

	void PTXToILTranslator::_translateSub(const ir::PTXInstruction& i)
	{
		switch (i.type)
		{
			// There's no isub instruction in IL so we need to use add
			case ir::PTXOperand::s32:
			case ir::PTXOperand::u32:
			case ir::PTXOperand::u64:
			{
				switch (i.b.addressMode)
				{
					case ir::PTXOperand::Immediate:
					{
						ir::ILIadd iadd;
						ir::PTXOperand b;

						b.addressMode = ir::PTXOperand::Immediate;
						b.imm_int = -1 * i.b.imm_int;

						iadd.d = _translate(i.d);
						iadd.a = _translate(i.a);
						iadd.b = _translate(b);

						_add(iadd);

						break;
					}
					case ir::PTXOperand::Register:
					{
						ir::ILOperand temp = _tempRegister();

						ir::ILInegate inegate;
						inegate.d = temp;
						inegate.a = _translate(i.b);
						_add(inegate);

						ir::ILIadd iadd;
						iadd.d = _translate(i.d);
						iadd.a = _translate(i.a);
						iadd.b = temp;
						_add(iadd);

						break;
					}
					default:
					{
						assertM(i.b.addressMode == ir::PTXOperand::Immediate,
								"Subtract operation not supported");
					}
				}
				break;
			}
			case ir::PTXOperand::f32:
			{
				ir::ILSub sub;

				sub.a = _translate(i.a);
				sub.b = _translate(i.b);
				sub.d = _translate(i.d);

				_add(sub);

				break;
			}
			default:
			{
				assertM(false, "Type "
						<< ir::PTXOperand::toString(i.type)
						<< " not supported in "
						<< i.toString());
			}
		}
	}

	void PTXToILTranslator::_translateVote(const ir::PTXInstruction& i)
    {
        switch(i.vote)
        {
            case ir::PTXInstruction::All:
            {
                ir::ILMov mov;
                mov.a = _translateLiteral(1);
                mov.d = _translate(i.d);
                _add(mov);
                break;
            }
            case ir::PTXInstruction::Ballot:
            {
                ir::ILMov mov;
                mov.a = _translateLiteral(0xFFFFFFFF);
                mov.d = _translate(i.d);
                _add(mov);
                break;
            }
			default:
			{
				assertM(false, "Type "
						<< ir::PTXInstruction::toString(i.vote)
						<< " not supported in "
						<< i.toString());
			}
        }
    }

	void PTXToILTranslator::_translateXor(const ir::PTXInstruction& i)
	{
		ir::ILIxor ixor;

		ixor.a = _translate(i.a);
		ixor.b = _translate(i.b);
		ixor.d = _translate(i.d);

		_add(ixor);
	}

	ir::ILOperand PTXToILTranslator::_translateLiteral(int l)
	{
		const LiteralMap::const_iterator it = _literals.find(l);
		if (it != _literals.end()) return it->second;

		ir::ILOperand op(ir::ILOperand::RegType_Literal);
		op.num = _literals.size();
		_literals.insert(std::make_pair(l, op));

		return op.x();
	}

	ir::ILOperand PTXToILTranslator::_translateConstantBuffer(
			const ir::PTXOperand o, unsigned int offset)
	{
		const std::string ident = o.identifier;
		std::stringstream stream;

		int i = 0;
		ir::Kernel::ParameterVector::const_iterator it;
		for (it = _ilKernel->arguments.begin() ; 
				it != _ilKernel->arguments.end() ; it++) {
			if (it->name == ident) break;
			i += it->arrayValues.size();
		}

		assertM(it != _ilKernel->arguments.end(), 
				"Argument " << ident << " not declared");

		ir::ILOperand op(ir::ILOperand::RegType_Const_Buf);
		op.num = 1;
		op.immediate_present = true;
		op.imm = i + o.offset + offset;

		return op.x();
	}

 	void PTXToILTranslator::_addKernelPrefix(const ATIExecutableKernel *k)
 	{
		report("Adding Kernel Prefix");

		if (_literals.size() > 0) {
			LiteralMap::const_iterator it;
			for (it = _literals.begin() ; it != _literals.end() ; it++) 
			{
				ir::ILStatement dcl_literal(ir::ILStatement::LiteralDcl);

				dcl_literal.operands.resize(1);
				dcl_literal.operands[0] = it->second;

				dcl_literal.arguments.resize(4);
				dcl_literal.arguments[0] = it->first;
				dcl_literal.arguments[1] = it->first;
				dcl_literal.arguments[2] = it->first;
				dcl_literal.arguments[3] = it->first;

				_ilKernel->_statements.push_front(dcl_literal);

				report("Added \'" << dcl_literal.toString() << "\'");
			}
		}

		if (_ilKernel->arguments.size() > 0) {
			ir::ILStatement dcl_cb1(ir::ILStatement::ConstantBufferDcl);

			dcl_cb1.operands.resize(1);
			dcl_cb1.operands[0].num = 1;
			dcl_cb1.operands[0].type = ir::ILOperand::RegType_Const_Buf;
			dcl_cb1.operands[0].immediate_present = true;
			dcl_cb1.operands[0].imm = _ilKernel->arguments.size();

			_ilKernel->_statements.push_front(dcl_cb1);

			report("Added \'" << dcl_cb1.toString() << "\'");
		}

		ir::ILStatement dcl_cb0(ir::ILStatement::ConstantBufferDcl);

		dcl_cb0.operands.resize(1);
		dcl_cb0.operands.resize(1);
		dcl_cb0.operands[0].num = 0;
		dcl_cb0.operands[0].type = ir::ILOperand::RegType_Const_Buf;
		dcl_cb0.operands[0].immediate_present = true;
		dcl_cb0.operands[0].imm = 2;

		_ilKernel->_statements.push_front(dcl_cb0);
		report("Added \'" << dcl_cb0.toString() << "\'");

		unsigned int totalSharedMemorySize = k->sharedMemorySize() +
			k->externSharedMemorySize() + k->voteMemorySize();

		if (totalSharedMemorySize > 0)
		{
			ir::ILStatement dcl_lds(ir::ILStatement::LocalDataShareDcl);

			dcl_lds.arguments.resize(1);
			dcl_lds.arguments[0] = totalSharedMemorySize;

			_ilKernel->_statements.push_front(dcl_lds);
		
			report("Added \'" << dcl_lds.toString() << "\'");
		}

 		_ilKernel->_statements.push_front(ir::ILStatement(
 					ir::ILStatement::OtherDeclarations));
 	}

	void PTXToILTranslator::_add(const ir::ILInstruction &i)
	{
		report("Added instruction '" << i.toString() << "'");
		_ilKernel->_statements.push_back(ir::ILStatement(i));
	}
}
