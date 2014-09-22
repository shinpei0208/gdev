/*!
 * \file PTXToSASSTranslator.cpp
 */

#include <sstream>
#include <iomanip>

#include <hydrazine/interface/debug.h>

#include <ocelot/ir/interface/PTXKernel.h>
#include <ocelot/ir/interface/SASSKernel.h>
#include <ocelot/ir/interface/SASSStatement.h>
#include <ocelot/ir/interface/SASSInstruction.h>
#include <ocelot/translator/interface/PTXToSASSTranslator.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif
#define REPORT_BASE 0

#define RZ	"RZ"
#define	R0	"R0"
#define	R0CC	"R0.CC"
#define	R1	"R1"
#define	PT	"pt"


namespace translator {
	PTXToSASSTranslator::PTXToSASSTranslator(OptimizationLevel l)
		: Translator(ir::Instruction::PTX, ir::Instruction::CAL, l) {
	}

	ir::Kernel* PTXToSASSTranslator::translate(const ir::Kernel* k) {
		return NULL;
	}

	ir::Kernel* PTXToSASSTranslator::translate(ir::PTXKernel* ptx) {
		report("Translate kernel: " << ptx->name);

		_reg32.clear();
		_reg64.clear();

		// pred
		_pred.clear();
		RegType n = 0;
		ir::PTXKernel::RegisterVector rv =
			ptx->getReferencedRegisters();
		for(ir::PTXKernel::RegisterVector::const_iterator r =
			rv.begin(); r != rv.end(); r++) {
			if(r->type == ir::PTXOperand::pred) {
				_pred[r->id] = n++;
			}
                }

		ControlTree ctree;
		_ptx = ptx;
		_sass = new ir::SASSKernel(*ptx);
		_sass->setParameters(*ptx);
		if (ptx->sharedMemorySize() > 0) {
			_sass->setSharedMemory(ptx);
		}

		_makeRange();
		ctree.analyze(*ptx);
		_translate(ctree.get_root_node());

		report("Translate kernel done");
		return _sass;
	}

	void PTXToSASSTranslator::_translate(const ControlTree::Node* n) {
		report("Translate node: " << n->label());

		switch (n->rtype()) {
		case ControlTree::Inst:
			_translate(
				static_cast<const ControlTree::InstNode*>(n));
				break;
		case ControlTree::Block:
			_translate(
				static_cast<const ControlTree::BlockNode*>(n));
				break;
		case ControlTree::IfThen:
			_translate(
				static_cast<const ControlTree::IfThenNode*>(n));
				break;
		case ControlTree::Natural: 
			_translate(
				static_cast<const ControlTree::NaturalNode*>(n));
				break;
		default:
			assertM(false, "invalid type: " << n->rtype());
		}
	}

	void PTXToSASSTranslator::_translate(const ControlTree::InstNode* n) {
		report("  inst: " << n->label());

		_sass->setLabel(n->label());

		for (ControlTree::InstructionList::const_iterator
			it = n->bb()->instructions.begin(), 
			end = n->bb()->instructions.end();
			it != end; it++) {
			_translate(static_cast<ir::PTXInstruction &>(**it));
		}
	}

	void PTXToSASSTranslator::_translate(const ControlTree::BlockNode* n) {
		report("  block: " << n->label());

		for (ControlTree::NodeList::const_iterator
			it = n->children().begin(),
			end = n->children().end();
			it != end; it++) {
			_translate(*it);
		}	
	}

	void PTXToSASSTranslator::_translate(const ControlTree::IfThenNode* n) {
		report("  ifthen: " << n->label());

		_translate(n->cond());

		report("  * true: " << n->label());
		_translate(n->ifTrue());
		if (n->ifFalse() != NULL) {
			report("  * false: " << n->label());
			_translate(n->ifFalse());
		}
	}

	void PTXToSASSTranslator::_translate(const ControlTree::NaturalNode* n) {
		report("  natural: " << n->label());

		ControlTree::NodeList::const_iterator end =
			(--n->children().end());
		for (ControlTree::NodeList::const_iterator it =
			n->children().begin(); it != end; it++) {
			_translate(*it);
		}
		_translate(*end);
	}

	void PTXToSASSTranslator::_translate(const ir::PTXInstruction &i) {
		report("  instr: " << i.toString());

		switch (i.opcode) {
		case ir::PTXInstruction::Add:
			_translateAdd(i);
			break;
		case ir::PTXInstruction::And:
			_translateAnd(i);
			break;
		case ir::PTXInstruction::Bar:
			_translateBar(i);
			break;
		case ir::PTXInstruction::Bra:
			_translateBra(i);
			break;
		case ir::PTXInstruction::Cvta:
			_translateCvta(i);
			break;
		case ir::PTXInstruction::Div:
			_translateDiv(i);
			break;
		case ir::PTXInstruction::Fma:
			_translateFma(i);
			break;
		case ir::PTXInstruction::Ld:
			_translateLd(i);
			break;
		case ir::PTXInstruction::Mad:
			_translateMad(i);
			break;
		case ir::PTXInstruction::Mov:
			_translateMov(i);
			break;
		case ir::PTXInstruction::Mul:
			_translateMul(i);
			break;
		case ir::PTXInstruction::Neg:
			_translateNeg(i);
			break;
		case ir::PTXInstruction::Or:
			_translateOr(i);
			break;
		case ir::PTXInstruction::Ret:
			_translateRet(i);
			break;
		case ir::PTXInstruction::SelP:
			_translateSelp(i);
			break;
		case ir::PTXInstruction::SetP:
			_translateSetp(i);
			break;
		case ir::PTXInstruction::Shl:
			_translateShl(i);
			break;
		case ir::PTXInstruction::Shr:
			_translateShr(i);
			break;
		case ir::PTXInstruction::St:
			_translateSt(i);
			break;
		case ir::PTXInstruction::Sub:
			_translateSub(i);
			break;
		default:
			assertM(false, "not implemented: " <<
				ir::PTXInstruction::toString(i.opcode));
		}
		_line++;
		_refreshRange();
	}

	void PTXToSASSTranslator::_translateAdd(const ir::PTXInstruction &i) {
		report("  add: bytes d=" << ir::PTXOperand::bytes(i.d.type) <<
			",a=" << ir::PTXOperand::bytes(i.a.type) <<
			",b=" << ir::PTXOperand::bytes(i.b.type) <<
			" : " << i.toString());
		if (i.d.addressMode == ir::PTXOperand::Register &&
			i.a.addressMode == ir::PTXOperand::Register) {
			switch (i.a.type) {
			case ir::PTXOperand::s8:
			case ir::PTXOperand::s16:
			case ir::PTXOperand::s32:
			case ir::PTXOperand::u8:
			case ir::PTXOperand::u16:
			case ir::PTXOperand::u32: {
				// 32bit integer
				ir::SASSStatement ss(ir::SASSStatement::Instr,
					ir::SASSInstruction::Iadd);
				ss.instr.addOperand(makeReg(i.d));
				ss.instr.addOperand(makeReg(i.a));
				switch (i.b.addressMode) {
				case ir::PTXOperand::Register:
					ss.instr.addOperand(makeReg(i.b));
					break;
				case ir::PTXOperand::Immediate:
					ss.instr.addOperand(makeImmediate(i.b.imm_int));
					break;
				default:
					assertM(false, "not implemented: " <<
						i.toString());
				}
				_sass->addStatement(ss);
				break;
			}
			case ir::PTXOperand::s64:
			case ir::PTXOperand::u64: {
				// 64bit integer
				ir::SASSStatement sl(ir::SASSStatement::Instr,
					ir::SASSInstruction::Iadd);
				sl.instr.addOperand(makeReg(i.d, 0, "CC"));
				sl.instr.addOperand(makeReg(i.a));
				switch (i.b.addressMode) {
				case ir::PTXOperand::Register:
					sl.instr.addOperand(makeReg(i.b));
					break;
				case ir::PTXOperand::Immediate:
					sl.instr.addOperand(makeImmediate(i.b.imm_int));
					break;
				default:
					assertM(false, "not implemented: " <<
						i.toString());
				}
				_sass->addStatement(sl);
				ir::SASSStatement sh(ir::SASSStatement::Instr,
					ir::SASSInstruction::Iadd);
				sh.instr.addOperand(makeReg(i.d, 1));
				sh.instr.addOperand(makeReg(i.a, 1));
				switch (i.b.addressMode) {
				case ir::PTXOperand::Register:
					sh.instr.addOperand(makeReg(i.b, 1));
					break;
				case ir::PTXOperand::Immediate:
					sh.instr.addOperand(RZ);
					break;
				default:
					assertM(false, "not implemented: " <<
						i.toString());
				}
				sh.instr.addModifier("X");
				_sass->addStatement(sh);
				break;
			}
			case ir::PTXOperand::f32: {
				// float
				if (i.b.addressMode != ir::PTXOperand::Register) {
					assertM(false, "not implemented: " <<
						i.toString());
				}
				ir::SASSStatement ss(ir::SASSStatement::Instr,
					ir::SASSInstruction::Fadd);
				ss.instr.addOperand(makeReg(i.d));
				ss.instr.addOperand(makeReg(i.a));
				ss.instr.addOperand(makeReg(i.b));
				_sass->addStatement(ss);
				break;
			}
			case ir::PTXOperand::f64: {
				// double
				if (i.b.addressMode != ir::PTXOperand::Register) {
					assertM(false, "not implemented: " <<
						i.toString());
				}
				ir::SASSStatement ss(ir::SASSStatement::Instr,
					ir::SASSInstruction::Dadd);
				ss.instr.addOperand(makeReg(i.d));
				ss.instr.addOperand(makeReg(i.a));
				ss.instr.addOperand(makeReg(i.b));
				_sass->addStatement(ss);
				break;
			}
			default:
				assertM(false, "not implemented: " <<
					ir::PTXOperand::toString(i.a.type));
			}
		} else {
			assertM(false, "not implemented: " << i.toString());
		}
	}

	void PTXToSASSTranslator::_translateAnd(const ir::PTXInstruction &i) {
		report("  and : " << i.toString());
		switch (i.d.type) {
		case ir::PTXOperand::pred: {
			ir::SASSStatement ss(ir::SASSStatement::Instr,
				ir::SASSInstruction::Psetp);
			ss.instr.addModifier("AND");
			ss.instr.addModifier("AND");
			ss.instr.addOperand(makeReg(i.d));
			ss.instr.addOperand(PT);
			ss.instr.addOperand(makeReg(i.a));
			ss.instr.addOperand(makeReg(i.b));
			ss.instr.addOperand(PT);
			_sass->addStatement(ss);
			break;
		}
		default: {
			ir::SASSStatement ss(ir::SASSStatement::Instr,
				ir::SASSInstruction::Lop);
			ss.instr.addModifier("AND");
			ss.instr.addOperand(makeReg(i.d));
			ss.instr.addOperand(makeReg(i.a));
			ss.instr.addOperand(makeReg(i.b));
			_sass->addStatement(ss);
		}
		}
	}

	void PTXToSASSTranslator::_translateBar(const ir::PTXInstruction &i) {
		report("  bar : " << i.toString());
		if (i.barrierOperation == ir::PTXInstruction::BarSync) {
			ir::SASSStatement ss(ir::SASSStatement::Instr,
				ir::SASSInstruction::Bar);
			ss.instr.addOperand(makeImmediate(0, false));
			ss.instr.addModifier("SYNC");
			_sass->addStatement(ss);
		} else {
			assertM(false, "not implemented: " << i.toString());
		}
	}

	void PTXToSASSTranslator::_translateBra(const ir::PTXInstruction &i) {
		report("  bra: pred=" << i.pg.toString() <<
			" d=" << i.d.toString() <<
			" : " << i.toString());
		if (i.d.addressMode == ir::PTXOperand::Label) {
			ir::SASSStatement ss(ir::SASSStatement::Instr,
				ir::SASSInstruction::Bra);
			if (i.pg.condition == ir::PTXOperand::Pred ||
				i.pg.condition == ir::PTXOperand::InvPred) {
				ss.instr.setPredicate(makeReg(i.pg));
			}
			ss.instr.addOperand(i.d.identifier);
			_sass->addStatement(ss);
		} else {
			assertM(false, "not implemented: " << i.toString());
		}
	}

	void PTXToSASSTranslator::_translateCvta(const ir::PTXInstruction &i) {
		report("  cvta: " << i.toString());
		if (i.addressSpace == ir::PTXInstruction::Global &&
			i.toAddrSpace &&
			ir::PTXOperand::bytes(i.d.type) == 8) {
			// 64bit
			ir::SASSStatement sl(ir::SASSStatement::Instr,
				ir::SASSInstruction::Mov);
			sl.instr.addOperand(makeReg(i.d));
			sl.instr.addOperand(makeReg(i.a));
			_sass->addStatement(sl);
			ir::SASSStatement sh(ir::SASSStatement::Instr,
				ir::SASSInstruction::Mov);
			sh.instr.addOperand(makeReg(i.d, 1));
			sh.instr.addOperand(makeReg(i.a, 1));
			_sass->addStatement(sh);
		} else {
			assertM(false, "not implemented: " << i.toString());
		}
	}

	void PTXToSASSTranslator::_translateDiv(const ir::PTXInstruction &i) {
		report("  div: " << i.toString());
		std::string typeStr = "U";

		// d = a * (1/b) ...
		switch (i.type) {
		case ir::PTXOperand::f32: {
			ir::SASSStatement s1(ir::SASSStatement::Instr,
				ir::SASSInstruction::Mufu);
			s1.instr.addModifier("RCP");
			s1.instr.addOperand(R0);
			s1.instr.addOperand(makeReg(i.b));
			_sass->addStatement(s1);
			ir::SASSStatement s2(ir::SASSStatement::Instr,
				ir::SASSInstruction::Fmul);
			s2.instr.addOperand(makeReg(i.d));
			s2.instr.addOperand(makeReg(i.a));
			s2.instr.addOperand(R0);
			_sass->addStatement(s2);
			break;
		}
		case ir::PTXOperand::s16:
		case ir::PTXOperand::s32:
		case ir::PTXOperand::s64:
			typeStr = "S";
			/* FALLTHROUGH */
		case ir::PTXOperand::u16:
		case ir::PTXOperand::u32:
		case ir::PTXOperand::u64: {
			typeStr += std::to_string(
				ir::PTXOperand::bytes(i.type)*8);
			ir::SASSStatement s1(ir::SASSStatement::Instr,
				ir::SASSInstruction::I2f);
			s1.instr.addModifier("F32");
			s1.instr.addModifier(typeStr);
			s1.instr.addOperand(makeReg(i.a));
			s1.instr.addOperand(makeReg(i.a));
			_sass->addStatement(s1);
			ir::SASSStatement s2(ir::SASSStatement::Instr,
				ir::SASSInstruction::I2f);
			s2.instr.addModifier("F32");
			s2.instr.addModifier(typeStr);
			s2.instr.addOperand(makeReg(i.b));
			s2.instr.addOperand(makeReg(i.b));
			_sass->addStatement(s2);
			ir::SASSStatement s3(ir::SASSStatement::Instr,
				ir::SASSInstruction::Mufu);
			s3.instr.addModifier("RCP");
			s3.instr.addOperand(R0);
			s3.instr.addOperand(makeReg(i.b));
			_sass->addStatement(s3);
			ir::SASSStatement s4(ir::SASSStatement::Instr,
				ir::SASSInstruction::Fmul);
			s4.instr.addOperand(makeReg(i.d));
			s4.instr.addOperand(makeReg(i.a));
			s4.instr.addOperand(R0);
			_sass->addStatement(s4);
			ir::SASSStatement s5(ir::SASSStatement::Instr,
				ir::SASSInstruction::F2i);
			s5.instr.addModifier(typeStr);
			s5.instr.addModifier("F32");
			s5.instr.addOperand(makeReg(i.d));
			s5.instr.addOperand(makeReg(i.d));
			_sass->addStatement(s5);
			break;
		}
		default:
			assertM(false, "not implemented: " << i.toString());
		}
	}

	void PTXToSASSTranslator::_translateFma(const ir::PTXInstruction &i) {
		report("  fma: " << i.toString());
		if (i.modifier & ir::PTXInstruction::rn &&
			ir::PTXOperand::bytes(i.a.type) == 4) {
			ir::SASSStatement ss(ir::SASSStatement::Instr,
				ir::SASSInstruction::Ffma);
			ss.instr.addOperand(makeReg(i.d));
			ss.instr.addOperand(makeReg(i.a));
			ss.instr.addOperand(makeReg(i.b));
			ss.instr.addOperand(makeReg(i.c));
			_sass->addStatement(ss);
		} else {
			assertM(false, "not implemented: " << i.toString());
		}
	}

	void PTXToSASSTranslator::_translateLd(const ir::PTXInstruction &i) {
		report("  ld: d=" << ir::PTXOperand::bytes(i.d.type) <<
			" : " << i.toString());
		if (i.addressSpace == ir::PTXInstruction::Param &&
			i.d.addressMode == ir::PTXOperand::Register) {
			switch (ir::PTXOperand::bytes(i.a.type)) {
			case 4: {
				// 32bit
				ir::SASSStatement ss(ir::SASSStatement::Instr,
					ir::SASSInstruction::Mov);
				ss.instr.addOperand(makeReg(i.d));
				ss.instr.addOperand(makeAddress(i.a));
				_sass->addStatement(ss);
				break;
			}
			case 8: {
				// 64bit
				ir::SASSStatement sl(ir::SASSStatement::Instr,
					ir::SASSInstruction::Mov);
				sl.instr.addOperand(makeReg(i.d));
				sl.instr.addOperand(makeAddress(i.a));
				_sass->addStatement(sl);
				ir::SASSStatement sh(ir::SASSStatement::Instr,
					ir::SASSInstruction::Mov);
				sh.instr.addOperand(makeReg(i.d, 1));
				sh.instr.addOperand(makeAddress(i.a, 1));
				_sass->addStatement(sh);
				break;
			}
			case 1:
			case 2: {
				// 8bit or 16bit
				ir::SASSStatement ss(ir::SASSStatement::Instr,
					ir::SASSInstruction::Ldc);
				ss.instr.addModifier("U" +
					std::to_string(ir::PTXOperand::bytes(i.a.type)*8));
				ss.instr.addOperand(makeReg(i.d));
				ss.instr.addOperand(makeAddress(i.a));
				_sass->addStatement(ss);
				break;
			}
			default:
				assertM(false, "not implemented: " <<
					ir::PTXOperand::toString(i.a.type));
			}
		} else if (i.addressSpace == ir::PTXInstruction::Global &&
			i.d.addressMode == ir::PTXOperand::Register &&
			ir::PTXOperand::bytes(i.d.type) == 4) {
			ir::SASSStatement ss(ir::SASSStatement::Instr,
				ir::SASSInstruction::Ld);
			ss.instr.addModifier("E");
			ss.instr.addOperand(makeReg(i.d));
			ss.instr.addOperand("[" + makeReg(i.a) + "]");
			_sass->addStatement(ss);
		} else if (i.addressSpace == ir::PTXInstruction::Shared &&
			i.d.addressMode == ir::PTXOperand::Register &&
			ir::PTXOperand::bytes(i.d.type) == 4) {
			ir::SASSStatement ss(ir::SASSStatement::Instr,
				ir::SASSInstruction::Lds);
			ss.instr.addOperand(makeReg(i.d));
			ss.instr.addOperand("[" + makeReg(i.a) + "]");
			_sass->addStatement(ss);
		} else {
			assertM(false, "not implemented: " << i.toString());
		}
	}

	void PTXToSASSTranslator::_translateMad(const ir::PTXInstruction &i) {
		report("  mad: i=" << ir::PTXOperand::toString(i.type) <<
			",d=" << ir::PTXOperand::toString(i.d.type) <<
			",a=" << ir::PTXOperand::toString(i.a.type) <<
			",b=" << ir::PTXOperand::toString(i.b.type) <<
			",c=" << ir::PTXOperand::toString(i.c.type) <<
			" : " << i.toString());
		if (i.d.addressMode == ir::PTXOperand::Register &&
			i.a.addressMode == ir::PTXOperand::Register &&
			i.b.addressMode == ir::PTXOperand::Register &&
			i.c.addressMode == ir::PTXOperand::Register &&
			ir::PTXOperand::bytes(i.d.type) == 4) {
			ir::SASSStatement ss(ir::SASSStatement::Instr,
				ir::SASSInstruction::Imad);
			ss.instr.addOperand(makeReg(i.d));
			ss.instr.addOperand(makeReg(i.a));
			ss.instr.addOperand(makeReg(i.b));
			ss.instr.addOperand(makeReg(i.c));
			if (i.type == ir::PTXOperand::u32) {
				ss.instr.addModifier("U32");
				ss.instr.addModifier("U32");
			}
			if (i.modifier & ir::PTXInstruction::hi) {
				ss.instr.addModifier("HI");
			}
			_sass->addStatement(ss);
		} else {
			assertM(false, "not implemented: " << i.toString());
		}
	}

	void PTXToSASSTranslator::_translateMov(const ir::PTXInstruction &i) {
		report("  mov: bytes d=" << ir::PTXOperand::bytes(i.d.type) <<
			",a=" << ir::PTXOperand::toString(i.a.addressMode) <<
			" : " << i.toString());
		if (i.d.addressMode == ir::PTXOperand::Register &&
			ir::PTXOperand::bytes(i.d.type) == 4) {
			if (i.a.addressMode == ir::PTXOperand::Special) {
				std::string spreg = makeSpecial(i.a);
				if (spreg.length() > 0) {
					ir::SASSStatement ss(ir::SASSStatement::Instr,
						ir::SASSInstruction::S2r);
					ss.instr.addOperand(makeReg(i.d));
					ss.instr.addOperand(spreg);
					_sass->addStatement(ss);
				} else {
					ir::SASSStatement ss(ir::SASSStatement::Instr,
						ir::SASSInstruction::Mov);
					int n;
					ss.instr.addOperand(makeReg(i.d));
					switch (i.a.special) {
					// gridDim
					case ir::PTXOperand::nctaId:
						n = 0x34;
						break;
					// blockDim
					case ir::PTXOperand::ntid:
						n = 0x28;
						break;
					default:
						assertM(false, "unsupported reg: " << i.a.toString());
					}
					switch(i.a.vIndex) {
					case ir::PTXOperand::ix:
						break;
					case ir::PTXOperand::iy:
						n += 4;
						break;
					case ir::PTXOperand::iz:
						n += 8;
						break;
					default:
						assertM(false, "unsupported reg: " << i.a.toString());
					}
					ss.instr.addOperand(makeImmAddress(n));
					_sass->addStatement(ss);
				}
			} else if (i.a.addressMode == ir::PTXOperand::Immediate) {
				ir::SASSStatement ss(ir::SASSStatement::Instr,
					ir::SASSInstruction::Mov32i);
				ss.instr.addOperand(makeReg(i.d));
				ss.instr.addOperand(makeImmediate(i.a.imm_int, false));
				_sass->addStatement(ss);
			} else if (i.a.addressMode == ir::PTXOperand::Register) {
				ir::SASSStatement ss(ir::SASSStatement::Instr,
					ir::SASSInstruction::Mov);
				ss.instr.addOperand(makeReg(i.d));
				ss.instr.addOperand(makeReg(i.a));
				_sass->addStatement(ss);
			} else {
				assertM(false, "not implemented: " << i.toString());
			}
		} else if (i.addressSpace == ir::PTXInstruction::Shared &&
			i.d.addressMode == ir::PTXOperand::Register &&
			i.a.addressMode == ir::PTXOperand::Address) {
			unsigned int addr =
				_sass->getLocalSharedMemory(i.a.identifier);
			assertM(addr != (unsigned int)-1,
				"cannot found: " << i.a.identifier);
			ir::SASSStatement sl(ir::SASSStatement::Instr,
				ir::SASSInstruction::Mov);
				sl.instr.addOperand(makeReg(i.d));
				sl.instr.addOperand(makeImmediate(addr));
				_sass->addStatement(sl);
			ir::SASSStatement sh(ir::SASSStatement::Instr,
				ir::SASSInstruction::Mov);
				sh.instr.addOperand(makeReg(i.d, 1));
				sh.instr.addOperand(RZ);
				_sass->addStatement(sh);
		} else {
			assertM(false, "not implemented: " << i.toString());
		}
	}

	void PTXToSASSTranslator::_translateMul(const ir::PTXInstruction &i) {
		report("  mul: bytes d=" << ir::PTXOperand::bytes(i.d.type) <<
			",a=" << ir::PTXOperand::bytes(i.a.type) <<
			",b=" << ir::PTXOperand::bytes(i.b.type) <<
			",modifier=" << i.modifier <<
			" : " << i.toString());
		if (i.a.addressMode == ir::PTXOperand::Register &&
			i.b.addressMode == ir::PTXOperand::Register &&
			i.d.addressMode == ir::PTXOperand::Register) {
			switch (i.a.type) {
			case ir::PTXOperand::s8:
			case ir::PTXOperand::s16:
			case ir::PTXOperand::s32:
			case ir::PTXOperand::u8:
			case ir::PTXOperand::u16:
			case ir::PTXOperand::u32: {
				// 32bit integer
				ir::SASSStatement ss(ir::SASSStatement::Instr,
					ir::SASSInstruction::Imul);
				ss.instr.addOperand(makeReg(i.d));
				ss.instr.addOperand(makeReg(i.a));
				ss.instr.addOperand(makeReg(i.b));
				_sass->addStatement(ss);
				break;
			}
			case ir::PTXOperand::s64:
			case ir::PTXOperand::u64: {
				// 64bit integer
				ir::SASSStatement s1(ir::SASSStatement::Instr,
					ir::SASSInstruction::Imul);
				s1.instr.addOperand(makeReg(i.d, 0, "CC"));
				s1.instr.addOperand(makeReg(i.b));
				s1.instr.addOperand(makeReg(i.a));
				s1.instr.addModifier("U32");
				s1.instr.addModifier("U32");
				_sass->addStatement(s1);
				ir::SASSStatement s2(ir::SASSStatement::Instr,
					ir::SASSInstruction::Imad);
				s2.instr.addOperand(R0CC);
				s2.instr.addOperand(makeReg(i.b));
				s2.instr.addOperand(makeReg(i.a));
				s2.instr.addOperand(RZ);
				s2.instr.addModifier("U32");
				s2.instr.addModifier("U32");
				s2.instr.addModifier("HI");
				s2.instr.addModifier("X");
				_sass->addStatement(s2);
				ir::SASSStatement s3(ir::SASSStatement::Instr,
					ir::SASSInstruction::Imad);
				s3.instr.addOperand(R0);
				s3.instr.addOperand(makeReg(i.a));
				s3.instr.addOperand(makeReg(i.b, 1));
				s3.instr.addOperand(R0);
				s3.instr.addModifier("U32");
				s3.instr.addModifier("U32");
				s3.instr.addModifier("X");
				_sass->addStatement(s3);
				ir::SASSStatement s4(ir::SASSStatement::Instr,
					ir::SASSInstruction::Imad);
				s4.instr.addOperand(makeReg(i.d, 1));
				s4.instr.addOperand(makeReg(i.b));
				s4.instr.addOperand(makeReg(i.a, 1));
				s4.instr.addOperand(R0);
				s4.instr.addModifier("U32");
				s4.instr.addModifier("U32");
				_sass->addStatement(s4);
				break;
				break;
			}
			case ir::PTXOperand::f32: {
				// float
				ir::SASSStatement ss(ir::SASSStatement::Instr,
					ir::SASSInstruction::Fmul);
				ss.instr.addOperand(makeReg(i.d));
				ss.instr.addOperand(makeReg(i.a));
				ss.instr.addOperand(makeReg(i.b));
				_sass->addStatement(ss);
				break;
			}
			case ir::PTXOperand::f64: {
				// double
				ir::SASSStatement ss(ir::SASSStatement::Instr,
					ir::SASSInstruction::Dmul);
				ss.instr.addOperand(makeReg(i.d));
				ss.instr.addOperand(makeReg(i.a));
				ss.instr.addOperand(makeReg(i.b));
				_sass->addStatement(ss);
				break;
			}
			default:
				assertM(false, "not implemented: " <<
					ir::PTXOperand::toString(i.a.type));
			}
		} else if (i.a.addressMode == ir::PTXOperand::Register &&
			i.b.addressMode == ir::PTXOperand::Immediate &&
			i.d.addressMode == ir::PTXOperand::Register) {
			ir::SASSStatement sl(ir::SASSStatement::Instr,
				ir::SASSInstruction::Imul);
			sl.instr.addOperand(makeReg(i.d));
			sl.instr.addOperand(makeReg(i.a));
			sl.instr.addOperand(std::to_string(i.b.imm_int));
			_sass->addStatement(sl);
			ir::SASSStatement sh(ir::SASSStatement::Instr,
				ir::SASSInstruction::Imul);
			sh.instr.addOperand(makeReg(i.d, 1));
			sh.instr.addOperand(makeReg(i.a, 1));
			sh.instr.addOperand(std::to_string(i.b.imm_int));
			sh.instr.addModifier("HI");
			_sass->addStatement(sh);
		} else {
			assertM(false, "not implemented: " << i.toString());
		}
	}

	void PTXToSASSTranslator::_translateNeg(const ir::PTXInstruction &i) {
		report("  neg : " << i.toString());
		if (i.d.addressMode == ir::PTXOperand::Register &&
			i.a.addressMode == ir::PTXOperand::Register &&
			i.a.type == ir::PTXOperand::s32) {
			ir::SASSStatement ss(ir::SASSStatement::Instr,
				ir::SASSInstruction::I2i);
			ss.instr.addOperand(makeReg(i.d));
			ss.instr.addOperand("-" + makeReg(i.a));
			ss.instr.addModifier("S32");
			ss.instr.addModifier("S32");
			_sass->addStatement(ss);
		} else {
			assertM(false, "not implemented: " << i.toString());
		}
	}

	void PTXToSASSTranslator::_translateOr(const ir::PTXInstruction &i) {
		report("  or : " << i.toString());
		switch (i.d.type) {
		case ir::PTXOperand::pred: {
			ir::SASSStatement ss(ir::SASSStatement::Instr,
				ir::SASSInstruction::Psetp);
			ss.instr.addModifier("OR");
			ss.instr.addModifier("OR");
			ss.instr.addOperand(makeReg(i.d));
			ss.instr.addOperand(makeReg(i.a));
			ss.instr.addOperand(makeReg(i.a));
			ss.instr.addOperand(makeReg(i.b));
			ss.instr.addOperand(makeReg(i.b));
			_sass->addStatement(ss);
			break;
		}
		default: {
			ir::SASSStatement ss(ir::SASSStatement::Instr,
				ir::SASSInstruction::Lop);
			ss.instr.addModifier("OR");
			ss.instr.addOperand(makeReg(i.d));
			ss.instr.addOperand(makeReg(i.a));
			ss.instr.addOperand(makeReg(i.b));
			_sass->addStatement(ss);
		}
		}
	}


	void PTXToSASSTranslator::_translateRet(const ir::PTXInstruction &i) {
		report("  ret: " << i.toString());
		ir::SASSStatement ss(ir::SASSStatement::Instr,
			ir::SASSInstruction::Exit);
		_sass->addStatement(ss);
	}

	void PTXToSASSTranslator::_translateSelp(const ir::PTXInstruction &i) {
		report("  selp: " << i.toString());
		if (i.d.addressMode == ir::PTXOperand::Register &&
			i.a.addressMode == ir::PTXOperand::Register &&
			i.b.addressMode == ir::PTXOperand::Register &&
			i.c.type == ir::PTXOperand::pred) {
			ir::SASSStatement ss(ir::SASSStatement::Instr,
				ir::SASSInstruction::Sel);
			ss.instr.addOperand(makeReg(i.d));
			ss.instr.addOperand(makeReg(i.a));
			ss.instr.addOperand(makeReg(i.b));
			ss.instr.addOperand(makeReg(i.c));
			_sass->addStatement(ss);
		} else {
			assertM(false, "not implemented: " << i.toString());
		}
	}

	void PTXToSASSTranslator::_translateSetp(const ir::PTXInstruction &i) {
		report("  setp: mod=" << i.modifier <<
			" comp=" << i.comparisonOperator <<
			" type=" << i.type <<
			" : " << i.toString());
		if (i.d.addressMode == ir::PTXOperand::Register &&
			i.a.addressMode == ir::PTXOperand::Register &&
			ir::PTXOperand::bytes(i.type) == 4) {
			ir::SASSStatement ss(ir::SASSStatement::Instr,
				ir::SASSInstruction::Isetp);
			ss.instr.addOperand(makeReg(i.d));
			ss.instr.addOperand(PT);
			ss.instr.addOperand(makeReg(i.a));
			switch (i.b.addressMode) {
			case ir::PTXOperand::Register:
				ss.instr.addOperand(makeReg(i.b));
				break;
			case ir::PTXOperand::Immediate:
				ss.instr.addOperand(makeImmediate(i.b.imm_int));
				break;
			default:
				assertM(false, "not implemented: " <<
					i.toString());
			}
			ss.instr.addOperand(PT);
			ss.instr.addModifier(
				makeComparison(i.comparisonOperator));
			if (i.type == ir::PTXOperand::u32) {
				ss.instr.addModifier("U32");
			}
			ss.instr.addModifier("AND");
			_sass->addStatement(ss);
		} else {
			assertM(false, "not implemented: " << i.toString());
		}
	}

	void PTXToSASSTranslator::_translateShl(const ir::PTXInstruction &i) {
		report("  shl: bytes d=" << ir::PTXOperand::bytes(i.d.type) <<
			",a=" << ir::PTXOperand::bytes(i.a.type) <<
			",b=" << ir::PTXOperand::bytes(i.b.type) <<
			" : " << i.toString());
		if (i.d.addressMode == ir::PTXOperand::Register &&
			i.a.addressMode == ir::PTXOperand::Register) {
			ir::SASSStatement ss(ir::SASSStatement::Instr,
				ir::SASSInstruction::Shl);
			ss.instr.addOperand(makeReg(i.d));
			ss.instr.addOperand(makeReg(i.a));
			switch (i.b.addressMode) {
			case ir::PTXOperand::Register:
				ss.instr.addOperand(makeReg(i.b));
				break;
			case ir::PTXOperand::Immediate:
				ss.instr.addOperand(makeImmediate(i.b.imm_int));
				break;
			default:
				assertM(false, "not implemented: " << i.toString());
			}
			_sass->addStatement(ss);
		} else {
			assertM(false, "not implemented: " << i.toString());
		}
	}

	void PTXToSASSTranslator::_translateShr(const ir::PTXInstruction &i) {
		report("  shr: " << i.toString());
		if (i.d.addressMode == ir::PTXOperand::Register &&
			i.a.addressMode == ir::PTXOperand::Register) {
			ir::SASSStatement ss(ir::SASSStatement::Instr,
				ir::SASSInstruction::Shr);
			if (i.d.type == ir::PTXOperand::u32) {
				ss.instr.addModifier("U32");
			}
			ss.instr.addOperand(makeReg(i.d));
			ss.instr.addOperand(makeReg(i.a));
			switch (i.b.addressMode) {
			case ir::PTXOperand::Register:
				ss.instr.addOperand(makeReg(i.b));
				break;
			case ir::PTXOperand::Immediate:
				ss.instr.addOperand(makeImmediate(i.b.imm_int));
				break;
			default:
				assertM(false, "not implemented: " << i.toString());
			}
			_sass->addStatement(ss);
		} else {
			assertM(false, "not implemented: " << i.toString());
		}
	}

	void PTXToSASSTranslator::_translateSt(const ir::PTXInstruction &i) {
		report("  st: a=" << ir::PTXOperand::bytes(i.a.type) <<
			" : " << i.toString());
		if (i.addressSpace == ir::PTXInstruction::Global &&
			i.a.addressMode == ir::PTXOperand::Register) {
			ir::SASSStatement ss(ir::SASSStatement::Instr,
				ir::SASSInstruction::St);
			ss.instr.addModifier("E");
			switch (ir::PTXOperand::bytes(i.type)) {
			case 4:
				// 32bit
				break;
			case 8:
				// 64bit
				ss.instr.addModifier("64");
				break;
			case 1:
			case 2:
				// 8bit or 16bit
				ss.instr.addModifier("U" +
					std::to_string(ir::PTXOperand::bytes(i.type)*8));
				break;
			default:
				assertM(false, "not implemented: " <<
					ir::PTXOperand::toString(i.a.type));
			}
			ss.instr.addOperand("[" + makeReg(i.d) + "]");
			ss.instr.addOperand(makeReg(i.a));
			_sass->addStatement(ss);
		} else if (i.addressSpace == ir::PTXInstruction::Shared &&
			i.a.addressMode == ir::PTXOperand::Register) {
			ir::SASSStatement ss(ir::SASSStatement::Instr,
				ir::SASSInstruction::Sts);
			ss.instr.addOperand("[" + makeReg(i.d) + "]");
			ss.instr.addOperand(makeReg(i.a));
			_sass->addStatement(ss);
		} else {
			assertM(false, "not implemented: " << i.toString());
		}
	}

	void PTXToSASSTranslator::_translateSub(const ir::PTXInstruction &i) {
		report("  sub: bytes d=" << ir::PTXOperand::bytes(i.d.type) <<
			",a=" << ir::PTXOperand::bytes(i.a.type) <<
			",b=" << ir::PTXOperand::bytes(i.b.type) <<
			" : " << i.toString());
		if (i.a.addressMode == ir::PTXOperand::Register &&
			i.b.addressMode == ir::PTXOperand::Register &&
			i.d.addressMode == ir::PTXOperand::Register) {
			switch (i.a.type) {
			case ir::PTXOperand::s8:
			case ir::PTXOperand::s16:
			case ir::PTXOperand::s32:
			case ir::PTXOperand::u8:
			case ir::PTXOperand::u16:
			case ir::PTXOperand::u32: {
				// 32bit integer
				ir::SASSStatement ss(ir::SASSStatement::Instr,
					ir::SASSInstruction::Iadd);
				ss.instr.addOperand(makeReg(i.d));
				ss.instr.addOperand(makeReg(i.a));
				ss.instr.addOperand("-" + makeReg(i.b));
				_sass->addStatement(ss);
				break;
			}
			case ir::PTXOperand::s64:
			case ir::PTXOperand::u64: {
				// 64bit integer
				ir::SASSStatement sl(ir::SASSStatement::Instr,
					ir::SASSInstruction::Iadd);
				sl.instr.addOperand(makeReg(i.d, 0, "CC"));
				sl.instr.addOperand(makeReg(i.a));
				sl.instr.addOperand("-" + makeReg(i.b));
				_sass->addStatement(sl);
				ir::SASSStatement sh(ir::SASSStatement::Instr,
					ir::SASSInstruction::Iadd);
				sh.instr.addOperand(makeReg(i.d, 1));
				sh.instr.addOperand(makeReg(i.a, 1));
				sh.instr.addOperand("-" + makeReg(i.b, 1));
				sh.instr.addModifier("X");
				_sass->addStatement(sh);
				break;
			}
			case ir::PTXOperand::f32: {
				// float
				ir::SASSStatement ss(ir::SASSStatement::Instr,
					ir::SASSInstruction::Fadd);
				ss.instr.addOperand(makeReg(i.d));
				ss.instr.addOperand(makeReg(i.a));
				ss.instr.addOperand("-" + makeReg(i.b));
				_sass->addStatement(ss);
				break;
			}
			case ir::PTXOperand::f64: {
				// double
				ir::SASSStatement ss(ir::SASSStatement::Instr,
					ir::SASSInstruction::Dadd);
				ss.instr.addOperand(makeReg(i.d));
				ss.instr.addOperand(makeReg(i.a));
				ss.instr.addOperand("-" + makeReg(i.b));
				_sass->addStatement(ss);
				break;
			}
			default:
				assertM(false, "not implemented: " <<
					ir::PTXOperand::toString(i.a.type));
			}
		} else {
			assertM(false, "not implemented: " << i.toString());
		}
	}

	std::string PTXToSASSTranslator::makeReg(ir::PTXOperand o, int a,
		std::string mod) {
		RegType r = o.reg;
		std::string ret = "";
		switch(o.type) {
		case ir::PTXOperand::pred:
			// pred
			if (_pred.find(r) != _pred.end()) {
				report("    pred " << r << "->" << _pred[r]);
				return ((o.condition == ir::PTXOperand::InvPred) ? "!P":"P") +
					std::to_string(_pred[r]) +
					((mod == "") ? "":"."+mod);
                	}
			break;
		case ir::PTXOperand::s64:
		case ir::PTXOperand::u64:
		case ir::PTXOperand::f64:
		case ir::PTXOperand::b64:
			// 64bit
			if (_rangemap[r|ptx64flag].isAssigned(_line)) {
				RegType sass = _rangemap[r|ptx64flag].sass;
				ret = "R" + std::to_string(sass+a) +
					((mod == "") ? "":"."+mod);
			} else {
				RegType sass = _assignRange64(r);
				ret = "R" + std::to_string(sass+a) +
					((mod == "") ? "":"."+mod);
			}
			break;
		default:
			// 32bit
			if (_rangemap[r].isAssigned(_line)) {
				ret = "R" + std::to_string(_rangemap[r].sass) +
					((mod == "") ? "":"."+mod);
			} else {
				RegType sass = _assignRange32(r);
				ret = "R" + std::to_string(sass) +
					((mod == "") ? "":"."+mod);
			}
		}
		assertM(ret != "", "invalid reg#: " << r);

		if (o.offset != 0) {
			std::stringstream ss;
			ss << ret << "+0x" << std::hex << o.offset;
			return ss.str();
		} else {
			return ret;
		}
	}

	std::string PTXToSASSTranslator::makeAddress(ir::PTXOperand o, int a) {
		int n = 0x140;
		assertM(_sass->isParameter(o.identifier),
			"invalid address: " << o.toString());
		return makeImmAddress(
			n + _sass->getParameterOffset(o.identifier) + a*4);
	}

	std::string PTXToSASSTranslator::makeImmAddress(int n) {
		std::stringstream ss;
		ss << "c [0x0] [" << makeImmediate(n) << "]";
		return ss.str();
	}

	std::string PTXToSASSTranslator::makeImmediate(int n, bool rz) {
		if (n == 0 && rz) {
			return RZ;
		} else {
			std::stringstream ss;
			ss << "0x" << std::hex << n;
			return ss.str();
		}
	}

	std::string PTXToSASSTranslator::makeSpecial(ir::PTXOperand o) {
		std::string ret;
		switch(o.special) {
		case ir::PTXOperand::ctaId:
			ret = "SR_CTAid_";
			break;
		case ir::PTXOperand::tid:
			ret = "SR_Tid_";
			break;
		case ir::PTXOperand::laneId:
			ret = "SR_LaneId";
			return ret;
		default:
			return "";
		}
		switch(o.vIndex) {
		case ir::PTXOperand::ix:
			ret += "X";
			break;
		case ir::PTXOperand::iy:
			ret += "Y";
			break;
		case ir::PTXOperand::iz:
			ret += "Z";
			break;
		default:
			assertM(false, "unsupported special: " << o.toString());
		}
		return ret;
	}

	std::string PTXToSASSTranslator::makeModifier(ir::PTXInstruction::Modifier m) {
		switch(m) {
		case ir::PTXInstruction::hi:
			return "HI";
		case ir::PTXInstruction::lo:
			return "LO";
		default:
			assertM(false, "unsupported modifier: "
				<< ir::PTXInstruction::toString(m));
		}
	}

	std::string PTXToSASSTranslator::makeComparison(ir::PTXInstruction::CmpOp c) {
		switch(c) {
		case ir::PTXInstruction::Eq:
			return "EQ";
		case ir::PTXInstruction::Ne:
			return "NE";
		case ir::PTXInstruction::Ge:
			return "GE";
		case ir::PTXInstruction::Gt:
			return "GT";
		case ir::PTXInstruction::Le:
			return "LE";
		case ir::PTXInstruction::Lt:
			return "LT";
		default:
			assertM(false, "unsupported cmpop: "
				<< ir::PTXInstruction::toString(c));
		}
	}


	PTXToSASSTranslator::RegisterRange::RegisterRange(int h, int t, RegType s)
		: head(h), tail(t), sass(s) {
	}

	bool PTXToSASSTranslator::RegisterRange::isAssigned(int l) {
		return (sass != sassInvalid && head <= l && l <= tail);
	}

	bool PTXToSASSTranslator::RegisterRange::isObsoleted(int l) {
		return (sass != sassInvalid && tail < l);
	}

	void PTXToSASSTranslator::_makeRange() {
		_line = 1;
		_rangemap.clear();
		ControlTree ctree;
		ctree.analyze(*_ptx);
		_makeRange(ctree.get_root_node());
		_line = 1;

#ifdef DEBUG_REG_RANGE
		for (std::map<RegType,RegisterRange>::iterator rr =
			_rangemap.begin(); rr != _rangemap.end(); rr++) {
			std::cout <<"result: " <<
				(rr->first & ~ptx64flag) <<
				", [" << rr->second.head << "-" <<
				rr->second.tail << "]";
		}
#endif /* DEBUG_REG_RANGE */
	}

	void PTXToSASSTranslator::_makeRange(const ControlTree::Node* n,
		int *from, int *to) {
		report("makeRange node: " << n->label());

		switch (n->rtype()) {
		case ControlTree::Inst:
			_makeRange(
				static_cast<const ControlTree::InstNode*>(n),
				from, to);
			break;
		case ControlTree::Block:
			_makeRange(
				static_cast<const ControlTree::BlockNode*>(n),
				from, to);
			break;
		case ControlTree::IfThen:
			_makeRange(
				static_cast<const ControlTree::IfThenNode*>(n),
				from, to);
			break;
		case ControlTree::Natural: 
			_makeRange(
				static_cast<const ControlTree::NaturalNode*>(n),
				from, to);
			break;
		default:
			assertM(false, "invalid type: " << n->rtype());
		}

		report("makeRange done node: " << n->label());
	}

	void PTXToSASSTranslator::_makeRange(const ControlTree::InstNode* n,
		int *from, int *to) {
		report("makeRange inst: " << n->label());

		for (ControlTree::InstructionList::const_iterator
			it = n->bb()->instructions.begin(), 
			end = n->bb()->instructions.end();
			it != end; it++) {
			_makeRange(
				static_cast<ir::PTXInstruction &>(**it),
				from, to);
		}

		report("makeRange done inst: " << n->label());
	}

	void PTXToSASSTranslator::_makeRange(const ControlTree::BlockNode* n,
		int *from, int *to) {
		report("makeRange block: " << n->label());

		for (ControlTree::NodeList::const_iterator
			it = n->children().begin(),
			end = n->children().end();
			it != end; it++) {
			_makeRange(*it, from, to);
		}	

		report("makeRange done block: " << n->label());
	}

	void PTXToSASSTranslator::_makeRange(const ControlTree::IfThenNode* n,
		int *from, int *to) {
		report("makeRange ifthen: " << n->label());

		_makeRange(n->cond(), from, to);

		report("  * true: " << n->label());
		_makeRange(n->ifTrue(), from, to);
		if (n->ifFalse() != NULL) {
			report("  * false: " << n->label());
			_makeRange(n->ifFalse(), from, to);
		}

		report("makeRange done ifthen: " << n->label());
	}

	void PTXToSASSTranslator::_makeRange(const ControlTree::NaturalNode* n,
		int *from, int *to) {
		report("makeRange natural: " << n->label());

		int lf = 0, lt = 0;

		ControlTree::NodeList::const_iterator end =
			(--n->children().end());
		for (ControlTree::NodeList::const_iterator it =
			n->children().begin(); it != end; it++) {
			_makeRange(*it, &lf, &lt);
		}
		_makeRange(*end, &lf, &lt);
		report("makeRange range: " << lf << "-" << lt << " : " <<
			n->label());
		_renewRange(lf, lt);
		if (from != NULL && *from == 0 && lf != 0) {
			*from = lf;
		}
		if (to != NULL && lt != 0) {
			*to = lt;
		}

		report("makeRange done natural: " << n->label());
	}

	void PTXToSASSTranslator::_makeRange(const ir::PTXInstruction &i,
		int *from, int *to) {
		report("makeRange instr[" << _line << "]: " << i.toString());
		_makeRange(i.d);
		_makeRange(i.a);
		_makeRange(i.b);
		_makeRange(i.c);
		if (from != NULL && *from == 0) {
			*from = _line;
		}
		if (to != NULL) {
			*to = _line;
		}
		_line++;
	}

	void PTXToSASSTranslator::_makeRange(const ir::PTXOperand &o) {
		if (o.addressMode != ir::PTXOperand::Register &&
			o.addressMode != ir::PTXOperand::Indirect) {
			return;
		}

		switch(o.type) {
		case ir::PTXOperand::pred:
			break;
		case ir::PTXOperand::s64:
		case ir::PTXOperand::u64:
		case ir::PTXOperand::f64:
		case ir::PTXOperand::b64: {
			// 64bit
			RegType ptx = o.reg | ptx64flag;
			if (_rangemap.find(ptx) == _rangemap.end()) {
				report("makeRange operand r=" << o.reg <<
					", add line=" << _line);
				RegisterRange r(_line, _line);
				_rangemap[ptx] = r;
			} else {
				report("makeRange operand r=" << o.reg <<
					", update " << _rangemap[ptx].head <<
					"-" << _line);
				_rangemap[ptx].tail = _line;
			}
			break;
		}
		default:
			// 32bit
			if (_rangemap.find(o.reg) == _rangemap.end()) {
				report("makeRange operand r=" << o.reg <<
					", add line=" << _line);
				RegisterRange r(_line, _line);
				_rangemap[o.reg] = r;
			} else {
				report("makeRange operand r=" << o.reg <<
					", update " << _rangemap[o.reg].head <<
					"-" << _line);
				_rangemap[o.reg].tail = _line;
			}
		}
	}

	void PTXToSASSTranslator::_renewRange(int from, int to) {
		for (std::map<RegType,RegisterRange>::iterator rr =
			_rangemap.begin(); rr != _rangemap.end(); rr++) {
			if (rr->second.head < from &&
				rr->second.tail >= from &&
				rr->second.tail < to) {
				report("renew: " <<
					(rr->first & ~ptx64flag) <<
					", [" << rr->second.head << "-" <<
					rr->second.tail << "]->" << to);
				rr->second.tail = to;
			}
		}
	}

	void PTXToSASSTranslator::_refreshRange() {
		for (std::map<RegType,RegisterRange>::iterator rr =
			_rangemap.begin(); rr != _rangemap.end(); rr++) {
			if (! rr->second.isObsoleted(_line)) {
				continue;
			}
			RegType sass = rr->second.sass;
			if (rr->first & ptx64flag) {
				assertM(_reg64.find(sass) != _reg64.end(),
					"ptx=" << (rr->first&~ptx64flag) <<
					", sass=" << sass << " is free!");
				_reg64.erase(sass);
				_reg64.erase(sass+1);
				report("refreshRange64 ptx=" <<
					(rr->first&~ptx64flag) <<
					" sass=" << sass << " line=" << _line);
			} else {
				assertM(_reg32.find(sass) != _reg32.end(),
					"ptx=" << rr->first << ", sass=" <<
					sass << " is free!");
				_reg32.erase(sass);
				report("refreshRange32 ptx=" << rr->first <<
					" sass=" << sass << " line=" << _line);
			}
			rr->second.sass = sassInvalid;
		}

#ifdef DEBUG_REG_ASSIGNED
		std::cout << "* reg:";
		for (RegType sass = sassRegStart; sass < sassRegEnd; sass++) {
			std::string st = "";
			if(_reg64.find(sass) != _reg64.end()) {
				st = (_reg32.find(sass) != _reg32.end()) ?
					"X":"6";
			} else {
				st = (_reg32.find(sass) != _reg32.end()) ?
					"3":".";
			}
			std::cout << st;
		}
		std::cout << std::endl;
#endif /* DEBUG_REG_ASSIGNED */
	}

	PTXToSASSTranslator::RegType PTXToSASSTranslator::_assignRange32(RegType ptx) {
		for (RegType sass = sassRegStart; sass < sassRegEnd; sass++) {
			if (_reg32.find(sass) == _reg32.end() &&
				_reg64.find(sass) == _reg64.end()) {
				_reg32[sass] = ptx;
				assertM(_rangemap[ptx].sass == sassInvalid,
					"reg ptx=" << ptx <<
					"(sass=" << sass << ") is not free.");
				_rangemap[ptx].sass = sass;
				report("assignRange32 ptx=" << ptx <<
					", sass=" << sass);
				return sass;
			}
		}
		assertM(false, "all registers are using...");
		return sassInvalid;
	}

	PTXToSASSTranslator::RegType PTXToSASSTranslator::_assignRange64(RegType ptx) {
		for (RegType sass = sassRegStart; sass < sassRegEnd; sass+=2) {
			if (_reg32.find(sass) == _reg32.end() &&
				_reg32.find(sass+1) == _reg32.end() &&
				_reg64.find(sass) == _reg64.end()) {
				_reg64[sass] = _reg64[sass+1] = ptx;
				assertM(_rangemap[ptx|ptx64flag].sass == sassInvalid,
					"reg ptx=" << ptx << "(sass=" <<
					sass << ") is not free.");
				_rangemap[ptx|ptx64flag].sass = sass;
				report("assignRange64 ptx=" << ptx <<
					", sass=" << sass);
				return sass;
			}
		}
		assertM(false, "all registers are using...");
		return sassInvalid;
	}
}
