/*! \file AffineLinearScan.cpp
	\author Diogo Nunes Sampaio <dnsampaio@gmail.com>
	\date Wednesday February 15, 2012
	\brief The source file for the AffineLinearScan class.
*/

#ifndef AFFINE_LINEAR_SCAN_CPP_
#define AFFINE_LINEAR_SCAN_CPP_

// Ocelot Includes
#include <ocelot/transforms/interface/AffineLinearScan.h>
#include <ocelot/transforms/interface/AffineRegister.h>
#include <ocelot/ir/interface/Module.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#ifdef NDEBUG
#undef NDEBUG
#endif

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#ifdef INFO
#undef INFO
#endif

#ifdef DEBUG
#undef DEBUG
#endif

#ifdef DEBUG_DETAILS
#undef DEBUG_DETAILS
#endif

#ifdef REPORT_ERROR_LEVEL
#undef REPORT_ERROR_LEVEL
#endif

#define REPORT_ERROR_LEVEL 5
#define REPORT_BASE 0
#if REPORT_BASE == 0
#define NDEBUG 1
#endif
#define BUG_INFO 5
#define INFO 4
#define DEBUG 3
#define DEBUG_DETAILS 2


namespace transforms
{

unsigned AffineLinearScan::MAX_WARPS = 32;

analysis::AffineAnalysis& AffineLinearScan::_afa()
{
	Analysis* aff = getAnalysis("AffineAnalysis");
	assert(aff != 0);

	return *static_cast<analysis::AffineAnalysis*>(aff);
}

void AffineLinearScan::_clear()
{
	LinearScanRegisterAllocationPass::_clear();
	_shared.clear();
}

void AffineLinearScan::_spill()
{
	if(_shared.bytes() > 0)
	{
		AffineRegister::warpPosition = _dfg().newRegister();
	}

	LinearScanRegisterAllocationPass::_spill();
}

void AffineLinearScan::_extendStack()
{
	_shared.declaration(_kernel->locals, MAX_WARPS);
	reportE(INFO, "Kernel " << _kernel->name << " requires " << _shared.bytes()
		<< " bytes of shared memory per warp, total of "
		<< MAX_WARPS * _shared.bytes() << '(' << MAX_WARPS << " warps)");
	LinearScanRegisterAllocationPass::_extendStack();
	reportE(DEBUG, "Writing warp local memory stack access information");
	
	if(_shared.bytes() == 0) return;
	/* warpid = (size_x * ( size_y * z + y ) + x) >> 5
	 * a = size_y
	 * b = z
	 * c = y
	 * a = mad a z c
	 * b = size_x
	 * c = x
	 * a = mad a b c
	 * a = shr a 5 (>>5 == /32)
	 * memPosition = memInitialPosition [ warpid * bytesPerWarp ]
	 */
	analysis::DataflowGraph::iterator block = _dfg().begin();
	RegisterId a, b, c;

	/* Use a AffineRegister temporary register of type u32 if available */
	if(AffineRegister::tempRegisters.count(ir::PTXOperand::DataType::u32) != 0)
	{
		a = AffineRegister::tempRegisters[ir::PTXOperand::DataType::u32];
	}
	else
	{
		a = _dfg().newRegister();
	}
	
	b = _dfg().newRegister();

	/* If memory size is 32 bits, can use warpPosition variable as temporary */
	if(_m->addressSize() == 32)
	{
		c = AffineRegister::warpPosition;
	}
	else
	{
		c = _dfg().newRegister();
	}
	
	// size_y = %ntid.y
	ir::PTXInstruction sizeY(ir::PTXInstruction::Mov);
	sizeY.d = ir::PTXOperand(ir::PTXOperand::Register,
		ir::PTXOperand::DataType::u32, a);
	sizeY.a = ir::PTXOperand(ir::PTXOperand::ntid,
		ir::PTXOperand::iy, ir::PTXOperand::u32);
	sizeY.type = ir::PTXOperand::DataType::u32;
	_dfg().insert(block, sizeY, 0);

	// z = %tid.z
	ir::PTXInstruction z(ir::PTXInstruction::Mov);
	z.d = ir::PTXOperand(ir::PTXOperand::Register,
		ir::PTXOperand::DataType::u32, b);
	z.a = ir::PTXOperand(ir::PTXOperand::tid,
		ir::PTXOperand::iz, ir::PTXOperand::u32);
	z.type = ir::PTXOperand::DataType::u32;
	_dfg().insert(block, z, 1);

	// y = %tid.y
	ir::PTXInstruction y(ir::PTXInstruction::Mov);
	y.d = ir::PTXOperand(ir::PTXOperand::Register,
		ir::PTXOperand::DataType::u32, c);
	y.a = ir::PTXOperand(ir::PTXOperand::tid,
		ir::PTXOperand::iy, ir::PTXOperand::u32);
	y.type = ir::PTXOperand::DataType::u32;
	_dfg().insert(block, y, 2);

	ir::PTXInstruction mad1(ir::PTXInstruction::Mad);
	mad1.d = sizeY.d;
	mad1.a = sizeY.d;
	mad1.b = z.d;
	mad1.c = y.d;
	mad1.type = ir::PTXOperand::DataType::u32;
	mad1.modifier = ir::PTXInstruction::Modifier::lo;
	_dfg().insert(block, mad1, 3);

	// size_x = %ntid.x
	ir::PTXInstruction sizeX(ir::PTXInstruction::Mov);
	sizeX.d = z.d;
	sizeX.a = ir::PTXOperand(ir::PTXOperand::ntid,
		ir::PTXOperand::ix, ir::PTXOperand::u32);
	sizeX.type = ir::PTXOperand::DataType::u32;
	_dfg().insert(block, sizeX, 4);

	// x = %tid.x
	ir::PTXInstruction x(ir::PTXInstruction::Mov);
	x.d = y.d;
	x.a = ir::PTXOperand(ir::PTXOperand::tid,
		ir::PTXOperand::ix, ir::PTXOperand::u32);
	x.type = ir::PTXOperand::DataType::u32;
	_dfg().insert(block, x, 5);

	// 1) warpid = size_x * size_y
	ir::PTXInstruction mad2(ir::PTXInstruction::Mad);
	mad2.d = mad1.d;
	mad2.a = mad1.d;
	mad2.b = sizeX.d;
	mad2.c = x.d;
	mad2.type = ir::PTXOperand::DataType::u32;
	mad2.modifier = ir::PTXInstruction::Modifier::lo;
	_dfg().insert(block, mad2, 6);

	// 5) warpid = [size_x * y + size_x * size_y * z + x] >> 5
	ir::PTXInstruction shr(ir::PTXInstruction::Shr);
	shr.d = mad2.d;
	shr.a = mad2.d;
	shr.b = ir::PTXOperand(5);
	shr.type = ir::PTXOperand::DataType::u32;
	_dfg().insert(block, shr, 7);

	// 6) position = warpid * stride
	ir::PTXInstruction position(ir::PTXInstruction::Mul);
	position.d = shr.d;
	position.a = shr.d;
	position.b = ir::PTXOperand(_shared.bytes());
	position.type = ir::PTXOperand::DataType::u32;
	position.modifier = ir::PTXInstruction::Modifier::lo;
	_dfg().insert(block, position, 8);

	//%memoryStart = stack name;
	ir::PTXInstruction memoryStart(ir::PTXInstruction::Mov);
	memoryStart.a = ir::PTXOperand(_shared.name() +
		"[" + position.d.toString() + "]");
	if(_m->addressSize() == 32)
	{
		memoryStart.d = x.d;
		memoryStart.type = ir::PTXOperand::DataType::u32;
	}
	else
	{
		memoryStart.d = ir::PTXOperand(ir::PTXOperand::Register,
			ir::PTXOperand::DataType::u64, AffineRegister::warpPosition);
		memoryStart.type = ir::PTXOperand::DataType::u64;
	}
	_dfg().insert(block, memoryStart, 9);

}

void AffineLinearScan::_addCoalesced(const RegisterId id,
	const analysis::DataflowGraph::Type type)
{
	_coalesced[id] = new AffineRegister(id, type, &_memoryStack, &_shared);
}

void AffineLinearScan::_coalesce()
{
	LinearScanRegisterAllocationPass::_coalesce();
	CoalescedRegisterMap::iterator affReg = _ssa.begin();
	while(affReg != _ssa.end())
	{
		AffineRegister &ar =
			static_cast<AffineRegister &>(*_coalesced[affReg->second]);
		reportE(DEBUG, "Coalesced: " << affReg->second);
		reportE(DEBUG, "Start: " << ar.state());
		reportE(DEBUG, "In state: " << _afa().state(affReg->first));
		ar.combineState(_afa().state(affReg->first));
		reportE(DEBUG, "Out state: " << ar.state());
		affReg++;
	}
}

AffineLinearScan::AffineLinearScan(unsigned regs) :
		LinearScanRegisterAllocationPass(regs-1,
			{"AffineAnalysis", "DataflowGraphAnalysis"}, 0),
		_shared("ocelot_affine_stack", MemoryArray::MemoryDirective::Shared,
			MemoryArray::StackAddressSpace::Shared),
		_m(NULL)
{
	
}

void AffineLinearScan::initialize(const ir::Module& m)
{
	reportE(DEBUG, "Running affine linear scan");
	_m = &m;
}

void AffineLinearScan::runOnKernel(ir::IRKernel& k)
{
	auto dfg = static_cast<analysis::DataflowGraph*>(
		getAnalysis("DataflowGraphAnalysis"));
	
	dfg->convertToSSAType(analysis::DataflowGraph::Gated);
	
	#if AFFINE_REGISTER_PROFILE_H_
	affineProfiler::resetSpillData();
	#endif
	
	AffineRegister::tempRegisters.clear();
	AffineRegister::warpPosition = 0;
	_shared.clear();
	
	LinearScanRegisterAllocationPass::runOnKernel(k);
	
	#if AFFINE_REGISTER_PROFILE_H_
	if(!k.function())
	{
		affineProfiler::printSpillResults(k.name);
	}
	#endif
}


void AffineLinearScan::finalize()
{
	_clear();
	reportE(DEBUG, "Finalizing affine linear scan");
}

}

#endif

