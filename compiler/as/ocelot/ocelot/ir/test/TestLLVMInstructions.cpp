/*!
	\file TestLLVMInstructions.cpp
	\date Monday July 27, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The source file for the TestLLVMInstructions class.
*/

#ifndef TEST_LLVM_INSTRUCTIONS_CPP_INCLUDED
#define TEST_LLVM_INSTRUCTIONS_CPP_INCLUDED

#include <ocelot/ir/test/TestLLVMInstructions.h>
#include <hydrazine/interface/ArgumentParser.h>

namespace test
{
	
	bool TestLLVMInstructions::check( const ir::Instruction& i, 
		const std::string& reference )
	{
		if( !i.valid().empty() )
		{
			status << "Instruction " << i.toString() << " is not valid: " 
				<< i.valid() << "\n";
			return false;
		}
		
		if( reference != i.toString() )
		{
			status << "Instruction \"" << i.toString() 
				<< "\" does not match reference \"" << reference << "\"\n";
			return false;
		}
		status << " Checked instruction \"" << i.toString() << "\"\n";
		return true;
	}

	bool TestLLVMInstructions::testAdd()
	{
		ir::LLVMAdd add;
		add.d.name = "<result>";
		add.d.type.category = Type::Element;
		add.d.type.type = Instruction::I32;

		add.a.constant = true;
		add.a.i32 = 4;
		add.a.type.category = Type::Element;
		add.a.type.type = Instruction::I32;
		
		add.b.name = "%var";
		add.b.type.category = Type::Element;
		add.b.type.type = Instruction::I32;

		std::string reference = "<result> = add i32 4, %var";
	
		if( !check( add, reference ) ) return false;

		status << "Add Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testAlloca()
	{
		ir::LLVMAlloca alloca;
		alloca.d.name = "%ptr";
		alloca.d.type.category = Type::Pointer;
		alloca.d.type.type = Instruction::I32;
		
		std::string reference = "%ptr = alloca i32";
		
		if( !check( alloca, reference) ) return false;

		alloca.elements = 4;
		reference = "%ptr = alloca i32, i32 4";
		
		if( !check( alloca, reference ) ) return false;
		
		alloca.alignment = 1024;
		reference = "%ptr = alloca i32, i32 4, align 1024";
		
		if( !check( alloca, reference ) ) return false;
		
		alloca.elements = 1;
		reference = "%ptr = alloca i32, align 1024";
		
		if( !check( alloca, reference ) ) return false;
		
		status << "Alloca Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testAnd()
	{
		ir::LLVMAnd And;
		And.d.name = "<result>";
		And.d.type.category = Type::Element;
		And.d.type.type = Instruction::I32;
		And.a.type.category = Type::Element;
		And.a.type.type = Instruction::I32;
		And.a.constant = true;
		And.a.i32 = 4;
		And.b.name = "%var";
		And.b.type.category = Type::Element;
		And.b.type.type = Instruction::I32;
		
		std::string reference = "<result> = and i32 4, %var";
		
		if( !check( And, reference ) ) return false;
		
		And.b.constant = true;
		And.a.i32 = 15;
		And.b.i32 = 40;
		
		reference = "<result> = and i32 15, 40";
		
		if( !check( And, reference ) ) return false;
		
		And.b.constant = true;
		And.a.i32 = 4;
		And.b.i32 = 8;
		
		reference = "<result> = and i32 4, 8";
		
		if( !check( And, reference ) ) return false;
		
		status << "And Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testAshr()
	{
		ir::LLVMAshr ashr;
		ashr.d.name = "<result>";
		ashr.d.type.category = Type::Element;
		ashr.d.type.type = Instruction::I32;
		ashr.a.type.category = Type::Element;
		ashr.a.type.type = Instruction::I32;
		ashr.a.constant = true;
		ashr.a.i32 = 4;
		
		ashr.b.type.category = Type::Element;
		ashr.b.type.type = Instruction::I32;
		ashr.b.constant = true;
		ashr.b.i32 = 1;
		
		std::string reference = "<result> = ashr i32 4, 1";
	
		if( !check( ashr, reference ) ) return false;
	
		ashr.b.type.type = Instruction::I8;
		ashr.a.type.type = Instruction::I8;
		ashr.d.type.type = Instruction::I8;
		
		ashr.a.i8 = -2;
		ashr.b.i8 = 1;
		
		reference = "<result> = ashr i8 -2, 1";
	
		if( !check( ashr, reference ) ) return false;
		
		ashr.b.type.type = Instruction::I32;
		ashr.a.type.type = Instruction::I32;
		ashr.d.type.type = Instruction::I32;
		
		ashr.d.type.category = Type::Vector;
		ashr.d.type.vector = 2;
		
		ashr.a.type.category = Type::Vector;
		ashr.a.type.vector = 2;
		ashr.a.values.resize( 2 );
		ashr.a.values[0].i32 = -2;
		ashr.a.values[1].i32 = 4;
		
		ashr.b.type.category = Type::Vector;
		ashr.b.type.vector = 2;
		ashr.b.values.resize( 2 );
		ashr.b.values[0].i32 = 1;
		ashr.b.values[1].i32 = 3;
		
		reference 
			= "<result> = ashr < 2 x i32 > < i32 -2, i32 4 >, < i32 1, i32 3 >";
	
		if( !check( ashr, reference ) ) return false;
		
		status << "Ashr Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testBitcast()	
	{
		ir::LLVMBitcast bitcast;
		bitcast.d.name = "%X";
		bitcast.d.type.category = Type::Element;
		bitcast.d.type.type = Instruction::I8;

		bitcast.a.type.category = Type::Element;
		bitcast.a.type.type = Instruction::I8;
		bitcast.a.constant = true;
		bitcast.a.i8 = -1;

		std::string reference = "%X = bitcast i8 -1 to i8";
		
		if( !check( bitcast, reference ) ) return false;
	
		bitcast.d.type.category = Type::Element;
		bitcast.d.type.type = Instruction::I64;
		bitcast.d.name = "%Z";
		
		bitcast.a.type.category = Type::Vector;
		bitcast.a.type.vector = 2;
		bitcast.a.type.type = Instruction::I32;
		bitcast.a.constant = false;
		bitcast.a.name = "%V";
	
		reference = "%Z = bitcast < 2 x i32 > %V to i64";

		if( !check( bitcast, reference ) ) return false;
	
		status << "Bitcast Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testBr()
	{
		ir::LLVMBr br;
		br.condition.name = "%cond";
		br.condition.type.category = Type::Element;
		br.condition.type.type = Instruction::I1;
		br.iftrue = "%IfEqual";
		br.iffalse = "%IfUnequal";
	
		std::string reference = "br i1 %cond, label %IfEqual, label %IfUnequal";
		
		if( !check( br, reference ) ) return false;
	
		status << "Br Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testCall()
	{
		ir::LLVMCall call;
		call.d.name = "%retval";
		call.d.type.category = Type::Element;
		call.d.type.type = Instruction::I32;
		call.name = "@test";
		call.parameters.resize( 1 );
		call.parameters[0].type.category = Type::Element;
		call.parameters[0].type.type = Instruction::I32;
		call.parameters[0].name = "%argc";
	
		std::string reference = "%retval = call i32 @test(i32 %argc)";

		if( !check( call, reference ) ) return false;
	
		call.d.type.category = Type::InvalidCategory;
		call.signature = "i32 (i8*, ...)*";
		call.name = "@printf";
		call.parameters.resize( 3 );
		call.parameters[0].type.category = Type::Pointer;
		call.parameters[0].type.type = Instruction::I8;
		call.parameters[0].name = "%msg";
		call.parameters[1].type.category = Type::Element;
		call.parameters[1].constant = true;
		call.parameters[1].type.type = Instruction::I32;
		call.parameters[1].i32 = 12;
		call.parameters[2].type.category = Type::Element;
		call.parameters[2].constant = true;
		call.parameters[2].type.type = Instruction::I8;
		call.parameters[2].i32 = 42;
		
		reference = "call i32 (i8*, ...)* @printf(i8* %msg, i32 12, i8 42)";
		
		if( !check( call, reference ) ) return false;
		
		call.d.type.category = Type::Element;
		call.d.name = "%X";
		call.tail = true;
		call.parameters.clear();
		call.signature.clear();
		call.name = "@foo";
		
		reference = "%X = tail call i32 @foo()";
		
		if( !check( call, reference ) ) return false;
		
		call.d.name = "%Y";
		call.convention = Instruction::FastCallingConvention;
		
		reference = "%Y = tail call fastcc i32 @foo()";
		
		if( !check( call, reference ) ) return false;
		
		call.d.type.category = Type::InvalidCategory;
		call.parameters.resize( 1 );
		call.parameters[0].type.category = Type::Element;
		call.parameters[0].type.type = Instruction::I8;
		call.parameters[0].constant = true;
		call.parameters[0].i8 = 97;
		call.parameters[0].attribute = Instruction::SignExtend;
		call.name = "@foo";
		call.tail = false;
		call.convention = Instruction::DefaultCallingConvention;
		
		reference = "call void @foo(i8 97 signext)";
		
		if( !check( call, reference ) ) return false;
		
		call.parameters.clear();
		call.d.type.category = Type::Structure;
		call.d.type.members.resize( 2 );
		call.d.type.label = "%struct.A";
		call.d.name = "%r";
		call.d.type.members[0].category = Type::Element;
		call.d.type.members[0].type = Instruction::I32;
		call.d.type.members[1].category = Type::Element;
		call.d.type.members[1].type = Instruction::I8;
		call.name = "@foo";
		
		reference = "%r = call %struct.A @foo()";
		
		if( !check( call, reference ) ) return false;
		
		call.d.type.members.clear();
		call.d.type.category = Type::InvalidCategory;
		call.functionAttributes = Instruction::NoReturn;
		
		reference = "call void @foo() noreturn";
		
		if( !check( call, reference ) ) return false;
		
		call.d.name = "%ZZ";
		call.d.type.category = Type::Element;
		call.d.type.type = Instruction::I32;
		call.functionAttributes = 0;
		call.d.attribute = Instruction::ZeroExtend;
		call.name = "@bar";
		
		reference = "%ZZ = call zeroext i32 @bar()";
		if( !check( call, reference ) ) return false;
		
		status << "Call Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testExtractelement()
	{
		ir::LLVMExtractelement ee;
		ee.d.name = "%result";
		ee.d.type.category = Type::Element;
		ee.d.type.type = Instruction::I32;
		ee.a.type.category = Type::Vector;
		ee.a.type.type = Instruction::I32;
		ee.a.type.vector = 4;
		ee.a.name = "%vec";
		ee.b.type.category = Type::Element;
		ee.b.type.type = Instruction::I32;
		ee.b.constant = true;
		ee.b.i32 = 0;
		
		std::string reference 
			= "%result = extractelement < 4 x i32 > %vec, i32 0";
		if( !check( ee, reference ) ) return false;
		
		status << "Extractelement Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testExtractvalue()
	{
		ir::LLVMExtractvalue ev;
		ev.d.name = "%result";
		ev.d.type.category = Type::Element;
		ev.d.type.type = Instruction::I32;
		ev.a.type.category = Type::Structure;
		ev.a.type.members.resize( 2 );
		ev.a.type.members[0].category = Type::Element;
		ev.a.type.members[0].type = Instruction::I32;
		ev.a.type.members[1].category = Type::Element;
		ev.a.type.members[1].type = Instruction::F32;
		ev.a.name = "%agg";
		
		ev.indices.push_back( 0 );
		
		std::string reference = "%result = extractvalue { i32, float } %agg, 0";
		if( !check( ev, reference ) ) return false;
		
		status << "Extractvalue Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testFadd()
	{
		ir::LLVMFadd i;
		
		i.d.name = "<result>";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::F32;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::F32;
		i.a.constant = true;
		i.a.f32 = 4;
		i.b.type.category = Type::Element;
		i.b.type.type = Instruction::F32;
		i.b.name = "%var";		
		
		std::string reference 
			= "<result> = fadd float 0x4010000000000000, %var";
		if( !check( i, reference ) ) return false;
		
		status << "Fadd Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testFcmp()
	{
		ir::LLVMFcmp i;
		
		i.d.name = "<result>";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::F32;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::F32;
		i.a.constant = true;
		i.a.f32 = 4;
		i.b.type.category = Type::Element;
		i.b.type.type = Instruction::F32;
		i.b.constant = true;
		i.b.f32 = 5;
		i.comparison = Instruction::Oeq;
		
		std::string reference = 
			"<result> = fcmp oeq float 0x4010000000000000, 0x4014000000000000";
		if( !check( i, reference ) ) return false;
		
		i.comparison = Instruction::One;
		
		reference = "<result> = fcmp one float 0x4010000000000000, 0x4014000000000000";
		if( !check( i, reference ) ) return false;
		
		i.comparison = Instruction::Olt;
		
		reference = "<result> = fcmp olt float 0x4010000000000000, 0x4014000000000000";
		if( !check( i, reference ) ) return false;
		
		i.comparison = Instruction::Ueq;
		i.a.f32 = 1;
		i.b.f32 = 2;
		
		reference = "<result> = fcmp ueq float 0x3ff0000000000000, 0x4000000000000000";
		if( !check( i, reference ) ) return false;
		
		status << "Fcmp Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testFdiv()
	{
		ir::LLVMFdiv i;
		
		i.d.name = "<result>";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::F32;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::F32;
		i.a.constant = true;
		i.a.f32 = 4;
		i.b.type.category = Type::Element;
		i.b.type.type = Instruction::F32;
		i.b.name = "%var";		
		
		std::string reference = "<result> = fdiv float 0x4010000000000000, %var";
		if( !check( i, reference ) ) return false;
		
		status << "Fdiv Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testFmul()
	{
		ir::LLVMFmul i;
		
		i.d.name = "<result>";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::F32;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::F32;
		i.a.constant = true;
		i.a.f32 = 4;
		i.b.type.category = Type::Element;
		i.b.type.type = Instruction::F32;
		i.b.name = "%var";		
		
		std::string reference = "<result> = fmul float 0x4010000000000000, %var";
		if( !check( i, reference ) ) return false;
		
		status << "Fmul Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testFpext()	
	{
		ir::LLVMFpext i;
		
		i.d.name = "%X";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::F64;
		i.a.f32 = 3.1415;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::F32;
		i.a.constant = true;		
		
		std::string reference = "%X = fpext float 0x400921cac0000000 to double";
		if( !check( i, reference ) ) return false;
		
		i.d.name = "%Y";
		i.d.type.type = Instruction::F32;
		i.a.f32 = 1;
		
		reference = "%Y = fpext float 0x3ff0000000000000 to float";
		if( !check( i, reference ) ) return false;
		
		status << "Fpext Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testFptosi()
	{
		ir::LLVMFptosi i;
		
		i.d.name = "%X";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I32;
		i.a.f64 = -123.0;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::F64;
		i.a.constant = true;		
		
		std::string reference = "%X = fptosi double 0xc05ec00000000000 to i32";
		if( !check( i, reference ) ) return false;
		
		status << "Fptosi Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testFptoui()
	{
		ir::LLVMFptoui i;
		
		i.d.name = "%X";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I32;
		i.a.f64 = 123.0;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::F64;
		i.a.constant = true;		
		
		std::string reference = "%X = fptoui double 0x405ec00000000000 to i32";
		if( !check( i, reference ) ) return false;
		
		status << "Fptoui Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testFptrunc()
	{
		ir::LLVMFptrunc i;
		
		i.d.name = "%X";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::F32;
		i.a.f64 = 123.0;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::F64;
		i.a.constant = true;		
		
		std::string 
			reference = "%X = fptrunc double 0x405ec00000000000 to float";
		if( !check( i, reference ) ) return false;
		
		status << "Fptrunc Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testFree()
	{
		ir::LLVMFree i;
		
		i.a.name = "%array";
		i.a.type.category = Type::Pointer;
		i.a.type.members.resize( 1 );
		i.a.type.members[0].category = Type::Array;
		i.a.type.members[0].vector = 4;
		i.a.type.members[0].type = Instruction::I8;
		
		std::string reference = "free [ 4 x i8 ]* %array";
		if( !check( i, reference ) ) return false;
		
		status << "Free Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testFrem()
	{
		ir::LLVMFrem i;
		
		i.d.name = "<result>";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::F32;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::F32;
		i.a.constant = true;
		i.a.f32 = 4;
		i.b.type.category = Type::Element;
		i.b.type.type = Instruction::F32;
		i.b.name = "%var";		
		
		std::string reference = "<result> = frem float 0x4010000000000000, %var";
		if( !check( i, reference ) ) return false;
		
		status << "Frem Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testFsub()
	{
		ir::LLVMFsub i;
		
		i.d.name = "<result>";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::F32;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::F32;
		i.a.constant = true;
		i.a.f32 = 0;
		i.b.type.category = Type::Element;
		i.b.type.type = Instruction::F32;
		i.b.name = "%val";		
		
		std::string reference = "<result> = fsub float 0x0, %val";
		if( !check( i, reference ) ) return false;
		
		status << "Fsub Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testGetelementptr()
	{
		ir::LLVMGetelementptr i;
		
		i.d.name = "%vptr";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I8;
		
		i.a.type.category = Type::Pointer;
		i.a.type.members.resize( 1 );
		i.a.type.members[0].category = Type::Structure;
		i.a.type.members[0].members.resize( 2 );
		i.a.type.members[0].members[0].category = Type::Element;
		i.a.type.members[0].members[0].type = Instruction::I32;
		i.a.type.members[0].members[1].category = Type::Vector;
		i.a.type.members[0].members[1].type = Instruction::I8;
		i.a.type.members[0].members[1].vector = 2;
		i.a.name = "%svptr";
		
		i.indices.push_back( 0 );
		i.indices.push_back( 1 );
		i.indices.push_back( 1 );

		std::string reference = 
			std::string( "%vptr = getelementptr { i32, < 2 x i8 > }* %svptr" ) 
			+ ", i32 0, i32 1, i32 1";
		if( !check( i, reference ) ) return false;
		
		i.d.name = "%eptr";
		i.a.name = "%aptr";
		i.a.type.members[0].members.resize( 0 );
		i.a.type.members[0].category = Type::Array;
		i.a.type.members[0].type = Instruction::I8;
		i.a.type.members[0].vector = 12;
		i.indices.clear();
		i.indices.push_back( 0 );
		i.indices.push_back( 1 );
				
		reference = "%eptr = getelementptr [ 12 x i8 ]* %aptr, i32 0, i32 1";
		if( !check( i, reference ) ) return false;
		
		status << "Getelementptr Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testIcmp()
	{
		ir::LLVMIcmp i;
		
		i.d.name = "<result>";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I1;
		i.a.type.category = Type::Pointer;
		i.a.type.type = Instruction::F32;
		i.a.name = "%X";
		i.b = i.a;
		i.comparison = Instruction::Ne;
		
		std::string reference = "<result> = icmp ne float* %X, %X";
		if( !check( i, reference ) ) return false;
		
		status << "Icmp Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testInsertelement()
	{
		ir::LLVMInsertelement i;
		
		i.d.name = "%result";
		i.d.type.category = Type::Vector;
		i.d.type.type = Instruction::I32;
		i.d.type.vector = 4;
		
		i.a.name = "%vec";
		i.a.type.category = Type::Vector;
		i.a.type.type = Instruction::I32;
		i.a.type.vector = 4;
		
		i.b.constant = true;
		i.b.i32 = 1;
		i.b.type.category = Type::Element;
		i.b.type.type = Instruction::I32;
		
		i.c.constant = true;
		i.c.i32 = 0;
		i.c.type.category = Type::Element;
		i.c.type.type = Instruction::I32;
		
		std::string reference 
			= "%result = insertelement < 4 x i32 > %vec, i32 1, i32 0";
		if( !check( i, reference ) ) return false;
		
		status << "Insertelement Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testInsertvalue()
	{
		ir::LLVMInsertvalue i;
		
		i.d.name = "%result";
		i.d.type.category = Type::Structure;
		i.d.type.members.resize( 2 );
		i.d.type.members[0].category = Type::Element;
		i.d.type.members[0].type = Instruction::I32;
		i.d.type.members[1].category = Type::Element;
		i.d.type.members[1].type = Instruction::F32;
		i.a = i.d;
		i.a.name = "%agg";
		i.b.type.category = Type::Element;
		i.b.type.type = Instruction::I32;
		i.b.constant = true;
		i.b.i32 = 1;
		i.indices.push_back( 0 );
		
		std::string reference 
			= "%result = insertvalue { i32, float } %agg, i32 1, 0";
		if( !check( i, reference ) ) return false;
		
		status << "Insertvalue Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testInttoptr()
	{
		ir::LLVMInttoptr i;
		
		i.d.name = "%Y";
		i.d.type.category = Type::Pointer;
		i.d.type.type = Instruction::I32;
		i.a.i64 = 0;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::I64;
		i.a.constant = true;		
		
		std::string reference = "%Y = inttoptr i64 0 to i32*";
		if( !check( i, reference ) ) return false;
		
		status << "Inttoptr Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testInvoke()
	{
		ir::LLVMInvoke i;
		
		i.d.name = "%retval";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I32;
		
		i.name = "@Test";
		i.parameters.resize( 1 );
		i.parameters[0].type.category = Type::Element;
		i.parameters[0].type.type = Instruction::I32;
		i.parameters[0].i32 = 15;
		i.parameters[0].constant = true;
		
		i.tolabel = "%Continue";
		i.unwindlabel = "%TestCleanup";
		
		std::string reference 
			= std::string( "%retval = invoke i32 @Test(i32 15) to label" ) 
			+ " %Continue unwind label %TestCleanup";
		if( !check( i, reference ) ) return false;
		
		status << "Invoke Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testLoad()
	{
		ir::LLVMLoad i;
		
		i.d.name = "%val";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I32;
		
		i.a.name = "%ptr";
		i.a.type.category = Type::Pointer;
		i.a.type.type = Instruction::I32;
		
		std::string reference = "%val = load i32* %ptr";
		if( !check( i, reference ) ) return false;
		
		status << "Load Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testLshr()
	{
		ir::LLVMLshr i;
		
		i.d.name = "<result>";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I8;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::I8;
		i.a.constant = true;
		i.a.i8 = -2;
		i.b.type.category = Type::Element;
		i.b.type.type = Instruction::I8;
		i.b.constant = true;
		i.b.i32 = 1;	
		
		std::string reference = "<result> = lshr i8 -2, 1";
		if( !check( i, reference ) ) return false;
		
		status << "Lshr Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testMalloc()
	{
		ir::LLVMMalloc i;
		
		i.d.name = "%array2";
		i.d.type.category = Type::Array;
		i.d.type.type = Instruction::I8;
		i.d.type.vector = 12;
		i.elements.name = "%size";
		i.elements.type.category = Type::Element;
		i.elements.type.type = Instruction::I32;
		
		std::string reference = "%array2 = malloc [ 12 x i8 ], i32 %size";
		if( !check( i, reference ) ) return false;
		
		status << "Malloc Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testMul()
	{
		ir::LLVMMul i;
		
		i.d.name = "<result>";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I32;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::I32;
		i.a.constant = true;
		i.a.i32 = 4;
		i.b.type.category = Type::Element;
		i.b.type.type = Instruction::I32;
		i.b.name = "%var";		
		
		std::string reference = "<result> = mul i32 4, %var";
		if( !check( i, reference ) ) return false;
		
		status << "Mul Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testOr()
	{
		ir::LLVMOr i;
		
		i.d.name = "<result>";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I32;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::I32;
		i.a.constant = true;
		i.a.i32 = 15;
		i.b.type.category = Type::Element;
		i.b.type.type = Instruction::I32;
		i.b.constant = true;		
		i.b.i32 = 40;
				
		std::string reference = "<result> = or i32 15, 40";
		if( !check( i, reference ) ) return false;
		
		status << "Or Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testPhi()
	{
		ir::LLVMPhi i;
		
		i.d.name = "%indvar";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I32;
		i.nodes.resize( 2 );

		i.nodes[0].operand = i.d;
		i.nodes[0].label = "%LoopHeader";
		i.nodes[0].operand.constant = true;
		i.nodes[0].operand.i32 = 0;
		
		i.nodes[1].operand = i.d;
		i.nodes[1].label = "%Loop";
		i.nodes[1].operand.name = "%nextindvar";
				
		std::string reference 
			= "%indvar = phi i32 [ 0, %LoopHeader ], [ %nextindvar, %Loop ]";
		if( !check( i, reference ) ) return false;
		
		status << "Phi Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testPtrtoint()
	{
		ir::LLVMPtrtoint i;
		
		i.d.name = "%Y";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I64;
		i.a.type.category = Type::Pointer;
		i.a.type.type = Instruction::I32;
		i.a.name = "%x";		
		
		std::string reference = "%Y = ptrtoint i32* %x to i64";
		if( !check( i, reference ) ) return false;
		
		status << "Prtoint Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testRet()
	{
		ir::LLVMRet i;
		
		std::string reference = "ret void";
		if( !check( i, reference ) ) return false;
		
		status << "Ret Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testSdiv()
	{
		ir::LLVMSdiv i;
		
		i.d.name = "<result>";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I32;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::I32;
		i.a.constant = true;
		i.a.i32 = 4;
		i.b.type.category = Type::Element;
		i.b.type.type = Instruction::I32;
		i.b.name = "%var";		
		
		std::string reference = "<result> = sdiv i32 4, %var";
		if( !check( i, reference ) ) return false;
		
		status << "Sdiv Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testSelect()
	{
		ir::LLVMSelect i;
		
		i.d.name = "%X";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I8;
		
		i.condition.type.category = Type::Element;
		i.condition.type.type = Instruction::I1;
		i.condition.constant = true;
		i.condition.i1 = 1;
		
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::I8;
		i.a.constant = true;
		i.a.i8 = 17;
		i.b.type.category = Type::Element;
		i.b.type.type = Instruction::I8;
		i.b.constant = true;
		i.b.i8 = 42;
		
		std::string reference = "%X = select i1 1, i8 17, i8 42";
		if( !check( i, reference ) ) return false;
		
		status << "Select Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testSext()
	{
		ir::LLVMSext i;
		
		i.d.name = "%X";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I16;
		i.a.i8 = -1;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::I8;
		i.a.constant = true;		
		
		std::string reference = "%X = sext i8 -1 to i16";
		if( !check( i, reference ) ) return false;
		
		status << "Sext Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testShl()
	{
		ir::LLVMShl i;
		
		i.d.name = "<result>";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I32;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::I32;
		i.a.constant = true;
		i.a.i32 = 1;
		i.b.type.category = Type::Element;
		i.b.type.type = Instruction::I32;
		i.b.constant = true;
		i.b.i32 = 32;
		
		std::string reference = "<result> = shl i32 1, 32";
		if( !check( i, reference ) ) return false;
		
		status << "Shl Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testShufflevector()
	{
		ir::LLVMShufflevector i;
		
		i.d.name = "%result";
		i.d.type.category = Type::Vector;
		i.d.type.type = Instruction::I32;
		i.d.type.vector = 4;
		
		i.a = i.d;
		i.a.name = "%v1";

		i.b = i.d;
		i.b.name = "%v2";
		
		i.mask.push_back( 0 );
		i.mask.push_back( 1 );
		i.mask.push_back( 2 );
		i.mask.push_back( 3 );
		i.mask.push_back( 4 );
		i.mask.push_back( 5 );
		i.mask.push_back( 6 );
		i.mask.push_back( 7 );
		
		std::string reference = std::string( "%result = shufflevector " ) 
			+ std::string( "< 4 x i32 > %v1, < 4 x i32 > %v2, < 8 x i32 > " ) 
			+ "< i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7 >";
		if( !check( i, reference ) ) return false;
		
		status << "ShuffleVector Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testSitofp()
	{
		ir::LLVMSitofp i;
		
		i.d.name = "%X";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::F32;
		i.a.i32 = 257;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::I32;
		i.a.constant = true;		
		
		std::string reference = "%X = sitofp i32 257 to float";
		if( !check( i, reference ) ) return false;
		
		status << "Sitofp Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testSrem()
	{
		ir::LLVMSrem i;
		
		i.d.name = "<result>";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I32;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::I32;
		i.a.constant = true;
		i.a.i32 = 4;
		i.b.type.category = Type::Element;
		i.b.type.type = Instruction::I32;
		i.b.name = "%var";		
		
		
		std::string reference = "<result> = srem i32 4, %var";
		if( !check( i, reference ) ) return false;
		
		status << "Srem Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testStore()
	{
		ir::LLVMStore i;
		
		i.d.name = "%ptr";
		i.d.type.category = Type::Pointer;
		i.d.type.type = Instruction::I32;
		
		i.a.constant = true;
		i.a.i32 = 3;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::I32;
		
		std::string reference = "store i32 3, i32* %ptr";
		if( !check( i, reference ) ) return false;
		
		status << "Store Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testSub()
	{
		ir::LLVMSub i;
		
		i.d.name = "<result>";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I32;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::I32;
		i.a.constant = true;
		i.a.i32 = 4;
		i.b.type.category = Type::Element;
		i.b.type.type = Instruction::I32;
		i.b.name = "%var";		
		
		std::string reference = "<result> = sub i32 4, %var";
		if( !check( i, reference ) ) return false;
		
		status << "Sub Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testSwitch()
	{
		ir::LLVMSwitch i;
		
		i.comparison.name = "%val";
		i.comparison.type.category = Type::Element;
		i.comparison.type.type = Instruction::I32;
		
		i.defaultTarget = "%otherwise";
		
		i.targets.resize( 3 );
		i.targets[0].label = "%onzero";
		i.targets[0].operand.type.category = Type::Element;
		i.targets[0].operand.type.type = Instruction::I32;
		i.targets[0].operand.i32 = 0;
		i.targets[0].operand.constant = true;
		
		i.targets[1].label = "%onone";
		i.targets[1].operand.type.category = Type::Element;
		i.targets[1].operand.type.type = Instruction::I32;
		i.targets[1].operand.i32 = 1;
		i.targets[1].operand.constant = true;
		
		i.targets[2].label = "%ontwo";
		i.targets[2].operand.type.category = Type::Element;
		i.targets[2].operand.type.type = Instruction::I32;
		i.targets[2].operand.i32 = 2;
		i.targets[2].operand.constant = true;
		
		std::string reference = std::string( "switch i32 %val, label " ) 
			+ std::string( "%otherwise [ i32 0, label %onzero i32 1, " ) 
			+ "label %onone i32 2, label %ontwo ]";
		if( !check( i, reference ) ) return false;
		
		status << "Switch Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testTrunc()
	{
		ir::LLVMTrunc i;
		
		i.d.name = "%Y";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I1;
		i.a.i32 = 123;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::I32;
		i.a.constant = true;		
		
		std::string reference = "%Y = trunc i32 123 to i1";
		if( !check( i, reference ) ) return false;
		
		status << "Trunc Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testUdiv()
	{
		ir::LLVMUdiv i;
		
		i.d.name = "<result>";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I32;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::I32;
		i.a.constant = true;
		i.a.i32 = 4;
		i.b.type.category = Type::Element;
		i.b.type.type = Instruction::I32;
		i.b.name = "%var";		
		
		std::string reference = "<result> = udiv i32 4, %var";
		if( !check( i, reference ) ) return false;
		
		status << "Udiv Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testUitofp()
	{
		ir::LLVMUitofp i;
		
		i.d.name = "%Y";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::F64;
		i.a.i8 = -1;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::I8;
		i.a.constant = true;		
		
		std::string reference = "%Y = uitofp i8 -1 to double";
		if( !check( i, reference ) ) return false;
		
		status << "Uitofp Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testUnreachable()
	{
		ir::LLVMUnreachable i;
		
		std::string reference = "unreachable";
		if( !check( i, reference ) ) return false;
		
		status << "Unreachable Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testUnwind()
	{
		ir::LLVMUnwind i;
		
		std::string reference = "unwind";
		if( !check( i, reference ) ) return false;
		
		status << "Unwind Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testUrem()
	{
		ir::LLVMUrem i;
		
		i.d.name = "<result>";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I32;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::I32;
		i.a.constant = true;
		i.a.i32 = 4;
		i.b.type.category = Type::Element;
		i.b.type.type = Instruction::I32;
		i.b.name = "%var";		
		
		std::string reference = "<result> = urem i32 4, %var";
		if( !check( i, reference ) ) return false;
		
		status << "Urem Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testVaArg()
	{
		ir::LLVMVaArg i;
		
		i.d.name = "%tmp";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I32;
		
		i.a.name = "%ap";
		i.a.type.category = Type::Pointer;
		i.a.type.members.resize( 1 );
		i.a.type.members[0].category = Type::Pointer;
		i.a.type.members[0].type = Instruction::I8;
		
		std::string reference = "%tmp = va_arg i8** %ap, i32";
		if( !check( i, reference ) ) return false;
		
		status << "VarArg Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testXor()
	{
		ir::LLVMXor i;
		
		i.d.name = "<result>";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I32;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::I32;
		i.a.constant = true;
		i.a.i32 = -1;
		i.b.type.category = Type::Element;
		i.b.type.type = Instruction::I32;
		i.b.name = "%V";		
		
		std::string reference = "<result> = xor i32 -1, %V";
		if( !check( i, reference ) ) return false;
		
		status << "Xor Instruction Passed\n";
		return true;
	}
	
	bool TestLLVMInstructions::testZext()
	{
		ir::LLVMZext i;
		
		i.d.name = "%X";
		i.d.type.category = Type::Element;
		i.d.type.type = Instruction::I64;
		i.a.i32 = 257;
		i.a.type.category = Type::Element;
		i.a.type.type = Instruction::I32;
		i.a.constant = true;
		
		std::string reference = "%X = zext i32 257 to i64";
		if( !check( i, reference ) ) return false;
		
		status << "Zext Instruction Passed\n";
		return true;
	}
	
	
	bool TestLLVMInstructions::doTest()
	{
		return testAdd() && testAlloca() && testAnd() && testAshr() 
			&& testBitcast() && testBr() && testCall() && testExtractelement() 
			&& testExtractvalue() && testFadd() && testFcmp() && testFdiv() 
			&& testFmul() && testFpext() && testFptosi() && testFptoui() 
			&& testFptrunc() && testFree() && testFrem() && testFsub() 
			&& testGetelementptr() && testIcmp() && testInsertelement() 
			&& testInsertvalue() && testInttoptr() && testInvoke() 
			&& testLoad() && testLshr() && testMalloc() && testMul() 
			&& testOr() && testPhi() && testPtrtoint() && testRet() 
			&& testSdiv() && testSelect() && testSext() && testShl() 
			&& testShufflevector() && testSitofp() && testSrem() 
			&& testStore() && testSub() && testSwitch() && testTrunc() 
			&& testUdiv() && testUitofp() && testUnreachable() && testUnwind() 
			&& testUrem() && testVaArg() && testXor() && testZext();
	}

	TestLLVMInstructions::TestLLVMInstructions()
	{
		name = "TestLLVMInstructions";
		
		description = "A test for the assembly code generation and automatic \
			verfication of individual LLVM instructions.\
		\
		Test Points:\
			1) For each instruction, generate several assembly strings using\
				the instruction's toString method, make sure that these pass\
				the valid() check, compare to references from the LLVM manual.";
	}
}

int main( int argc, char** argv )
{
	hydrazine::ArgumentParser parser( argc, argv );
	test::TestLLVMInstructions test;
	parser.description( test.testDescription() );
	
	parser.parse( "-v", "--verbose", test.verbose, false, 
		"Print out info after the test is over." );
	parser.parse( "-s", "--seed", test.seed, 0, 
		"The random seed for repeatability.  0 imples seed with time." );
	parser.parse();
	
	test.test();
	
	return test.passed();
}

#endif

