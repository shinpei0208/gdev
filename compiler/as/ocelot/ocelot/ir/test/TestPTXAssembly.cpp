/*! \file TestPTXAssembly.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Tuesday May 11, 2010
	\brief The source file for the TestPTXAssembly class.
*/

#ifndef TEST_PTX_ASSEMBLY_CPP_INCLUDED
#define TEST_PTX_ASSEMBLY_CPP_INCLUDED

#include <ocelot/ir/test/TestPTXAssembly.h>
#include <ocelot/ir/interface/PTXInstruction.h>
#include <ocelot/executive/interface/TextureOperations.h>

#include <ocelot/api/interface/ocelot.h>

#include <hydrazine/interface/Casts.h>
#include <hydrazine/interface/ArgumentParser.h>
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/string.h>
#include <hydrazine/interface/math.h>
#include <hydrazine/interface/FloatingPoint.h>

#include <ocelot/cuda/interface/cuda_runtime.h>

#include <limits>
#include <climits>
#include <cmath>
#include <cstdint>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif 

#define REPORT_BASE 0

#define VERBOSE_TEST_CONFIGURATION 0

#define PTX_VERSION_AND_TARGET ".version 3.0\n.target sm_30\n"

////////////////////////////////////////////////////////////////////////////////
// HELPER FUNCTIONS
template<typename T>
bool issubnormal(T r0)
{
	return false;
}

bool issubnormal(float r0)
{
	return !std::isnormal(r0) && !std::isnan(r0)
		&& !std::isinf(r0) && r0 != (float)0;
}

bool issubnormal(double r0)
{
	return !std::isnormal(r0) && !std::isnan(r0)
		&& !std::isinf(r0) && r0 != (double)0;
}

bool compareFloat(float a, float b, int eps)
{
	if(std::isnan(a) && std::isnan(b)) return true;
	
	bool signa = std::copysign(1.0, a) > 0;
	bool signb = std::copysign(1.0, b) > 0;
	
	if(std::isinf(a) && std::isinf(b) && signa == signb) return true;

	float difference = 0.0f;
	
	if(std::abs(a) < std::pow(10.0f, 4))
	{
		difference = std::abs(a - b);
	}
	else
	{
		difference = std::abs(1.0f - a/b);
	}
	
	float epsilon = std::numeric_limits<float>::epsilon()
		* std::pow(2.0, eps);

	report("Difference is " << difference << " epsilon is "
		<< epsilon << " eps is " << eps);

	return difference <= epsilon;
}

bool compareDouble(double a, double b, int eps)
{
	if(std::isnan(a) && std::isnan(b)) return true;
	
	bool signa = std::copysign(1.0, a) > 0;
	bool signb = std::copysign(1.0, b) > 0;
	
	if(std::isinf(a) && std::isinf(b) && signa == signb) return true;

	float difference = 0.0;
	
	if(std::abs(a) < std::pow(10.0, 15))
	{
		difference = std::abs(a - b);
	}
	else
	{
		difference = std::abs(1.0 - a/b);
	}
	
	double epsilon = std::numeric_limits<double>::epsilon()
		* std::pow(2.0, eps);

	report("Difference is " << difference << " epsilon is "
		<< epsilon << " eps is " << eps);

	return difference <= epsilon;
}

template<typename T>
bool isFloat()
{
	if(typeid(T) == typeid(float) || typeid(T) == typeid(double)) return true;
	return false;
}

template<typename T>
T getParameter(void* in, uint32_t offset)
{
	return *(T*)((char*)in + offset);
}

template<typename T>
void setParameter(void* output, uint32_t offset, T value)
{
//	std::cout << "setParameter(offset = " << offset << ", value = " << value << ")" << std::endl;
	*(T*)((char*)output + offset) = value;
}

template <typename type, uint32_t size>
char* uniformRandom(test::Test::MersenneTwister& generator)
{
	type* allocation = new type[size];
	char* result = (char*) allocation;

	for(uint32_t i = 0; i < size * sizeof(type); ++i)
	{
		result[i] = generator();
	}
	
	return result;
}

template <typename type, uint32_t size>
char* uniformNonZero(test::Test::MersenneTwister& generator)
{
	type* allocation = new type[size];
	char* result = (char*) allocation;

	for(uint32_t i = 0; i < size * sizeof(type); ++i)
	{
		char value = generator();
		result[i] = (value == 0) ? 1 : value;
	}
	
	return result;
}

template <typename type, uint32_t size>
char* uniformFloat(test::Test::MersenneTwister& generator)
{
	char* allocation = new char[size*sizeof(type)];
	type* result = (type*) allocation;

	for(uint32_t i = 0; i < size; ++i)
	{
		uint32_t fptype = generator();
		
		if(fptype & 0x100)
		{
			if(fptype & 0x10)
			{
				result[i] = std::numeric_limits<type>::signaling_NaN();
			}
			else
			{
				result[i] = std::numeric_limits<type>::denorm_min();
			}
		}
		else
		{
			if(fptype & 0x10)
			{
				result[i] = std::numeric_limits<type>::infinity();	
			}
			else
			{
				if(fptype & 0x1)
				{
					result[i] = hydrazine::bit_cast<type>(generator());
				}
				else
				{
					result[i] = (type)generator();
				}
			}
		}
	}
	
	return allocation;
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST Function Pointer Array
std::string testFunctionPointerArray_PTX()
{
	std::stringstream ptx;

	ptx << PTX_VERSION_AND_TARGET;
	ptx << "\n";

	ptx << ".visible .func (.param .u32 return) add0(.param .u32 a)\n";
	ptx << ".visible .func (.param .u32 return) add1(.param .u32 a)\n";
	ptx << ".visible .func (.param .u32 return) add2(.param .u32 a)\n";
	ptx << ".visible .func (.param .u32 return) add3(.param .u32 a)\n";
	ptx << "\n";

	ptx << ".global .u64 functionPointerArray[4] ="
		" { add0, add1, add2, add3 };\n";
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg .u32 %r<3>;                             \n";
	ptx << "\t.reg .u64 %functionPointerArrayBase;         \n";
	ptx << "\t.reg .u64 %functionPointer;                  \n";
	ptx << "\t.reg .u64 %offset;                           \n";
	ptx << "\t.param .u32 operandA;                        \n";
	ptx << "\t.param .u32 result;                          \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	ptx << "\tld.global.u32 %r0, [%rIn];                   \n";
	ptx << "\tst.param.u32 [operandA], %r0;                \n";
	ptx << "\tcvt.u64.u32 %offset, %tid.x;                 \n";
	ptx << "\tmul.lo.u64 %offset, %offset, 8;              \n";
	ptx << "\tmov.u64 %functionPointerArrayBase, functionPointerArray;\n";
	ptx << "\tadd.u64 %functionPointerArrayBase, "
		"%functionPointerArrayBase, %offset;\n";
	ptx << "\tld.global.u64 %functionPointer, [%functionPointerArrayBase];\n";
	ptx << "\tprototype: .callprototype (.param .u32 _)    \n";
	ptx << "\t    _ (.param .u32 _);                       \n";
	ptx << "\tcall (result), %functionPointer,             \n";
	ptx << "\t    (operandA), prototype;                   \n";
	ptx << "\tld.param.u32 %r2, [result];                  \n";
	ptx << "\tcvt.u64.u32 %offset, %tid.x;                 \n";
	ptx << "\tmul.lo.u64 %offset, %offset, 4;              \n";
	ptx << "\tadd.u64 %rOut, %offset, %rOut;               \n";
	ptx << "\tst.global.u32 [%rOut], %r2;                  \n";
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";

	ptx << ".visible .func (.param .u32 return) add0(.param .u32 a) \n";
	ptx << "{\t                                 \n";
	ptx << "\t.reg .u32 %r<3>;                  \n";
	ptx << "\tld.param.u32 %r0, [a];            \n";
	ptx << "\tadd.u32 %r0, %r0, 0;              \n";
	ptx << "\tst.param.u32 [return], %r0;       \n";
	ptx << "\tret 0;                            \n";
	ptx << "}                                   \n";

	ptx << ".visible .func (.param .u32 return) add1(.param .u32 a) \n";
	ptx << "{\t                                 \n";
	ptx << "\t.reg .u32 %r<3>;                  \n";
	ptx << "\tld.param.u32 %r0, [a];            \n";
	ptx << "\tadd.u32 %r0, %r0, 1;              \n";
	ptx << "\tst.param.u32 [return], %r0;       \n";
	ptx << "\tret 0;                            \n";
	ptx << "}                                   \n";

	ptx << ".visible .func (.param .u32 return) add2(.param .u32 a) \n";
	ptx << "{\t                                 \n";
	ptx << "\t.reg .u32 %r<3>;                  \n";
	ptx << "\tld.param.u32 %r0, [a];            \n";
	ptx << "\tadd.u32 %r0, %r0, 2;              \n";
	ptx << "\tst.param.u32 [return], %r0;       \n";
	ptx << "\tret 0;                            \n";
	ptx << "}                                   \n";

	ptx << ".visible .func (.param .u32 return) add3(.param .u32 a) \n";
	ptx << "{\t                                 \n";
	ptx << "\t.reg .u32 %r<3>;                  \n";
	ptx << "\tld.param.u32 %r0, [a];            \n";
	ptx << "\tadd.u32 %r0, %r0, 3;              \n";
	ptx << "\tst.param.u32 [return], %r0;       \n";
	ptx << "\tret 0;                            \n";
	ptx << "}                                   \n";
	
	return ptx.str();
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST CVTA
std::string testCvta_PTX(ir::PTXOperand::DataType type, 
	ir::PTXInstruction::AddressSpace space)
{
	std::stringstream ptx;

	std::string typeString   = "." + ir::PTXOperand::toString(type);
	std::string addressSpace = "." + ir::PTXInstruction::toString(space);
	
	ptx << PTX_VERSION_AND_TARGET;
	ptx << "\n";

	if(space != ir::PTXInstruction::Local)
	{	
		ptx << addressSpace << " " << typeString << " variable;\n";
	}

	ptx << "\n";
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg .u64 %address;                          \n";
	ptx << "\t.reg " << typeString    << " %rt;            \n";
	ptx << "\t.reg .pred %pd;                              \n";
	if(space == ir::PTXInstruction::Local)
	{	
		ptx << addressSpace << " " << typeString << " variable;\n";
	}

	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	ptx << "\tld.global" << typeString << " %rt, [%rIn];   \n";
	
	ptx << "\tst" << addressSpace << typeString << " [variable], %rt;     \n";
	ptx << "\tcvta" << addressSpace << typeString << " %address, variable;\n";
	ptx << "\tld" << typeString << " %rt, [%address];     \n";

	ptx << "\tst" << addressSpace << typeString << " [variable], %rt;     \n";
	ptx << "\tcvta.to" << addressSpace << typeString << " %address, %address;\n";
	ptx << "\tld" << addressSpace << typeString << " %rt, [%address];     \n";

	ptx << "\tst.global" << typeString << " [%rOut], %rt; \n";
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

template<typename dtype>
void testCvta_REF(void* output, void* input)
{
	dtype r0 = getParameter<dtype>(input, 0);

	setParameter(output, 0, r0);
}

test::TestPTXAssembly::TypeVector testCvta_INOUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST LOCAL MEMORY
std::string testLocalMemory_PTX(ir::PTXOperand::DataType dtype,
	bool global, bool scoped)
{
	std::stringstream ptx;

	std::string dTypeString = "." + ir::PTXOperand::toString(dtype);

	ptx << PTX_VERSION_AND_TARGET;

	std::stringstream local;

	local << ".local " << dTypeString << " localMemory";

	if(global)
	{
		ptx << local.str() << ";\n";
	}

	if(scoped)
	{
		if(global)
		{
			ptx << ".visible .func (.param " << dTypeString
				<< " return) function() \n";
			ptx << "{\t                                                 \n";
			ptx << "\t.reg " << dTypeString << " %r<4>;                 \n";
			ptx << "\t                                                  \n";
			ptx << "\tld.local" << dTypeString << " %r0, [localMemory]; \n";
			ptx << "\tst.param" << dTypeString << " [return], %r0;      \n";
			ptx << "\tret;                                              \n";
			ptx << "}\t                                                 \n";
		}
		else
		{
			ptx << ".visible .func () function()                        \n";
			ptx << "{\t                                                 \n";
			ptx << "\t" << local.str() << ";                            \n";
			ptx << "\t.reg " << dTypeString << " %r<4>;                 \n";
			ptx << "\t                                                  \n";
			ptx << "\tst.local" << dTypeString << " [localMemory], 0;   \n";
			ptx << "\tret;                                              \n";
			ptx << "}\t                                                 \n";
		}
	}

	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg " << dTypeString << " %r<4>;            \n";

	if(scoped && global)
	{
		ptx << "\t.param " << dTypeString << " result;\n";
	}

	if(!global)
	{
		ptx << "\t" << local.str() << ";                   \n";
	}
	
	ptx << "\t                                             \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	ptx << "\tld.global" << dTypeString << " %r0, [%rIn];  \n";


	if(scoped)
	{
		if(global)
		{
			ptx << "\tst.local" << dTypeString << " [localMemory], %r0;  \n";
			ptx << "\tcall.uni (result), function;                       \n";
			ptx << "\tld.param" << dTypeString << " %r0, [result];       \n";
		}
		else
		{
			ptx << "\tst.local" << dTypeString << " [localMemory], %r0;  \n";
			ptx << "\tcall.uni function;                                 \n";
			ptx << "\tld.local" << dTypeString << " %r0, [localMemory];  \n";
		}
	}
	else
	{
		ptx << "\tst.local" << dTypeString << " [localMemory], %r0; \n";
		ptx << "\tld.local" << dTypeString << " %r0, [localMemory]; \n";
	}

	ptx << "\tst.global" << dTypeString << " [%rOut], %r0; \n";
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

template<typename dtype>
void testLocalMemory_REF(void* output, void* input)
{
	dtype r0 = getParameter<dtype>(input, 0);
	
	setParameter(output, 0, r0);
}

test::TestPTXAssembly::TypeVector testLocalMemory_INOUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST TEXTURES
std::string testTex_PTX(ir::PTXInstruction::Geometry dim,
	ir::PTXOperand::DataType dtype,
	ir::PTXOperand::DataType samplerType)
{
	std::stringstream ptx;

	std::string samplerTypeString = "." + ir::PTXOperand::toString(samplerType);
	std::string dTypeString       = "." + ir::PTXOperand::toString(dtype);

	ptx << PTX_VERSION_AND_TARGET;
	ptx << ".global .texref Texture;\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg " << samplerTypeString << " %rs<4>;     \n";
	ptx << "\t.reg " << dTypeString << " %rd<4>;           \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	ptx << "\tld.global" << samplerTypeString << " %rs0, [%rIn];   \n";
	if(dim >= ir::PTXInstruction::_2d)
	{
		ptx << "\tld.global" << samplerTypeString << " %rs1, [%rIn + "
			<< ir::PTXOperand::bytes(samplerType) << "];   \n";
	}
	if(dim >= ir::PTXInstruction::_3d)
	{
		ptx << "\tld.global" << samplerTypeString << " %rs2, [%rIn + "
			<< 2 * ir::PTXOperand::bytes(samplerType) << "];   \n";
	}
	
	ptx << "\ttex." << ir::PTXInstruction::toString(dim) << ".v4"
		<< dTypeString << samplerTypeString 
		<< " {%rd0, %rd1, %rd2, %rd3}, [Texture, {%rs0";
	
	if(dim >= ir::PTXInstruction::_2d)
	{
		ptx << ", %rs1";
	}
	if(dim >= ir::PTXInstruction::_3d)
	{
		ptx << ", %rs2, %rs3}];\n";
	}
	
	ptx << "\tst.global" << dTypeString << " [%rOut], %rd0;\n";
	ptx << "\tst.global" << dTypeString << " [%rOut + " 
		<< ir::PTXOperand::bytes(dtype) << "], %rd1;\n";
	ptx << "\tst.global" << dTypeString << " [%rOut + " 
		<< 2 * ir::PTXOperand::bytes(dtype) << "], %rd2;\n";
	ptx << "\tst.global" << dTypeString << " [%rOut + " 
		<< 3 * ir::PTXOperand::bytes(dtype) << "], %rd3;\n";
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

template<typename dtype, typename stype, ir::PTXInstruction::Geometry dim>
void testTex_REF(void* output, void* input)
{
	stype r[3];

	const ir::Texture& texture = *getParameter<const ir::Texture*>(input, 0);
	
	r[0] = getParameter<stype>(input, sizeof(const ir::Texture*));
	
	if(dim >= ir::PTXInstruction::_2d)
	{
		r[1] = getParameter<stype>(input,
			sizeof(const ir::Texture*) + sizeof(stype));
	}
	if(dim == ir::PTXInstruction::_3d)
	{
		r[2] = getParameter<stype>(input,
			sizeof(const ir::Texture*) + 2 * sizeof(stype));
	}

	dtype d[4];

	if(dim == ir::PTXInstruction::_1d)
	{
		d[0] = executive::tex::sample<0, dtype>(texture, r[0]);
		d[1] = executive::tex::sample<1, dtype>(texture, r[0]);
		d[2] = executive::tex::sample<2, dtype>(texture, r[0]);
		d[3] = executive::tex::sample<3, dtype>(texture, r[0]);
	}
	else if(dim == ir::PTXInstruction::_2d)
	{
		d[0] = executive::tex::sample<0, dtype>(texture, r[0], r[1]);
		d[1] = executive::tex::sample<1, dtype>(texture, r[0], r[1]);
		d[2] = executive::tex::sample<2, dtype>(texture, r[0], r[1]);
		d[3] = executive::tex::sample<3, dtype>(texture, r[0], r[1]);
	}
	else
	{
		d[0] = executive::tex::sample<0, dtype>(texture, r[0], r[1], r[2]);
		d[1] = executive::tex::sample<1, dtype>(texture, r[0], r[1], r[2]);
		d[2] = executive::tex::sample<2, dtype>(texture, r[0], r[1], r[2]);
		d[3] = executive::tex::sample<3, dtype>(texture, r[0], r[1], r[2]);
	}
	
	setParameter(output, 0 * sizeof(dtype), d[0]);
	setParameter(output, 1 * sizeof(dtype), d[1]);
	setParameter(output, 2 * sizeof(dtype), d[2]);
	setParameter(output, 3 * sizeof(dtype), d[3]);
}

test::TestPTXAssembly::TypeVector testTex_IN(
	test::TestPTXAssembly::DataType type, ir::PTXInstruction::Geometry dim)
{
	return test::TestPTXAssembly::TypeVector(dim, type);
}

test::TestPTXAssembly::TypeVector testTex_OUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(4, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST CVT
std::string testCvt_PTX(ir::PTXOperand::DataType dtype, 
	ir::PTXOperand::DataType stype, bool ftz, bool sat, bool round,
	bool rmi = false)
{
	std::stringstream ptx;

	std::string sTypeString = "." + ir::PTXOperand::toString(stype);
	std::string dTypeString = "." + ir::PTXOperand::toString(dtype);

	ptx << PTX_VERSION_AND_TARGET;
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg " << sTypeString    << " %rs;           \n";
	ptx << "\t.reg " << dTypeString    << " %rd;           \n";
	ptx << "\t.reg .pred %pd;                              \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	ptx << "\tld.global" << sTypeString << " %rs, [%rIn];  \n";
	
	if(ir::PTXOperand::isFloat(stype) && !ir::PTXOperand::isFloat(dtype))
	{
		ptx << "\ttestp.notanumber" << sTypeString << " %pd, %rs;  \n";
		ptx << "\t@%pd mov" << sTypeString << " %rs, 0.0;  \n";
	
	}
	
	ptx << "\tcvt";
	
	if(round)
	{
		if(ir::PTXOperand::isFloat(stype))
		{
			if(rmi)
			{
				ptx << ".rmi";
			}
			else
			{
				ptx << ".rzi";
			}
		}
		else
		{
			if(rmi)
			{
				ptx << ".rm";
			}
			else
			{
				ptx << ".rz";
			}
		}
	}
	
	if(ftz) ptx << ".ftz";
	if(sat) ptx << ".sat";
	
	ptx << dTypeString << sTypeString << " %rd, %rs;       \n";
	
	ptx << "\tst.global" << dTypeString << " [%rOut], %rd; \n";
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

template<typename dtype, typename stype, bool ftz, bool sat, bool round,
	bool rmi = false>
void testCvt_REF(void* output, void* input)
{
	stype r0 = getParameter<stype>(input, 0);

	if(isFloat<stype>() && !isFloat<dtype>())
	{
		if(isnan(r0)) r0 = (stype)0;
	}

	if(ftz)
	{
		if(issubnormal(r0)) r0 = (stype)0;
	}

	dtype r1 = r0;

	if(sat)
	{
		if(typeid(float) != typeid(dtype) && typeid(double) != typeid(dtype))
		{
			r1 = (stype)std::numeric_limits<dtype>::max() > r0
				? r0 : std::numeric_limits<dtype>::max();
			r1 = (stype)std::numeric_limits<dtype>::min() < r0
				? r0 : std::numeric_limits<dtype>::min();
		}
	}

	if(round)
	{
		if(isFloat<stype>() && typeid(stype) == typeid(dtype))
		{
			if(rmi)
			{
				r1 = floor(r0);
			}
			else if(typeid(double) == typeid(dtype))
			{
				r1 = hydrazine::trunc((double)r0);
			}
			else
			{
				r1 = hydrazine::trunc((float)r0);
			}
		}
		else if(isFloat<stype>())
		{
			int mode = hydrazine::fegetround();
		
			if(rmi)
			{
				hydrazine::fesetround(FE_DOWNWARD);
			}
			else
			{
				hydrazine::fesetround(FE_TOWARDZERO);
			}
		
			r1 = r0;
		
			hydrazine::fesetround(mode);
		}
	}

	if(isFloat<stype>() && !isFloat<dtype>())
	{
		if((stype)std::numeric_limits<dtype>::max() < r0)
		{
			r1 = std::numeric_limits<dtype>::max();
		}
		if((stype)std::numeric_limits<dtype>::min() > r0)
		{
			r1 = std::numeric_limits<dtype>::min();	
		}
	}
	
	if(sat)
	{
		if(typeid(float) == typeid(dtype))
		{
			if(isnan(r1)) r1 = 0.0f;

			r1 = std::min((dtype)1.0f, r1);
			r1 = std::max((dtype)0.0f, r1);
		}
		else if(typeid(double) == typeid(dtype))
		{
			if(isnan(r1)) r1 = 0.0;

			r1 = std::min((dtype)1.0, r1);
			r1 = std::max((dtype)0.0, r1);
		}
	}
	
	setParameter(output, 0, r1);
}

test::TestPTXAssembly::TypeVector testCvt_INOUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST MOV LABELS
std::string testMovLabel_PTX(ir::PTXInstruction::AddressSpace space, bool index, 
	ir::PTXInstruction::Vec v)
{
	std::stringstream ptx;

	std::string spaceString = "." + ir::PTXInstruction::toString(space);
	std::string vecString;
	
	if(v != ir::PTXOperand::v1)
	{
		vecString = "." + ir::PTXInstruction::toString(v);
	}
	
	ptx << PTX_VERSION_AND_TARGET;

	if(space != ir::PTXInstruction::Param
		&& space != ir::PTXInstruction::Local)
	{	
		if(index)
		{
			ptx << spaceString << " " << vecString << " .u32 global[2];\n";
		}
		else
		{
			ptx << spaceString << " " << vecString << " .u32 global;\n";
		}
	}

	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";

	if(space == ir::PTXInstruction::Param
		|| space == ir::PTXInstruction::Local)
	{	
		if(index)
		{
			ptx << spaceString << " " << vecString << " .u32 global[2];\n";
		}
		else
		{
			ptx << spaceString << " " << vecString << " .u32 global;\n";
		}
	}

	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg .u32 %r<4>;                             \n";
	ptx << "\t.reg .u64 %address;                          \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	
	ptx << "\tld.global.u32 %r0, [%rIn];                   \n";

	if(index)
	{
		ptx << "\tmov.u64 %address, global[1];             \n";
	}
	else
	{
		ptx << "\tmov.u64 %address, global;                \n";
	}

	switch(v)
	{
	case ir::PTXOperand::v1:
	{
		ptx << "\tst" << spaceString << ".u32 [%address], %r0; \n";
		break;
	}
	case ir::PTXOperand::v2:
	{
		ptx << "\tst" << spaceString << ".v2.u32 [%address], {%r0, %r0}; \n";
		break;
	}
	case ir::PTXOperand::v4:
	{
		ptx << "\tst" << spaceString
			<< ".v4.u32 [%address], {%r0, %r0, %r0, %r0}; \n";
		break;
	}
	default: break;
	}
	
	if(index)
	{
		ptx << "\tmov.u64 %address, global + 4;            \n";
	}

	switch(v)
	{
	case ir::PTXOperand::v1:
	{
		ptx << "\tld" << spaceString << ".u32 %r0, [%address]; \n";
		break;
	}
	case ir::PTXOperand::v2:
	{
		ptx << "\tld" << spaceString << ".v2.u32 {%r0, %r1}, [%address]; \n";
		ptx << "\tand.b32 %r0, %r0, %r1;\n";
		break;
	}
	case ir::PTXOperand::v4:
	{
		ptx << "\tld" << spaceString 
			<< ".v4.u32 {%r0, %r1, %r2, %r3}, [%address]; \n";
		ptx << "\tand.b32 %r0, %r0, %r1;\n";
		ptx << "\tand.b32 %r0, %r0, %r2;\n";
		ptx << "\tand.b32 %r0, %r0, %r3;\n";
	break;
	}
	default: break;
	}

	ptx << "\tst.global.u32 [%rOut], %r0;                  \n";
	
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

void testMovLabel_REF(void* output, void* input)
{
	uint32_t r0 = getParameter<unsigned>(input, 0);

	setParameter(output, 0, r0);
}

test::TestPTXAssembly::TypeVector testMovLabel_INOUT()
{
	return test::TestPTXAssembly::TypeVector(1, test::TestPTXAssembly::I32);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST MOV
std::string testMov_PTX(ir::PTXOperand::DataType type)
{
	std::stringstream ptx;

	std::string typeString = "." + ir::PTXOperand::toString(type);

	ptx << PTX_VERSION_AND_TARGET;
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg " << typeString    << " %r<2>;          \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	
	if(type == ir::PTXOperand::pred)
	{
		ptx << "\t.reg .u8  %rb0;                          \n";
		ptx << "\t.reg .u16 %rs0;                          \n";
		ptx << "\tld.global.u8 %rb0, [%rIn];               \n";
		ptx << "\tcvt.u16.u8 %rs0, %rb0;                   \n";
		ptx << "\tand.b16 %rs0, %rs0, 0x1;                 \n";
		ptx << "\tsetp.ne.u16 %r0, %rs0, 0;                \n";
		ptx << "\tmov" << typeString << " %r1, %r0;        \n";
		ptx << "\tselp.u16   %rs0, 1, 0, %r1;              \n";
		ptx << "\tcvt.u8.u16 %rb0, %rs0;                   \n";
		ptx << "\tst.global.u8 [%rOut], %rb0;              \n";
	}
	else
	{
		ptx << "\tld.global" << typeString << " %r0, [%rIn];   \n";
		ptx << "\tmov" << typeString << " %r1, %r0;            \n";
		ptx << "\tst.global" << typeString << " [%rOut], %r1;  \n";
	}
	
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

template<typename type>
void testMov_REF(void* output, void* input)
{
	if(typeid(bool) == typeid(type))
	{
		char r0 = getParameter<char>(input, 0);
	
		char r1 = ((char)r0 & 0x1) == 0 ? 0 : 1;

		setParameter<char>(output, 0, r1);
	}
	else
	{
		type r0 = getParameter<type>(input, 0);
	
		type r1 = r0;

		setParameter(output, 0, r1);
	}
}

test::TestPTXAssembly::TypeVector testMov_INOUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST LOGICAL OPs
std::string testLops_PTX(ir::PTXInstruction::Opcode opcode,
	ir::PTXOperand::DataType type)
{
	std::stringstream ptx;

	std::string typeString = "." + ir::PTXOperand::toString(type);

	ptx << PTX_VERSION_AND_TARGET;
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	if( opcode == ir::PTXInstruction::Shr 
		|| opcode == ir::PTXInstruction::Shl )
	{
		ptx << "\t.reg " << typeString    << " %r0, %r2;   \n";
		ptx << "\t.reg .u32 %r1;                           \n";
	}
	else
	{
		ptx << "\t.reg " << typeString    << " %r<3>;      \n";
	}
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	
	if(type == ir::PTXOperand::pred)
	{
		ptx << "\t.reg .u8  %rb<2>;                        \n";
		ptx << "\t.reg .u16 %rs<2>;                        \n";
		ptx << "\tld.global.u8 %rb0, [%rIn];               \n";
		ptx << "\tcvt.u16.u8 %rs0, %rb0;                   \n";
		ptx << "\tsetp.lt.u16 %r0, %rs0, 64;               \n";
		if(opcode == ir::PTXInstruction::And
			|| opcode == ir::PTXInstruction::Or
			|| opcode == ir::PTXInstruction::Xor)
		{
			ptx << "\tld.global.u8 %rb1, [%rIn + " 
				<< ir::PTXOperand::bytes(type) << "];          \n";
			ptx << "\tcvt.u16.u8 %rs1, %rb1;                   \n";
			ptx << "\tsetp.lt.u16 %r1, %rs1, 64;               \n";
		}
	}
	else
	{
		ptx << "\tld.global" << typeString << " %r0, [%rIn];   \n";
		if( opcode == ir::PTXInstruction::Shr 
			|| opcode == ir::PTXInstruction::Shl )
		{
			ptx << "\tld.global.u32 %r1, [%rIn + " 
				<< std::max((size_t)ir::PTXOperand::bytes(type), sizeof(uint32_t)) 
				<< "];              \n";
			ptx << "\trem.u32 %r1, %r1, " 
				<< 8 * ir::PTXOperand::bytes(type) << ";\n";
		}
		else if(opcode == ir::PTXInstruction::And
			|| opcode == ir::PTXInstruction::Or
			|| opcode == ir::PTXInstruction::Xor)
		{
			ptx << "\tld.global" << typeString << " %r1, [%rIn + " 
				<< ir::PTXOperand::bytes(type) << "];              \n";
		}
	}
	
	ptx << "\t" << ir::PTXInstruction::toString(opcode)
		<< typeString << " %r2, %r0";

	if(opcode == ir::PTXInstruction::And
		|| opcode == ir::PTXInstruction::Or
		|| opcode == ir::PTXInstruction::Xor
		|| opcode == ir::PTXInstruction::Shr
		|| opcode == ir::PTXInstruction::Shl)
	{
		ptx << ", %r1";
	}

	ptx << ";\n";

	if(type == ir::PTXOperand::pred)
	{
		ptx << "\tselp.u16   %rs0, 1, 0, %r2;                    \n";
		ptx << "\tcvt.u8.u16 %rb0, %rs0;                         \n";
		ptx << "\tst.global.u8 [%rOut], %rb0;                    \n";
	}
	else
	{
		ptx << "\tst.global" << typeString << " [%rOut], %r2;    \n";
	}
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

template<ir::PTXInstruction::Opcode opcode, typename type>
void testLops_REF(void* output, void* input)
{
	typedef uint64_t U64;

	switch(opcode)
	{
	case ir::PTXInstruction::And:
	{
		type r0 = getParameter<type>(input, 0);
		type r1 = getParameter<type>(input, sizeof(type));

		if(typeid(type) == typeid(uint8_t))
		{
			r1 = r1 < 64;
			r0 = r0 < 64;
		}

		U64 a = hydrazine::bit_cast<U64>(r0);
		U64 b = hydrazine::bit_cast<U64>(r1);
		
		U64 d = a & b;
		
		setParameter(output, 0, hydrazine::bit_cast<type>(d));
		break;
	}
	case ir::PTXInstruction::Or:
	{
		type r0 = getParameter<type>(input, 0);
		type r1 = getParameter<type>(input, sizeof(type));

		if(typeid(type) == typeid(uint8_t))
		{
			r1 = r1 < 64;
			r0 = r0 < 64;
		}

		U64 a = hydrazine::bit_cast<U64>(r0);
		U64 b = hydrazine::bit_cast<U64>(r1);
		
		U64 d = a | b;
		
		setParameter(output, 0, hydrazine::bit_cast<type>(d));
		break;
	}
	case ir::PTXInstruction::Xor:
	{
		type r0 = getParameter<type>(input, 0);
		type r1 = getParameter<type>(input, sizeof(type));

		if(typeid(type) == typeid(uint8_t))
		{
			r1 = r1 < 64;
			r0 = r0 < 64;
		}

		U64 a = hydrazine::bit_cast<U64>(r0);
		U64 b = hydrazine::bit_cast<U64>(r1);
		
		U64 d = a ^ b;
		
		setParameter(output, 0, hydrazine::bit_cast<type>(d));
		break;
	}
	case ir::PTXInstruction::Shr:
	{
		type r0 = getParameter<type>(input, 0);
		uint32_t r1 = getParameter<uint32_t>(input,
			std::max(sizeof(type), sizeof(uint32_t)));

		if(typeid(type) == typeid(uint8_t))
		{
			r1 = r1 < 64;
			r0 = r0 < 64;
		}

		type d = r0 >> (r1 % (sizeof(type) * 8));
		
		setParameter(output, 0, d);
		break;
	}
	case ir::PTXInstruction::Shl:
	{
		type r0 = getParameter<type>(input, 0);
		uint32_t r1 = getParameter<uint32_t>(input,
			std::max(sizeof(type), sizeof(uint32_t)));

		type d = r0 << (r1 % (sizeof(type) * 8));
		
		setParameter(output, 0, d);
		break;
	}
	case ir::PTXInstruction::Not:
	{
		type r0 = getParameter<type>(input, 0);

		if(typeid(type) == typeid(uint8_t))
		{
			r0 = r0 < 64;
		}

		U64 a = hydrazine::bit_cast<U64>(r0);
		
		U64 d = ~a;
		
		if(typeid(type) == typeid(uint8_t))
		{
			d = d & 1;
		}
		
		setParameter(output, 0, hydrazine::bit_cast<type>(d));
		break;
	}
	case ir::PTXInstruction::CNot:
	{
		type r0 = getParameter<type>(input, 0);

		if(typeid(type) == typeid(uint8_t))
		{
			r0 = r0 < 64;
		}

		U64 a = hydrazine::bit_cast<U64>(r0);
		
		U64 d = a == 0 ? 1 : 0;
		
		setParameter(output, 0, hydrazine::bit_cast<type>(d));
		break;
	}
	default: assertM(false, "Invalid opcode for logical op.");
	}
	
}

test::TestPTXAssembly::TypeVector testLops_IN(
	ir::PTXInstruction::Opcode opcode, test::TestPTXAssembly::DataType type)
{
	if(opcode == ir::PTXInstruction::And
		|| opcode == ir::PTXInstruction::Or
		|| opcode == ir::PTXInstruction::Xor)
	{
		return test::TestPTXAssembly::TypeVector(2, type);
	}
	else if(opcode == ir::PTXInstruction::Shr
		|| opcode == ir::PTXInstruction::Shl)
	{
		test::TestPTXAssembly::TypeVector types(1, type);
		
		types.push_back(test::TestPTXAssembly::I32);
		
		return types;
	}
	else
	{
		return test::TestPTXAssembly::TypeVector(1, type);
	}
}

test::TestPTXAssembly::TypeVector testLops_OUT(
	test::TestPTXAssembly::DataType type)
{	
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST SLCT
std::string testSlct_PTX(ir::PTXOperand::DataType type,
	bool sourceFloat, bool ftz)
{
	std::stringstream ptx;

	std::string typeString = "." + ir::PTXOperand::toString(type);

	std::string cmpTypeString;
	
	if(sourceFloat)
	{
		cmpTypeString = ".f32";
	}
	else
	{
		cmpTypeString = ".s32";
	}
	
	ptx << PTX_VERSION_AND_TARGET;
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg " << typeString    << " %r<3>;          \n";
	ptx << "\t.reg " << cmpTypeString << " %rc;            \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	ptx << "\tld.global" << typeString << " %r0, [%rIn];   \n";
	ptx << "\tld.global" << typeString << " %r1, [%rIn + " 
		<< ir::PTXOperand::bytes(type) << "];              \n";
	ptx << "\tld.global" << cmpTypeString << " %rc, [%rIn + " 
		<< 2 * ir::PTXOperand::bytes(type) << "];          \n";

	ptx << "\tslct";
	if(ftz) ptx << ".ftz";
	
	ptx << typeString << cmpTypeString << " %r2, %r0, %r1, %rc; \n";
	
	ptx << "\tst.global" << typeString << " [%rOut], %r2;  \n";
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

template<typename type, bool sourceFloat, bool ftz>
void testSlct_REF(void* output, void* input)
{
	type r0 = getParameter<type>(input, 0);
	type r1 = getParameter<type>(input, sizeof(type));
	bool predicate = false;
	
	if(sourceFloat)
	{
		float value = getParameter<float>(input, 2 * sizeof(type));
	
		if(ftz)
		{
			if(issubnormal(value)) value = 0;
		}
		
		predicate = value >= 0.0f;
	}
	else
	{
		int value = getParameter<int>(input, 2 * sizeof(type));
		
		predicate = value >= 0;
	}
	
	if(predicate)
	{
		setParameter(output, 0, r0);
	}
	else
	{
		setParameter(output, 0, r1);
	}
}

test::TestPTXAssembly::TypeVector testSlct_IN(
	test::TestPTXAssembly::DataType type, bool sourceFloat)
{
	test::TestPTXAssembly::TypeVector types(2, type);
	
	if(sourceFloat)
	{
		types.push_back(test::TestPTXAssembly::FP32);
	}
	else
	{
		types.push_back(test::TestPTXAssembly::I32);
	}

	return types;
}

test::TestPTXAssembly::TypeVector testSlct_OUT(
	test::TestPTXAssembly::DataType type)
{	
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST SELP
std::string testSelP_PTX(ir::PTXOperand::DataType type)
{
	std::stringstream ptx;

	std::string typeString = "." + ir::PTXOperand::toString(type);
	
	ptx << PTX_VERSION_AND_TARGET;
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg " << typeString << " %r<3>;             \n";
	ptx << "\t.reg .b8 %b0;                                \n";
	ptx << "\t.reg .u16 %s0;                               \n";
	ptx << "\t.reg .pred %p0;                              \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	ptx << "\tld.global" << typeString << " %r0, [%rIn];   \n";
	ptx << "\tld.global" << typeString << " %r1, [%rIn + " 
		<< ir::PTXOperand::bytes(type) << "];              \n";
	ptx << "\tld.global.u8 %b0, [%rIn + " 
		<< 2 * ir::PTXOperand::bytes(type) << "];          \n";
	ptx << "\tcvt.u16.u8 %s0, %b0;                         \n";
	ptx << "\tsetp.lt.u16 %p0, %s0, 128;                   \n";

	ptx << "\tselp" << typeString << " %r2, %r0, %r1, %p0; \n";
	
	ptx << "\tst.global" << typeString << " [%rOut], %r2;  \n";
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

template<typename type>
void testSelP_REF(void* output, void* input)
{
	type r0 = getParameter<type>(input, 0);
	type r1 = getParameter<type>(input, sizeof(type));
	bool predicate = getParameter<uint8_t>(
		input, 2 * sizeof(type)) < 128;

	
	if(predicate)
	{
		setParameter(output, 0, r0);
	}
	else
	{
		setParameter(output, 0, r1);
	}
}

test::TestPTXAssembly::TypeVector testSelP_IN(
	test::TestPTXAssembly::DataType type)
{
	test::TestPTXAssembly::TypeVector types(2, type);
	
	types.push_back(test::TestPTXAssembly::I8);

	return types;
}

test::TestPTXAssembly::TypeVector testSelP_OUT(
	test::TestPTXAssembly::DataType type)
{	
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST SETP
std::string testSetP_PTX(ir::PTXOperand::DataType stype,
	ir::PTXOperand::PredicateCondition c, 
	ir::PTXInstruction::BoolOp boolOp, ir::PTXInstruction::CmpOp cmpOp,
	bool ftz, bool pq)
{
	std::stringstream ptx;

	std::string sTypeString = "." + ir::PTXOperand::toString(stype);
	
	ptx << PTX_VERSION_AND_TARGET;
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg " << sTypeString << " %rs<2>;           \n";
	ptx << "\t.reg .u16 %rh<2>;                            \n";
	ptx << "\t.reg .u8 %b0;                                \n";
	ptx << "\t.reg .u8 %rd<2>;                             \n";
	ptx << "\t.reg .u16 %s0;                               \n";
	ptx << "\t.reg .pred %p<3>;                            \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	ptx << "\tld.global" << sTypeString << " %rs0, [%rIn]; \n";
	ptx << "\tld.global" << sTypeString << " %rs1, [%rIn + " 
		<< ir::PTXOperand::bytes(stype) << "];             \n";
	ptx << "\tld.global.u8 %b0, [%rIn + " 
		<< 2 * ir::PTXOperand::bytes(stype) << "];         \n";
	ptx << "\tcvt.u16.u8 %s0, %b0;                         \n";
	ptx << "\tsetp.lt.u16 %p0, %s0, 128;                   \n";

	ptx << "\tsetp." << ir::PTXInstruction::toString(cmpOp);

	if(boolOp != ir::PTXInstruction::BoolOp_Invalid)
	{
		ptx << "." << ir::PTXInstruction::toString(boolOp);
	}
	
	if(ftz) ptx << ".ftz";
	
	ptx << sTypeString;
	
	ptx << " %p1";
	if(pq) ptx << " | %p2";

	ptx << ", %rs0, %rs1";
	
	if(boolOp != ir::PTXInstruction::BoolOp_Invalid)
	{
		ptx << ", ";
		
		if(c == ir::PTXOperand::InvPred)
		{
			ptx << "!%p0";
		}
		else
		{
			ptx << "%p0";
		}
	}
	
	ptx << ";\n";
	
	ptx << "\tselp.u16 %rh0, 1, 0, %p1;                      \n";
	ptx << "\tcvt.u8.u16 %rd0, %rh0;                         \n";
	ptx << "\tst.global.u8 [%rOut], %rd0;                    \n";
	if(pq)
	{
		ptx << "\tselp.u16   %rh1, 1, 0, %p2;                    \n";
		ptx << "\tcvt.u8.u16 %rd1, %rh1;                         \n";
		ptx << "\tst.global.u8 [%rOut + 1], %rd1;                \n";
	}
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

template<typename stype, ir::PTXOperand::PredicateCondition c, 
	ir::PTXInstruction::BoolOp boolOp, ir::PTXInstruction::CmpOp cmpOp,
	bool ftz, bool pq>
void testSetP_REF(void* output, void* input)
{
	stype r0 = getParameter<stype>(input, 0);
	stype r1 = getParameter<stype>(input, sizeof(stype));
	bool predicate = getParameter<uint8_t>(
		input, 2 * sizeof(stype)) < 128;

	if(ftz)
	{
		if(issubnormal(r0)) r0 = 0.0f;
		if(issubnormal(r1)) r1 = 0.0f;
	}

	if(c == ir::PTXOperand::InvPred) predicate = !predicate;

	bool comparison = false;
	
	switch(cmpOp)
	{
	case ir::PTXInstruction::Eq:
	{
		comparison = !std::isnan(r0) && !std::isnan(r1) && r0 == r1;
		break;
	}
	case ir::PTXInstruction::Ne:
	{
		comparison = !std::isnan(r0) && !std::isnan(r1) && r0 != r1;
		break;
	}
	case ir::PTXInstruction::Lo:
	case ir::PTXInstruction::Lt:
	{
		comparison = !std::isnan(r0) && !std::isnan(r1) && r0 < r1;
		break;
	}
	case ir::PTXInstruction::Ls:
	case ir::PTXInstruction::Le:
	{
		comparison = !std::isnan(r0) && !std::isnan(r1) && r0 <= r1;
		break;
	}
	case ir::PTXInstruction::Hi:
	case ir::PTXInstruction::Gt:
	{
		comparison = !std::isnan(r0) && !std::isnan(r1) && r0 > r1;
		break;
	}
	case ir::PTXInstruction::Hs:
	case ir::PTXInstruction::Ge:
	{
		comparison = !std::isnan(r0) && !std::isnan(r1) && r0 >= r1;
		break;
	}
	case ir::PTXInstruction::Equ:
	{
		comparison = std::isnan(r0) || std::isnan(r1) || r0 == r1;
		break;
	}
	case ir::PTXInstruction::Neu:
	{
		comparison = std::isnan(r0) || std::isnan(r1) || r0 != r1;
		break;
	}
	case ir::PTXInstruction::Ltu:
	{
		comparison = std::isnan(r0) || std::isnan(r1) || r0 < r1;
		break;
	}
	case ir::PTXInstruction::Leu:
	{
		comparison = std::isnan(r0) || std::isnan(r1) || r0 <= r1;
		break;
	}
	case ir::PTXInstruction::Gtu:
	{
		comparison = std::isnan(r0) || std::isnan(r1) || r0 > r1;
		break;
	}
	case ir::PTXInstruction::Geu:
	{
		comparison = std::isnan(r0) || std::isnan(r1) || r0 >= r1;
		break;
	}
	case ir::PTXInstruction::Num:
	{
		comparison = !std::isnan(r0) && !std::isnan(r1);
		break;
	}
	case ir::PTXInstruction::Nan:
	{
		comparison = std::isnan(r0) || std::isnan(r1);
		break;
	}
	default: break;
	}

	bool inverse = !comparison;
	
	switch(boolOp)
	{
	case ir::PTXInstruction::BoolAnd:
	{
		comparison = comparison && predicate;
		inverse    = inverse    && predicate;
		break;
	}
	case ir::PTXInstruction::BoolOr:
	{
		comparison = comparison || predicate;
		inverse    = inverse    || predicate;
		break;
	}
	case ir::PTXInstruction::BoolXor:
	{
		comparison = comparison ^ predicate;
		inverse    = inverse    ^ predicate;
		break;
	}
	default: break;
	}
	
	setParameter(output, 0, comparison);
	if(pq) setParameter(output, sizeof(bool), inverse);
}

test::TestPTXAssembly::TypeVector testSetP_IN(
	test::TestPTXAssembly::DataType type)
{
	test::TestPTXAssembly::TypeVector types(2, type);
	
	types.push_back(test::TestPTXAssembly::I8);

	return types;
}

test::TestPTXAssembly::TypeVector testSetP_OUT(bool pq)
{
	if(pq)
	{
		return test::TestPTXAssembly::TypeVector(2, test::TestPTXAssembly::I8);
	}
	else
	{
		return test::TestPTXAssembly::TypeVector(1, test::TestPTXAssembly::I8);
	}
}
/////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST SET
std::string testSet_PTX(ir::PTXOperand::DataType dtype,
	ir::PTXOperand::DataType stype, ir::PTXOperand::PredicateCondition c, 
	ir::PTXInstruction::BoolOp boolOp, ir::PTXInstruction::CmpOp cmpOp,
	bool ftz)
{
	std::stringstream ptx;

	std::string dTypeString = "." + ir::PTXOperand::toString(dtype);
	std::string sTypeString = "." + ir::PTXOperand::toString(stype);
	
	ptx << PTX_VERSION_AND_TARGET;
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg " << sTypeString << " %rs<2>;           \n";
	ptx << "\t.reg " << dTypeString << " %rd0;             \n";
	ptx << "\t.reg .u8 %b0;                                \n";
	ptx << "\t.reg .u16 %s0;                               \n";
	ptx << "\t.reg .pred %p0;                              \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	ptx << "\tld.global" << sTypeString << " %rs0, [%rIn]; \n";
	ptx << "\tld.global" << sTypeString << " %rs1, [%rIn + " 
		<< ir::PTXOperand::bytes(stype) << "];             \n";
	ptx << "\tld.global.u8 %b0, [%rIn + " 
		<< 2 * ir::PTXOperand::bytes(stype) << "];         \n";
	ptx << "\tcvt.u16.u8 %s0, %b0;                         \n";
	ptx << "\tsetp.lt.u16 %p0, %s0, 128;                   \n";

	ptx << "\tset." << ir::PTXInstruction::toString(cmpOp);

	if(boolOp != ir::PTXInstruction::BoolOp_Invalid)
	{
		ptx << "." << ir::PTXInstruction::toString(boolOp);
	}
	
	if(ftz) ptx << ".ftz";
	
	ptx << dTypeString << sTypeString;
	
	ptx << " %rd0, %rs0, %rs1";
	
	if(boolOp != ir::PTXInstruction::BoolOp_Invalid)
	{
		ptx << ", ";
		
		if(c == ir::PTXOperand::InvPred)
		{
			ptx << "!%p0";
		}
		else
		{
			ptx << "%p0";
		}
	}
	
	ptx << ";\n";
	
	ptx << "\tst.global" << dTypeString << " [%rOut], %rd0;\n";
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

template<typename dtype, typename stype, ir::PTXOperand::PredicateCondition c, 
	ir::PTXInstruction::BoolOp boolOp, ir::PTXInstruction::CmpOp cmpOp,
	bool ftz>
void testSet_REF(void* output, void* input)
{
	stype r0 = getParameter<stype>(input, 0);
	stype r1 = getParameter<stype>(input, sizeof(stype));
	bool predicate = getParameter<uint8_t>(
		input, 2 * sizeof(stype)) < 128;

	if(ftz)
	{
		if(issubnormal(r0)) r0 = 0.0f;
		if(issubnormal(r1)) r1 = 0.0f;
	}

	if(c == ir::PTXOperand::InvPred) predicate = !predicate;

	bool comparison = false;
	
	switch(cmpOp)
	{
	case ir::PTXInstruction::Eq:
	{
		comparison = !std::isnan(r0) && !std::isnan(r1) && r0 == r1;
		break;
	}
	case ir::PTXInstruction::Ne:
	{
		comparison = !std::isnan(r0) && !std::isnan(r1) && r0 != r1;
		break;
	}
	case ir::PTXInstruction::Lo:
	case ir::PTXInstruction::Lt:
	{
		comparison = !std::isnan(r0) && !std::isnan(r1) && r0 < r1;
		break;
	}
	case ir::PTXInstruction::Ls:
	case ir::PTXInstruction::Le:
	{
		comparison = !std::isnan(r0) && !std::isnan(r1) && r0 <= r1;
		break;
	}
	case ir::PTXInstruction::Hi:
	case ir::PTXInstruction::Gt:
	{
		comparison = !std::isnan(r0) && !std::isnan(r1) && r0 > r1;
		break;
	}
	case ir::PTXInstruction::Hs:
	case ir::PTXInstruction::Ge:
	{
		comparison = !std::isnan(r0) && !std::isnan(r1) && r0 >= r1;
		break;
	}
	case ir::PTXInstruction::Equ:
	{
		comparison = std::isnan(r0) || std::isnan(r1) || r0 == r1;
		break;
	}
	case ir::PTXInstruction::Neu:
	{
		comparison = std::isnan(r0) || std::isnan(r1) || r0 != r1;
		break;
	}
	case ir::PTXInstruction::Ltu:
	{
		comparison = std::isnan(r0) || std::isnan(r1) || r0 < r1;
		break;
	}
	case ir::PTXInstruction::Leu:
	{
		comparison = std::isnan(r0) || std::isnan(r1) || r0 <= r1;
		break;
	}
	case ir::PTXInstruction::Gtu:
	{
		comparison = std::isnan(r0) || std::isnan(r1) || r0 > r1;
		break;
	}
	case ir::PTXInstruction::Geu:
	{
		comparison = std::isnan(r0) || std::isnan(r1) || r0 >= r1;
		break;
	}
	case ir::PTXInstruction::Num:
	{
		comparison = !std::isnan(r0) && !std::isnan(r1);
		break;
	}
	case ir::PTXInstruction::Nan:
	{
		comparison = std::isnan(r0) || std::isnan(r1);
		break;
	}
	default: break;
	}
	
	switch(boolOp)
	{
	case ir::PTXInstruction::BoolAnd:
	{
		comparison = comparison && predicate;
		break;
	}
	case ir::PTXInstruction::BoolOr:
	{
		comparison = comparison || predicate;
		break;
	}
	case ir::PTXInstruction::BoolXor:
	{
		comparison = comparison ^ predicate;
		break;
	}
	default: break;
	}
	
	dtype result;
	
	if(typeid(float) == typeid(dtype))
	{
		result = comparison ? 1.0f : 0.0f;
	}
	else
	{
		result = comparison ? 0xffffffff : 0x00000000;
	}
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testSet_IN(
	test::TestPTXAssembly::DataType type)
{
	test::TestPTXAssembly::TypeVector types(2, type);
	
	types.push_back(test::TestPTXAssembly::I8);

	return types;
}

test::TestPTXAssembly::TypeVector testSet_OUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST Special Functions
std::string testSpecial_PTX(ir::PTXInstruction::Opcode opcode, bool ftz)
{
	std::stringstream ptx;
	
	ptx << PTX_VERSION_AND_TARGET;
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg .f32 %f<2>;                             \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	ptx << "\tld.global.f32 %f0, [%rIn];                   \n";

	if(opcode == ir::PTXInstruction::Cos || opcode == ir::PTXInstruction::Sin)
	{
		ptx << "\tabs.f32 %f0, %f0;                        \n";
		ptx << "\tmin.f32 %f0, %f0, 0f3f4ccccd;            \n";
	}

	ptx << "\t" << ir::PTXInstruction::toString(opcode) << ".approx";

	if(ftz) ptx << ".ftz";
	
	ptx << ".f32 %f1, %f0;\n";
	
	ptx << "\tst.global.f32 [%rOut], %f1;                  \n";
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

template<ir::PTXInstruction::Opcode opcode, bool ftz>
void testSpecial_REF(void* output, void* input)
{
	static_assert(opcode == ir::PTXInstruction::Cos
		|| opcode == ir::PTXInstruction::Sin
		|| opcode == ir::PTXInstruction::Lg2
		|| opcode == ir::PTXInstruction::Ex2, "Invalid special opcode.");

	float r0 = getParameter<float>(input, 0);

	if(ftz)
	{
		if(issubnormal(r0)) r0 = 0.0f;
	}

	float result = 0.0f;

	switch(opcode)
	{
		case ir::PTXInstruction::Cos:
		{
			if(std::isinf(r0))
			{
				if(r0 < 0)
				{
					result = std::numeric_limits<float>::signaling_NaN();
				}
				else
				{
					result = std::cos(0.8);
				}
			}
			else if(std::isnan(r0))
			{
				result = std::cos(0.8);
			}
			else
			{
				result = std::cos(std::min(std::abs(r0), 0.8f));
			}
			break;
		}
		case ir::PTXInstruction::Sin:
		{
			if(std::isinf(r0) || std::isnan(r0))
			{
				if(r0 < 0)
				{
					result = std::numeric_limits<float>::signaling_NaN();
				}
				else
				{
					result = std::sin(0.8);
				}
			}
			else if(std::isnan(r0))
			{
				result = std::sin(0.8);
			}
			else
			{
				result = std::sin(std::min(std::abs(r0), 0.8f));
			}
			break;
		}
		case ir::PTXInstruction::Lg2:
		{
			if(std::isinf(r0) && std::copysign(1.0f, r0) < 0)
			{
				result = std::numeric_limits<float>::signaling_NaN();
			}
			else if(r0 == 0.0f)
			{
				result = -std::numeric_limits<float>::infinity();
			}
			else
			{
				result = std::log2f(r0);
			}
			break;
		}
		case ir::PTXInstruction::Ex2:
		{
			if(std::isinf(r0))
			{
				if(r0 < 0)
				{
					result = 0.0f;
				}
				else
				{
					result = r0;
				}
			}
			else
			{
				result = std::exp2f(r0);
			}
			break;
		}
		default: break;
	}
	
	if(ftz)
	{
		if(issubnormal(result)) result = 0.0f;
	}
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testSpecial_INOUT()
{
	return test::TestPTXAssembly::TypeVector(1, test::TestPTXAssembly::FP32);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST RSQRT
std::string testRsqrt_PTX(ir::PTXOperand::DataType type, bool ftz)
{
	std::stringstream ptx;
	
	std::string typeString;
	
	if(type == ir::PTXOperand::f32)
	{
		typeString = ".f32";
	}
	else
	{
		typeString = ".f64";
	}
	
	ptx << PTX_VERSION_AND_TARGET;
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg " << typeString << " %f<2>;             \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	ptx << "\tld.global" << typeString << " %f0, [%rIn];   \n";

	ptx << "\trsqrt.approx";
	
	if(ftz) ptx << ".ftz";
	
	ptx << typeString << " %f1, %f0;\n";
	
	ptx << "\tst.global" << typeString << " [%rOut], %f1;  \n";
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

template<typename type, bool ftz>
void testRsqrt_REF(void* output, void* input)
{
	static_assert(sizeof(type) == 4 || sizeof(type) == 8, "only f32/f64 valid");

	type r0 = getParameter<type>(input, 0);

	if(ftz)
	{
		if(issubnormal(r0)) r0 = 0.0f;
	}

	type result = 0;

	if(std::isinf(r0))
	{
		result = (type)0;
	}
	else
	{
		result = (type)1 / std::sqrt(r0);
	}
	
	if(ftz)
	{
		if(issubnormal(result)) result = 0.0f;
	}
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testRsqrt_INOUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// TEST RCP/SQRT
std::string testRcpSqrt_PTX(ir::PTXOperand::DataType type, bool sqrt, 
	bool approx, bool ftz)
{
	std::stringstream ptx;
	
	std::string typeString;
	
	if(type == ir::PTXOperand::f32)
	{
		typeString = ".f32";
	}
	else
	{
		typeString = ".f64";
	}
	
	ptx << PTX_VERSION_AND_TARGET;
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg " << typeString << " %f<2>;             \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	ptx << "\tld.global" << typeString << " %f0, [%rIn];   \n";

	if(sqrt)
	{
		ptx << "\tsqrt";
	}	
	else
	{
		ptx << "\trcp";
	}
	
	if(approx)
	{
		ptx << ".approx";
	}
	else
	{
		ptx << ".rn";
	}
	
	if(ftz) ptx << ".ftz";
	
	ptx << typeString << " %f1, %f0;\n";
	
	ptx << "\tst.global" << typeString << " [%rOut], %f1;  \n";
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

template<typename type, bool sqrt, bool approx, bool ftz>
void testRcpSqrt_REF(void* output, void* input)
{
	typedef uint64_t uint;
	static_assert(sizeof(type) == 4 || sizeof(type) == 8, "only f32/f64 valid");

	type r0 = getParameter<type>(input, 0);

	if(ftz)
	{
		if(issubnormal(r0)) r0 = std::copysign((type)0, r0);
	}

	type result = 0;

	if(sqrt)
	{
		if(r0 < (type)0 || std::isnan(r0))
		{
			result = std::numeric_limits<type>::signaling_NaN();
		}
		else
		{
			result = std::sqrt(r0);
		}
	}
	else
	{
		if(approx && std::isinf(r0))
		{
			result = std::copysign((type)0, r0);
		}
		else
		{
			if(sizeof(type) == 8 && approx)
			{
				if(r0 == 0.0)
				{
					result = std::numeric_limits<double>::infinity();
				}
				else if(r0 == -0.0)
				{
					result = -std::numeric_limits<double>::infinity();
				}
				else
				{
					uint32_t upper = 0;
					uint32_t lower = 0;
					uint word = hydrazine::bit_cast<uint>(r0);
				
					upper = word >> 32;
					lower = 0;
								
					word = lower | ((uint)upper << 32);
				
					result = 1.0 / hydrazine::bit_cast<type>(word);
				}
			}
			else
			{
				result = (type)1 / r0;
			}
		}
	}
	
	if(ftz)
	{
		if(issubnormal(result)) result = (type)0;
	}
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testRcpSqrt_INOUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST FMIN/FMAX
std::string testFMinMax_PTX(ir::PTXOperand::DataType type, bool min, bool ftz)
{
	std::stringstream ptx;
	
	std::string typeString;
	
	if(type == ir::PTXOperand::f32)
	{
		typeString = ".f32";
	}
	else
	{
		typeString = ".f64";
	}
	
	ptx << PTX_VERSION_AND_TARGET;
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg " << typeString << " %f<3>;             \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	ptx << "\tld.global" << typeString << " %f0, [%rIn];   \n";
	ptx << "\tld.global" << typeString << " %f1, [%rIn + "
		<< ir::PTXOperand::bytes(type) << "];              \n";
	
	if(min)
	{
		ptx << "\tmin";
	}
	else
	{
		ptx << "\tmax";
	}
	
	if(ftz) ptx << ".ftz";
	
	ptx << typeString << " %f2, %f0, %f1;  \n";
	
	ptx << "\tst.global" << typeString << " [%rOut], %f2;  \n";
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

template<typename type, bool min, bool ftz>
void testFMinMax_REF(void* output, void* input)
{
	static_assert(sizeof(type) == 4 || sizeof(type) == 8, "only f32/f64 valid");
	static_assert(sizeof(type) != 8 || !ftz, "ftz only valid for f32");

	type r0 = getParameter<type>(input, 0);
	type r1 = getParameter<type>(input, sizeof(type));

	if(ftz)
	{
		if(issubnormal(r0)) r0 = 0.0f;
		if(issubnormal(r1)) r1 = 0.0f;
	}

	type result = 0;

	if(std::isnan(r0))
	{	
		result = r1;
	}
	else if(std::isnan(r1))
	{
		result = r0;
	}
	else
	{
		if(min)
		{
			result = std::min(r0, r1);
		}
		else
		{
			result = std::max(r0, r1);
		}
	}
	
	if(ftz)
	{
		if(issubnormal(result)) result = 0.0f;
	}
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testFMinMax_IN(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(2, type);
}

test::TestPTXAssembly::TypeVector testFMinMax_OUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST ABS/NEG
std::string testAbsNeg_PTX(ir::PTXOperand::DataType type, bool neg, bool ftz)
{
	std::stringstream ptx;
	
	std::string typeString;
	
	if(type == ir::PTXOperand::f32)
	{
		typeString = ".f32";
	}
	else
	{
		typeString = ".f64";
	}
	
	ptx << PTX_VERSION_AND_TARGET;
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg " << typeString << " %f<2>;             \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	ptx << "\tld.global" << typeString << " %f0, [%rIn];   \n";
	
	if(neg)
	{
		ptx << "\tneg";
	}
	else
	{
		ptx << "\tabs";
	}
	
	if(ftz) ptx << ".ftz";
	
	ptx << typeString << " %f1, %f0;  \n";
	
	ptx << "\tst.global" << typeString << " [%rOut], %f1;  \n";
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

template<typename type, bool neg, bool ftz>
void testAbsNeg_REF(void* output, void* input)
{
	static_assert(sizeof(type) == 4 || sizeof(type) == 8, "only f32/f64 valid");
	static_assert(sizeof(type) != 8 || !ftz, "ftz only valid for f32");

	type r0 = getParameter<type>(input, 0);

	if(ftz)
	{
		if(issubnormal(r0)) r0 = 0.0f;
	}

	type result = 0;
	
	if(neg)
	{
		result = -r0;
	}
	else
	{
		result = std::abs(r0);
	}
	if(ftz)
	{
		if(issubnormal(result)) result = 0.0f;
	}
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testAbsNeg_INOUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST DIV
std::string testFdiv_PTX(ir::PTXOperand::DataType type, int modifier)
{
	bool sat = modifier & ir::PTXInstruction::sat;
	bool ftz = modifier & ir::PTXInstruction::ftz;
	std::string roundingString = ir::PTXInstruction::roundingMode(
		(ir::PTXInstruction::Modifier) modifier);
	
	std::stringstream ptx;
	
	std::string typeString;
	
	if(type == ir::PTXOperand::f32)
	{
		typeString = ".f32";
	}
	else
	{
		typeString = ".f64";
	}
	
	ptx << PTX_VERSION_AND_TARGET;
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg " << typeString << " %f<3>;             \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	ptx << "\tld.global" << typeString << " %f0, [%rIn];   \n";
	ptx << "\tld.global" << typeString << " %f1, [%rIn + "
		<< ir::PTXOperand::bytes(type) << "];              \n";
	
	if(modifier & ir::PTXInstruction::approx)
	{
		ptx << "\tdiv.approx";
	}
	else if(modifier & ir::PTXInstruction::full)
	{
		ptx << "\tdiv.full";
	}
	else
	{
		ptx << "\tdiv.rn";
	}
	
	if(!roundingString.empty()) ptx << "." << roundingString;
	
	if(ftz) ptx << ".ftz";
	if(sat) ptx << ".sat";
	
	ptx << typeString << " %f2, %f0, %f1;  \n";
	
	ptx << "\tst.global" << typeString << " [%rOut], %f2;  \n";
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

template<typename type, int modifier>
void testFdiv_REF(void* output, void* input)
{
	static_assert(sizeof(type) == 4 || sizeof(type) == 8, "only f32/f64 valid");
	static_assert(sizeof(type) != 8 || !(modifier & ir::PTXInstruction::ftz),
		"ftz only valid for f32");

	type r0 = getParameter<type>(input, 0);
	type r1 = getParameter<type>(input, sizeof(type));

	if(modifier & ir::PTXInstruction::ftz)
	{
		if(issubnormal(r0)) r0 = std::copysign(0.0f, r0);
		if(issubnormal(r1)) r1 = std::copysign(0.0f, r1);
	}

	type result = 0;
	
	if(modifier & ir::PTXInstruction::approx)
	{
		if(issubnormal(r0) || issubnormal(r1))
		{
			result = r0 / r1;
		}
		else
		{
			result = r0 * ( 1.0f / r1 );
		}
	}
	else
	{
		result = r0 / r1;
	}

	if(modifier & ir::PTXInstruction::ftz)
	{
		if(issubnormal(result)) result = std::copysign(0.0f, result);
	}
	
	if(modifier & ir::PTXInstruction::sat)
	{
		if(result < 0.0f) result = 0.0f;
		if(result > 1.0f) result = 1.0f;
		if(std::isnan(result)) result = 0.0f;
	}
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testFdiv_IN(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(2, type);
}

test::TestPTXAssembly::TypeVector testFdiv_OUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST FMA
std::string testFma_PTX(ir::PTXOperand::DataType type, int modifier, bool mad)
{
	bool sat = modifier & ir::PTXInstruction::sat;
	bool ftz = modifier & ir::PTXInstruction::ftz;
	std::string roundingString = ir::PTXInstruction::roundingMode(
		(ir::PTXInstruction::Modifier) modifier);
	
	std::stringstream ptx;
	
	std::string typeString;
	
	if(type == ir::PTXOperand::f32)
	{
		typeString = ".f32";
	}
	else
	{
		typeString = ".f64";
	}
	
	ptx << PTX_VERSION_AND_TARGET;
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg " << typeString << " %f<4>;             \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	ptx << "\tld.global" << typeString << " %f0, [%rIn];   \n";
	ptx << "\tld.global" << typeString << " %f1, [%rIn + "
		<< ir::PTXOperand::bytes(type) << "];              \n";
	ptx << "\tld.global" << typeString << " %f2, [%rIn + "
		<< 2 * ir::PTXOperand::bytes(type) << "];          \n";
	
	if(mad)
	{
		ptx << "\tmad.rn";
	}
	else
	{
		ptx << "\tfma.rn";
	}
	
	if(!roundingString.empty()) ptx << "." << roundingString;
	
	if(ftz) ptx << ".ftz";
	if(sat) ptx << ".sat";
	
	ptx << typeString << " %f3, %f0, %f1, %f2;  \n";
	
	ptx << "\tst.global" << typeString << " [%rOut], %f3;  \n";
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

template<typename type, int modifier>
void testFma_REF(void* output, void* input)
{
	static_assert(sizeof(type) == 4 || sizeof(type) == 8, "only f32/f64 valid");
	static_assert(sizeof(type) != 8 || !(modifier & ir::PTXInstruction::sat),
		"sat only valid for f32");
	static_assert(sizeof(type) != 8 || !(modifier & ir::PTXInstruction::ftz),
		"ftz only valid for f32");

	type r0 = getParameter<type>(input, 0);
	type r1 = getParameter<type>(input, sizeof(type));
	type r2 = getParameter<type>(input, 2*sizeof(type));

	if(modifier & ir::PTXInstruction::ftz)
	{
		if(issubnormal(r0)) r0 = 0.0f;
		if(issubnormal(r1)) r1 = 0.0f;
		if(issubnormal(r2)) r2 = 0.0f;
	}

	type result = r0 * r1 + r2;
		
	if(modifier & ir::PTXInstruction::ftz)
	{
		if(issubnormal(result)) result = 0.0f;
	}
	
	if(modifier & ir::PTXInstruction::sat)
	{
		if(result < 0.0f) result = 0.0f;
		if(result > 1.0f) result = 1.0f;
		if(std::isnan(result)) result = 0.0f;
	}
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testFma_IN(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(3, type);
}

test::TestPTXAssembly::TypeVector testFma_OUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST FMUL
std::string testFmul_PTX(ir::PTXOperand::DataType type, int modifier)
{
	bool sat = modifier & ir::PTXInstruction::sat;
	bool ftz = modifier & ir::PTXInstruction::ftz;
	std::string roundingString = ir::PTXInstruction::roundingMode(
		(ir::PTXInstruction::Modifier) modifier);
	
	std::stringstream ptx;
	
	std::string typeString;
	
	if(type == ir::PTXOperand::f32)
	{
		typeString = ".f32";
	}
	else
	{
		typeString = ".f64";
	}
	
	ptx << PTX_VERSION_AND_TARGET;
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg " << typeString << " %f<3>;             \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	ptx << "\tld.global" << typeString << " %f0, [%rIn];   \n";
	ptx << "\tld.global" << typeString << " %f1, [%rIn + "
		<< ir::PTXOperand::bytes(type) << "];              \n";
	
	ptx << "\tmul";
	
	if(!roundingString.empty()) ptx << "." << roundingString;
	
	if(ftz) ptx << ".ftz";
	if(sat) ptx << ".sat";
	
	ptx << typeString << " %f2, %f0, %f1;  \n";
	
	ptx << "\tst.global" << typeString << " [%rOut], %f2;  \n";
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

template<typename type, int modifier>
void testFmul_REF(void* output, void* input)
{
	static_assert(sizeof(type) == 4 || sizeof(type) == 8, "only f32/f64 valid");
	static_assert(sizeof(type) != 8 || !(modifier & ir::PTXInstruction::sat),
		"sat only valid for f32");
	static_assert(sizeof(type) != 8 || !(modifier & ir::PTXInstruction::ftz),
		"ftz only valid for f32");

	type r0 = getParameter<type>(input, 0);
	type r1 = getParameter<type>(input, sizeof(type));

	if(modifier & ir::PTXInstruction::ftz)
	{
		if(issubnormal(r0)) r0 = 0.0f;
		if(issubnormal(r1)) r1 = 0.0f;
	}


	type result = r0 * r1;
		
	if(modifier & ir::PTXInstruction::ftz)
	{
		if(issubnormal(result)) result = 0.0f;
	}
	
	if(modifier & ir::PTXInstruction::sat)
	{
		if(result < 0.0f) result = 0.0f;
		if(result > 1.0f) result = 1.0f;
		if(std::isnan(result)) result = 0.0f;
	}
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testFmul_IN(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(2, type);
}

test::TestPTXAssembly::TypeVector testFmul_OUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST FADD
std::string testFadd_PTX(ir::PTXOperand::DataType type, int modifier, bool add)
{
	bool sat = modifier & ir::PTXInstruction::sat;
	bool ftz = modifier & ir::PTXInstruction::ftz;
	std::string roundingString = ir::PTXInstruction::roundingMode(
		(ir::PTXInstruction::Modifier) modifier);
	
	std::stringstream ptx;
	
	std::string typeString;
	
	if(type == ir::PTXOperand::f32)
	{
		typeString = ".f32";
	}
	else
	{
		typeString = ".f64";
	}
	
	ptx << PTX_VERSION_AND_TARGET;
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg " << typeString << " %f<3>;             \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	ptx << "\tld.global" << typeString << " %f0, [%rIn];   \n";
	ptx << "\tld.global" << typeString << " %f1, [%rIn + "
		<< ir::PTXOperand::bytes(type) << "];              \n";
	
	if(add)
	{
		ptx << "\tadd";
	}
	else
	{
		ptx << "\tsub";
	}
	
	if(!roundingString.empty()) ptx << "." << roundingString;
	
	if(ftz) ptx << ".ftz";
	if(sat) ptx << ".sat";
	
	ptx << typeString << " %f2, %f0, %f1;  \n";
	
	ptx << "\tst.global" << typeString << " [%rOut], %f2;  \n";
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

template<typename type, int modifier, bool add>
void testFadd_REF(void* output, void* input)
{
	static_assert(sizeof(type) == 4 || sizeof(type) == 8, "only f32/f64 valid");
	static_assert(sizeof(type) != 8 || !(modifier & ir::PTXInstruction::sat),
		"sat only valid for f32");
	static_assert(sizeof(type) != 8 || !(modifier & ir::PTXInstruction::ftz),
		"ftz only valid for f32");

	type r0 = getParameter<type>(input, 0);
	type r1 = getParameter<type>(input, sizeof(type));

	if(modifier & ir::PTXInstruction::ftz)
	{
		if(issubnormal(r0)) r0 = 0.0f;
		if(issubnormal(r1)) r1 = 0.0f;
	}


	type result = 0;
	
	if(add)
	{
		result = r0 + r1;
	}
	else
	{
		result = r0 - r1;
	}
	
	if(modifier & ir::PTXInstruction::ftz)
	{
		if(issubnormal(result)) result = 0.0f;
	}
	
	if(modifier & ir::PTXInstruction::sat)
	{
		if(result < 0.0f) result = 0.0f;
		if(result > 1.0f) result = 1.0f;
		if(std::isnan(result)) result = 0.0f;
	}
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testFadd_IN(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(2, type);
}

test::TestPTXAssembly::TypeVector testFadd_OUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST COPYSIGN
std::string testCopysign_PTX(ir::PTXOperand::DataType type)
{
	std::stringstream ptx;
	
	std::string typeString;
	
	if(type == ir::PTXOperand::f32)
	{
		typeString = ".f32";
	}
	else
	{
		typeString = ".f64";
	}
	
	ptx << PTX_VERSION_AND_TARGET;
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg " << typeString << " %f<3>;             \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	ptx << "\tld.global" << typeString << " %f0, [%rIn];   \n";
	ptx << "\tld.global" << typeString << " %f1, [%rIn + "
		<< ir::PTXOperand::bytes(type) << "];              \n";
	ptx << "\tcopysign" << typeString << " %f2, %f0, %f1;  \n";
	ptx << "\tst.global" << typeString << " [%rOut], %f2;  \n";
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

template<typename type>
void testCopysign_REF(void* output, void* input)
{
	static_assert(sizeof(type) == 4 || sizeof(type) == 8, "only f32/f64");
	
	type r0 = getParameter<type>(input, 0);
	type r1 = getParameter<type>(input, sizeof(type));

	if(sizeof(type) == 4)
	{
		const uint32_t bits = 31;
		const uint32_t mask = 1 << bits;
		uint32_t result = hydrazine::bit_cast<uint32_t>(r0) & mask;

		result = result | ( hydrazine::bit_cast<uint32_t>(r1) & ~mask );

		setParameter(output, 0, result);
	}
	else
	{
		const uint32_t bits = 63;
		const uint64_t mask = (
			(uint64_t) 1 << bits);
		uint64_t result = hydrazine::bit_cast<
			uint64_t>(r0) & mask;

		result = result | ( hydrazine::bit_cast<
			uint64_t>(r1) & ~mask );

		setParameter(output, 0, result);
	}
}

test::TestPTXAssembly::TypeVector testCopysign_IN(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(2, type);
}

test::TestPTXAssembly::TypeVector testCopysign_OUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST TESTP
std::string testTestp_PTX(ir::PTXOperand::DataType type, 
	ir::PTXInstruction::FloatingPointMode mode)
{
	std::stringstream ptx;
	
	std::string typeString;
	
	if(type == ir::PTXOperand::f32)
	{
		typeString = ".f32";
	}
	else
	{
		typeString = ".f64";
	}
	
	ptx << PTX_VERSION_AND_TARGET;
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg " << typeString << " %f0;               \n";
	ptx << "\t.reg .u32 %r0;                               \n";
	ptx << "\t.reg .pred %p0;                              \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	ptx << "\tld.global" << typeString << " %f0, [%rIn];   \n";
	ptx << "\ttestp." << ir::PTXInstruction::toString(mode) 
		<< typeString << " %p0, %f0;\n";
	ptx << "\tselp.u32 %r0, 0xffffffff, 0x0, %p0;          \n";
	ptx << "\tst.global.u32 [%rOut], %r0;                  \n";
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";
	
	return ptx.str();
}

template<typename type, ir::PTXInstruction::FloatingPointMode mode>
void testTestP_REF(void* output, void* input)
{
	type r0 = getParameter<type>(input, 0);

	bool condition = false;

	switch(mode)
	{
	default:
	case ir::PTXInstruction::Finite:
	{
		condition = !std::isinf(r0) && !std::isnan(r0);
	}
	break;
	case ir::PTXInstruction::Infinite:
	{
		condition = std::isinf(r0);
	}
	break;
	case ir::PTXInstruction::Number:
	{
		condition = !std::isnan(r0);
	}
	break;
	case ir::PTXInstruction::NotANumber:
	{
		condition = std::isnan(r0);
	}
	break;
	case ir::PTXInstruction::Normal:
	{
		condition = std::isnormal(r0);
	}
	break;
	case ir::PTXInstruction::SubNormal:
	{
		condition = issubnormal(r0);
	}
	break;
	}

	uint32_t result = condition ? 0xffffffff : 0x0;

	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testTestp_IN(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}

test::TestPTXAssembly::TypeVector testTestp_OUT()
{
	return test::TestPTXAssembly::TypeVector(1, test::TestPTXAssembly::I32);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST CUDA NESTED PARALLELISM
std::string testCudaNestedParallelism_PTX()
{
	std::stringstream ptx;

	ptx <<
		"// testCudaNestedParallelism_PTX()\n"
		".version 3.1\n"
		".target sm_35\n"
		".address_size 64\n"
		"\n"
		".extern .func  (.param .b64 func_retval0) cudaGetParameterBuffer\n"
		"(\n"
		"	.param .b64 cudaGetParameterBuffer_param_0,\n"
		"	.param .b64 cudaGetParameterBuffer_param_1\n"
		");\n"
		"\n"
		".extern .func  (.param .b32 func_retval0) cudaLaunchDevice\n"
		"(\n"
		"	.param .b64 cudaLaunchDevice_param_0,\n"
		"	.param .b64 cudaLaunchDevice_param_1,\n"
		"	.param .align 4 .b8 cudaLaunchDevice_param_2[12],\n"
		"	.param .align 4 .b8 cudaLaunchDevice_param_3[12],\n"
		"	.param .b32 cudaLaunchDevice_param_4,\n"
		"	.param .b64 cudaLaunchDevice_param_5\n"
		");\n"
		"\n"
		".visible .entry _Z10nestedCalliPi(\n"
		"	.param .u32 _Z10nestedCalliPi_param_0,\n"
		"	.param .u64 _Z10nestedCalliPi_param_1\n"
		")\n"
		"{\n"
		"	.reg .pred 	%p<3>;\n"
		"	.reg .s32 	%r<9>;\n"
		"	.reg .s64 	%rd<10>;\n"
		"\n"
		"\n"
		"	ld.param.u32 	%r1, [_Z10nestedCalliPi_param_0];\n"
		"	ld.param.u64 	%rd2, [_Z10nestedCalliPi_param_1];\n"
		"	cvta.to.global.u64 	%rd3, %rd2;\n"
		"	mul.wide.s32 	%rd4, %r1, 4;\n"
		"	add.s64 	%rd5, %rd3, %rd4;\n"
		"	st.global.u32 	[%rd5], %r1;\n"
		"	setp.lt.s32 	%p1, %r1, 1;\n"
		"	@%p1 bra 	BB0_3;\n"
		"\n"
		"	mov.u64 	%rd6, 4;\n"
		"	mov.u64 	%rd7, 16;\n"
		"	// Callseq Start 0\n"
		"	{\n"
		"	.reg .b32 temp_param_reg;\n"
		"	.param .b64 param0;\n"
		"	st.param.b64	[param0+0], %rd6;\n"
		"	.param .b64 param1;\n"
		"	st.param.b64	[param1+0], %rd7;\n"
		"	.param .b64 retval0;\n"
		"	call.uni (retval0), \n"
		"	cudaGetParameterBuffer, \n"
		"	(\n"
		"	param0,\n" 
		"	param1\n"
		"	);\n"
		"	ld.param.b64	%rd1, [retval0+0];\n"
		"	}\n"
		"	// Callseq End 0\n"
		"	setp.eq.s64 	%p2, %rd1, 0;\n"
		"	@%p2 bra 	BB0_3;\n"
		"\n"
		"	add.s32 	%r3, %r1, -1;\n"
		"	st.u32 	[%rd1], %r3;\n"
		"	st.u64 	[%rd1+8], %rd2;\n"
		"	mov.u32 	%r6, 1;\n"
		"	mov.u32 	%r7, 0;\n"
		"	mov.u64 	%rd8, 0;\n"
		"	mov.u64 	%rd9, _Z10nestedCalliPi;\n"
		"	// Callseq Start 1\n"
		"	{\n"
		"	.reg .b32 temp_param_reg;\n"
		"	.param .b64 param0;\n"
		"	st.param.b64	[param0+0], %rd9;\n"
		"	.param .b64 param1;\n"
		"	st.param.b64	[param1+0], %rd1;\n"
		"	.param .align 4 .b8 param2[12];\n"
		"	st.param.b32	[param2+0], %r6;\n"
		"	st.param.b32	[param2+4], %r6;\n"
		"	st.param.b32	[param2+8], %r6;\n"
		"	.param .align 4 .b8 param3[12];\n"
		"	st.param.b32	[param3+0], %r6;\n"
		"	st.param.b32	[param3+4], %r6;\n"
		"	st.param.b32	[param3+8], %r6;\n"
		"	.param .b32 param4;\n"
		"	st.param.b32	[param4+0], %r7;\n"
		"	.param .b64 param5;\n"
		"	st.param.b64	[param5+0], %rd8;\n"
		"	.param .b32 retval0;\n"
		"	call.uni (retval0), \n"
		"	cudaLaunchDevice, \n"
		"	(\n"
		"	param0, \n"
		"	param1, \n"
		"	param2, \n"
		"	param3, \n"
		"	param4,\n" 
		"	param5\n"
		"	);\n"
		"	ld.param.b32	%r8, [retval0+0];\n"
		"	}\n"
		"	// Callseq End 1\n"
		"\n"
		"BB0_3:\n"
		"	ret;\n"
		"}\n"
		"\n"
		".visible .entry test(\n"
		"	.param .u64 out, .param .u64 in)\n"
		"{\n"
		"	.reg .pred 	%p<2>;\n"
		"	.reg .s32 	%r<8>;\n"
		"	.reg .s64 	%rd<8>;\n"
		"\n"
		"\n"
		"	ld.param.u64 	%rd2, [out];\n"
		"	mov.u64 	%rd3, 4;\n"
		"	mov.u64 	%rd4, 16;\n"
		"	// Callseq Start 2\n"
		"	{\n"
		"	.reg .b32 temp_param_reg;\n"
		"	.param .b64 param0;\n"
		"	st.param.b64	[param0+0], %rd3;\n"
		"	.param .b64 param1;\n"
		"	st.param.b64	[param1+0], %rd4;\n"
		"	.param .b64 retval0;\n"
		"	call.uni (retval0), \n"
		"	cudaGetParameterBuffer, \n"
		"	(\n"
		"	param0,\n" 
		"	param1\n"
		"	);\n"
		"	ld.param.b64	%rd1, [retval0+0];\n"
		"	}\n"
		"	// Callseq End 2\n"
		"	setp.eq.s64 	%p1, %rd1, 0;\n"
		"	@%p1 bra 	BB1_2;\n"
		"\n"
		"   ld.param.u64 %rd7, [in];\n"
		"   ld.global.u32 %r7, [%rd7];\n"
		"   rem.u32 %r7, %r7, 10;\n"
		"	mov.u32 	%r1, %r7;\n"
		"	st.u32 	[%rd1], %r1;\n"
		"	st.u64 	[%rd1+8], %rd2;\n"
		"	mov.u32 	%r4, 1;\n"
		"	mov.u32 	%r5, 0;\n"
		"	mov.u64 	%rd5, 0;\n"
		"	mov.u64 	%rd6, _Z10nestedCalliPi;\n"
		"	// Callseq Start 3\n"
		"	{\n"
		"	.reg .b32 temp_param_reg;\n"
		"	.param .b64 param0;\n"
		"	st.param.b64	[param0+0], %rd6;\n"
		"	.param .b64 param1;\n"
		"	st.param.b64	[param1+0], %rd1;\n"
		"	.param .align 4 .b8 param2[12];\n"
		"	st.param.b32	[param2+0], %r4;\n"
		"	st.param.b32	[param2+4], %r4;\n"
		"	st.param.b32	[param2+8], %r4;\n"
		"	.param .align 4 .b8 param3[12];\n"
		"	st.param.b32	[param3+0], %r4;\n"
		"	st.param.b32	[param3+4], %r4;\n"
		"	st.param.b32	[param3+8], %r4;\n"
		"	.param .b32 param4;\n"
		"	st.param.b32	[param4+0], %r5;\n"
		"	.param .b64 param5;\n"
		"	st.param.b64	[param5+0], %rd5;\n"
		"	.param .b32 retval0;\n"
		"	call.uni (retval0), \n"
		"	cudaLaunchDevice, \n"
		"	(\n"
		"	param0, \n"
		"	param1, \n"
		"	param2, \n"
		"	param3, \n"
		"	param4,\n" 
		"	param5\n"
		"	);\n"
		"	ld.param.b32	%r6, [retval0+0];\n"
		"	}\n"
		"	// Callseq End 3\n"
		"\n"
		"BB1_2:\n"
		"	ret;\n"
		"}\n"; 
   
	return ptx.str();
}

void testCudaNestedParallelism_REF(void* output, void* input)
{
	for(uint32_t i = 0; i < 10; ++i)
	{
		setParameter(output, i * sizeof(uint32_t), 0);
	}
	
	uint32_t limit = getParameter<uint32_t>(input, 0) % 10;

	for(uint32_t i = 0; i <= limit; ++i)
	{
		setParameter(output, i * sizeof(uint32_t), i);
	}
}

test::TestPTXAssembly::TypeVector testCudaNestedParallelism_IN()
{
	return test::TestPTXAssembly::TypeVector(1, test::TestPTXAssembly::I32);
}

test::TestPTXAssembly::TypeVector testCudaNestedParallelism_OUT()
{
	return test::TestPTXAssembly::TypeVector(10, test::TestPTXAssembly::I32);
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// TEST INDIRECT FUNCTION CALLS
std::string testIndirectFunctionCall_PTX()
{
	std::stringstream ptx;
	
	ptx << PTX_VERSION_AND_TARGET;
	ptx << ".visible .func (.param .u32 return) add0(.param .u32 a)\n";
	ptx << ".visible .func (.param .u32 return) add1(.param .u32 a)\n";
	ptx << ".visible .func (.param .u32 return) add2(.param .u32 a)\n";
	ptx << ".visible .func (.param .u32 return) add3(.param .u32 a)\n";
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in)   \n";
	ptx << "{\t                                            \n";
	ptx << "\t.reg .u64 %rIn, %rOut;                       \n";
	ptx << "\t.reg .u32 %r<3>;                             \n";
	ptx << "\t.reg .u64 %functionPointer;                  \n";
	ptx << "\t.reg .pred %p0;                              \n";
	ptx << "\t.reg .u32 %thread;                           \n";
	ptx << "\t.reg .u64 %offset;                           \n";
	ptx << "\t.param .u32 operandA;                        \n";
	ptx << "\t.param .u32 result;                          \n";
	ptx << "\tld.param.u64 %rIn, [in];                     \n";
	ptx << "\tld.param.u64 %rOut, [out];                   \n";
	ptx << "\tld.global.u32 %r0, [%rIn];                   \n";
	ptx << "\tst.param.u32 [operandA], %r0;                \n";
	ptx << "\tmov.u32 %thread, %tid.x;                     \n";
	ptx << "\tmov.u64 %functionPointer, add0;              \n";
	ptx << "\tsetp.eq.u32 %p0, %thread, 1;                 \n";
	ptx << "\t@%p0 mov.u64 %functionPointer, add1;         \n";
	ptx << "\tsetp.eq.u32 %p0, %thread, 2;                 \n";
	ptx << "\t@%p0 mov.u64 %functionPointer, add2;         \n";
	ptx << "\tsetp.eq.u32 %p0, %thread, 3;                 \n";
	ptx << "\t@%p0 mov.u64 %functionPointer, add3;         \n";
	ptx << "\tprototype: .callprototype (.param .u32 _)    \n";
	ptx << "\t    _ (.param .u32 _);                       \n";
	ptx << "\tcall (result), %functionPointer,             \n";
	ptx << "\t    (operandA), prototype;                   \n";
	ptx << "\tld.param.u32 %r2, [result];                  \n";
	ptx << "\tcvt.u64.u32 %offset, %thread;                \n";
	ptx << "\tmul.lo.u64 %offset, %offset, 4;              \n";
	ptx << "\tadd.u64 %rOut, %offset, %rOut;               \n";
	ptx << "\tst.global.u32 [%rOut], %r2;                  \n";
	ptx << "\texit;                                        \n";
	ptx << "}                                              \n";
	ptx << "                                               \n";

	ptx << ".visible .func (.param .u32 return) add0(.param .u32 a) \n";
	ptx << "{\t                                 \n";
	ptx << "\t.reg .u32 %r<3>;                  \n";
	ptx << "\tld.param.u32 %r0, [a];            \n";
	ptx << "\tadd.u32 %r0, %r0, 0;              \n";
	ptx << "\tst.param.u32 [return], %r0;       \n";
	ptx << "\tret 0;                            \n";
	ptx << "}                                   \n";

	ptx << ".visible .func (.param .u32 return) add1(.param .u32 a) \n";
	ptx << "{\t                                 \n";
	ptx << "\t.reg .u32 %r<3>;                  \n";
	ptx << "\tld.param.u32 %r0, [a];            \n";
	ptx << "\tadd.u32 %r0, %r0, 1;              \n";
	ptx << "\tst.param.u32 [return], %r0;       \n";
	ptx << "\tret 0;                            \n";
	ptx << "}                                   \n";

	ptx << ".visible .func (.param .u32 return) add2(.param .u32 a) \n";
	ptx << "{\t                                 \n";
	ptx << "\t.reg .u32 %r<3>;                  \n";
	ptx << "\tld.param.u32 %r0, [a];            \n";
	ptx << "\tadd.u32 %r0, %r0, 2;              \n";
	ptx << "\tst.param.u32 [return], %r0;       \n";
	ptx << "\tret 0;                            \n";
	ptx << "}                                   \n";

	ptx << ".visible .func (.param .u32 return) add3(.param .u32 a) \n";
	ptx << "{\t                                 \n";
	ptx << "\t.reg .u32 %r<3>;                  \n";
	ptx << "\tld.param.u32 %r0, [a];            \n";
	ptx << "\tadd.u32 %r0, %r0, 3;              \n";
	ptx << "\tst.param.u32 [return], %r0;       \n";
	ptx << "\tret 0;                            \n";
	ptx << "}                                   \n";
	
	return ptx.str();
}

void testIndirectFunctionCall_REF(void* output, void* input)
{
	uint32_t r0 = getParameter<uint32_t>(input, 0);

	setParameter(output, 0, (uint32_t)r0);
	setParameter(output, sizeof(uint32_t), (uint32_t)(r0 + 1));
	setParameter(output, 2 * sizeof(uint32_t), (uint32_t)(r0 + 2));
	setParameter(output, 3 * sizeof(uint32_t), (uint32_t)(r0 + 3));
}

test::TestPTXAssembly::TypeVector testIndirectFunctionCall_IN()
{
	return test::TestPTXAssembly::TypeVector(1, test::TestPTXAssembly::I32);
}

test::TestPTXAssembly::TypeVector testIndirectFunctionCall_OUT()
{
	return test::TestPTXAssembly::TypeVector(4, test::TestPTXAssembly::I32);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST RECURSIVE FUNCTION CALLS
std::string testRecursiveFunctionCall_PTX()
{
	std::stringstream ptx;
	
	ptx << PTX_VERSION_AND_TARGET;
	ptx << ".visible .func (.param .u32 return) count(.param .u32 a)\n";
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in) \n";
	ptx << "{\t                                 \n";
	ptx << "\t.reg .u64 %rIn, %rOut;            \n";
	ptx << "\t.reg .u32 %r<3>;                  \n";
	ptx << "\t.param .u32 operandA;             \n";
	ptx << "\t.param .u32 result;               \n";
	ptx << "\tld.param.u64 %rIn, [in];          \n";
	ptx << "\tld.param.u64 %rOut, [out];        \n";
	ptx << "\tld.global.u32 %r0, [%rIn];        \n";
	ptx << "\trem.u32 %r0, %r0, 10;             \n";
	ptx << "\tst.param.u32 [operandA], %r0;     \n";
	ptx << "\tcall (result), count, (operandA); \n";
	ptx << "\tld.param.u32 %r2, [result];       \n";
	ptx << "\tst.global.u32 [%rOut], %r2;       \n";
	ptx << "\texit;                             \n";
	ptx << "}                                   \n";
	ptx << "                                    \n";

	ptx << ".visible .func (.param .u32 return) count(.param .u32 a) \n";
	ptx << "{\t                                 \n";
	ptx << "\t.reg .u32 %r<4>;                  \n";
	ptx << "\t.reg .pred %p0;                   \n";
	ptx << "\t.param .u32 operandA;             \n";
	ptx << "\t.param .u32 result;               \n";
	ptx << "\tld.param.u32 %r0, [a];            \n";
	ptx << "\tsetp.ne.u32 %p0, %r0, 0;          \n";
	ptx << "\tmov.u32 %r3, %r0;       \n";
	ptx << "\tsub.u32 %r0, %r0, 1;              \n";
	ptx << "\tst.param.u32 [operandA], %r0;     \n";
	ptx << "\t@%p0 call (result), count, (operandA); \n";
	ptx << "\t@%p0 ld.param.u32 %r3, [result];  \n";
	ptx << "\tst.param.u32 [return], %r3;       \n";
	ptx << "\tret 0;                            \n";
	ptx << "}                                   \n";
	
	return ptx.str();
}

void testRecursiveFunctionCall_REF(void* output, void* input)
{
	setParameter(output, 0, (uint32_t)0);
}

test::TestPTXAssembly::TypeVector testRecursiveFunctionCall_IN()
{
	return test::TestPTXAssembly::TypeVector(1, test::TestPTXAssembly::I32);
}

test::TestPTXAssembly::TypeVector testRecursiveFunctionCall_OUT()
{
	return test::TestPTXAssembly::TypeVector(1, test::TestPTXAssembly::I32);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST DIVERGENT FUNCTION CALLS
std::string testDivergentFunctionCall_PTX()
{
	std::stringstream ptx;
	
	ptx << PTX_VERSION_AND_TARGET;
	ptx << ".visible .func (.param .u32 return) " 
		<< "add(.param .u32 a, .param .u32 b)\n";
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in) \n";
	ptx << "{\t\n";
	ptx << "\t.reg .u64 %rIn, %rOut; \n";
	ptx << "\t.reg .u32 %r<3>;               \n";
	ptx << "\t.reg .pred %less;              \n";
	ptx << "\t.reg .u32 %thread;             \n";
	ptx << "\t.reg .u64 %offset;             \n";
	ptx << "\t.param .u32 operandA;          \n";
	ptx << "\t.param .u32 operandB;          \n";
	ptx << "\t.param .u32 result;            \n";
	ptx << "\tmov.u32 %thread, %tid.x;       \n";
	ptx << "\tld.param.u64 %rIn, [in];       \n";
	ptx << "\tld.param.u64 %rOut, [out];     \n";
	ptx << "\tld.global.u32 %r0, [%rIn];     \n";
	ptx << "\tld.global.u32 %r1, [%rIn + 4]; \n";
	ptx << "\tst.param.u32 [operandA], %r0; \n";
	ptx << "\tst.param.u32 [operandB], %r1; \n";
	ptx << "\tmov.u32 %r2, %r1; \n";
	ptx << "\tsetp.lt.u32 %less, %thread, 1; \n";
	ptx << "\t@%less call (result), add, (operandA, operandB); \n";
	ptx << "\t@%less ld.param.u32 %r2, [result]; \n";
	ptx << "\tcvt.u64.u32 %offset, %thread; \n";
	ptx << "\tmul.lo.u64 %offset, %offset, 4; \n";
	ptx << "\tadd.u64 %rOut, %rOut, %offset; \n";
	ptx << "\tst.global.u32 [%rOut], %r2; \n";
	ptx << "\texit; \n";
	ptx << "}\n";
	ptx << "\n";

	ptx << ".visible .func (.param .u32 return) " 
		<< "add(.param .u32 a, .param .u32 b) \n";
	ptx << "{\t\n";
	ptx << "\t.reg .u32 %r<3>; \n";
	ptx << "\tld.param.u32 %r0, [a];\n";
	ptx << "\tld.param.u32 %r1, [b];\n";
	ptx << "\tadd.u32 %r2, %r1, %r0;\n";
	ptx << "\tst.param.u32 [return], %r2;\n";
	ptx << "\tret 0;\n";
	ptx << "}\n";
	
	return ptx.str();
}

void testDivergentFunctionCall_REF(void* output, void* input)
{
	uint32_t r0 = getParameter<uint32_t>(input, 0);
	uint32_t r1 = getParameter<uint32_t>(input, sizeof(uint32_t));
	
	uint32_t result = r0 + r1;
	
	setParameter(output, 0, result);
	setParameter(output, sizeof(uint32_t), r1);
}

test::TestPTXAssembly::TypeVector testDivergentFunctionCall_IN()
{
	return test::TestPTXAssembly::TypeVector(2, test::TestPTXAssembly::I32);
}

test::TestPTXAssembly::TypeVector testDivergentFunctionCall_OUT()
{
	return test::TestPTXAssembly::TypeVector(2, test::TestPTXAssembly::I32);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST FUNCTION CALLS
std::string testFunctionCalls_PTX(bool uni)
{
	std::stringstream ptx;
	
	ptx << PTX_VERSION_AND_TARGET;
	ptx << ".visible .func (.param .u32 return) " 
		<< "add(.param .u32 a, .param .u32 b)\n";
	ptx << "\n";
	
	ptx << ".entry test(.param .u64 out, .param .u64 in) \n";
	ptx << "{\t\n";
	ptx << "\t.reg .u64 %rIn, %rOut; \n";
	ptx << "\t.reg .u32 %r<3>; \n";
	ptx << "\t.param .u32 operandA;\n";
	ptx << "\t.param .u32 operandB;\n";
	ptx << "\t.param .u32 result;\n";
	ptx << "\tld.param.u64 %rIn, [in]; \n";
	ptx << "\tld.param.u64 %rOut, [out]; \n";
	ptx << "\tld.global.u32 %r0, [%rIn]; \n";
	ptx << "\tld.global.u32 %r1, [%rIn + 4]; \n";
	ptx << "\tst.param.u32 [operandA], %r0; \n";
	ptx << "\tst.param.u32 [operandB], %r1; \n";
	ptx << "\tcall";
	if(uni) ptx << ".uni";
	ptx << " (result), add, (operandA, operandB); \n";
	ptx << "\tld.param.u32 %r2, [result]; \n";
	ptx << "\tst.global.u32 [%rOut], %r2; \n";
	ptx << "\texit; \n";
	ptx << "}\n";
	ptx << "\n";

	ptx << ".visible .func (.param .u32 return) " 
		<< "add(.param .u32 a, .param .u32 b) \n";
	ptx << "{\t\n";
	ptx << "\t.reg .u32 %r<3>; \n";
	ptx << "\tld.param.u32 %r0, [a];\n";
	ptx << "\tld.param.u32 %r1, [b];\n";
	ptx << "\tadd.u32 %r2, %r1, %r0;\n";
	ptx << "\tst.param.u32 [return], %r2;\n";
	ptx << "\tret 0;\n";
	ptx << "}\n";
	
	return ptx.str();
}

void testFunctionCalls_REF(void* output, void* input)
{
	uint32_t r0 = getParameter<uint32_t>(input, 0);
	uint32_t r1 = getParameter<uint32_t>(input, sizeof(uint32_t));
	
	uint32_t result = r0 + r1;
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testFunctionCalls_IN()
{
	return test::TestPTXAssembly::TypeVector(2, test::TestPTXAssembly::I32);
}

test::TestPTXAssembly::TypeVector testFunctionCalls_OUT()
{
	return test::TestPTXAssembly::TypeVector(1, test::TestPTXAssembly::I32);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST VECTOR ELEMENTS
std::string testVectorElements_PTX()
{
	std::stringstream result;

	result << PTX_VERSION_AND_TARGET;
	result << ".entry test(.param .u64 out, .param .u64 in) \n";
	result << "{\t\n";
	result << "\t.reg .u64 %rIn, %rOut; \n";
	result << "\t.reg .v4 .u32 %rv<2>; \n";
	result << "\tld.param.u64 %rIn, [in]; \n";
	result << "\tld.param.u64 %rOut, [out]; \n";
	result << "\tld.global.v4.u32 %rv0, [%rIn]; \n";
	result << "\tmov.u32 %rv1.x, %rv0.y; \n";
	result << "\tmov.u32 %rv1.y, %rv0.z; \n";
	result << "\tmov.u32 %rv1.z, %rv0.w; \n";
	result << "\tmov.u32 %rv1.w, %rv0.x; \n";
	result << "\tst.global.v4.u32 [%rOut], %rv1; \n";
	result << "\texit; \n";
	result << "}\n";
	
	return result.str();
}

void testVectorElements_REF(void* output, void* input)
{
	uint32_t r0 = getParameter<uint32_t>(input, 0);
	uint32_t r1 = getParameter<uint32_t>(input, sizeof(uint32_t));
	uint32_t r2 = getParameter<uint32_t>(input, 2*sizeof(uint32_t));
	uint32_t r3 = getParameter<uint32_t>(input, 3*sizeof(uint32_t));
		
	setParameter(output, 0, r1);
	setParameter(output, sizeof(uint32_t), r2);
	setParameter(output, 2*sizeof(uint32_t), r3);
	setParameter(output, 3*sizeof(uint32_t), r0);
}

test::TestPTXAssembly::TypeVector testVectorElements_IN()
{
	return test::TestPTXAssembly::TypeVector(4, test::TestPTXAssembly::I32);
}

test::TestPTXAssembly::TypeVector testVectorElements_OUT()
{
	return test::TestPTXAssembly::TypeVector(4, test::TestPTXAssembly::I32);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST ADD/SUB
std::string testAdd_PTX(ir::PTXOperand::DataType type, bool sat, bool sub)
{
	std::stringstream result;
	std::string typeString = "." + ir::PTXOperand::toString(type);

	result << PTX_VERSION_AND_TARGET;
	result << ".entry test(.param .u64 out, .param .u64 in) \n";
	result << "{\t\n";
	result << "\t.reg .u64 %rIn, %rOut; \n";
	result << "\t.reg " << typeString << " %r<3>; \n";
	result << "\tld.param.u64 %rIn, [in]; \n";
	result << "\tld.param.u64 %rOut, [out]; \n";
	result << "\tld.global" << typeString << " %r0, [%rIn]; \n";
	result << "\tld.global" << typeString << " %r1, [%rIn + " 
		<< ir::PTXOperand::bytes(type) << "]; \n";
	if(sub)
	{
		result << "\tsub";
	}
	else
	{
		result << "\tadd";
	}
	if(sat) result << ".sat";
	result << typeString << " %r2, %r0, %r1; \n";
	result << "\tst.global" << typeString << " [%rOut], %r2; \n";
	result << "\texit; \n";
	result << "}\n";
	
	return result.str();
}

template <typename type, bool sat, bool sub>
void testAdd_REF(void* output, void* input)
{
	type r0 = getParameter<type>(input, 0);
	type r1 = getParameter<type>(input, sizeof(type));
	
	type result = 0;
	
	if(sat)
	{
		int64_t t0 = r0;
		int64_t t1 = r1;
		
		int64_t tresult = 0;
		if(sub)
		{
			tresult = t0 - t1;
		}
		else
		{
			tresult = t0 + t1;
		}
		tresult = std::max(tresult, (int64_t)INT_MIN);
		tresult = std::min(tresult, (int64_t)INT_MAX);
		
		result = (type)tresult;
	}
	else
	{
		if(sub)
		{
			result = r0 - r1;
		}
		else
		{
			result = r0 + r1;
		}
	}
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testAdd_IN(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(2, type);
}

test::TestPTXAssembly::TypeVector testAdd_OUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST Carry
std::string testCarry_PTX(ir::PTXOperand::DataType type, bool sub)
{
	std::stringstream stream;
	std::string typeString = "." + ir::PTXOperand::toString(type);
	
	stream << PTX_VERSION_AND_TARGET;
	stream << ".entry test(.param .u64 out, .param .u64 in)\n";
	stream << "{\n";
	stream << "\t.reg .u64 %rIn, %rOut;\n";
	stream << "\t.reg " << typeString << " %r<5>;\n";
	stream << "\tld.param.u64 %rIn, [in];\n";
	stream << "\tld.param.u64 %rOut, [out];\n";
	stream << "\tld.global" << typeString << " %r0, [%rIn]; \n";
	stream << "\tld.global" << typeString << " %r1, [%rIn + " 
		<< ir::PTXOperand::bytes(type) << "]; \n";
	
	if(sub)
	{
		stream << "\tsub.cc"  << typeString << " %r2, %r1, %r0;\n";
		stream << "\tsubc.cc" << typeString << " %r3, %r2, %r0;\n";
		stream << "\tsubc"    << typeString << " %r4, %r3, %r0;\n";
	}
	else
	{
		stream << "\tadd.cc"  << typeString << " %r2, %r1, %r0;\n";
		stream << "\taddc.cc" << typeString << " %r3, %r2, %r0;\n";
		stream << "\taddc"    << typeString << " %r4, %r3, %r0;\n";
	}
	
	stream << "\tst.global" << typeString << " [%rOut], %r4;\n";
	stream << "\texit;\n";
	stream << "}\n";
	
	return stream.str();
}

template <typename type, bool sub>
void testCarry_REF(void* output, void* input)
{
	type r0 = getParameter<type>(input, 0);
	type r1 = getParameter<type>(input, sizeof(type));
	
	type result = 0;
	
	type t0 = r0;
	type t1 = r1;
	
	type carry   = 0;
	type tresult = 0;
	
	if(sub) t0 = -t0;
	hydrazine::add(tresult, carry, t1, t0, carry);
	if(sub) carry += -1;
	hydrazine::add(tresult, carry, tresult, t0, carry);
	if(sub) carry += -1;
	hydrazine::add(tresult, carry, tresult, t0, carry);

	result = tresult;
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testCarry_IN(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(2, type);
}

test::TestPTXAssembly::TypeVector testCarry_OUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST MUL
enum MulType
{
	MulHi,
	MulLo,
	MulWide
};

std::string testMul_PTX(ir::PTXOperand::DataType type, MulType op)
{
	std::stringstream stream;
	std::string typeString = "." + ir::PTXOperand::toString(type);
	std::string dTypeString = "." + ir::PTXOperand::toString(type);

	if(op == MulWide)
	{
		switch(type)
		{
			case ir::PTXOperand::u16: dTypeString = ".u32"; break;
			case ir::PTXOperand::u32: dTypeString = ".u64"; break;
			case ir::PTXOperand::s16: dTypeString = ".s32"; break;
			case ir::PTXOperand::s32: dTypeString = ".s64"; break;
			default: assertM(false, "Invalid data type.");
		}
	}
	stream << PTX_VERSION_AND_TARGET;
	stream << ".entry test(.param .u64 out, .param .u64 in)\n";
	stream << "{\n";
	stream << "\t.reg .u64 %rIn, %rOut;\n";
	stream << "\t.reg " << typeString << " %r<2>;\n";
	stream << "\t.reg " << dTypeString << " %r2;\n";
	stream << "\tld.param.u64 %rIn, [in];\n";
	stream << "\tld.param.u64 %rOut, [out];\n";
	stream << "\tld.global" << typeString << " %r0, [%rIn]; \n";
	stream << "\tld.global" << typeString << " %r1, [%rIn + " 
		<< ir::PTXOperand::bytes(type) << "]; \n";
	
	if( op == MulHi )
	{
		stream << "\tmul.hi" << typeString << " %r2, %r1, %r0;\n";
	}
	else if( op == MulLo )
	{
		stream << "\tmul.lo" << typeString << " %r2, %r1, %r0;\n";
	}
	else
	{
		stream << "\tmul.wide" << typeString << " %r2, %r1, %r0;\n";
	}

	
	stream << "\tst.global" << dTypeString << " [%rOut], %r2;\n";
	stream << "\texit;\n";
	stream << "}\n";
	
	return stream.str();
}

template <typename type, MulType op>
void testMul_REF(void* output, void* input)
{
	type r0 = getParameter<type>(input, 0);
	type r1 = getParameter<type>(input, sizeof(type));
	type hi;
	type lo;
	
	hydrazine::multiplyHiLo(hi, lo, r0, r1);
	
	if(op == MulWide)
	{
		setParameter(output, 0, lo);
		setParameter(output, sizeof(type), hi);
	}
	else if(op == MulLo)
	{
		setParameter(output, 0, lo);
	}
	else
	{
		setParameter(output, 0, hi);
	}
}

test::TestPTXAssembly::TypeVector testMul_IN(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(2, type);
}

test::TestPTXAssembly::TypeVector testMul_OUT(
	test::TestPTXAssembly::DataType type, MulType op)
{
	if(op != MulWide)
	{
		return test::TestPTXAssembly::TypeVector(1, type);
	}
	else
	{
		switch(type)
		{
			case test::TestPTXAssembly::I16:
			{
				return test::TestPTXAssembly::TypeVector(1, 
					test::TestPTXAssembly::I32);
			}
			case test::TestPTXAssembly::I32:
			{
				return test::TestPTXAssembly::TypeVector(1, 
					test::TestPTXAssembly::I64);
			}
			default: assertM(false, "Invalid data type for wide multiply.");
		}		
	}
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST MAD
std::string testMad_PTX(ir::PTXOperand::DataType type, MulType op, bool sat)
{
	std::stringstream stream;
	std::string typeString = "." + ir::PTXOperand::toString(type);
	std::string dTypeString = "." + ir::PTXOperand::toString(type);

	if(op == MulWide)
	{
		switch(type)
		{
			case ir::PTXOperand::u16: dTypeString = ".u32"; break;
			case ir::PTXOperand::u32: dTypeString = ".u64"; break;
			case ir::PTXOperand::s16: dTypeString = ".s32"; break;
			case ir::PTXOperand::s32: dTypeString = ".s64"; break;
			default: assertM(false, "Invalid data type.");
		}
	}
	
	stream << PTX_VERSION_AND_TARGET;
	stream << ".entry test(.param .u64 out, .param .u64 in)\n";
	stream << "{\n";
	stream << "\t.reg .u64 %rIn, %rOut;\n";
	stream << "\t.reg " << typeString << " %r<2>;\n";
	stream << "\t.reg " << dTypeString << " %r2;\n";
	stream << "\t.reg " << dTypeString << " %r3;\n";
	stream << "\tld.param.u64 %rIn, [in];\n";
	stream << "\tld.param.u64 %rOut, [out];\n";
	stream << "\tld.global" << typeString << " %r0, [%rIn]; \n";
	stream << "\tld.global" << typeString << " %r1, [%rIn + " 
		<< ir::PTXOperand::bytes(type) << "]; \n";
	stream << "\tld.global" << dTypeString << " %r2, [%rIn + " 
		<< 2 * ir::PTXOperand::bytes(type) << "]; \n";
	
	if( op == MulHi )
	{
		if( sat )
		{
			stream << "\tmad.hi.sat" << typeString << " %r3, %r0, %r1, %r2;\n";
		}
		else
		{
			stream << "\tmad.hi" << typeString << " %r3, %r0, %r1, %r2;\n";
		}
	}
	else if( op == MulLo )
	{
		stream << "\tmad.lo" << typeString << " %r3, %r0, %r1, %r2;\n";
	}
	else
	{
		stream << "\tmad.wide" << typeString << " %r3, %r0, %r1, %r2;\n";
	}

	
	stream << "\tst.global" << dTypeString << " [%rOut], %r3;\n";
	stream << "\texit;\n";
	stream << "}\n";
	
	return stream.str();
}

template <typename type, MulType op, bool sat>
void testMad_REF(void* output, void* input)
{
	type r0 = getParameter<type>(input, 0);
	type r1 = getParameter<type>(input, sizeof(type));
	type hi;
	type lo;
	
	hydrazine::multiplyHiLo(hi, lo, r0, r1);
	
	if(op == MulWide)
	{
		type r2 = getParameter<type>(input, 2 * sizeof(type));
		type r3 = getParameter<type>(input, 3 * sizeof(type));
		hydrazine::addHiLo(hi, lo, r2);
		hi += r3;
		setParameter(output, 0, lo);
		setParameter(output, sizeof(type), hi);
	}
	else if(op == MulLo)
	{
		type r2 = getParameter<type>(input, 2 * sizeof(type));
		lo += r2;
		setParameter(output, 0, lo);
	}
	else
	{
		type r2 = getParameter<type>(input, 2 * sizeof(type));
		if(sat)
		{
			int64_t t = (int64_t)hi + (int64_t)r2;
			t = std::max(t, (int64_t)INT_MIN);
			t = std::min(t, (int64_t)INT_MAX);
			hi = t;
		}
		else
		{
			hi += r2;
		}
		setParameter(output, 0, hi);
	}
}

test::TestPTXAssembly::TypeVector testMad_IN(
	test::TestPTXAssembly::DataType type, MulType op)
{
	test::TestPTXAssembly::TypeVector vector(2, type);

	if(op != MulWide)
	{
		vector.push_back( type );
	}
	else
	{
		switch(type)
		{
			case test::TestPTXAssembly::I16:
			{
				vector.push_back( test::TestPTXAssembly::I32 );
				break;
			}
			case test::TestPTXAssembly::I32:
			{
				vector.push_back( test::TestPTXAssembly::I64 );
				break;
			}
			default: assertM(false, "Invalid data type for wide multiply.");
		}	
	}
	
	return vector;
}

test::TestPTXAssembly::TypeVector testMad_OUT(
	test::TestPTXAssembly::DataType type, MulType op)
{
	if(op != MulWide)
	{
		return test::TestPTXAssembly::TypeVector(1, type);
	}
	else
	{
		switch(type)
		{
			case test::TestPTXAssembly::I16:
			{
				return test::TestPTXAssembly::TypeVector(1, 
					test::TestPTXAssembly::I32);
			}
			case test::TestPTXAssembly::I32:
			{
				return test::TestPTXAssembly::TypeVector(1, 
					test::TestPTXAssembly::I64);
			}
			default: assertM(false, "Invalid data type for wide multiply.");
		}		
	}
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST SAD
std::string testSad_PTX(ir::PTXOperand::DataType type)
{
	std::stringstream result;
	std::string typeString = "." + ir::PTXOperand::toString(type);

	result << PTX_VERSION_AND_TARGET;
	result << ".entry test(.param .u64 out, .param .u64 in) \n";
	result << "{\t\n";
	result << "\t.reg .u64 %rIn, %rOut; \n";
	result << "\t.reg " << typeString << " %r<4>; \n";
	result << "\tld.param.u64 %rIn, [in]; \n";
	result << "\tld.param.u64 %rOut, [out]; \n";
	result << "\tld.global" << typeString << " %r0, [%rIn]; \n";
	result << "\tld.global" << typeString << " %r1, [%rIn + " 
		<< ir::PTXOperand::bytes(type) << "]; \n";
	result << "\tld.global" << typeString << " %r2, [%rIn + " 
		<< 2 * ir::PTXOperand::bytes(type) << "]; \n";
	result << "\tsad" << typeString << " %r3, %r0, %r1, %r2; \n";
	result << "\tst.global" << typeString << " [%rOut], %r3; \n";
	result << "\texit; \n";
	result << "}\n";
	
	return result.str();
}

template <typename type>
void testSad_REF(void* output, void* input)
{
	type r0 = getParameter<type>(input, 0);
	type r1 = getParameter<type>(input, sizeof(type));
	type r2 = getParameter<type>(input, 2 * sizeof(type));
	
	type result = r2 + ( ( r0 < r1 ) ? ( r1 - r0 ) : ( r0 - r1 ) );
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testSad_IN(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(3, type);
}

test::TestPTXAssembly::TypeVector testSad_OUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST DIV
std::string testDiv_PTX(ir::PTXOperand::DataType type)
{
	std::stringstream result;
	std::string typeString = "." + ir::PTXOperand::toString(type);

	result << PTX_VERSION_AND_TARGET;
	result << ".entry test(.param .u64 out, .param .u64 in) \n";
	result << "{\t\n";
	result << "\t.reg .u64 %rIn, %rOut; \n";
	result << "\t.reg " << typeString << " %r<4>; \n";
	result << "\tld.param.u64 %rIn, [in]; \n";
	result << "\tld.param.u64 %rOut, [out]; \n";
	result << "\tld.global" << typeString << " %r0, [%rIn]; \n";
	result << "\tld.global" << typeString << " %r1, [%rIn + " 
		<< ir::PTXOperand::bytes(type) << "]; \n";
	result << "\tdiv" << typeString << " %r2, %r0, %r1; \n";
	result << "\tst.global" << typeString << " [%rOut], %r2; \n";
	result << "\texit; \n";
	result << "}\n";
	
	return result.str();
}

template <typename type>
void testDiv_REF(void* output, void* input)
{
	type r0 = getParameter<type>(input, 0);
	type r1 = getParameter<type>(input, sizeof(type));
	
	type result = r0 / r1;
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testDiv_IN(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(2, type);
}

test::TestPTXAssembly::TypeVector testDiv_OUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST REM
std::string testRem_PTX(ir::PTXOperand::DataType type)
{
	std::stringstream result;
	std::string typeString = "." + ir::PTXOperand::toString(type);

	result << PTX_VERSION_AND_TARGET;
	result << ".entry test(.param .u64 out, .param .u64 in) \n";
	result << "{\t\n";
	result << "\t.reg .u64 %rIn, %rOut; \n";
	result << "\t.reg " << typeString << " %r<4>; \n";
	result << "\tld.param.u64 %rIn, [in]; \n";
	result << "\tld.param.u64 %rOut, [out]; \n";
	result << "\tld.global" << typeString << " %r0, [%rIn]; \n";
	result << "\tld.global" << typeString << " %r1, [%rIn + " 
		<< ir::PTXOperand::bytes(type) << "]; \n";
	result << "\trem" << typeString << " %r2, %r0, %r1; \n";
	result << "\tst.global" << typeString << " [%rOut], %r2; \n";
	result << "\texit; \n";
	result << "}\n";
	
	return result.str();
}

template <typename type>
void testRem_REF(void* output, void* input)
{
	type r0 = getParameter<type>(input, 0);
	type r1 = getParameter<type>(input, sizeof(type));
	
	type result = r0 % r1;
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testRem_IN(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(2, type);
}

test::TestPTXAssembly::TypeVector testRem_OUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST ABS
std::string testAbs_PTX(ir::PTXOperand::DataType type)
{
	std::stringstream result;
	std::string typeString = "." + ir::PTXOperand::toString(type);

	result << PTX_VERSION_AND_TARGET;
	result << ".entry test(.param .u64 out, .param .u64 in) \n";
	result << "{\t\n";
	result << "\t.reg .u64 %rIn, %rOut; \n";
	result << "\t.reg " << typeString << " %r<2>; \n";
	result << "\tld.param.u64 %rIn, [in]; \n";
	result << "\tld.param.u64 %rOut, [out]; \n";
	result << "\tld.global" << typeString << " %r0, [%rIn]; \n";
	result << "\tabs" << typeString << " %r1, %r0; \n";
	result << "\tst.global" << typeString << " [%rOut], %r1; \n";
	result << "\texit; \n";
	result << "}\n";
	
	return result.str();
}

template <typename type>
void testAbs_REF(void* output, void* input)
{
	type r0 = getParameter<type>(input, 0);
	
	type result = std::abs( r0 );
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testAbs_IN(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}

test::TestPTXAssembly::TypeVector testAbs_OUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST NEG
std::string testNeg_PTX(ir::PTXOperand::DataType type)
{
	std::stringstream result;
	std::string typeString = "." + ir::PTXOperand::toString(type);

	result << PTX_VERSION_AND_TARGET;
	result << ".entry test(.param .u64 out, .param .u64 in) \n";
	result << "{\t\n";
	result << "\t.reg .u64 %rIn, %rOut; \n";
	result << "\t.reg " << typeString << " %r<2>; \n";
	result << "\tld.param.u64 %rIn, [in]; \n";
	result << "\tld.param.u64 %rOut, [out]; \n";
	result << "\tld.global" << typeString << " %r0, [%rIn]; \n";
	result << "\tneg" << typeString << " %r1, %r0; \n";
	result << "\tst.global" << typeString << " [%rOut], %r1; \n";
	result << "\texit; \n";
	result << "}\n";
	
	return result.str();
}

template <typename type>
void testNeg_REF(void* output, void* input)
{
	type r0 = getParameter<type>(input, 0);
	
	type result = -r0;
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testNeg_IN(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}

test::TestPTXAssembly::TypeVector testNeg_OUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST MIN-MAX
std::string testMinMax_PTX(ir::PTXOperand::DataType type, bool max)
{
	std::stringstream result;
	std::string typeString = "." + ir::PTXOperand::toString(type);

	result << PTX_VERSION_AND_TARGET;
	result << ".entry test(.param .u64 out, .param .u64 in) \n";
	result << "{\t\n";
	result << "\t.reg .u64 %rIn, %rOut; \n";
	result << "\t.reg " << typeString << " %r<3>; \n";
	result << "\tld.param.u64 %rIn, [in]; \n";
	result << "\tld.param.u64 %rOut, [out]; \n";
	result << "\tld.global" << typeString << " %r0, [%rIn]; \n";
	result << "\tld.global" << typeString << " %r1, [%rIn + " 
		<< ir::PTXOperand::bytes(type) << "]; \n";
	if(max)
	{
		result << "\tmax" << typeString << " %r2, %r0, %r1; \n";
	}
	else
	{
		result << "\tmin" << typeString << " %r2, %r0, %r1; \n";
	}
	result << "\tst.global" << typeString << " [%rOut], %r2; \n";
	result << "\texit; \n";
	result << "}\n";
	
	return result.str();
}

template <typename type, bool max>
void testMinMax_REF(void* output, void* input)
{
	type r0 = getParameter<type>(input, 0);
	type r1 = getParameter<type>(input, sizeof(type));
	
	type result = 0;
	
	if(max)
	{
		result = std::max(r0, r1);
	}
	else
	{
		result = std::min(r0, r1);
	}

	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testMinMax_IN(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(2, type);
}

test::TestPTXAssembly::TypeVector testMinMax_OUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST POPC
std::string testPopc_PTX(ir::PTXOperand::DataType type)
{
	std::stringstream result;
	std::string typeString = "." + ir::PTXOperand::toString(type);

	result << PTX_VERSION_AND_TARGET;
	result << ".entry test(.param .u64 out, .param .u64 in) \n";
	result << "{\t\n";
	result << "\t.reg .u64 %rIn, %rOut; \n";
	result << "\t.reg " << typeString << " %r0; \n";
	result << "\t.reg .u32 %r1; \n";
	result << "\tld.param.u64 %rIn, [in]; \n";
	result << "\tld.param.u64 %rOut, [out]; \n";
	result << "\tld.global" << typeString << " %r0, [%rIn]; \n";
	result << "\tpopc" << typeString << " %r1, %r0; \n";
	result << "\tst.global.u32 [%rOut], %r1; \n";
	result << "\texit; \n";
	result << "}\n";
	
	return result.str();
}

template <typename type>
void testPopc_REF(void* output, void* input)
{
	type r0 = getParameter<type>(input, 0);
	
	uint32_t result = hydrazine::popc( r0 );
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testPopc_IN(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}

test::TestPTXAssembly::TypeVector testPopc_OUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST CLZ
std::string testClz_PTX(ir::PTXOperand::DataType type)
{
	std::stringstream result;
	std::string typeString = "." + ir::PTXOperand::toString(type);

	result << PTX_VERSION_AND_TARGET;
	result << ".entry test(.param .u64 out, .param .u64 in) \n";
	result << "{\t\n";
	result << "\t.reg .u64 %rIn, %rOut; \n";
	result << "\t.reg " << typeString << " %r0; \n";
	result << "\t.reg .u32 %r1; \n";
	result << "\tld.param.u64 %rIn, [in]; \n";
	result << "\tld.param.u64 %rOut, [out]; \n";
	result << "\tld.global" << typeString << " %r0, [%rIn]; \n";
	result << "\tclz" << typeString << " %r1, %r0; \n";
	result << "\tst.global.u32 [%rOut], %r1; \n";
	result << "\texit; \n";
	result << "}\n";
	
	return result.str();
}

template <typename type>
void testClz_REF(void* output, void* input)
{
	type r0 = getParameter<type>(input, 0);
	
	uint32_t result = hydrazine::countLeadingZeros( r0 );
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testClz_IN(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}

test::TestPTXAssembly::TypeVector testClz_OUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST BFIND
std::string testBfind_PTX(ir::PTXOperand::DataType type, bool shift)
{
	std::stringstream result;
	std::string typeString = "." + ir::PTXOperand::toString(type);

	result << PTX_VERSION_AND_TARGET;
	result << ".entry test(.param .u64 out, .param .u64 in) \n";
	result << "{\t\n";
	result << "\t.reg .u64 %rIn, %rOut; \n";
	result << "\t.reg " << typeString << " %r0; \n";
	result << "\t.reg .u32 %r1; \n";
	result << "\tld.param.u64 %rIn, [in]; \n";
	result << "\tld.param.u64 %rOut, [out]; \n";
	result << "\tld.global" << typeString << " %r0, [%rIn]; \n";
	if(shift)
	{
		result << "\tbfind.shiftamt" << typeString << " %r1, %r0; \n";
	}
	else
	{
		result << "\tbfind" << typeString << " %r1, %r0; \n";
	}
	result << "\tst.global.u32 [%rOut], %r1; \n";
	result << "\texit; \n";
	result << "}\n";
	
	return result.str();
}

template <typename type, bool shift>
void testBfind_REF(void* output, void* input)
{
	type r0 = getParameter<type>(input, 0);
	
	uint32_t result = hydrazine::bfind(r0, shift);
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testBfind_IN(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}

test::TestPTXAssembly::TypeVector testBfind_OUT()
{
	return test::TestPTXAssembly::TypeVector(1, test::TestPTXAssembly::I32);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST BREV
std::string testBrev_PTX(ir::PTXOperand::DataType type)
{
	std::stringstream result;
	std::string typeString = "." + ir::PTXOperand::toString(type);

	result << PTX_VERSION_AND_TARGET;
	result << ".entry test(.param .u64 out, .param .u64 in) \n";
	result << "{\t\n";
	result << "\t.reg .u64 %rIn, %rOut; \n";
	result << "\t.reg " << typeString << " %r<2>; \n";
	result << "\tld.param.u64 %rIn, [in]; \n";
	result << "\tld.param.u64 %rOut, [out]; \n";
	result << "\tld.global" << typeString << " %r0, [%rIn]; \n";
	result << "\tbrev" << typeString << " %r1, %r0; \n";
	result << "\tst.global" << typeString << " [%rOut], %r1; \n";
	result << "\texit; \n";
	result << "}\n";
	
	return result.str();
}

template <typename type>
void testBrev_REF(void* output, void* input)
{
	type r0 = getParameter<type>(input, 0);
	
	type result = hydrazine::brev(r0);
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testBrev_IN(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}

test::TestPTXAssembly::TypeVector testBrev_OUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST BFI
std::string testBfi_PTX(ir::PTXOperand::DataType type)
{
	std::stringstream result;
	std::string typeString = "." + ir::PTXOperand::toString(type);

	result << PTX_VERSION_AND_TARGET;
	result << ".entry test(.param .u64 out, .param .u64 in) \n";
	result << "{\t\n";
	result << "\t.reg .u64 %rIn, %rOut; \n";
	result << "\t.reg " << typeString << " %r<3>; \n";
	result << "\t.reg .u32 %r3, %r4; \n";
	result << "\tld.param.u64 %rIn, [in]; \n";
	result << "\tld.param.u64 %rOut, [out]; \n";
	result << "\tld.global" << typeString << " %r0, [%rIn]; \n";
	result << "\tld.global" << typeString << " %r1, [%rIn + " 
		<< ir::PTXOperand::bytes(type) << "]; \n";
	result << "\tld.global.u32 %r3, [%rIn + " 
		<< (2 * ir::PTXOperand::bytes(type)) << "]; \n";
	result << "\tld.global.u32 %r4, [%rIn + " 
		<< (2 * ir::PTXOperand::bytes(type) 
		+ ir::PTXOperand::bytes(ir::PTXOperand::u32)) << "]; \n";
	result << "\tmin.u32 %r3, %r3, 63; \n";
	result << "\tmin.u32 %r4, %r4, 63; \n";
	result << "\tbfi" << typeString << " %r2, %r0, %r1, %r3, %r4; \n";
	result << "\tst.global" << typeString << " [%rOut], %r2; \n";
	result << "\texit; \n";
	result << "}\n";
	
	return result.str();
}

template <typename type>
void testBfi_REF(void* output, void* input)
{
	type r0 = getParameter<type>(input, 0);
	type r1 = getParameter<type>(input, sizeof(type));
	uint32_t r2 = getParameter<uint32_t>(input, 2 * sizeof(type));
	uint32_t r3 = getParameter<uint32_t>(input, 2 * sizeof(type) 
		+ sizeof(uint32_t));
	
	r2 = std::min((uint32_t)63, r2);
	r3 = std::min((uint32_t)63, r3);
	
	type result = hydrazine::bitFieldInsert(r0, r1, r2, r3);
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testBfi_IN(
	test::TestPTXAssembly::DataType type)
{
	test::TestPTXAssembly::TypeVector input(4, type);
	
	input[2] = test::TestPTXAssembly::I32;
	input[3] = test::TestPTXAssembly::I32;
	
	return input;
}

test::TestPTXAssembly::TypeVector testBfi_OUT(
	test::TestPTXAssembly::DataType type)
{
	return test::TestPTXAssembly::TypeVector(1, type);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// TEST PRMT
std::string testPrmt_PTX(ir::PTXInstruction::PermuteMode mode)
{
	std::string modeString;
	
	if( mode != ir::PTXInstruction::DefaultPermute )
	{
		modeString = "." + ir::PTXInstruction::toString( mode );
	}
	
	std::stringstream result;

	result << PTX_VERSION_AND_TARGET;
	result << ".entry test(.param .u64 out, .param .u64 in) \n";
	result << "{\t\n";
	result << "\t.reg .u64 %rIn, %rOut; \n";
	result << "\t.reg .u32 %r<4>; \n";
	result << "\tld.param.u64 %rIn, [in]; \n";
	result << "\tld.param.u64 %rOut, [out]; \n";
	result << "\tld.global.u32 %r0, [%rIn]; \n";
	result << "\tld.global.u32 %r1, [%rIn + 4]; \n";
	result << "\tld.global.u32 %r2, [%rIn + 8]; \n";
	result << "\tprmt.b32" << modeString << " %r3, %r0, %r1, %r2; \n";
	result << "\tst.global.u32 [%rOut], %r3; \n";
	result << "\texit; \n";
	result << "}\n";
	
	return result.str();
}

template <ir::PTXInstruction::PermuteMode mode>
void testPrmt_REF(void* output, void* input)
{
	uint32_t r0 = getParameter<uint32_t>(input, 0);
	uint32_t r1 = getParameter<uint32_t>(input, sizeof(uint32_t));
	uint32_t r2 = getParameter<uint32_t>(input, 
		2 * sizeof(uint32_t));
	
	uint32_t result = 0;
	
	switch( mode )
	{
		case ir::PTXInstruction::ForwardFourExtract:
		{
			result = hydrazine::permute<hydrazine::ForwardFourExtract>(
				r0,r1,r2);
			break;
		}
		case ir::PTXInstruction::BackwardFourExtract:
		{
			result = hydrazine::permute<hydrazine::BackwardFourExtract>(
				r0,r1,r2);
			break;
		}
		case ir::PTXInstruction::ReplicateEight:
		{
			result = hydrazine::permute<hydrazine::ReplicateEight>(r0,r1,r2);
			break;
		}
		case ir::PTXInstruction::EdgeClampLeft:
		{
			result = hydrazine::permute<hydrazine::EdgeClampLeft>(r0,r1,r2);
			break;
		}
		case ir::PTXInstruction::EdgeClampRight:
		{
			result = hydrazine::permute<hydrazine::EdgeClampRight>(r0,r1,r2);
			break;
		}
		case ir::PTXInstruction::ReplicateSixteen:
		{
			result = hydrazine::permute<hydrazine::ReplicateSixteen>(r0,r1,r2);
			break;
		}
		case ir::PTXInstruction::DefaultPermute:
		{
			result = hydrazine::permute<hydrazine::DefaultPermute>(
				r0,r1,r2);
			break;
		}
	}
	
	setParameter(output, 0, result);
}

test::TestPTXAssembly::TypeVector testPrmt_IN()
{
	return test::TestPTXAssembly::TypeVector(3, test::TestPTXAssembly::I32);
}

test::TestPTXAssembly::TypeVector testPrmt_OUT()
{
	return test::TestPTXAssembly::TypeVector(1, test::TestPTXAssembly::I32);
}
////////////////////////////////////////////////////////////////////////////////


namespace test
{
	static uint32_t align(uint32_t address, uint32_t bytes)
	{
		uint32_t remainder = address % bytes;
		if(remainder == 0) return address;
		return address + bytes - remainder;
	}

	uint32_t TestPTXAssembly::bytes(DataType t)
	{
		switch(t)
		{
			case I8: return 1;
			case I16: return 2;
			case I32: return 4;
			case I64: return 8;
			case FP32: return 4;
			case FP64: return 8;
		}
		return 1;
	}
	
	void TestPTXAssembly::_writeArguments(std::ostream &out, 
		const TypeVector &types, char *parameters) {
		
		uint32_t index = 0;
		for(TypeVector::const_iterator type = types.begin(); type != types.end(); ++type)
		{
			switch(*type)
			{
				case I8:
				{
					out << (index ? ", " : "") << getParameter<char>(parameters, index);
					break;
				}
				case I16:
				{
					out << (index ? ", " : "") << getParameter<short>(parameters, index);
					break;
				}				
				case I32:
				{
					out << (index ? ", " : "") << getParameter<int>(parameters, index);
					break;
				}
				case I64:
				{
					out << (index ? ", " : "") << getParameter<int64_t>(parameters, index);
					break;
				}
				case FP32:
				{
					out << (index ? ", " : "") << getParameter<float>(parameters, index);
					break;
				}
				case FP64:
				{
					out << (index ? ", " : "") << getParameter<double>(parameters, index);
					break;
				}
				default: break;
			}
			index += bytes(*type);
		}
	}
	
	bool TestPTXAssembly::_doOneTest(const TestHandle& test, uint32_t seed)
	{
		random.seed(seed);
		
		report("::_doOneTest(" << test.name << ", seed = " << seed << ")");
		
		uint32_t inputSize = 0;
		uint32_t outputSize = 0;
		
		for(TypeVector::const_iterator type = test.inputTypes.begin(); 
			type != test.inputTypes.end(); ++type)
		{
			inputSize = align(inputSize, bytes(*type));
			inputSize += bytes(*type);
		}
		
		for(TypeVector::const_iterator type = test.outputTypes.begin(); 
			type != test.outputTypes.end(); ++type)
		{
			outputSize = align(outputSize, bytes(*type));
			outputSize += bytes(*type);
		}
		
		char* inputBlock = (*test.generator)(random);
		char* outputBlock = new char[outputSize];
		char* referenceBlock = new char[outputSize];
		
		bool pass = true;
		int devices = 0;
		int device = 0;
		
		cudaGetDeviceCount(&devices);
		
		try
		{
			std::stringstream stream(test.ptx);
			ocelot::registerPTXModule(stream, test.name);
			
			uint64_t deviceInput  = 0;
			uint64_t deviceOutput = 0;
						
			if(devices > 0) device = random() % devices;
			cudaSetDevice(device);
			
			if(veryVerbose)
			{
				cudaDeviceProp properties;
				cudaGetDeviceProperties(&properties, device);
			
				std::cout << "\n-----------------------\n";
				std::cout << " Running Test '" << test.name 
					<< "' on device - " << device << " - '" 
					<< properties.name << "'\n";
			}
			
			#if VERBOSE_TEST_CONFIGURATION
			status << "\n---------------------\n";
			status << "\nBefore test " << test.name << "\n";
			status << "Input parameters: ";
			_writeArguments(status, test.inputTypes, inputBlock);
			status << "\nReference: ";
			_writeArguments(status, test.outputTypes, referenceBlock);
			status << "\nOutput parameters: ";
			_writeArguments(status, test.outputTypes, outputBlock);
			status << "\n";
			#endif
			
			cudaMalloc((void**)&deviceInput, inputSize);
			cudaMalloc((void**)&deviceOutput, outputSize);
			cudaMemset((void*)deviceOutput, 0, outputSize);
			
			cudaMemcpy((void*)deviceInput, inputBlock, 
				inputSize, cudaMemcpyHostToDevice);
				
			cudaSetupArgument(&deviceOutput, 8, 0);
			cudaSetupArgument(&deviceInput, 8, 8);
			cudaConfigureCall( dim3( test.ctas, 1, 1 ), 
				dim3( test.threads, 1, 1 ), 0, 0 );
			ocelot::launch(test.name, "test");
			
			cudaThreadSynchronize();
			
			cudaMemcpy(outputBlock, (void*)deviceOutput, 
				outputSize, cudaMemcpyDeviceToHost);
			
			cudaFree((void*)deviceInput);
			cudaFree((void*)deviceOutput);
			
			#if VERBOSE_TEST_CONFIGURATION
			status << "After test\n";
			status << "Output parameters: ";
			_writeArguments(status, test.outputTypes, outputBlock);
			status << "\n";
			#endif
			
			ocelot::unregisterModule(test.name);
		}
		catch(const std::exception& e)
		{
			status << " Test '" << test.name 
				<< "' failed during CUDA run with exception - \n" 
				<< e.what() << "\n";
				
			cudaDeviceProp properties;
			cudaGetDeviceProperties(&properties, device);
			
			status << "  On device - " << device << " - '" 
				<< properties.name << "'\n";
			
			pass = false;
		}
		
		if(pass)
		{
			try
			{
				(*test.reference)(referenceBlock, inputBlock);
			}
			catch(const hydrazine::Exception& e)
			{
				status << " Failed during reference run with exception - " 
					<< e.what() << "\n";

				cudaDeviceProp properties;
				cudaGetDeviceProperties(&properties, device);
			
				status << "  On device - " << device << " - '" 
					<< properties.name << "'\n";
				pass = false;
			}
		}
		
		uint32_t index = 0;
		for(TypeVector::const_iterator type = test.outputTypes.begin(); 
			type != test.outputTypes.end() && pass; ++type)
		{
			switch(*type)
			{
				case I8:
				{
					char computed = getParameter<char>(outputBlock, index);
					char reference = getParameter<char>(referenceBlock, index);
					if(computed != reference)
					{
						pass = false;
						status << " Output parameter " 
							<< std::distance(test.outputTypes.begin(), type) 
							<< " (I8) computed value - " << (int)computed 
							<< " does not match reference value - " 
							<< (int)reference << "\n";
						cudaDeviceProp properties;
						cudaGetDeviceProperties(&properties, device);
			
						status << "  On device - " << device << " - '" 
							<< properties.name << "'\n";
					}
					break;
				}
				case I16:
				{
					short computed = getParameter<short>(outputBlock, index);
					short reference = getParameter<short>(
						referenceBlock, index);
					if(computed != reference)
					{
						pass = false;
						status << " Output parameter " 
							<< std::distance(test.outputTypes.begin(), type) 
							<< " (I16) computed value - " << computed 
							<< " does not match reference value - " 
							<< reference << "\n";
						cudaDeviceProp properties;
						cudaGetDeviceProperties(&properties, device);
			
						status << "  On device - " << device << " - '" 
							<< properties.name << "'\n";
					}
					break;
				}				
				case I32:
				{
					int computed = getParameter<int>(outputBlock, index);
					int reference = getParameter<int>(referenceBlock, index);
					if(computed != reference)
					{
						pass = false;
						status << "Test '" << test.name << "' (seed " 
							<< seed << "): failed, output parameter " 
							<< std::distance(test.outputTypes.begin(), type) 
							<< " (I32) computed value - " << computed 
							<< " does not match reference value - " 
							<< reference << "\n";
						cudaDeviceProp properties;
						cudaGetDeviceProperties(&properties, device);
			
						status << "  On device - " << device << " - '" 
							<< properties.name << "'\n";
					}
					break;
				}
				case I64:
				{
					int64_t computed = getParameter<int64_t>(
						outputBlock, index);
					int64_t reference = getParameter<int64_t>(
						referenceBlock, index);
					if(computed != reference)
					{
						pass = false;
						status << "Test '" << test.name << "' (seed " 
							<< seed << "): failed, output parameter " 
							<< std::distance(test.outputTypes.begin(), type) 
							<< " (I64) computed value - " << computed 
							<< " does not match reference value - " 
							<< reference << "\n";
						cudaDeviceProp properties;
						cudaGetDeviceProperties(&properties, device);
			
						status << "  On device - " << device << " - '" 
							<< properties.name << "'\n";
					}
					break;
				}
				case FP32:
				{
					float computed = getParameter<float>(outputBlock, index);
					float reference = getParameter<float>(
						referenceBlock, index);
					if(!compareFloat(computed, reference, test.epsilon))
					{
						pass = false;
						status << " Output parameter " 
							<< std::distance(test.outputTypes.begin(), type) 
							<< " (F32) computed value - " << computed 
							<< " does not match reference value - " 
							<< reference << "\n";
						cudaDeviceProp properties;
						cudaGetDeviceProperties(&properties, device);
			
						status << "  On device - " << device << " - '" 
							<< properties.name << "'\n";
					}
					break;
				}
				case FP64:
				{
					double computed = getParameter<double>(outputBlock, index);
					double reference = getParameter<double>(
						referenceBlock, index);
					if(!compareDouble(computed, reference, test.epsilon))
					{
						pass = false;
						status << " Output parameter " 
							<< std::distance(test.outputTypes.begin(), type) 
							<< " (F64) computed value - " << computed 
							<< " does not match reference value - " 
							<< reference << "\n";
						cudaDeviceProp properties;
						cudaGetDeviceProperties(&properties, device);
			
						status << "  On device - " << device << " - '" 
							<< properties.name << "'\n";
					}
					break;
				}

			}
			index += bytes(*type);
		}
		
		if (!pass) {
			status << "Input parameters: ";
			_writeArguments(status, test.inputTypes, inputBlock);
			status << "\nOutput parameters: ";
			_writeArguments(status, test.outputTypes, outputBlock);
			status << "\nReference: ";
			_writeArguments(status, test.outputTypes, referenceBlock);
			
			status << "\n\n/* " << test.name << " */\n" << test.ptx << "\n";
		}
		
		delete[] inputBlock;
		delete[] outputBlock;
		delete[] referenceBlock;
	
	
		return pass;
	}

	void TestPTXAssembly::_loadTests(ir::Instruction::Architecture ISA)
	{
		//
		// Some tests necessarily fail for particular backends.
		//
		report("TestPTXAssembly::_loadTests(" << ir::Instruction::toString(ISA) << ")");

		add("TestVectorElements-u32", testVectorElements_REF, 
			testVectorElements_PTX(), testVectorElements_OUT(), 
			testVectorElements_IN(), uniformRandom<uint32_t, 4>, 1, 1);
		
		add("TestAdd-u16", testAdd_REF<uint16_t, false, false>, 
			testAdd_PTX(ir::PTXOperand::u16, false, false), 
			testAdd_OUT(I16), testAdd_IN(I16), 
			uniformRandom<uint16_t, 2>, 1, 1);
		add("TestAdd-s16", testAdd_REF<int16_t, false, false>, 
			testAdd_PTX(ir::PTXOperand::s16, false, false), 
			testAdd_OUT(I16), testAdd_IN(I16), 
			uniformRandom<int16_t, 2>, 1, 1);
		add("TestAdd-u32", testAdd_REF<uint32_t, false, false>, 
			testAdd_PTX(ir::PTXOperand::u32, false, false), 
			testAdd_OUT(I32), testAdd_IN(I32), 
			uniformRandom<uint32_t, 2>, 1, 1);
		add("TestAdd-s32", testAdd_REF<int32_t, false, false>, 
			testAdd_PTX(ir::PTXOperand::s32, false, false), 
			testAdd_OUT(I32), testAdd_IN(I32), 
			uniformRandom<int32_t, 2>, 1, 1);
		add("TestAdd-s32-sat", testAdd_REF<int32_t, true, false>, 
			testAdd_PTX(ir::PTXOperand::s32, true, false), 
			testAdd_OUT(I32), testAdd_IN(I32), 
			uniformRandom<int32_t, 2>, 1, 1);
		add("TestAdd-u64", testAdd_REF<uint64_t, false, false>, 
			testAdd_PTX(ir::PTXOperand::u64, false, false), 
			testAdd_OUT(I64), testAdd_IN(I64), 
			uniformRandom<uint64_t, 2>, 1, 1);
		add("TestAdd-s64", testAdd_REF<int64_t, false, false>, 
			testAdd_PTX(ir::PTXOperand::s64, false, false), 
			testAdd_OUT(I64), testAdd_IN(I64), 
			uniformRandom<int64_t, 2>, 1, 1);

		add("TestSub-u16", testAdd_REF<uint16_t, false, true>, 
			testAdd_PTX(ir::PTXOperand::u16, false, true), 
			testAdd_OUT(I16), testAdd_IN(I16), 
			uniformRandom<uint16_t, 2>, 1, 1);
		add("TestSub-s16", testAdd_REF<int16_t, false, true>, 
			testAdd_PTX(ir::PTXOperand::s16, false, true), 
			testAdd_OUT(I16), testAdd_IN(I16), 
			uniformRandom<int16_t, 2>, 1, 1);
		add("TestSub-u32", testAdd_REF<uint32_t, false, true>, 
			testAdd_PTX(ir::PTXOperand::u32, false, true), 
			testAdd_OUT(I32), testAdd_IN(I32), 
			uniformRandom<uint32_t, 2>, 1, 1);
		add("TestSub-s32", testAdd_REF<int32_t, false, true>, 
			testAdd_PTX(ir::PTXOperand::s32, false, true), 
			testAdd_OUT(I32), testAdd_IN(I32), 
			uniformRandom<int32_t, 2>, 1, 1);
		add("TestSub-s32-sat", testAdd_REF<int32_t, true, true>, 
			testAdd_PTX(ir::PTXOperand::s32, true, true), 
			testAdd_OUT(I32), testAdd_IN(I32), 
			uniformRandom<int32_t, 2>, 1, 1);
		add("TestSub-u64", testAdd_REF<uint64_t, false, true>, 
			testAdd_PTX(ir::PTXOperand::u64, false, true), 
			testAdd_OUT(I64), testAdd_IN(I64), 
			uniformRandom<uint64_t, 2>, 1, 1);
		add("TestSub-s64", testAdd_REF<int64_t, false, true>, 
			testAdd_PTX(ir::PTXOperand::s64, false, true), 
			testAdd_OUT(I64), testAdd_IN(I64), 
			uniformRandom<int64_t, 2>, 1, 1);

		add("TestSub-Carry-s32", testCarry_REF<int32_t, true>, 
			testCarry_PTX(ir::PTXOperand::s32, true), 
			testCarry_OUT(I32), testCarry_IN(I32), 
			uniformRandom<int32_t, 2>, 1, 1);
		add("TestSub-Carry-u32", testCarry_REF<uint32_t, true>, 
			testCarry_PTX(ir::PTXOperand::u32, true), 
			testCarry_OUT(I32), testCarry_IN(I32), 
			uniformRandom<uint32_t, 2>, 1, 1);

		add("TestAdd-Carry-s32", testCarry_REF<int32_t, false>, 
			testCarry_PTX(ir::PTXOperand::s32, false), 
			testCarry_OUT(I32), testCarry_IN(I32), 
			uniformRandom<int32_t, 2>, 1, 1);
		add("TestAdd-Carry-u32", testCarry_REF<uint32_t, false>, 
			testCarry_PTX(ir::PTXOperand::u32, false), 
			testCarry_OUT(I32), testCarry_IN(I32), 
			uniformRandom<uint32_t, 2>, 1, 1);	

		add("TestMul-Lo-u16", testMul_REF<uint16_t, MulLo>, 
			testMul_PTX(ir::PTXOperand::u16, MulLo), 
			testMul_OUT(I16, MulLo), testMul_IN(I16), 
			uniformRandom<uint16_t, 2>, 1, 1);
		add("TestMul-Hi-u16", testMul_REF<uint16_t, MulHi>, 
			testMul_PTX(ir::PTXOperand::u16, MulHi), 
			testMul_OUT(I16, MulHi), testMul_IN(I16), 
			uniformRandom<uint16_t, 2>, 1, 1);
		add("TestMul-Wide-u16", testMul_REF<uint16_t, MulWide>, 
			testMul_PTX(ir::PTXOperand::u16, MulWide), 
			testMul_OUT(I16, MulWide), testMul_IN(I16), 
			uniformRandom<uint16_t, 2>, 1, 1);

		add("TestMul-Lo-s16", testMul_REF<short, MulLo>, 
			testMul_PTX(ir::PTXOperand::s16, MulLo), 
			testMul_OUT(I16, MulLo), testMul_IN(I16), 
			uniformRandom<short, 2>, 1, 1);
		add("TestMul-Hi-s16", testMul_REF<short, MulHi>, 
			testMul_PTX(ir::PTXOperand::s16, MulHi), 
			testMul_OUT(I16, MulHi), testMul_IN(I16), 
			uniformRandom<short, 2>, 1, 1);
		add("TestMul-Wide-s16", testMul_REF<short, MulWide>, 
			testMul_PTX(ir::PTXOperand::s16, MulWide), 
			testMul_OUT(I16, MulWide), testMul_IN(I16), 
			uniformRandom<short, 2>, 1, 1);

		add("TestMul-Lo-u32", testMul_REF<uint32_t, MulLo>, 
			testMul_PTX(ir::PTXOperand::u32, MulLo), 
			testMul_OUT(I32, MulLo), testMul_IN(I32), 
			uniformRandom<uint32_t, 2>, 1, 1);
		add("TestMul-Hi-u32", testMul_REF<uint32_t, MulHi>, 
			testMul_PTX(ir::PTXOperand::u32, MulHi), 
			testMul_OUT(I32, MulHi), testMul_IN(I32), 
			uniformRandom<uint32_t, 2>, 1, 1);
		add("TestMul-Wide-u32", testMul_REF<uint32_t, MulWide>, 
			testMul_PTX(ir::PTXOperand::u32, MulWide), 
			testMul_OUT(I32, MulWide), testMul_IN(I32), 
			uniformRandom<uint32_t, 2>, 1, 1);

		add("TestMul-Lo-s32", testMul_REF<int, MulLo>, 
			testMul_PTX(ir::PTXOperand::s32, MulLo), 
			testMul_OUT(I32, MulLo), testMul_IN(I32), 
			uniformRandom<int, 2>, 1, 1);
		add("TestMul-Hi-s32", testMul_REF<int, MulHi>, 
			testMul_PTX(ir::PTXOperand::s32, MulHi), 
			testMul_OUT(I32, MulHi), testMul_IN(I32), 
			uniformRandom<int, 2>, 1, 1);
		add("TestMul-Wide-s32", testMul_REF<int, MulWide>, 
			testMul_PTX(ir::PTXOperand::s32, MulWide), 
			testMul_OUT(I32, MulWide), testMul_IN(I32), 
			uniformRandom<int, 2>, 1, 1);

		add("TestMul-Lo-u64", testMul_REF<uint64_t, MulLo>, 
			testMul_PTX(ir::PTXOperand::u64, MulLo), 
			testMul_OUT(I64, MulLo), testMul_IN(I64), 
			uniformRandom<uint64_t, 2>, 1, 1);
		add("TestMul-Hi-u64", testMul_REF<uint64_t, MulHi>, 
			testMul_PTX(ir::PTXOperand::u64, MulHi), 
			testMul_OUT(I64, MulHi), testMul_IN(I64), 
			uniformRandom<uint64_t, 2>, 1, 1);

		add("TestMul-Lo-s64", testMul_REF<int64_t, MulLo>, 
			testMul_PTX(ir::PTXOperand::s64, MulLo), 
			testMul_OUT(I64, MulLo), testMul_IN(I64), 
			uniformRandom<int64_t, 2>, 1, 1);
		add("TestMul-Hi-s64", testMul_REF<int64_t, MulHi>, 
			testMul_PTX(ir::PTXOperand::s64, MulHi), 
			testMul_OUT(I64, MulHi), testMul_IN(I64), 
			uniformRandom<int64_t, 2>, 1, 1);

		add("TestMad-Lo-u16", testMad_REF<uint16_t, MulLo, false>, 
			testMad_PTX(ir::PTXOperand::u16, MulLo, false), 
			testMad_OUT(I16, MulLo), testMad_IN(I16, MulLo), 
			uniformRandom<uint16_t, 3>, 1, 1);
		add("TestMad-Hi-u16", testMad_REF<uint16_t, MulHi, false>, 
			testMad_PTX(ir::PTXOperand::u16, MulHi, false), 
			testMad_OUT(I16, MulHi), testMad_IN(I16, MulHi), 
			uniformRandom<uint16_t, 3>, 1, 1);
		add("TestMad-Wide-u16", testMad_REF<uint16_t, MulWide, false>, 
			testMad_PTX(ir::PTXOperand::u16, MulWide, false), 
			testMad_OUT(I16, MulWide), testMad_IN(I16, MulWide), 
			uniformRandom<uint16_t, 4>, 1, 1);

		add("TestMad-Lo-s16", testMad_REF<short, MulLo, false>, 
			testMad_PTX(ir::PTXOperand::s16, MulLo, false), 
			testMad_OUT(I16, MulLo), testMad_IN(I16, MulLo), 
			uniformRandom<short, 3>, 1, 1);
		add("TestMad-Hi-s16", testMad_REF<short, MulHi, false>, 
			testMad_PTX(ir::PTXOperand::s16, MulHi, false), 
			testMad_OUT(I16, MulHi), testMad_IN(I16, MulHi), 
			uniformRandom<short, 3>, 1, 1);
		add("TestMad-Wide-s16", testMad_REF<short, MulWide, false>, 
			testMad_PTX(ir::PTXOperand::s16, MulWide, false), 
			testMad_OUT(I16, MulWide), testMad_IN(I16, MulWide), 
			uniformRandom<short, 4>, 1, 1);

		add("TestMad-Lo-u32", testMad_REF<uint32_t, MulLo, false>, 
			testMad_PTX(ir::PTXOperand::u32, MulLo, false), 
			testMad_OUT(I32, MulLo), testMad_IN(I32, MulLo), 
			uniformRandom<uint32_t, 3>, 1, 1);
		add("TestMad-Hi-u32", testMad_REF<uint32_t, MulHi, false>, 
			testMad_PTX(ir::PTXOperand::u32, MulHi, false), 
			testMad_OUT(I32, MulHi), testMad_IN(I32, MulHi), 
			uniformRandom<uint32_t, 3>, 1, 1);
		add("TestMad-Wide-u32", testMad_REF<uint32_t, MulWide, false>, 
			testMad_PTX(ir::PTXOperand::u32, MulWide, false), 
			testMad_OUT(I32, MulWide), testMad_IN(I32, MulWide), 
			uniformRandom<uint32_t, 4>, 1, 1);

		add("TestMad-Lo-s32", testMad_REF<int, MulLo, false>, 
			testMad_PTX(ir::PTXOperand::s32, MulLo, false), 
			testMad_OUT(I32, MulLo), testMad_IN(I32, MulLo), 
			uniformRandom<int, 3>, 1, 1);
		add("TestMad-Hi-s32", testMad_REF<int, MulHi, false>, 
			testMad_PTX(ir::PTXOperand::s32, MulHi, false), 
			testMad_OUT(I32, MulHi), testMad_IN(I32, MulHi), 
			uniformRandom<int, 3>, 1, 1);
		add("TestMad-Sat-Hi-s32", testMad_REF<int, MulHi, true>, 
			testMad_PTX(ir::PTXOperand::s32, MulHi, true), 
			testMad_OUT(I32, MulHi), testMad_IN(I32, MulHi), 
			uniformRandom<int, 3>, 1, 1);
		add("TestMad-Wide-s32", testMad_REF<int, MulWide, false>, 
			testMad_PTX(ir::PTXOperand::s32, MulWide, false), 
			testMad_OUT(I32, MulWide), testMad_IN(I32, MulWide), 
			uniformRandom<int, 4>, 1, 1);

		add("TestMad-Lo-u64", testMad_REF<uint64_t, MulLo, false>,
			testMad_PTX(ir::PTXOperand::u64, MulLo, false), 
			testMad_OUT(I64, MulLo), testMad_IN(I64, MulLo), 
			uniformRandom<uint64_t, 3>, 1, 1);
		add("TestMad-Hi-u64", testMad_REF<uint64_t, MulHi, false>,
			testMad_PTX(ir::PTXOperand::u64, MulHi, false), 
			testMad_OUT(I64, MulHi), testMad_IN(I64, MulHi), 
			uniformRandom<uint64_t, 3>, 1, 1);

		add("TestMad-Lo-s64", testMad_REF<int64_t, MulLo, false>, 
			testMad_PTX(ir::PTXOperand::s64, MulLo, false), 
			testMad_OUT(I64, MulLo), testMad_IN(I64, MulLo), 
			uniformRandom<int64_t, 3>, 1, 1);
		add("TestMad-Hi-s64", testMad_REF<int64_t, MulHi, false>, 
			testMad_PTX(ir::PTXOperand::s64, MulHi, false), 
			testMad_OUT(I64, MulHi), testMad_IN(I64, MulHi), 
			uniformRandom<int64_t, 3>, 1, 1);

		add("TestSad-u16", testSad_REF<uint16_t>, 
			testSad_PTX(ir::PTXOperand::u16), 
			testSad_OUT(I16), testSad_IN(I16), 
			uniformRandom<uint16_t, 3>, 1, 1);
		add("TestSad-s16", testSad_REF<short>, 
			testSad_PTX(ir::PTXOperand::s16), 
			testSad_OUT(I16), testSad_IN(I16), 
			uniformRandom<short, 3>, 1, 1);

		add("TestSad-u32", testSad_REF<uint32_t>, 
			testSad_PTX(ir::PTXOperand::u32), 
			testSad_OUT(I32), testSad_IN(I32), 
			uniformRandom<uint32_t, 3>, 1, 1);
		add("TestSad-s32", testSad_REF<int>, 
			testSad_PTX(ir::PTXOperand::s32), 
			testSad_OUT(I32), testSad_IN(I32), 
			uniformRandom<int, 3>, 1, 1);

		add("TestSad-u64", testSad_REF<uint64_t>, 
			testSad_PTX(ir::PTXOperand::u64), 
			testSad_OUT(I64), testSad_IN(I64), 
			uniformRandom<uint64_t, 3>, 1, 1);
		add("TestSad-s64", testSad_REF<int64_t>, 
			testSad_PTX(ir::PTXOperand::s64), 
			testSad_OUT(I64), testSad_IN(I64), 
			uniformRandom<int64_t, 3>, 1, 1);

		if (ISA == ir::Instruction::Emulated || ISA == ir::Instruction::LLVM || 
			ISA == ir::Instruction::VIR || ISA == ir::Instruction::CAL) {
			//
			// NVIDIA driver NVIDIA-Linux-x86_64-304.43 seems to compile PTX incorrectly
			// causing these tests to fail. div.{s,u}16 seems to be broken.
			//
			add("TestDiv-u16", testDiv_REF<uint16_t>, 
				testDiv_PTX(ir::PTXOperand::u16), 
				testDiv_OUT(I16), testDiv_IN(I16), 
				uniformNonZero<uint16_t, 2>, 1, 1);
			add("TestDiv-s16", testDiv_REF<short>, 
				testDiv_PTX(ir::PTXOperand::s16), 
				testDiv_OUT(I16), testDiv_IN(I16), 
				uniformNonZero<short, 2>, 1, 1);
		}

		add("TestDiv-u32", testDiv_REF<uint32_t>, 
			testDiv_PTX(ir::PTXOperand::u32), 
			testDiv_OUT(I32), testDiv_IN(I32), 
			uniformNonZero<uint32_t, 2>, 1, 1);
		add("TestDiv-s32", testDiv_REF<int>, 
			testDiv_PTX(ir::PTXOperand::s32), 
			testDiv_OUT(I32), testDiv_IN(I32), 
			uniformNonZero<int, 2>, 1, 1);

		add("TestDiv-u64", testDiv_REF<uint64_t>, 
			testDiv_PTX(ir::PTXOperand::u64), 
			testDiv_OUT(I64), testDiv_IN(I64), 
			uniformNonZero<uint64_t, 2>, 1, 1);
		add("TestDiv-s64", testDiv_REF<int64_t>, 
			testDiv_PTX(ir::PTXOperand::s64), 
			testDiv_OUT(I64), testDiv_IN(I64), 
			uniformNonZero<int64_t, 2>, 1, 1);
			
		if (ISA == ir::Instruction::Emulated || ISA == ir::Instruction::LLVM || 
			ISA == ir::Instruction::VIR || ISA == ir::Instruction::CAL) {
			//
			// NVIDIA driver NVIDIA-Linux-x86_64-304.43 seems to compile PTX incorrectly
			// causing these tests to fail. rem.{s,u}16 seems to be broken.
			//
			add("TestRem-u16", testRem_REF<uint16_t>, 
				testRem_PTX(ir::PTXOperand::u16), 
				testRem_OUT(I16), testRem_IN(I16), 
				uniformNonZero<uint16_t, 2>, 1, 1);
			add("TestRem-s16", testRem_REF<short>, 
				testRem_PTX(ir::PTXOperand::s16), 
				testRem_OUT(I16), testRem_IN(I16), 
				uniformNonZero<short, 2>, 1, 1);
		}

		add("TestRem-u32", testRem_REF<uint32_t>, 
			testRem_PTX(ir::PTXOperand::u32), 
			testRem_OUT(I32), testRem_IN(I32), 
			uniformNonZero<uint32_t, 2>, 1, 1);
		add("TestRem-s32", testRem_REF<int>, 
			testRem_PTX(ir::PTXOperand::s32), 
			testRem_OUT(I32), testRem_IN(I32), 
			uniformNonZero<int, 2>, 1, 1);
		add("TestRem-u64", testRem_REF<uint64_t>, 
			testRem_PTX(ir::PTXOperand::u64), 
			testRem_OUT(I64), testRem_IN(I64), 
			uniformNonZero<uint64_t, 2>, 1, 1);
		add("TestRem-s64", testRem_REF<int64_t>, 
			testRem_PTX(ir::PTXOperand::s64), 
			testRem_OUT(I64), testRem_IN(I64), 
			uniformNonZero<int64_t, 2>, 1, 1);

		add("TestAbs-s16", testAbs_REF<short>, 
			testAbs_PTX(ir::PTXOperand::s16), 
			testAbs_OUT(I16), testAbs_IN(I16), 
			uniformRandom<short, 1>, 1, 1);
		add("TestAbs-s32", testAbs_REF<int>, 
			testAbs_PTX(ir::PTXOperand::s32), 
			testAbs_OUT(I32), testAbs_IN(I32), 
			uniformRandom<int, 1>, 1, 1);
		add("TestAbs-s64", testAbs_REF<int64_t>, 
			testAbs_PTX(ir::PTXOperand::s64), 
			testAbs_OUT(I64), testAbs_IN(I64), 
			uniformRandom<int64_t, 1>, 1, 1);

		add("TestNeg-s16", testNeg_REF<short>, 
			testNeg_PTX(ir::PTXOperand::s16), 
			testNeg_OUT(I16), testNeg_IN(I16), 
			uniformRandom<short, 1>, 1, 1);
		add("TestNeg-s32", testNeg_REF<int>, 
			testNeg_PTX(ir::PTXOperand::s32), 
			testNeg_OUT(I32), testNeg_IN(I32), 
			uniformRandom<int, 1>, 1, 1);
		add("TestNeg-s64", testNeg_REF<int64_t>, 
			testNeg_PTX(ir::PTXOperand::s64), 
			testNeg_OUT(I64), testNeg_IN(I64), 
			uniformRandom<int64_t, 1>, 1, 1);

		add("TestMax-u16", testMinMax_REF<uint16_t, true>, 
			testMinMax_PTX(ir::PTXOperand::u16, true), 
			testMinMax_OUT(I16), testMinMax_IN(I16), 
			uniformRandom<uint16_t, 2>, 1, 1);
		add("TestMax-u32", testMinMax_REF<uint32_t, true>, 
			testMinMax_PTX(ir::PTXOperand::u32, true), 
			testMinMax_OUT(I32), testMinMax_IN(I32), 
			uniformRandom<uint32_t, 2>, 1, 1);
		add("TestMax-u64", testMinMax_REF<uint64_t, true>, 
			testMinMax_PTX(ir::PTXOperand::u64, true), 
			testMinMax_OUT(I64), testMinMax_IN(I64), 
			uniformRandom<uint64_t, 2>, 1, 1);
		add("TestMax-s16", testMinMax_REF<short, true>, 
			testMinMax_PTX(ir::PTXOperand::s16, true), 
			testMinMax_OUT(I16), testMinMax_IN(I16), 
			uniformRandom<short, 2>, 1, 1);
		add("TestMax-s32", testMinMax_REF<int, true>, 
			testMinMax_PTX(ir::PTXOperand::s32, true), 
			testMinMax_OUT(I32), testMinMax_IN(I32), 
			uniformRandom<int, 2>, 1, 1);
		add("TestMax-s64", testMinMax_REF<int64_t, true>, 
			testMinMax_PTX(ir::PTXOperand::s64, true), 
			testMinMax_OUT(I64), testMinMax_IN(I64), 
			uniformRandom<int64_t, 2>, 1, 1);

		add("TestMin-u16", testMinMax_REF<uint16_t, false>, 
			testMinMax_PTX(ir::PTXOperand::u16, false), 
			testMinMax_OUT(I16), testMinMax_IN(I16), 
			uniformRandom<uint16_t, 2>, 1, 1);
		add("TestMin-u32", testMinMax_REF<uint32_t, false>, 
			testMinMax_PTX(ir::PTXOperand::u32, false), 
			testMinMax_OUT(I32), testMinMax_IN(I32), 
			uniformRandom<uint32_t, 2>, 1, 1);
		add("TestMin-u64", testMinMax_REF<uint64_t, false>, 
			testMinMax_PTX(ir::PTXOperand::u64, false), 
			testMinMax_OUT(I64), testMinMax_IN(I64), 
			uniformRandom<uint64_t, 2>, 1, 1);
		add("TestMin-s16", testMinMax_REF<short, false>, 
			testMinMax_PTX(ir::PTXOperand::s16, false), 
			testMinMax_OUT(I16), testMinMax_IN(I16), 
			uniformRandom<short, 2>, 1, 1);
		add("TestMin-s32", testMinMax_REF<int, false>, 
			testMinMax_PTX(ir::PTXOperand::s32, false), 
			testMinMax_OUT(I32), testMinMax_IN(I32), 
			uniformRandom<int, 2>, 1, 1);
		add("TestMin-s64", testMinMax_REF<int64_t, false>, 
			testMinMax_PTX(ir::PTXOperand::s64, false), 
			testMinMax_OUT(I64), testMinMax_IN(I64), 
			uniformRandom<int64_t, 2>, 1, 1);

		add("TestPopc-b32", testPopc_REF<uint32_t>, 
			testPopc_PTX(ir::PTXOperand::b32), 
			testPopc_OUT(I32), testPopc_IN(I32), 
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestPopc-b64", testPopc_REF<uint64_t>, 
			testPopc_PTX(ir::PTXOperand::b64), 
			testPopc_OUT(I32), testPopc_IN(I64), 
			uniformRandom<uint64_t, 1>, 1, 1);

		add("TestClz-b32", testClz_REF<uint32_t>, 
			testClz_PTX(ir::PTXOperand::b32), 
			testClz_OUT(I32), testClz_IN(I32), 
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestClz-b64", testClz_REF<uint64_t>, 
			testClz_PTX(ir::PTXOperand::b64), 
			testClz_OUT(I32), testClz_IN(I64), 
			uniformRandom<uint64_t, 1>, 1, 1);

		add("TestBfind-u32", testBfind_REF<uint32_t, false>, 
			testBfind_PTX(ir::PTXOperand::u32, false), testBfind_OUT(), 
			testBfind_IN(I32), uniformRandom<uint32_t, 1>, 1, 1);
		add("TestBfind-u64", testBfind_REF<uint64_t, false>, 
			testBfind_PTX(ir::PTXOperand::u64, false), testBfind_OUT(), 
			testBfind_IN(I64), uniformRandom<uint64_t, 1>, 1, 1);
		add("TestBfind-s32", testBfind_REF<int, false>, 
			testBfind_PTX(ir::PTXOperand::s32, false), testBfind_OUT(), 
			testBfind_IN(I32), uniformRandom<int, 1>, 1, 1);
		add("TestBfind-s64", testBfind_REF<int64_t, false>, 
			testBfind_PTX(ir::PTXOperand::s64, false), testBfind_OUT(), 
			testBfind_IN(I64), uniformRandom<int64_t, 1>, 1, 1);

		add("TestBfind-shiftamount-u32", testBfind_REF<uint32_t, true>, 
			testBfind_PTX(ir::PTXOperand::u32, true), testBfind_OUT(), 
			testBfind_IN(I32), uniformRandom<uint32_t, 1>, 1, 1);
		add("TestBfind-shiftamount-u64", 
			testBfind_REF<uint64_t, true>, 
			testBfind_PTX(ir::PTXOperand::u64, true), testBfind_OUT(), 
			testBfind_IN(I64), uniformRandom<uint64_t, 1>, 1, 1);
		add("TestBfind-shiftamount-s32", testBfind_REF<int, true>, 
			testBfind_PTX(ir::PTXOperand::s32, true), testBfind_OUT(), 
			testBfind_IN(I32), uniformRandom<int, 1>, 1, 1);
		add("TestBfind-shiftamount-s64", testBfind_REF<int64_t, true>, 
			testBfind_PTX(ir::PTXOperand::s64, true), testBfind_OUT(), 
			testBfind_IN(I64), uniformRandom<int64_t, 1>, 1, 1);

		add("TestBrev-b32", testBrev_REF<uint32_t>, 
			testBrev_PTX(ir::PTXOperand::b32), testBrev_OUT(I32), 
			testBrev_IN(I32), uniformRandom<uint32_t, 1>, 1, 1);
		add("TestBrev-b64", testBrev_REF<uint64_t>, 
			testBrev_PTX(ir::PTXOperand::b64), testBrev_OUT(I64), 
			testBrev_IN(I64), uniformRandom<uint64_t, 1>, 1, 1);

		add("TestBfi-b32", testBfi_REF<uint32_t>, 
			testBfi_PTX(ir::PTXOperand::b32), testBfi_OUT(I32), 
			testBfi_IN(I32), uniformRandom<uint32_t, 4>, 1, 1);
		add("TestBfi-b64", testBfi_REF<uint64_t>, 
			testBfi_PTX(ir::PTXOperand::b64), testBfi_OUT(I64), 
			testBfi_IN(I64), uniformRandom<uint64_t, 3>, 1, 1);
	
		add("TestPrmt-b32", testPrmt_REF<ir::PTXInstruction::DefaultPermute>, 
			testPrmt_PTX(ir::PTXInstruction::DefaultPermute), testPrmt_OUT(), 
			testPrmt_IN(), uniformRandom<uint32_t, 3>, 1, 1);
		add("TestPrmt-f4e-b32", 
			testPrmt_REF<ir::PTXInstruction::ForwardFourExtract>, 
			testPrmt_PTX(ir::PTXInstruction::ForwardFourExtract), 
			testPrmt_OUT(), testPrmt_IN(), uniformRandom<uint32_t, 3>, 
			1, 1);
		add("TestPrmt-b4e-b32", 
			testPrmt_REF<ir::PTXInstruction::BackwardFourExtract>, 
			testPrmt_PTX(ir::PTXInstruction::BackwardFourExtract), 
			testPrmt_OUT(), testPrmt_IN(), uniformRandom<uint32_t, 3>, 
			1, 1);
		add("TestPrmt-rc8-b32", 
			testPrmt_REF<ir::PTXInstruction::ReplicateEight>, 
			testPrmt_PTX(ir::PTXInstruction::ReplicateEight), testPrmt_OUT(), 
			testPrmt_IN(), uniformRandom<uint32_t, 3>, 1, 1);
		add("TestPrmt-ecl-b32", testPrmt_REF<ir::PTXInstruction::EdgeClampLeft>, 
			testPrmt_PTX(ir::PTXInstruction::EdgeClampLeft), testPrmt_OUT(), 
			testPrmt_IN(), uniformRandom<uint32_t, 3>, 1, 1);
		add("TestPrmt-ecr-b32", 
			testPrmt_REF<ir::PTXInstruction::EdgeClampRight>, 
			testPrmt_PTX(ir::PTXInstruction::EdgeClampRight), testPrmt_OUT(), 
			testPrmt_IN(), uniformRandom<uint32_t, 3>, 1, 1);
		add("TestPrmt-rc16-b32", 
			testPrmt_REF<ir::PTXInstruction::ReplicateSixteen>, 
			testPrmt_PTX(ir::PTXInstruction::ReplicateSixteen), testPrmt_OUT(), 
			testPrmt_IN(), uniformRandom<uint32_t, 3>, 1, 1);
			
		if (ISA == ir::Instruction::Emulated || ISA == ir::Instruction::LLVM) {
		
			add("TestCudaNestedParallelism", testCudaNestedParallelism_REF, 
				testCudaNestedParallelism_PTX(), testCudaNestedParallelism_OUT(), 
				testCudaNestedParallelism_IN(), uniformRandom<uint32_t, 1>, 1, 1);
		}
		
		add("TestCall-Uni", testFunctionCalls_REF, 
			testFunctionCalls_PTX(true), testFunctionCalls_OUT(), 
			testFunctionCalls_IN(), uniformRandom<uint32_t, 2>, 1, 1);
		add("TestCall-Nondivergent", testFunctionCalls_REF, 
			testFunctionCalls_PTX(false), testFunctionCalls_OUT(), 
			testFunctionCalls_IN(), uniformRandom<uint32_t, 2>, 1, 1);
		add("TestCall-Divergent", testDivergentFunctionCall_REF, 
			testDivergentFunctionCall_PTX(), testDivergentFunctionCall_OUT(), 
			testDivergentFunctionCall_IN(), 
			uniformRandom<uint32_t, 4>, 2, 1);
		add("TestCall-Recursive", testRecursiveFunctionCall_REF, 
			testRecursiveFunctionCall_PTX(), testRecursiveFunctionCall_OUT(), 
			testRecursiveFunctionCall_IN(), 
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestCall-Indirect", testIndirectFunctionCall_REF, 
			testIndirectFunctionCall_PTX(), testIndirectFunctionCall_OUT(), 
			testIndirectFunctionCall_IN(), 
			uniformRandom<uint32_t, 1>, 4, 1);


		if (ISA == ir::Instruction::Emulated || ISA == ir::Instruction::LLVM || 
			ISA == ir::Instruction::VIR || ISA == ir::Instruction::CAL) {
			
			add("TestFunctionPointerArray", testIndirectFunctionCall_REF, 
				testFunctionPointerArray_PTX(), testIndirectFunctionCall_OUT(), 
				testIndirectFunctionCall_IN(),
				uniformRandom<uint32_t, 1>, 4, 1);
		}
		
		add("TestTestP-f32-Finite", 
			testTestP_REF<float, ir::PTXInstruction::Finite>, 
			testTestp_PTX(ir::PTXOperand::f32, ir::PTXInstruction::Finite), 
			testTestp_OUT(), testTestp_IN(FP32), uniformFloat<float, 1>, 1, 1);
		add("TestTestP-f32-Infinite", 
			testTestP_REF<float, ir::PTXInstruction::Infinite>, 
			testTestp_PTX(ir::PTXOperand::f32, ir::PTXInstruction::Infinite), 
			testTestp_OUT(), testTestp_IN(FP32), uniformFloat<float, 1>, 1, 1);
		add("TestTestP-f32-Number", 
			testTestP_REF<float, ir::PTXInstruction::Number>, 
			testTestp_PTX(ir::PTXOperand::f32, ir::PTXInstruction::Number), 
			testTestp_OUT(), testTestp_IN(FP32), uniformFloat<float, 1>, 1, 1);
		add("TestTestP-f32-NotANumber", 
			testTestP_REF<float, ir::PTXInstruction::NotANumber>, 
			testTestp_PTX(ir::PTXOperand::f32, ir::PTXInstruction::NotANumber), 
			testTestp_OUT(), testTestp_IN(FP32), uniformFloat<float, 1>, 1, 1);
		add("TestTestP-f32-Normal", 
			testTestP_REF<float, ir::PTXInstruction::Normal>, 
			testTestp_PTX(ir::PTXOperand::f32, ir::PTXInstruction::Normal), 
			testTestp_OUT(), testTestp_IN(FP32), uniformFloat<float, 1>, 1, 1);
		add("TestTestP-f32-SubNormal", 
			testTestP_REF<float, ir::PTXInstruction::SubNormal>, 
			testTestp_PTX(ir::PTXOperand::f32, ir::PTXInstruction::SubNormal), 
			testTestp_OUT(), testTestp_IN(FP32), uniformFloat<float, 1>, 1, 1);

		add("TestTestP-f64-Finite", 
			testTestP_REF<double, ir::PTXInstruction::Finite>, 
			testTestp_PTX(ir::PTXOperand::f64, ir::PTXInstruction::Finite), 
			testTestp_OUT(), testTestp_IN(FP64), uniformFloat<double, 1>, 1, 1);
		add("TestTestP-f64-Infinite", 
			testTestP_REF<double, ir::PTXInstruction::Infinite>, 
			testTestp_PTX(ir::PTXOperand::f64, ir::PTXInstruction::Infinite), 
			testTestp_OUT(), testTestp_IN(FP64), uniformFloat<double, 1>, 1, 1);
		add("TestTestP-f64-Number", 
			testTestP_REF<double, ir::PTXInstruction::Number>, 
			testTestp_PTX(ir::PTXOperand::f64, ir::PTXInstruction::Number), 
			testTestp_OUT(), testTestp_IN(FP64), uniformFloat<double, 1>, 1, 1);
		add("TestTestP-f64-NotANumber", 
			testTestP_REF<double, ir::PTXInstruction::NotANumber>, 
			testTestp_PTX(ir::PTXOperand::f64, ir::PTXInstruction::NotANumber), 
			testTestp_OUT(), testTestp_IN(FP64), uniformFloat<double, 1>, 1, 1);
		add("TestTestP-f64-Normal", 
			testTestP_REF<double, ir::PTXInstruction::Normal>, 
			testTestp_PTX(ir::PTXOperand::f64, ir::PTXInstruction::Normal), 
			testTestp_OUT(), testTestp_IN(FP64), uniformFloat<double, 1>, 1, 1);
		add("TestTestP-f64-SubNormal", 
			testTestP_REF<double, ir::PTXInstruction::SubNormal>, 
			testTestp_PTX(ir::PTXOperand::f64, ir::PTXInstruction::SubNormal), 
			testTestp_OUT(), testTestp_IN(FP64), uniformFloat<double, 1>, 1, 1);

		add("TestCopysign-f32", testCopysign_REF<float>, 
			testCopysign_PTX(ir::PTXOperand::f32), testCopysign_OUT(FP32), 
			testCopysign_IN(FP32), uniformFloat<float, 2>, 1, 1);
		add("TestCopysign-f64", testCopysign_REF<double>, 
			testCopysign_PTX(ir::PTXOperand::f64), testCopysign_OUT(FP64), 
			testCopysign_IN(FP64), uniformFloat<double, 2>, 1, 1);

		add("TestAdd-f32", testFadd_REF<float, 0, true>, 
			testFadd_PTX(ir::PTXOperand::f32, 0, true), testFadd_OUT(FP32), 
			testFadd_IN(FP32), uniformFloat<float, 2>, 1, 1);
		add("TestAdd-f32-sat",
			testFadd_REF<float, ir::PTXInstruction::sat, true>, 
			testFadd_PTX(ir::PTXOperand::f32, ir::PTXInstruction::sat, true),
			testFadd_OUT(FP32), testFadd_IN(FP32),
			uniformFloat<float, 2>, 1, 1);
		add("TestAdd-f32-ftz",
			testFadd_REF<float, ir::PTXInstruction::ftz, true>, 
			testFadd_PTX(ir::PTXOperand::f32, ir::PTXInstruction::ftz, true),
			testFadd_OUT(FP32), testFadd_IN(FP32),
			uniformFloat<float, 2>, 1, 1);
		add("TestAdd-f32-ftz-sat", testFadd_REF<float, 
			ir::PTXInstruction::ftz | ir::PTXInstruction::sat, true>, 
			testFadd_PTX(ir::PTXOperand::f32, 
			ir::PTXInstruction::ftz | ir::PTXInstruction::sat, true),
			testFadd_OUT(FP32), testFadd_IN(FP32),
			uniformFloat<float, 2>, 1, 1);
		add("TestAdd-f64", testFadd_REF<double, 0, true>, 
			testFadd_PTX(ir::PTXOperand::f64, 0, true), testFadd_OUT(FP64), 
			testFadd_IN(FP64), uniformFloat<double, 2>, 1, 1);

		add("TestSub-f32", testFadd_REF<float, 0, false>, 
			testFadd_PTX(ir::PTXOperand::f32, 0, false), testFadd_OUT(FP32), 
			testFadd_IN(FP32), uniformFloat<float, 2>, 1, 1);
		add("TestSub-f32-sat",
			testFadd_REF<float, ir::PTXInstruction::sat, false>, 
			testFadd_PTX(ir::PTXOperand::f32, ir::PTXInstruction::sat, false),
			testFadd_OUT(FP32), testFadd_IN(FP32),
			uniformFloat<float, 2>, 1, 1);
		add("TestSub-f32-ftz",
			testFadd_REF<float, ir::PTXInstruction::ftz, false>, 
			testFadd_PTX(ir::PTXOperand::f32, ir::PTXInstruction::ftz, false),
			testFadd_OUT(FP32), testFadd_IN(FP32),
			uniformFloat<float, 2>, 1, 1);
		add("TestSub-f32-ftz-sat", testFadd_REF<float, 
			ir::PTXInstruction::ftz | ir::PTXInstruction::sat, false>, 
			testFadd_PTX(ir::PTXOperand::f32, 
			ir::PTXInstruction::ftz | ir::PTXInstruction::sat, false),
			testFadd_OUT(FP32), testFadd_IN(FP32),
			uniformFloat<float, 2>, 1, 1);
		add("TestSub-f64", testFadd_REF<double, 0, false>, 
			testFadd_PTX(ir::PTXOperand::f64, 0, false), testFadd_OUT(FP64), 
			testFadd_IN(FP64), uniformFloat<double, 2>, 1, 1);

		add("TestMul-f32", testFmul_REF<float, 0>, 
			testFmul_PTX(ir::PTXOperand::f32, 0), testFmul_OUT(FP32), 
			testFmul_IN(FP32), uniformFloat<float, 2>, 1, 1);
		add("TestMul-f32-sat",
			testFmul_REF<float, ir::PTXInstruction::sat>, 
			testFmul_PTX(ir::PTXOperand::f32, ir::PTXInstruction::sat),
			testFmul_OUT(FP32), testFmul_IN(FP32),
			uniformFloat<float, 2>, 1, 1);
		add("TestMul-f32-ftz",
			testFmul_REF<float, ir::PTXInstruction::ftz>, 
			testFmul_PTX(ir::PTXOperand::f32, ir::PTXInstruction::ftz),
			testFmul_OUT(FP32), testFmul_IN(FP32),
			uniformFloat<float, 2>, 1, 1);
		add("TestMul-f32-ftz-sat", testFmul_REF<float, 
			ir::PTXInstruction::ftz | ir::PTXInstruction::sat>, 
			testFmul_PTX(ir::PTXOperand::f32, 
			ir::PTXInstruction::ftz | ir::PTXInstruction::sat),
			testFmul_OUT(FP32), testFmul_IN(FP32),
			uniformFloat<float, 2>, 1, 1);
		add("TestMul-f64", testFmul_REF<double, 0>, 
			testFmul_PTX(ir::PTXOperand::f64, 0), testFmul_OUT(FP64), 
			testFmul_IN(FP64), uniformFloat<double, 2>, 1, 1);

		add("TestMad-f32", testFma_REF<float, 0>, 
			testFma_PTX(ir::PTXOperand::f32, 0, true), testFma_OUT(FP32), 
			testFma_IN(FP32), uniformFloat<float, 3>, 1, 1);
		add("TestMad-f32-sat",
			testFma_REF<float, ir::PTXInstruction::sat>, 
			testFma_PTX(ir::PTXOperand::f32, ir::PTXInstruction::sat, true),
			testFma_OUT(FP32), testFma_IN(FP32),
			uniformFloat<float, 3>, 1, 1);
		add("TestMad-f32-ftz",
			testFma_REF<float, ir::PTXInstruction::ftz>, 
			testFma_PTX(ir::PTXOperand::f32, ir::PTXInstruction::ftz, true),
			testFma_OUT(FP32), testFma_IN(FP32),
			uniformFloat<float, 3>, 1, 1);
		add("TestMad-f32-ftz-sat", testFma_REF<float, 
			ir::PTXInstruction::ftz | ir::PTXInstruction::sat>, 
			testFma_PTX(ir::PTXOperand::f32, 
			ir::PTXInstruction::ftz | ir::PTXInstruction::sat, true),
			testFma_OUT(FP32), testFma_IN(FP32),
			uniformFloat<float, 3>, 1, 1);
		add("TestMad-f64", testFma_REF<double, 0>, 
			testFma_PTX(ir::PTXOperand::f64, 0, true), testFma_OUT(FP64), 
			testFma_IN(FP64), uniformFloat<double, 3>, 1, 1);

		add("TestFma-f32", testFma_REF<float, 0>, 
			testFma_PTX(ir::PTXOperand::f32, 0, false), testFma_OUT(FP32), 
			testFma_IN(FP32), uniformFloat<float, 3>, 1, 1);
		add("TestFma-f32-sat",
			testFma_REF<float, ir::PTXInstruction::sat>, 
			testFma_PTX(ir::PTXOperand::f32, ir::PTXInstruction::sat, false),
			testFma_OUT(FP32), testFma_IN(FP32),
			uniformFloat<float, 3>, 1, 1);
		add("TestFma-f32-ftz",
			testFma_REF<float, ir::PTXInstruction::ftz>, 
			testFma_PTX(ir::PTXOperand::f32, ir::PTXInstruction::ftz, false),
			testFma_OUT(FP32), testFma_IN(FP32),
			uniformFloat<float, 3>, 1, 1);
		add("TestFma-f32-ftz-sat", testFma_REF<float, 
			ir::PTXInstruction::ftz | ir::PTXInstruction::sat>, 
			testFma_PTX(ir::PTXOperand::f32, 
			ir::PTXInstruction::ftz | ir::PTXInstruction::sat, false),
			testFma_OUT(FP32), testFma_IN(FP32),
			uniformFloat<float, 3>, 1, 1);
		add("TestFma-f64", testFma_REF<double, 0>, 
			testFma_PTX(ir::PTXOperand::f64, 0, false), testFma_OUT(FP64), 
			testFma_IN(FP64), uniformFloat<double, 3>, 1, 1);

		add("TestDiv-f32", testFdiv_REF<float, 0>,
			testFdiv_PTX(ir::PTXOperand::f32, 0), testFdiv_OUT(FP32), 
			testFdiv_IN(FP32), uniformFloat<float, 2>, 1, 1);
		add("TestDiv-f32-ftz", testFdiv_REF<float, ir::PTXInstruction::ftz>,
			testFdiv_PTX(ir::PTXOperand::f32, ir::PTXInstruction::ftz), 
			testFdiv_OUT(FP32), testFdiv_IN(FP32),
			uniformFloat<float, 2>, 1, 1);
		add("TestDiv-f32-approx",
			testFdiv_REF<float, ir::PTXInstruction::approx>,
			testFdiv_PTX(ir::PTXOperand::f32, ir::PTXInstruction::approx), 
			testFdiv_OUT(FP32), testFdiv_IN(FP32),
			uniformFloat<float, 2>, 1, 1, 5);
		add("TestDiv-f32-full", testFdiv_REF<float, ir::PTXInstruction::full>,
			testFdiv_PTX(ir::PTXOperand::f32, ir::PTXInstruction::full), 
			testFdiv_OUT(FP32), testFdiv_IN(FP32),
			uniformFloat<float, 2>, 1, 1, 8);
		add("TestDiv-f32-approx-ftz",
			testFdiv_REF<float,
				ir::PTXInstruction::approx | ir::PTXInstruction::ftz>,
			testFdiv_PTX(ir::PTXOperand::f32,
				ir::PTXInstruction::approx | ir::PTXInstruction::ftz), 
			testFdiv_OUT(FP32), testFdiv_IN(FP32),
			uniformFloat<float, 2>, 1, 1, 5);
		add("TestDiv-f32-full-ftz", 
			testFdiv_REF<float,
				ir::PTXInstruction::full | ir::PTXInstruction::ftz>,
			testFdiv_PTX(ir::PTXOperand::f32,
				ir::PTXInstruction::full | ir::PTXInstruction::ftz), 
			testFdiv_OUT(FP32), testFdiv_IN(FP32),
			uniformFloat<float, 2>, 1, 1, 8);
		add("TestDiv-f64", testFdiv_REF<double, 0>,
			testFdiv_PTX(ir::PTXOperand::f64, 0), testFdiv_OUT(FP64), 
			testFdiv_IN(FP64), uniformFloat<double, 2>, 1, 1);

		add("TestAbs-f32", testAbsNeg_REF<float, false, false>,
			testAbsNeg_PTX(ir::PTXOperand::f32, false, false),
			testAbsNeg_INOUT(FP32), testAbsNeg_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestAbs-f32-ftz", testAbsNeg_REF<float, false, true>,
			testAbsNeg_PTX(ir::PTXOperand::f32, false, true),
			testAbsNeg_INOUT(FP32), testAbsNeg_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestAbs-f64", testAbsNeg_REF<double, false, false>,
			testAbsNeg_PTX(ir::PTXOperand::f64, false, false),
			testAbsNeg_INOUT(FP64), testAbsNeg_INOUT(FP64),
			uniformFloat<double, 1>, 1, 1);

		add("TestNeg-f32", testAbsNeg_REF<float, true, false>,
			testAbsNeg_PTX(ir::PTXOperand::f32, true, false),
			testAbsNeg_INOUT(FP32), testAbsNeg_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestNeg-f32-ftz", testAbsNeg_REF<float, true, true>,
			testAbsNeg_PTX(ir::PTXOperand::f32, true, true),
			testAbsNeg_INOUT(FP32), testAbsNeg_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestNeg-f64", testAbsNeg_REF<double, true, false>,
			testAbsNeg_PTX(ir::PTXOperand::f64, true, false),
			testAbsNeg_INOUT(FP64), testAbsNeg_INOUT(FP64),
			uniformFloat<double, 1>, 1, 1);
			
		add("TestMin-f32", testFMinMax_REF<float, true, false>,
			testFMinMax_PTX(ir::PTXOperand::f32, true, false),
			testFMinMax_OUT(FP32), testFMinMax_IN(FP32),
			uniformFloat<float, 2>, 1, 1);
		add("TestMin-f32-ftz", testFMinMax_REF<float, true, true>,
			testFMinMax_PTX(ir::PTXOperand::f32, true, true),
			testFMinMax_OUT(FP32), testFMinMax_IN(FP32),
			uniformFloat<float, 2>, 1, 1);
		add("TestMin-f64", testFMinMax_REF<double, true, false>,
			testFMinMax_PTX(ir::PTXOperand::f64, true, false),
			testFMinMax_OUT(FP64), testFMinMax_IN(FP64),
			uniformFloat<double, 2>, 1, 1);
			
		add("TestMax-f32", testFMinMax_REF<float, false, false>,
			testFMinMax_PTX(ir::PTXOperand::f32, false, false),
			testFMinMax_OUT(FP32), testFMinMax_IN(FP32),
			uniformFloat<float, 2>, 1, 1);
		add("TestMax-f32-ftz", testFMinMax_REF<float, false, true>,
			testFMinMax_PTX(ir::PTXOperand::f32, false, true),
			testFMinMax_OUT(FP32), testFMinMax_IN(FP32),
			uniformFloat<float, 2>, 1, 1);
		add("TestMax-f64", testFMinMax_REF<double, false, false>,
			testFMinMax_PTX(ir::PTXOperand::f64, false, false),
			testFMinMax_OUT(FP64), testFMinMax_IN(FP64),
			uniformFloat<double, 2>, 1, 1);

		add("TestRcp-f32", testRcpSqrt_REF<float, false, false, false>,
			testRcpSqrt_PTX(ir::PTXOperand::f32, false, false, false),
			testRcpSqrt_INOUT(FP32), testRcpSqrt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestRcp-f32-approx", testRcpSqrt_REF<float, false, true, false>,
			testRcpSqrt_PTX(ir::PTXOperand::f32, false, true, false),
			testRcpSqrt_INOUT(FP32), testRcpSqrt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1, 15);
		add("TestRcp-f32-ftz", testRcpSqrt_REF<float, false, false, true>,
			testRcpSqrt_PTX(ir::PTXOperand::f32, false, false, true),
			testRcpSqrt_INOUT(FP32), testRcpSqrt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestRcp-f32-approx-ftz", testRcpSqrt_REF<float, false, true, true>,
			testRcpSqrt_PTX(ir::PTXOperand::f32, false, true, true),
			testRcpSqrt_INOUT(FP32), testRcpSqrt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1, 15);
		add("TestRcp-f64", testRcpSqrt_REF<double, false, false, false>,
			testRcpSqrt_PTX(ir::PTXOperand::f64, false, false, false),
			testRcpSqrt_INOUT(FP64), testRcpSqrt_INOUT(FP64),
			uniformFloat<double, 1>, 1, 1, 10);
		add("TestRcp-f64-approx-ftz",
			testRcpSqrt_REF<double, false, true, true>,
			testRcpSqrt_PTX(ir::PTXOperand::f64, false, true, true),
			testRcpSqrt_INOUT(FP64), testRcpSqrt_INOUT(FP64),
			uniformFloat<double, 1>, 1, 1, 10);

		add("TestSqrt-f32", testRcpSqrt_REF<float, true, false, false>,
			testRcpSqrt_PTX(ir::PTXOperand::f32, true, false, false),
			testRcpSqrt_INOUT(FP32), testRcpSqrt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestSqrt-f32-approx", testRcpSqrt_REF<float, true, true, false>,
			testRcpSqrt_PTX(ir::PTXOperand::f32, true, true, false),
			testRcpSqrt_INOUT(FP32), testRcpSqrt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1, 15);
		add("TestSqrt-f32-ftz", testRcpSqrt_REF<float, true, false, true>,
			testRcpSqrt_PTX(ir::PTXOperand::f32, true, false, true),
			testRcpSqrt_INOUT(FP32), testRcpSqrt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestSqrt-f32-approx-ftz", testRcpSqrt_REF<float, true, true, true>,
			testRcpSqrt_PTX(ir::PTXOperand::f32, true, true, true),
			testRcpSqrt_INOUT(FP32), testRcpSqrt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1, 15);
		add("TestSqrt-f64", testRcpSqrt_REF<double, true, false, false>,
			testRcpSqrt_PTX(ir::PTXOperand::f64, true, false, false),
			testRcpSqrt_INOUT(FP64), testRcpSqrt_INOUT(FP64),
			uniformFloat<double, 1>, 1, 1);

		add("TestRsqrt-f32", testRsqrt_REF<float, false>,
			testRsqrt_PTX(ir::PTXOperand::f32, false),
			testRsqrt_INOUT(FP32), testRsqrt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1, 15);
		add("TestRsqrt-f32-ftz", testRsqrt_REF<float, true>,
			testRsqrt_PTX(ir::PTXOperand::f32, true),
			testRsqrt_INOUT(FP32), testRsqrt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1, 15);
		add("TestRsqrt-f64", testRsqrt_REF<double, false>,
			testRsqrt_PTX(ir::PTXOperand::f64, false),
			testRsqrt_INOUT(FP64), testRsqrt_INOUT(FP64),
			uniformFloat<double, 1>, 1, 1, 5);

		add("TestCos-f32",
			testSpecial_REF<ir::PTXInstruction::Cos, false>,
			testSpecial_PTX(ir::PTXInstruction::Cos, false),
			testSpecial_INOUT(), testSpecial_INOUT(),
			uniformFloat<float, 1>, 1, 1);
		add("TestCos-f32-ftz",
			testSpecial_REF<ir::PTXInstruction::Cos, true>,
			testSpecial_PTX(ir::PTXInstruction::Cos, true),
			testSpecial_INOUT(), testSpecial_INOUT(),
			uniformFloat<float, 1>, 1, 1);

		add("TestSin-f32",
			testSpecial_REF<ir::PTXInstruction::Sin, false>,
			testSpecial_PTX(ir::PTXInstruction::Sin, false),
			testSpecial_INOUT(), testSpecial_INOUT(),
			uniformFloat<float, 1>, 1, 1, 3);
		add("TestSin-f32-ftz",
			testSpecial_REF<ir::PTXInstruction::Sin, true>,
			testSpecial_PTX(ir::PTXInstruction::Sin, true),
			testSpecial_INOUT(), testSpecial_INOUT(),
			uniformFloat<float, 1>, 1, 1, 3);

		add("TestEx2-f32",
			testSpecial_REF<ir::PTXInstruction::Ex2, false>,
			testSpecial_PTX(ir::PTXInstruction::Ex2, false),
			testSpecial_INOUT(), testSpecial_INOUT(),
			uniformFloat<float, 1>, 1, 1);
		add("TestEx2-f32-ftz",
			testSpecial_REF<ir::PTXInstruction::Ex2, true>,
			testSpecial_PTX(ir::PTXInstruction::Ex2, true),
			testSpecial_INOUT(), testSpecial_INOUT(),
			uniformFloat<float, 1>, 1, 1);

		add("TestLg2-f32",
			testSpecial_REF<ir::PTXInstruction::Lg2, false>,
			testSpecial_PTX(ir::PTXInstruction::Lg2, false),
			testSpecial_INOUT(), testSpecial_INOUT(),
			uniformFloat<float, 1>, 1, 1, 10);
		add("TestLg2-f32-ftz",
			testSpecial_REF<ir::PTXInstruction::Lg2, true>,
			testSpecial_PTX(ir::PTXInstruction::Lg2, true),
			testSpecial_INOUT(), testSpecial_INOUT(),
			uniformFloat<float, 1>, 1, 1, 10);

		add("TestSet-lt-u32-u16",
			testSet_REF<uint32_t, uint16_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lt,
				false>,
			testSet_PTX(ir::PTXOperand::u32, ir::PTXOperand::u16, 
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Lt, false),
			testSet_OUT(I32), testSet_IN(I16),
			uniformRandom<char, 2*sizeof(uint16_t) + sizeof(bool)>, 1, 1);
		add("TestSet-lt-s32-u16",
			testSet_REF<int, uint16_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lt,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::u16, 
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Lt, false),
			testSet_OUT(I32), testSet_IN(I16),
			uniformRandom<char, 2*sizeof(uint16_t) + sizeof(bool)>, 1, 1);
		add("TestSet-lt-f32-u16",
			testSet_REF<float, uint16_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lt,
				false>,
			testSet_PTX(ir::PTXOperand::f32, ir::PTXOperand::u16, 
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Lt, false),
			testSet_OUT(FP32), testSet_IN(I16),
			uniformRandom<char, 2*sizeof(uint16_t) + sizeof(bool)>, 1, 1);
		add("TestSet-lt-s32-u32",
			testSet_REF<int, uint32_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lt,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::u32, 
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Lt, false),
			testSet_OUT(I32), testSet_IN(I32),
			uniformRandom<char, 2*sizeof(uint32_t) + sizeof(bool)>, 1, 1);
		add("TestSet-lt-s32-u64",
			testSet_REF<int, uint64_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lt,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::u64,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Lt, false),
			testSet_OUT(I32), testSet_IN(I64),
			uniformRandom<char, 2*sizeof(uint64_t)
				+ sizeof(bool)>, 1, 1);
		add("TestSet-lt-s32-s16",
			testSet_REF<int, short, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lt,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::s16,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Lt, false),
			testSet_OUT(I32), testSet_IN(I16),
			uniformRandom<char, 2*sizeof(short) + sizeof(bool)>, 1, 1);
		add("TestSet-lt-s32-s32",
			testSet_REF<int, int, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lt,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::s32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Lt, false),
			testSet_OUT(I32), testSet_IN(I32),
			uniformRandom<char, 2*sizeof(int) + sizeof(bool)>, 1, 1);
		add("TestSet-lt-s32-s64",
			testSet_REF<int, int64_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lt,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::s64,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Lt, false),
			testSet_OUT(I32), testSet_IN(I64),
			uniformRandom<char, 2*sizeof(int64_t) + sizeof(bool)>, 1, 1);
		add("TestSet-lt-s32-f32",
			testSet_REF<int, float, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lt,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::f32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Lt, false),
			testSet_OUT(I32), testSet_IN(FP32), uniformFloat<float, 3>, 1, 1);
		add("TestSet-lt-s32-f64",
			testSet_REF<int, double, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lt,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::f64,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Lt, false),
			testSet_OUT(I32), testSet_IN(FP64), uniformFloat<double, 3>, 1, 1);
		add("TestSet-lt-ftz-s32-f32",
			testSet_REF<int, float, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lt,
				true>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::f32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Lt, true),
			testSet_OUT(I32), testSet_IN(FP32),
			uniformFloat<float, 3>, 1, 1);

		add("TestSet-Eq-s32-s32",
			testSet_REF<int, int, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Eq,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::s32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Eq, false),
			testSet_OUT(I32), testSet_IN(I32),
			uniformRandom<char, 2*sizeof(int) + sizeof(bool)>, 1, 1);
		add("TestSet-Ne-s32-s32",
			testSet_REF<int, int, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Ne,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::s32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Ne, false),
			testSet_OUT(I32), testSet_IN(I32),
			uniformRandom<char, 2*sizeof(int) + sizeof(bool)>, 1, 1);
		add("TestSet-Le-s32-s32",
			testSet_REF<int, int, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Le,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::s32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Le, false),
			testSet_OUT(I32), testSet_IN(I32),
			uniformRandom<char, 2*sizeof(int) + sizeof(bool)>, 1, 1);
		add("TestSet-Gt-s32-s32",
			testSet_REF<int, int, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Gt,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::s32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Gt, false),
			testSet_OUT(I32), testSet_IN(I32),
			uniformRandom<char, 2*sizeof(int) + sizeof(bool)>, 1, 1);
		add("TestSet-Ge-s32-s32",
			testSet_REF<int, int, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Ge,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::s32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Ge, false),
			testSet_OUT(I32), testSet_IN(I32),
			uniformRandom<char, 2*sizeof(int) + sizeof(bool)>, 1, 1);
		add("TestSet-Lo-s32-u32",
			testSet_REF<int, uint32_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lo,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::u32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Lo, false),
			testSet_OUT(I32), testSet_IN(I32),
			uniformRandom<char, sizeof(uint32_t)
				+ sizeof(int) + sizeof(bool)>, 1, 1);
		add("TestSet-Ls-s32-u32",
			testSet_REF<int, uint32_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Ls,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::u32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Ls, false),
			testSet_OUT(I32), testSet_IN(I32),
			uniformRandom<char, sizeof(uint32_t)
				+ sizeof(int) + sizeof(bool)>, 1, 1);
		add("TestSet-Hi-s32-u32",
			testSet_REF<int, uint32_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Hi,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::u32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Hi, false),
			testSet_OUT(I32), testSet_IN(I32),
			uniformRandom<char, sizeof(uint32_t)
				+ sizeof(int) + sizeof(bool)>, 1, 1);
		add("TestSet-Hs-s32-u32",
			testSet_REF<int, uint32_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Hs,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::u32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Hs, false),
			testSet_OUT(I32), testSet_IN(I32),
			uniformRandom<char, sizeof(uint32_t)
				+ sizeof(int) + sizeof(bool)>, 1, 1);

		add("TestSet-Eq-s32-f32",
			testSet_REF<int, float, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Eq,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::f32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Eq, false),
			testSet_OUT(I32), testSet_IN(FP32),
			uniformFloat<float, 3>, 1, 1);
		add("TestSet-Ne-s32-f32",
			testSet_REF<int, float, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Ne,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::f32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Ne, false),
			testSet_OUT(I32), testSet_IN(FP32),
			uniformFloat<float, 3>, 1, 1);
		add("TestSet-Lt-s32-f32",
			testSet_REF<int, float, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lt,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::f32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Lt, false),
			testSet_OUT(I32), testSet_IN(FP32),
			uniformFloat<float, 3>, 1, 1);
		add("TestSet-Le-s32-f32",
			testSet_REF<int, float, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Le,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::f32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Le, false),
			testSet_OUT(I32), testSet_IN(FP32),
			uniformFloat<float, 3>, 1, 1);
		add("TestSet-Gt-s32-f32",
			testSet_REF<int, float, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Gt,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::f32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Gt, false),
			testSet_OUT(I32), testSet_IN(FP32),
			uniformFloat<float, 3>, 1, 1);
		add("TestSet-Ge-s32-f32",
			testSet_REF<int, float, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Ge,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::f32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Ge, false),
			testSet_OUT(I32), testSet_IN(FP32),
			uniformFloat<float, 3>, 1, 1);
		add("TestSet-Lo-s32-u32",
			testSet_REF<int, uint32_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lo,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::u32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Lo, false),
			testSet_OUT(I32), testSet_IN(I32),
			uniformRandom<char, sizeof(uint32_t)
				+ sizeof(int) + sizeof(bool)>, 1, 1);
		add("TestSet-Ls-s32-u32",
			testSet_REF<int, uint32_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Ls,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::u32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Ls, false),
			testSet_OUT(I32), testSet_IN(I32),
			uniformRandom<char, sizeof(uint32_t)
				+ sizeof(int) + sizeof(bool)>, 1, 1);
		add("TestSet-Hi-s32-u32",
			testSet_REF<int, uint32_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Hi,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::u32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Hi, false),
			testSet_OUT(I32), testSet_IN(I32),
			uniformRandom<char, sizeof(uint32_t)
				+ sizeof(int) + sizeof(bool)>, 1, 1);
		add("TestSet-Hs-s32-u32",
			testSet_REF<int, uint32_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Hs,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::u32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Hs, false),
			testSet_OUT(I32), testSet_IN(I32),
			uniformRandom<char, sizeof(uint32_t)
				+ sizeof(int) + sizeof(bool)>, 1, 1);
		add("TestSet-Equ-s32-f32",
			testSet_REF<int, float, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Equ,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::f32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Equ, false),
			testSet_OUT(I32), testSet_IN(FP32),
			uniformFloat<float, 3>, 1, 1);
		add("TestSet-Neu-s32-f32",
			testSet_REF<int, float, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Neu,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::f32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Neu, false),
			testSet_OUT(I32), testSet_IN(FP32),
			uniformFloat<float, 3>, 1, 1);
		add("TestSet-Ltu-s32-f32",
			testSet_REF<int, float, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Ltu,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::f32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Ltu, false),
			testSet_OUT(I32), testSet_IN(FP32),
			uniformFloat<float, 3>, 1, 1);
		add("TestSet-Leu-s32-f32",
			testSet_REF<int, float, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Leu,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::f32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Leu, false),
			testSet_OUT(I32), testSet_IN(FP32),
			uniformFloat<float, 3>, 1, 1);
		add("TestSet-Gtu-s32-f32",
			testSet_REF<int, float, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Gtu,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::f32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Gtu, false),
			testSet_OUT(I32), testSet_IN(FP32),
			uniformFloat<float, 3>, 1, 1);
		add("TestSet-Geu-s32-f32",
			testSet_REF<int, float, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Geu,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::f32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Geu, false),
			testSet_OUT(I32), testSet_IN(FP32),
			uniformFloat<float, 3>, 1, 1);
		add("TestSet-Num-s32-f32",
			testSet_REF<int, float, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Num,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::f32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Num, false),
			testSet_OUT(I32), testSet_IN(FP32),
			uniformFloat<float, 3>, 1, 1);
		add("TestSet-Nan-s32-f32",
			testSet_REF<int, float, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Nan,
				false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::f32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOp_Invalid,
				ir::PTXInstruction::Nan, false),
			testSet_OUT(I32), testSet_IN(FP32),
			uniformFloat<float, 3>, 1, 1);

		add("TestSet-Lt-s32-s32-And",
			testSet_REF<int, int, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolAnd, ir::PTXInstruction::Lt, false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::s32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolAnd,
				ir::PTXInstruction::Lt, false),
			testSet_OUT(I32), testSet_IN(I32),
			uniformRandom<char, 2*sizeof(int) + sizeof(bool)>, 1, 1);
		add("TestSet-Lt-s32-s32-Or",
			testSet_REF<int, int, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOr, ir::PTXInstruction::Lt, false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::s32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolOr,
				ir::PTXInstruction::Lt, false),
			testSet_OUT(I32), testSet_IN(I32),
			uniformRandom<char, 2*sizeof(int) + sizeof(bool)>, 1, 1);
		add("TestSet-Lt-s32-s32-Xor",
			testSet_REF<int, int, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolXor, ir::PTXInstruction::Lt, false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::s32,
				ir::PTXOperand::Pred, ir::PTXInstruction::BoolXor,
				ir::PTXInstruction::Lt, false),
			testSet_OUT(I32), testSet_IN(I32),
			uniformRandom<char, 2*sizeof(int) + sizeof(bool)>, 1, 1);
		add("TestSet-Lt-s32-s32-And-Inv",
			testSet_REF<int, int, ir::PTXOperand::InvPred, 
				ir::PTXInstruction::BoolAnd, ir::PTXInstruction::Lt, false>,
			testSet_PTX(ir::PTXOperand::s32, ir::PTXOperand::s32,
				ir::PTXOperand::InvPred, ir::PTXInstruction::BoolAnd,
				ir::PTXInstruction::Lt, false),
			testSet_OUT(I32), testSet_IN(I32),
			uniformRandom<char, 2*sizeof(int) + sizeof(bool)>, 1, 1);

		add("TestSetP-lt-u16",
			testSetP_REF<uint16_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lt,
				false, false>,
			testSetP_PTX(ir::PTXOperand::u16, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, 
				ir::PTXInstruction::Lt, false, false),
			testSetP_OUT(false), testSetP_IN(I16),
			uniformRandom<char, 2*sizeof(uint16_t) + sizeof(bool)>, 1, 1);
		add("TestSetP-lt-u32",
			testSetP_REF<uint32_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lt,
				false, false>,
			testSetP_PTX(ir::PTXOperand::u32, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, 
				ir::PTXInstruction::Lt, false, false),
			testSetP_OUT(false), testSetP_IN(I32),
			uniformRandom<char, 2*sizeof(uint32_t) + sizeof(bool)>, 1, 1);
		add("TestSetP-lt-u32-pq",
			testSetP_REF<uint32_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lt,
				false, true>,
			testSetP_PTX(ir::PTXOperand::u32, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, 
				ir::PTXInstruction::Lt, false, true),
			testSetP_OUT(true), testSetP_IN(I32),
			uniformRandom<char, 2*sizeof(uint32_t) + sizeof(bool)>, 1, 1);
		add("TestSetP-lt-u64",
			testSetP_REF<uint64_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lt,
				false, false>,
			testSetP_PTX(ir::PTXOperand::u64, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, 
				ir::PTXInstruction::Lt, false, false),
			testSetP_OUT(false), testSetP_IN(I64),
			uniformRandom<char,
				2*sizeof(uint64_t) + sizeof(bool)>, 1, 1);
		add("TestSetP-lt-s16",
			testSetP_REF<short, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lt,
				false, false>,
			testSetP_PTX(ir::PTXOperand::s16, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, 
				ir::PTXInstruction::Lt, false, false),
			testSetP_OUT(false), testSetP_IN(I16),
			uniformRandom<char, 2*sizeof(short) + sizeof(bool)>, 1, 1);
		add("TestSetP-lt-s32",
			testSetP_REF<int, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lt,
				false, false>,
			testSetP_PTX(ir::PTXOperand::s32, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, 
				ir::PTXInstruction::Lt, false, false),
			testSetP_OUT(false), testSetP_IN(I32),
			uniformRandom<char, 2*sizeof(int) + sizeof(bool)>, 1, 1);
		add("TestSetP-lt-s64",
			testSetP_REF<int64_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lt,
				false, false>,
			testSetP_PTX(ir::PTXOperand::s64, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, 
				ir::PTXInstruction::Lt, false, false),
			testSetP_OUT(false), testSetP_IN(I64),
			uniformRandom<char, 2*sizeof(int64_t) + sizeof(bool)>, 1, 1);
		add("TestSetP-lt-f32",
			testSetP_REF<float, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lt,
				false, false>,
			testSetP_PTX(ir::PTXOperand::f32, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, 
				ir::PTXInstruction::Lt, false, false),
			testSetP_OUT(false), testSetP_IN(FP32),
			uniformRandom<char, 2*sizeof(float) + sizeof(bool)>, 1, 1);
		add("TestSetP-lt-f64",
			testSetP_REF<double, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, ir::PTXInstruction::Lt,
				false, false>,
			testSetP_PTX(ir::PTXOperand::f64, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOp_Invalid, 
				ir::PTXInstruction::Lt, false, false),
			testSetP_OUT(false), testSetP_IN(FP64),
			uniformRandom<char, 2*sizeof(double) + sizeof(bool)>, 1, 1);
		
		add("TestSetP-lt-u32-pq-and",
			testSetP_REF<uint32_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolAnd, ir::PTXInstruction::Lt,
				false, true>,
			testSetP_PTX(ir::PTXOperand::u32, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolAnd, 
				ir::PTXInstruction::Lt, false, true),
			testSetP_OUT(true), testSetP_IN(I32),
			uniformRandom<char, 2*sizeof(uint32_t) + sizeof(bool)>, 1, 1);
		add("TestSetP-lt-u32-pq-or",
			testSetP_REF<uint32_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOr, ir::PTXInstruction::Lt,
				false, true>,
			testSetP_PTX(ir::PTXOperand::u32, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolOr, 
				ir::PTXInstruction::Lt, false, true),
			testSetP_OUT(true), testSetP_IN(I32),
			uniformRandom<char, 2*sizeof(uint32_t) + sizeof(bool)>, 1, 1);
		add("TestSetP-lt-u32-pq-xor",
			testSetP_REF<uint32_t, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolXor, ir::PTXInstruction::Lt,
				false, true>,
			testSetP_PTX(ir::PTXOperand::u32, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolXor, 
				ir::PTXInstruction::Lt, false, true),
			testSetP_OUT(true), testSetP_IN(I32),
			uniformRandom<char, 2*sizeof(uint32_t) + sizeof(bool)>, 1, 1);
		add("TestSetP-lt-f32-pq-and-ftz",
			testSetP_REF<float, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolAnd, ir::PTXInstruction::Lt,
				true, true>,
			testSetP_PTX(ir::PTXOperand::f32, ir::PTXOperand::Pred, 
				ir::PTXInstruction::BoolAnd, 
				ir::PTXInstruction::Lt, true, true),
			testSetP_OUT(true), testSetP_IN(FP32),
			uniformRandom<char, 2*sizeof(float) + sizeof(bool)>, 1, 1);
		
		add("TestSelp-u16", testSelP_REF<uint16_t>, 
			testSelP_PTX(ir::PTXOperand::u16), 
			testSelP_OUT(I16), testSelP_IN(I16), 
			uniformRandom<char, 2*sizeof(uint16_t) + sizeof(bool)>, 1, 1);
		add("TestSelp-u32", testSelP_REF<uint32_t>, 
			testSelP_PTX(ir::PTXOperand::u32), 
			testSelP_OUT(I32), testSelP_IN(I32), 
			uniformRandom<char, 2*sizeof(uint32_t) + sizeof(bool)>, 1, 1);
		add("TestSelp-u64", testSelP_REF<uint64_t>, 
			testSelP_PTX(ir::PTXOperand::u64), 
			testSelP_OUT(I64), testSelP_IN(I64), 
			uniformRandom<char,
				2*sizeof(uint64_t) + sizeof(bool)>, 1, 1);
		add("TestSelp-s16", testSelP_REF<short>, 
			testSelP_PTX(ir::PTXOperand::s16), 
			testSelP_OUT(I16), testSelP_IN(I16), 
			uniformRandom<char, 2*sizeof(short) + sizeof(bool)>, 1, 1);
		add("TestSelp-s32", testSelP_REF<int>, 
			testSelP_PTX(ir::PTXOperand::u32), 
			testSelP_OUT(I32), testSelP_IN(I32), 
			uniformRandom<char, 2*sizeof(int) + sizeof(bool)>, 1, 1);
		add("TestSelp-s64", testSelP_REF<int64_t>, 
			testSelP_PTX(ir::PTXOperand::u64), 
			testSelP_OUT(I64), testSelP_IN(I64), 
			uniformRandom<char, 2*sizeof(int64_t) + sizeof(bool)>, 1, 1);
		add("TestSelp-f32", testSelP_REF<float>, 
			testSelP_PTX(ir::PTXOperand::f32), 
			testSelP_OUT(FP32), testSelP_IN(FP32), 
			uniformFloat<float, 3>, 1, 1);
		add("TestSelp-f64", testSelP_REF<double>, 
			testSelP_PTX(ir::PTXOperand::u64), 
			testSelP_OUT(I64), testSelP_IN(I64), 
			uniformFloat<double, 3>, 1, 1);
		
		add("TestSlct-u16-s32", testSlct_REF<uint16_t, false, false>,
			testSlct_PTX(ir::PTXOperand::u16, false, false),
			testSlct_OUT(I16), testSlct_IN(I16, false),
			uniformRandom<char, 2*sizeof(uint16_t)
				+ sizeof(uint32_t)>, 1, 1);
		add("TestSlct-u32-s32", testSlct_REF<uint32_t, false, false>,
			testSlct_PTX(ir::PTXOperand::u32, false, false),
			testSlct_OUT(I32), testSlct_IN(I32, false),
			uniformRandom<char, 2*sizeof(uint32_t)
				+ sizeof(uint32_t)>, 1, 1);
		add("TestSlct-u64-s32", 
			testSlct_REF<uint64_t, false, false>,
			testSlct_PTX(ir::PTXOperand::u64, false, false),
			testSlct_OUT(I64), testSlct_IN(I64, false),
			uniformRandom<char, 2*sizeof(uint64_t)
				+ sizeof(uint32_t)>, 1, 1);
		add("TestSlct-s16-s32", testSlct_REF<short, false, false>,
			testSlct_PTX(ir::PTXOperand::s16, false, false),
			testSlct_OUT(I16), testSlct_IN(I16, false),
			uniformRandom<char, 2*sizeof(short)	+ sizeof(uint32_t)>, 1, 1);
		add("TestSlct-s32-s32", testSlct_REF<int, false, false>,
			testSlct_PTX(ir::PTXOperand::u32, false, false),
			testSlct_OUT(I32), testSlct_IN(I32, false),
			uniformRandom<char, 2*sizeof(int) + sizeof(uint32_t)>, 1, 1);
		add("TestSlct-s64-s32", 
			testSlct_REF<int64_t, false, false>,
			testSlct_PTX(ir::PTXOperand::s64, false, false),
			testSlct_OUT(I64), testSlct_IN(I64, false),
			uniformRandom<char, 2*sizeof(int64_t)
				+ sizeof(uint32_t)>, 1, 1);
		add("TestSlct-f32-s32", testSlct_REF<float, false, false>,
			testSlct_PTX(ir::PTXOperand::f32, false, false),
			testSlct_OUT(FP32), testSlct_IN(FP32, false),
			uniformFloat<float, 3>, 1, 1);
		add("TestSlct-f64-s32", testSlct_REF<double, false, false>,
			testSlct_PTX(ir::PTXOperand::f64, false, false),
			testSlct_OUT(FP64), testSlct_IN(FP64, false),
			uniformFloat<double, 3>, 1, 1);

		
		add("TestSlct-u16-f32", testSlct_REF<uint16_t, true, false>,
			testSlct_PTX(ir::PTXOperand::u16, true, false),
			testSlct_OUT(I16), testSlct_IN(I16, true),
			uniformRandom<char, 2*sizeof(uint16_t)
				+ sizeof(uint32_t)>, 1, 1);
		add("TestSlct-u32-f32", testSlct_REF<uint32_t, true, false>,
			testSlct_PTX(ir::PTXOperand::u32, true, false),
			testSlct_OUT(I32), testSlct_IN(I32, true),
			uniformRandom<char, 2*sizeof(uint32_t)
				+ sizeof(uint32_t)>, 1, 1);
		add("TestSlct-u64-f32", 
			testSlct_REF<uint64_t, true, false>,
			testSlct_PTX(ir::PTXOperand::u64, true, false),
			testSlct_OUT(I64), testSlct_IN(I64, true),
			uniformRandom<char, 2*sizeof(uint64_t)
				+ sizeof(uint32_t)>, 1, 1);
		add("TestSlct-s16-f32", testSlct_REF<short, true, false>,
			testSlct_PTX(ir::PTXOperand::s16, true, false),
			testSlct_OUT(I16), testSlct_IN(I16, true),
			uniformRandom<char, 2*sizeof(short)	+ sizeof(uint32_t)>, 1, 1);
		add("TestSlct-s32-f32", testSlct_REF<int, true, false>,
			testSlct_PTX(ir::PTXOperand::u32, true, false),
			testSlct_OUT(I32), testSlct_IN(I32, true),
			uniformRandom<char, 2*sizeof(int) + sizeof(uint32_t)>, 1, 1);
		add("TestSlct-s64-f32", 
			testSlct_REF<int64_t, true, false>,
			testSlct_PTX(ir::PTXOperand::s64, true, false),
			testSlct_OUT(I64), testSlct_IN(I64, true),
			uniformRandom<char, 2*sizeof(int64_t)
				+ sizeof(uint32_t)>, 1, 1);
		add("TestSlct-f32-f32", testSlct_REF<float, true, false>,
			testSlct_PTX(ir::PTXOperand::f32, true, false),
			testSlct_OUT(FP32), testSlct_IN(FP32, true),
			uniformFloat<float, 3>, 1, 1);
		add("TestSlct-f64-f32", testSlct_REF<double, true, false>,
			testSlct_PTX(ir::PTXOperand::f64, true, false),
			testSlct_OUT(FP64), testSlct_IN(FP64, true),
			uniformFloat<double, 3>, 1, 1);
		add("TestSlct-f32-f32-ftz", testSlct_REF<float, true, true>,
			testSlct_PTX(ir::PTXOperand::f32, true, true),
			testSlct_OUT(FP32), testSlct_IN(FP32, true),
			uniformFloat<float, 3>, 1, 1);

		add("TestAnd-pred",
			testLops_REF<ir::PTXInstruction::And, uint8_t>,
			testLops_PTX(ir::PTXInstruction::And, ir::PTXOperand::pred),
			testLops_OUT(I8), testLops_IN(ir::PTXInstruction::And, I8),
			uniformRandom<bool, 3>, 1, 1);
		add("TestAnd-b16",
			testLops_REF<ir::PTXInstruction::And, uint16_t>,
			testLops_PTX(ir::PTXInstruction::And, ir::PTXOperand::b16),
			testLops_OUT(I16), testLops_IN(ir::PTXInstruction::And, I16),
			uniformRandom<uint16_t, 3>, 1, 1);
		add("TestAnd-b32",
			testLops_REF<ir::PTXInstruction::And, uint32_t>,
			testLops_PTX(ir::PTXInstruction::And, ir::PTXOperand::b32),
			testLops_OUT(I32), testLops_IN(ir::PTXInstruction::And, I32),
			uniformRandom<uint32_t, 3>, 1, 1);
		add("TestAnd-b64",
			testLops_REF<ir::PTXInstruction::And, uint64_t>,
			testLops_PTX(ir::PTXInstruction::And, ir::PTXOperand::b64),
			testLops_OUT(I64), testLops_IN(ir::PTXInstruction::And, I64),
			uniformRandom<uint64_t, 3>, 1, 1);
		add("TestOr-pred", testLops_REF<ir::PTXInstruction::Or, uint8_t>,
			testLops_PTX(ir::PTXInstruction::Or, ir::PTXOperand::pred),
			testLops_OUT(I8), testLops_IN(ir::PTXInstruction::Or, I8),
			uniformRandom<bool, 3>, 1, 1);
		add("TestOr-b16",
			testLops_REF<ir::PTXInstruction::Or, uint16_t>,
			testLops_PTX(ir::PTXInstruction::Or, ir::PTXOperand::b16),
			testLops_OUT(I16), testLops_IN(ir::PTXInstruction::Or, I16),
			uniformRandom<uint16_t, 3>, 1, 1);
		add("TestOr-b32",
			testLops_REF<ir::PTXInstruction::Or, uint32_t>,
			testLops_PTX(ir::PTXInstruction::Or, ir::PTXOperand::b32),
			testLops_OUT(I32), testLops_IN(ir::PTXInstruction::Or, I32),
			uniformRandom<uint32_t, 3>, 1, 1);
		add("TestOr-b64",
			testLops_REF<ir::PTXInstruction::Or, uint64_t>,
			testLops_PTX(ir::PTXInstruction::Or, ir::PTXOperand::b64),
			testLops_OUT(I64), testLops_IN(ir::PTXInstruction::Or, I64),
			uniformRandom<uint64_t, 3>, 1, 1);
		add("TestXor-pred",
			testLops_REF<ir::PTXInstruction::Xor, uint8_t>,
			testLops_PTX(ir::PTXInstruction::Xor, ir::PTXOperand::pred),
			testLops_OUT(I8), testLops_IN(ir::PTXInstruction::Xor, I8),
			uniformRandom<bool, 3>, 1, 1);
		add("TestXor-b16",
			testLops_REF<ir::PTXInstruction::Xor, uint16_t>,
			testLops_PTX(ir::PTXInstruction::Xor, ir::PTXOperand::b16),
			testLops_OUT(I16), testLops_IN(ir::PTXInstruction::Xor, I16),
			uniformRandom<uint16_t, 3>, 1, 1);
		add("TestXor-b32",
			testLops_REF<ir::PTXInstruction::Xor, uint32_t>,
			testLops_PTX(ir::PTXInstruction::Xor, ir::PTXOperand::b32),
			testLops_OUT(I32), testLops_IN(ir::PTXInstruction::Xor, I32),
			uniformRandom<uint32_t, 3>, 1, 1);
		add("TestXor-b64",
			testLops_REF<ir::PTXInstruction::Xor, uint64_t>,
			testLops_PTX(ir::PTXInstruction::Xor, ir::PTXOperand::b64),
			testLops_OUT(I64), testLops_IN(ir::PTXInstruction::Xor, I64),
			uniformRandom<uint64_t, 3>, 1, 1);
		add("TestNot-pred",
			testLops_REF<ir::PTXInstruction::Not,uint8_t>,
			testLops_PTX(ir::PTXInstruction::Not, ir::PTXOperand::pred),
			testLops_OUT(I8), testLops_IN(ir::PTXInstruction::Not, I8),
			uniformRandom<bool, 2>, 1, 1);
		add("TestNot-b16",
			testLops_REF<ir::PTXInstruction::Not, uint16_t>,
			testLops_PTX(ir::PTXInstruction::Not, ir::PTXOperand::b16),
			testLops_OUT(I16), testLops_IN(ir::PTXInstruction::Not, I16),
			uniformRandom<uint16_t, 2>, 1, 1);
		add("TestNot-b32", testLops_REF<ir::PTXInstruction::Not, uint32_t>,
			testLops_PTX(ir::PTXInstruction::Not, ir::PTXOperand::b32),
			testLops_OUT(I32), testLops_IN(ir::PTXInstruction::Not, I32),
			uniformRandom<uint32_t, 2>, 1, 1);
		add("TestNot-b64",
			testLops_REF<ir::PTXInstruction::Not, uint64_t>,
			testLops_PTX(ir::PTXInstruction::Not, ir::PTXOperand::b64),
			testLops_OUT(I64), testLops_IN(ir::PTXInstruction::Not, I64),
			uniformRandom<uint64_t, 2>, 1, 1);
		add("TestCNot-b16",
			testLops_REF<ir::PTXInstruction::CNot, uint16_t>,
			testLops_PTX(ir::PTXInstruction::CNot, ir::PTXOperand::b16),
			testLops_OUT(I16), testLops_IN(ir::PTXInstruction::CNot, I16),
			uniformRandom<uint16_t, 2>, 1, 1);
		add("TestCNot-b32", testLops_REF<ir::PTXInstruction::CNot, uint32_t>,
			testLops_PTX(ir::PTXInstruction::CNot, ir::PTXOperand::b32),
			testLops_OUT(I32), testLops_IN(ir::PTXInstruction::CNot, I32),
			uniformRandom<uint32_t, 2>, 1, 1);
		add("TestCNot-b64",
			testLops_REF<ir::PTXInstruction::CNot, uint64_t>,
			testLops_PTX(ir::PTXInstruction::CNot, ir::PTXOperand::b64),
			testLops_OUT(I64), testLops_IN(ir::PTXInstruction::CNot, I64),
			uniformRandom<uint64_t, 2>, 1, 1);

		add("TestShl-b16",
			testLops_REF<ir::PTXInstruction::Shl, uint16_t>,
			testLops_PTX(ir::PTXInstruction::Shl, ir::PTXOperand::b16),
			testLops_OUT(I16), testLops_IN(ir::PTXInstruction::Shl, I16),
			uniformRandom<uint16_t, 4>, 1, 1);
		add("TestShl-b32", testLops_REF<ir::PTXInstruction::Shl, uint32_t>,
			testLops_PTX(ir::PTXInstruction::Shl, ir::PTXOperand::b32),
			testLops_OUT(I32), testLops_IN(ir::PTXInstruction::Shl, I32),
			uniformRandom<uint32_t, 3>, 1, 1);
		add("TestShl-b64",
			testLops_REF<ir::PTXInstruction::Shl, uint64_t>,
			testLops_PTX(ir::PTXInstruction::Shl, ir::PTXOperand::b64),
			testLops_OUT(I64), testLops_IN(ir::PTXInstruction::Shl, I64),
			uniformRandom<uint64_t, 3>, 1, 1);

		add("TestShr-u16",
			testLops_REF<ir::PTXInstruction::Shr, uint16_t>,
			testLops_PTX(ir::PTXInstruction::Shr, ir::PTXOperand::u16),
			testLops_OUT(I16), testLops_IN(ir::PTXInstruction::Shr, I16),
			uniformRandom<uint16_t, 4>, 1, 1);
		add("TestShr-u32", testLops_REF<ir::PTXInstruction::Shr, uint32_t>,
			testLops_PTX(ir::PTXInstruction::Shr, ir::PTXOperand::u32),
			testLops_OUT(I32), testLops_IN(ir::PTXInstruction::Shr, I32),
			uniformRandom<uint32_t, 3>, 1, 1);
		add("TestShr-u64",
			testLops_REF<ir::PTXInstruction::Shr, uint64_t>,
			testLops_PTX(ir::PTXInstruction::Shr, ir::PTXOperand::u64),
			testLops_OUT(I64), testLops_IN(ir::PTXInstruction::Shr, I64),
			uniformRandom<uint64_t, 3>, 1, 1);
		add("TestShr-s16",
			testLops_REF<ir::PTXInstruction::Shr, short>,
			testLops_PTX(ir::PTXInstruction::Shr, ir::PTXOperand::s16),
			testLops_OUT(I16), testLops_IN(ir::PTXInstruction::Shr, I16),
			uniformRandom<short, 4>, 1, 1);
		add("TestShr-s32", testLops_REF<ir::PTXInstruction::Shr, int>,
			testLops_PTX(ir::PTXInstruction::Shr, ir::PTXOperand::s32),
			testLops_OUT(I32), testLops_IN(ir::PTXInstruction::Shr, I32),
			uniformRandom<int, 3>, 1, 1);
		add("TestShr-s64",
			testLops_REF<ir::PTXInstruction::Shr, int64_t>,
			testLops_PTX(ir::PTXInstruction::Shr, ir::PTXOperand::s64),
			testLops_OUT(I64), testLops_IN(ir::PTXInstruction::Shr, I64),
			uniformRandom<int64_t, 3>, 1, 1);

		add("TestMov-pred", testMov_REF<bool>,
			testMov_PTX(ir::PTXOperand::pred), testMov_INOUT(I8),
			testMov_INOUT(I8), uniformRandom<char, 1>, 1, 1);
		add("TestMov-u16", testMov_REF<uint16_t>,
			testMov_PTX(ir::PTXOperand::u16), testMov_INOUT(I16),
			testMov_INOUT(I16), uniformRandom<uint16_t, 1>, 1, 1);
		add("TestMov-u32", testMov_REF<uint32_t>,
			testMov_PTX(ir::PTXOperand::u32), testMov_INOUT(I32),
			testMov_INOUT(I32), uniformRandom<uint16_t, 1>, 1, 1);
		add("TestMov-u64", testMov_REF<uint64_t>,
			testMov_PTX(ir::PTXOperand::u64), testMov_INOUT(I64),
			testMov_INOUT(I64), uniformRandom<uint64_t, 1>, 1, 1);
		add("TestMov-s16", testMov_REF<short>,
			testMov_PTX(ir::PTXOperand::s16), testMov_INOUT(I16),
			testMov_INOUT(I16), uniformRandom<short, 1>, 1, 1);
		add("TestMov-s32", testMov_REF<int>,
			testMov_PTX(ir::PTXOperand::s32), testMov_INOUT(I32),
			testMov_INOUT(I32), uniformRandom<int, 1>, 1, 1);
		add("TestMov-s64", testMov_REF<int64_t>,
			testMov_PTX(ir::PTXOperand::s64), testMov_INOUT(I64),
			testMov_INOUT(I64), uniformRandom<int64_t, 1>, 1, 1);
		add("TestMov-f32", testMov_REF<int>,
			testMov_PTX(ir::PTXOperand::f32), testMov_INOUT(FP32),
			testMov_INOUT(FP32), uniformFloat<float, 1>, 1, 1);
		add("TestMov-f64", testMov_REF<int64_t>,
			testMov_PTX(ir::PTXOperand::f64), testMov_INOUT(FP64),
			testMov_INOUT(FP64), uniformFloat<double, 1>, 1, 1);

		add("TestMovLabel-global-v1", testMovLabel_REF,
			testMovLabel_PTX(ir::PTXInstruction::Global,
				false, ir::PTXOperand::v1), 
			testMovLabel_INOUT(), testMovLabel_INOUT(),
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestMovLabel-global-v1-index", testMovLabel_REF,
			testMovLabel_PTX(ir::PTXInstruction::Global,
				true, ir::PTXOperand::v1), 
			testMovLabel_INOUT(), testMovLabel_INOUT(),
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestMovLabel-global-v2", testMovLabel_REF,
			testMovLabel_PTX(ir::PTXInstruction::Global,
				false, ir::PTXOperand::v2), 
			testMovLabel_INOUT(), testMovLabel_INOUT(),
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestMovLabel-global-v4", testMovLabel_REF,
			testMovLabel_PTX(ir::PTXInstruction::Global,
				false, ir::PTXOperand::v4), 
			testMovLabel_INOUT(), testMovLabel_INOUT(),
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestMovLabel-local-v1", testMovLabel_REF,
			testMovLabel_PTX(ir::PTXInstruction::Local,
				false, ir::PTXOperand::v1), 
			testMovLabel_INOUT(), testMovLabel_INOUT(),
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestMovLabel-local-v1-index", testMovLabel_REF,
			testMovLabel_PTX(ir::PTXInstruction::Local,
				true, ir::PTXOperand::v1), 
			testMovLabel_INOUT(), testMovLabel_INOUT(),
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestMovLabel-local-v2", testMovLabel_REF,
			testMovLabel_PTX(ir::PTXInstruction::Local,
				false, ir::PTXOperand::v2), 
			testMovLabel_INOUT(), testMovLabel_INOUT(),
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestMovLabel-local-v4", testMovLabel_REF,
			testMovLabel_PTX(ir::PTXInstruction::Local,
				false, ir::PTXOperand::v4), 
			testMovLabel_INOUT(), testMovLabel_INOUT(),
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestMovLabel-shared-v1", testMovLabel_REF,
			testMovLabel_PTX(ir::PTXInstruction::Shared,
				false, ir::PTXOperand::v1),
			testMovLabel_INOUT(), testMovLabel_INOUT(),
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestMovLabel-shared-v1-index", testMovLabel_REF,
			testMovLabel_PTX(ir::PTXInstruction::Shared,
				true, ir::PTXOperand::v1),
			testMovLabel_INOUT(), testMovLabel_INOUT(),
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestMovLabel-shared-v4", testMovLabel_REF,
			testMovLabel_PTX(ir::PTXInstruction::Shared,
				false, ir::PTXOperand::v4),
			testMovLabel_INOUT(), testMovLabel_INOUT(),
			uniformRandom<uint32_t, 1>, 1, 1);

		add("TestCvt-u8-u8",
			testCvt_REF<uint8_t, uint8_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u8,
				ir::PTXOperand::u8, false, false, false),
			testCvt_INOUT(I8), testCvt_INOUT(I8),
			uniformRandom<uint8_t, 1>, 1, 1);
		add("TestCvt-u8-u16",
			testCvt_REF<uint8_t, uint16_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u8,
				ir::PTXOperand::u16, false, false, false),
			testCvt_INOUT(I8), testCvt_INOUT(I16),
			uniformRandom<uint16_t, 1>, 1, 1);
		add("TestCvt-u8-u32",
			testCvt_REF<uint8_t, uint32_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u8,
				ir::PTXOperand::u32, false, false, false),
			testCvt_INOUT(I8), testCvt_INOUT(I32),
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestCvt-u8-u64",
			testCvt_REF<uint8_t, uint64_t,
				false, false, false>,
			testCvt_PTX(ir::PTXOperand::u8,
				ir::PTXOperand::u64, false, false, false),
			testCvt_INOUT(I8), testCvt_INOUT(I64),
			uniformRandom<uint64_t, 1>, 1, 1);
		add("TestCvt-u8-s8",
			testCvt_REF<uint8_t, char, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u8,
				ir::PTXOperand::s8, false, false, false),
			testCvt_INOUT(I8), testCvt_INOUT(I8),
			uniformRandom<char, 1>, 1, 1);
		add("TestCvt-u8-s16",
			testCvt_REF<uint8_t, short, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u8,
				ir::PTXOperand::s16, false, false, false),
			testCvt_INOUT(I8), testCvt_INOUT(I16),
			uniformRandom<short, 1>, 1, 1);
		add("TestCvt-u8-s32",
			testCvt_REF<uint8_t, int, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u8,
				ir::PTXOperand::s32, false, false, false),
			testCvt_INOUT(I8), testCvt_INOUT(I32),
			uniformRandom<int, 1>, 1, 1);
		add("TestCvt-u8-s64",
			testCvt_REF<uint8_t, int64_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u8,
				ir::PTXOperand::s64, false, false, false),
			testCvt_INOUT(I8), testCvt_INOUT(I64),
			uniformRandom<int64_t, 1>, 1, 1);
		add("TestCvt-u8-f32",
			testCvt_REF<uint8_t, float, false, false, true>,
			testCvt_PTX(ir::PTXOperand::u8,
				ir::PTXOperand::f32, false, false, true),
			testCvt_INOUT(I8), testCvt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestCvt-u8-f64",
			testCvt_REF<uint8_t, double, false, false, true>,
			testCvt_PTX(ir::PTXOperand::u8,
				ir::PTXOperand::f64, false, false, true),
			testCvt_INOUT(I8), testCvt_INOUT(FP64),
			uniformFloat<double, 1>, 1, 1);

		add("TestCvt-u16-u8",
			testCvt_REF<uint16_t, uint8_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u16,
				ir::PTXOperand::u8, false, false, false),
			testCvt_INOUT(I16), testCvt_INOUT(I8),
			uniformRandom<uint8_t, 1>, 1, 1);
		add("TestCvt-u16-u16",
			testCvt_REF<uint16_t, uint16_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u16,
				ir::PTXOperand::u16, false, false, false),
			testCvt_INOUT(I16), testCvt_INOUT(I16),
			uniformRandom<uint16_t, 1>, 1, 1);
		add("TestCvt-u16-u32",
			testCvt_REF<uint16_t, uint32_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u16,
				ir::PTXOperand::u32, false, false, false),
			testCvt_INOUT(I16), testCvt_INOUT(I32),
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestCvt-u16-u64",
			testCvt_REF<uint16_t, uint64_t,
				false, false, false>,
			testCvt_PTX(ir::PTXOperand::u16,
				ir::PTXOperand::u64, false, false, false),
			testCvt_INOUT(I16), testCvt_INOUT(I64),
			uniformRandom<uint64_t, 1>, 1, 1);
		add("TestCvt-u16-s8",
			testCvt_REF<uint16_t, char, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u16,
				ir::PTXOperand::s8, false, false, false),
			testCvt_INOUT(I16), testCvt_INOUT(I8),
			uniformRandom<char, 1>, 1, 1);
		add("TestCvt-u16-s16",
			testCvt_REF<uint16_t, short, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u16,
				ir::PTXOperand::s16, false, false, false),
			testCvt_INOUT(I16), testCvt_INOUT(I16),
			uniformRandom<short, 1>, 1, 1);
		add("TestCvt-u16-s32",
			testCvt_REF<uint16_t, int, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u16,
				ir::PTXOperand::s32, false, false, false),
			testCvt_INOUT(I16), testCvt_INOUT(I32),
			uniformRandom<int, 1>, 1, 1);
		add("TestCvt-u16-s64",
			testCvt_REF<uint16_t, int64_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u16,
				ir::PTXOperand::s64, false, false, false),
			testCvt_INOUT(I16), testCvt_INOUT(I64),
			uniformRandom<int64_t, 1>, 1, 1);
		add("TestCvt-u16-f32",
			testCvt_REF<uint16_t, float, false, false, true>,
			testCvt_PTX(ir::PTXOperand::u16,
				ir::PTXOperand::f32, false, false, true),
			testCvt_INOUT(I16), testCvt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestCvt-u16-f64",
			testCvt_REF<uint16_t, double, false, false, true>,
			testCvt_PTX(ir::PTXOperand::u16,
				ir::PTXOperand::f64, false, false, true),
			testCvt_INOUT(I16), testCvt_INOUT(FP64),
			uniformFloat<double, 1>, 1, 1);

		add("TestCvt-u32-u8",
			testCvt_REF<uint32_t, uint8_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u32,
				ir::PTXOperand::u8, false, false, false),
			testCvt_INOUT(I32), testCvt_INOUT(I8),
			uniformRandom<uint8_t, 1>, 1, 1);
		add("TestCvt-u32-u16",
			testCvt_REF<uint32_t, uint16_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u32,
				ir::PTXOperand::u16, false, false, false),
			testCvt_INOUT(I32), testCvt_INOUT(I16),
			uniformRandom<uint16_t, 1>, 1, 1);
		add("TestCvt-u32-u32",
			testCvt_REF<uint32_t, uint32_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u32,
				ir::PTXOperand::u32, false, false, false),
			testCvt_INOUT(I32), testCvt_INOUT(I32),
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestCvt-u32-u64",
			testCvt_REF<uint32_t, uint64_t,
				false, false, false>,
			testCvt_PTX(ir::PTXOperand::u32,
				ir::PTXOperand::u64, false, false, false),
			testCvt_INOUT(I32), testCvt_INOUT(I64),
			uniformRandom<uint64_t, 1>, 1, 1);
		add("TestCvt-u32-s8",
			testCvt_REF<uint32_t, char, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u32,
				ir::PTXOperand::s8, false, false, false),
			testCvt_INOUT(I32), testCvt_INOUT(I8),
			uniformRandom<char, 1>, 1, 1);
		add("TestCvt-u32-s16",
			testCvt_REF<uint32_t, short, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u32,
				ir::PTXOperand::s16, false, false, false),
			testCvt_INOUT(I32), testCvt_INOUT(I16),
			uniformRandom<short, 1>, 1, 1);
		add("TestCvt-u32-s32",
			testCvt_REF<uint32_t, int, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u32,
				ir::PTXOperand::s32, false, false, false),
			testCvt_INOUT(I32), testCvt_INOUT(I32),
			uniformRandom<int, 1>, 1, 1);
		add("TestCvt-u32-s64",
			testCvt_REF<uint32_t, int64_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u32,
				ir::PTXOperand::s64, false, false, false),
			testCvt_INOUT(I32), testCvt_INOUT(I64),
			uniformRandom<int64_t, 1>, 1, 1);
		add("TestCvt-u32-f32",
			testCvt_REF<uint32_t, float, false, false, true>,
			testCvt_PTX(ir::PTXOperand::u32,
				ir::PTXOperand::f32, false, false, true),
			testCvt_INOUT(I32), testCvt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestCvt-u32-f32-ftz-sat-rzi",
			testCvt_REF<uint32_t, float, true, true, true>,
			testCvt_PTX(ir::PTXOperand::u32,
				ir::PTXOperand::f32, true, true, true),
			testCvt_INOUT(I32), testCvt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestCvt-u32-f32-rmi",
			testCvt_REF<uint32_t, float, false, false, true, true>,
			testCvt_PTX(ir::PTXOperand::u32,
				ir::PTXOperand::f32, false, false, true, true),
			testCvt_INOUT(I32), testCvt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestCvt-u32-f64-rzi",
			testCvt_REF<uint32_t, double, false, false, true>,
			testCvt_PTX(ir::PTXOperand::u32,
				ir::PTXOperand::f64, false, false, true),
			testCvt_INOUT(I32), testCvt_INOUT(FP64),
			uniformFloat<double, 1>, 1, 1);

		add("TestCvt-u64-u8",
			testCvt_REF<uint64_t, uint8_t,
				false, false, false>,
			testCvt_PTX(ir::PTXOperand::u64,
				ir::PTXOperand::u8, false, false, false),
			testCvt_INOUT(I64), testCvt_INOUT(I8),
			uniformRandom<uint8_t, 1>, 1, 1);
		add("TestCvt-u64-u16",
			testCvt_REF<uint64_t,
				uint16_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u64,
				ir::PTXOperand::u16, false, false, false),
			testCvt_INOUT(I64), testCvt_INOUT(I16),
			uniformRandom<uint16_t, 1>, 1, 1);
		add("TestCvt-u64-u32",
			testCvt_REF<uint64_t, uint32_t,
				false, false, false>,
			testCvt_PTX(ir::PTXOperand::u64,
				ir::PTXOperand::u32, false, false, false),
			testCvt_INOUT(I64), testCvt_INOUT(I32),
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestCvt-u64-u64",
			testCvt_REF<uint64_t, uint64_t,
				false, false, false>,
			testCvt_PTX(ir::PTXOperand::u64,
				ir::PTXOperand::u64, false, false, false),
			testCvt_INOUT(I64), testCvt_INOUT(I64),
			uniformRandom<uint64_t, 1>, 1, 1);
		add("TestCvt-u64-s8",
			testCvt_REF<uint64_t, char, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u64,
				ir::PTXOperand::s8, false, false, false),
			testCvt_INOUT(I64), testCvt_INOUT(I8),
			uniformRandom<char, 1>, 1, 1);
		add("TestCvt-u64-s16",
			testCvt_REF<uint64_t, short, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u64,
				ir::PTXOperand::s16, false, false, false),
			testCvt_INOUT(I64), testCvt_INOUT(I16),
			uniformRandom<short, 1>, 1, 1);
		add("TestCvt-u64-s32",
			testCvt_REF<uint64_t, int, false, false, false>,
			testCvt_PTX(ir::PTXOperand::u64,
				ir::PTXOperand::s32, false, false, false),
			testCvt_INOUT(I64), testCvt_INOUT(I32),
			uniformRandom<int, 1>, 1, 1);
		add("TestCvt-u64-s64",
			testCvt_REF<uint64_t, int64_t,
				false, false, false>,
			testCvt_PTX(ir::PTXOperand::u64,
				ir::PTXOperand::s64, false, false, false),
			testCvt_INOUT(I64), testCvt_INOUT(I64),
			uniformRandom<int64_t, 1>, 1, 1);
		add("TestCvt-u64-f32",
			testCvt_REF<uint64_t, float, false, false, true>,
			testCvt_PTX(ir::PTXOperand::u64,
				ir::PTXOperand::f32, false, false, true),
			testCvt_INOUT(I64), testCvt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestCvt-u64-f64",
			testCvt_REF<uint64_t, double, false, false, true>,
			testCvt_PTX(ir::PTXOperand::u64,
				ir::PTXOperand::f64, false, false, true),
			testCvt_INOUT(I64), testCvt_INOUT(FP64),
			uniformFloat<double, 1>, 1, 1);

		add("TestCvt-s8-u8",
			testCvt_REF<char, uint8_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s8,
				ir::PTXOperand::u8, false, false, false),
			testCvt_INOUT(I8), testCvt_INOUT(I8),
			uniformRandom<uint8_t, 1>, 1, 1);
		add("TestCvt-s8-u16",
			testCvt_REF<char, uint16_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s8,
				ir::PTXOperand::u16, false, false, false),
			testCvt_INOUT(I8), testCvt_INOUT(I16),
			uniformRandom<uint16_t, 1>, 1, 1);
		add("TestCvt-s8-u32",
			testCvt_REF<char, uint32_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s8,
				ir::PTXOperand::u32, false, false, false),
			testCvt_INOUT(I8), testCvt_INOUT(I32),
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestCvt-s8-u64",
			testCvt_REF<char, uint64_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s8,
				ir::PTXOperand::u64, false, false, false),
			testCvt_INOUT(I8), testCvt_INOUT(I64),
			uniformRandom<uint64_t, 1>, 1, 1);
		add("TestCvt-s8-s8",
			testCvt_REF<char, char, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s8,
				ir::PTXOperand::s8, false, false, false),
			testCvt_INOUT(I8), testCvt_INOUT(I8),
			uniformRandom<char, 1>, 1, 1);
		add("TestCvt-s8-s16",
			testCvt_REF<char, short, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s8,
				ir::PTXOperand::s16, false, false, false),
			testCvt_INOUT(I8), testCvt_INOUT(I16),
			uniformRandom<short, 1>, 1, 1);
		add("TestCvt-s8-s32",
			testCvt_REF<char, int, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s8,
				ir::PTXOperand::s32, false, false, false),
			testCvt_INOUT(I8), testCvt_INOUT(I32),
			uniformRandom<int, 1>, 1, 1);
		add("TestCvt-s8-s64",
			testCvt_REF<char, int64_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s8,
				ir::PTXOperand::s64, false, false, false),
			testCvt_INOUT(I8), testCvt_INOUT(I64),
			uniformRandom<int64_t, 1>, 1, 1);
		add("TestCvt-s8-f32",
			testCvt_REF<char, float, false, false, true>,
			testCvt_PTX(ir::PTXOperand::s8,
				ir::PTXOperand::f32, false, false, true),
			testCvt_INOUT(I8), testCvt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestCvt-s8-f64",
			testCvt_REF<char, double, false, false, true>,
			testCvt_PTX(ir::PTXOperand::s8,
				ir::PTXOperand::f64, false, false, true),
			testCvt_INOUT(I8), testCvt_INOUT(FP64),
			uniformFloat<double, 1>, 1, 1);

		add("TestCvt-s16-u8",
			testCvt_REF<short, uint8_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s16,
				ir::PTXOperand::u8, false, false, false),
			testCvt_INOUT(I16), testCvt_INOUT(I8),
			uniformRandom<uint8_t, 1>, 1, 1);
		add("TestCvt-s16-u16",
			testCvt_REF<short, uint16_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s16,
				ir::PTXOperand::u16, false, false, false),
			testCvt_INOUT(I16), testCvt_INOUT(I16),
			uniformRandom<uint16_t, 1>, 1, 1);
		add("TestCvt-s16-u32",
			testCvt_REF<short, uint32_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s16,
				ir::PTXOperand::u32, false, false, false),
			testCvt_INOUT(I16), testCvt_INOUT(I32),
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestCvt-s16-u64",
			testCvt_REF<short, uint64_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s16,
				ir::PTXOperand::u64, false, false, false),
			testCvt_INOUT(I16), testCvt_INOUT(I64),
			uniformRandom<uint64_t, 1>, 1, 1);
		add("TestCvt-s16-s8",
			testCvt_REF<short, char, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s16,
				ir::PTXOperand::s8, false, false, false),
			testCvt_INOUT(I16), testCvt_INOUT(I8),
			uniformRandom<char, 1>, 1, 1);
		add("TestCvt-s16-s16",
			testCvt_REF<short, short, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s16,
				ir::PTXOperand::s16, false, false, false),
			testCvt_INOUT(I16), testCvt_INOUT(I16),
			uniformRandom<short, 1>, 1, 1);
		add("TestCvt-s16-s32",
			testCvt_REF<short, int, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s16,
				ir::PTXOperand::s32, false, false, false),
			testCvt_INOUT(I16), testCvt_INOUT(I32),
			uniformRandom<int, 1>, 1, 1);
		add("TestCvt-s16-s64",
			testCvt_REF<short, int64_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s16,
				ir::PTXOperand::s64, false, false, false),
			testCvt_INOUT(I16), testCvt_INOUT(I64),
			uniformRandom<int64_t, 1>, 1, 1);
		add("TestCvt-s16-f32",
			testCvt_REF<short, float, false, false, true>,
			testCvt_PTX(ir::PTXOperand::s16,
				ir::PTXOperand::f32, false, false, true),
			testCvt_INOUT(I16), testCvt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestCvt-s16-f64",
			testCvt_REF<short, double, false, false, true>,
			testCvt_PTX(ir::PTXOperand::s16,
				ir::PTXOperand::f64, false, false, true),
			testCvt_INOUT(I16), testCvt_INOUT(FP64),
			uniformFloat<double, 1>, 1, 1);

		add("TestCvt-s32-u8",
			testCvt_REF<int, uint8_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s32,
				ir::PTXOperand::u8, false, false, false),
			testCvt_INOUT(I32), testCvt_INOUT(I8),
			uniformRandom<uint8_t, 1>, 1, 1);
		add("TestCvt-s32-u16",
			testCvt_REF<int, uint16_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s32,
				ir::PTXOperand::u16, false, false, false),
			testCvt_INOUT(I32), testCvt_INOUT(I16),
			uniformRandom<uint16_t, 1>, 1, 1);
		add("TestCvt-s32-u32",
			testCvt_REF<int, uint32_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s32,
				ir::PTXOperand::u32, false, false, false),
			testCvt_INOUT(I32), testCvt_INOUT(I32),
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestCvt-s32-u64",
			testCvt_REF<int, uint64_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s32,
				ir::PTXOperand::u64, false, false, false),
			testCvt_INOUT(I32), testCvt_INOUT(I64),
			uniformRandom<uint64_t, 1>, 1, 1);
		add("TestCvt-s32-s8",
			testCvt_REF<int, char, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s32,
				ir::PTXOperand::s8, false, false, false),
			testCvt_INOUT(I32), testCvt_INOUT(I8),
			uniformRandom<char, 1>, 1, 1);
		add("TestCvt-s32-s16",
			testCvt_REF<int, short, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s32,
				ir::PTXOperand::s16, false, false, false),
			testCvt_INOUT(I32), testCvt_INOUT(I16),
			uniformRandom<short, 1>, 1, 1);
		add("TestCvt-s32-s32",
			testCvt_REF<int, int, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s32,
				ir::PTXOperand::s32, false, false, false),
			testCvt_INOUT(I32), testCvt_INOUT(I32),
			uniformRandom<int, 1>, 1, 1);
		add("TestCvt-s32-s64",
			testCvt_REF<int, int64_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s32,
				ir::PTXOperand::s64, false, false, false),
			testCvt_INOUT(I32), testCvt_INOUT(I64),
			uniformRandom<int64_t, 1>, 1, 1);
		add("TestCvt-s32-f32",
			testCvt_REF<int, float, false, false, true>,
			testCvt_PTX(ir::PTXOperand::s32,
				ir::PTXOperand::f32, false, false, true),
			testCvt_INOUT(I32), testCvt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestCvt-s32-f64",
			testCvt_REF<int, double, false, false, true>,
			testCvt_PTX(ir::PTXOperand::s32,
				ir::PTXOperand::f64, false, false, true),
			testCvt_INOUT(I32), testCvt_INOUT(FP64),
			uniformFloat<double, 1>, 1, 1);

		add("TestCvt-s64-u8",
			testCvt_REF<int64_t, uint8_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s64,
				ir::PTXOperand::u8, false, false, false),
			testCvt_INOUT(I64), testCvt_INOUT(I8),
			uniformRandom<uint8_t, 1>, 1, 1);
		add("TestCvt-s64-u16",
			testCvt_REF<int64_t, uint16_t, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s64,
				ir::PTXOperand::u16, false, false, false),
			testCvt_INOUT(I64), testCvt_INOUT(I16),
			uniformRandom<uint16_t, 1>, 1, 1);
		add("TestCvt-s64-u32",
			testCvt_REF<int64_t, uint32_t,
				false, false, false>,
			testCvt_PTX(ir::PTXOperand::s64,
				ir::PTXOperand::u32, false, false, false),
			testCvt_INOUT(I64), testCvt_INOUT(I32),
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestCvt-s64-u64",
			testCvt_REF<int64_t, uint64_t,
				false, false, false>,
			testCvt_PTX(ir::PTXOperand::s64,
				ir::PTXOperand::u64, false, false, false),
			testCvt_INOUT(I64), testCvt_INOUT(I64),
			uniformRandom<uint64_t, 1>, 1, 1);
		add("TestCvt-s64-s8",
			testCvt_REF<int64_t, char, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s64,
				ir::PTXOperand::s8, false, false, false),
			testCvt_INOUT(I64), testCvt_INOUT(I8),
			uniformRandom<char, 1>, 1, 1);
		add("TestCvt-s64-s16",
			testCvt_REF<int64_t, short, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s64,
				ir::PTXOperand::s16, false, false, false),
			testCvt_INOUT(I64), testCvt_INOUT(I16),
			uniformRandom<short, 1>, 1, 1);
		add("TestCvt-s64-s32",
			testCvt_REF<int64_t, int, false, false, false>,
			testCvt_PTX(ir::PTXOperand::s64,
				ir::PTXOperand::s32, false, false, false),
			testCvt_INOUT(I64), testCvt_INOUT(I32),
			uniformRandom<int, 1>, 1, 1);
		add("TestCvt-s64-s64",
			testCvt_REF<int64_t, int64_t,
				false, false, false>,
			testCvt_PTX(ir::PTXOperand::s64,
				ir::PTXOperand::s64, false, false, false),
			testCvt_INOUT(I64), testCvt_INOUT(I64),
			uniformRandom<int64_t, 1>, 1, 1);
		add("TestCvt-s64-f32",
			testCvt_REF<int64_t, float, false, false, true>,
			testCvt_PTX(ir::PTXOperand::s64,
				ir::PTXOperand::f32, false, false, true),
			testCvt_INOUT(I64), testCvt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestCvt-s64-f64",
			testCvt_REF<int64_t, double, false, false, true>,
			testCvt_PTX(ir::PTXOperand::s64,
				ir::PTXOperand::f64, false, false, true),
			testCvt_INOUT(I64), testCvt_INOUT(FP64),
			uniformFloat<double, 1>, 1, 1);

		add("TestCvt-f32-u8",
			testCvt_REF<float, uint8_t, false, false, true>,
			testCvt_PTX(ir::PTXOperand::f32,
				ir::PTXOperand::u8, false, false, true),
			testCvt_INOUT(FP32), testCvt_INOUT(I8),
			uniformRandom<uint8_t, 1>, 1, 1);
		add("TestCvt-f32-u16",
			testCvt_REF<float, uint16_t, false, false, true>,
			testCvt_PTX(ir::PTXOperand::f32,
				ir::PTXOperand::u16, false, false, true),
			testCvt_INOUT(FP32), testCvt_INOUT(I16),
			uniformRandom<uint16_t, 1>, 1, 1);
		add("TestCvt-f32-u32",
			testCvt_REF<float, uint32_t, false, false, true>,
			testCvt_PTX(ir::PTXOperand::f32,
				ir::PTXOperand::u32, false, false, true),
			testCvt_INOUT(FP32), testCvt_INOUT(I32),
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestCvt-f32-u64",
			testCvt_REF<float, uint64_t, false, false, true>,
			testCvt_PTX(ir::PTXOperand::f32,
				ir::PTXOperand::u64, false, false, true),
			testCvt_INOUT(FP32), testCvt_INOUT(I64),
			uniformRandom<uint64_t, 1>, 1, 1);
		add("TestCvt-f32-s8",
			testCvt_REF<float, char, false, false, true>,
			testCvt_PTX(ir::PTXOperand::f32,
				ir::PTXOperand::s8, false, false, true),
			testCvt_INOUT(FP32), testCvt_INOUT(I8),
			uniformRandom<char, 1>, 1, 1);
		add("TestCvt-f32-s16",
			testCvt_REF<float, short, false, false, true>,
			testCvt_PTX(ir::PTXOperand::f32,
				ir::PTXOperand::s16, false, false, true),
			testCvt_INOUT(FP32), testCvt_INOUT(I16),
			uniformRandom<short, 1>, 1, 1);
		add("TestCvt-f32-s32",
			testCvt_REF<float, int, false, false, true>,
			testCvt_PTX(ir::PTXOperand::f32,
				ir::PTXOperand::s32, false, false, true),
			testCvt_INOUT(FP32), testCvt_INOUT(I32),
			uniformRandom<int, 1>, 1, 1);
		add("TestCvt-f32-s64",
			testCvt_REF<float, int64_t, false, false, true>,
			testCvt_PTX(ir::PTXOperand::f32,
				ir::PTXOperand::s64, false, false, true),
			testCvt_INOUT(FP32), testCvt_INOUT(I64),
			uniformRandom<int64_t, 1>, 1, 1);

		add("TestCvt-f32-f32-sat",
			testCvt_REF<float, float, false, true, false>,
			testCvt_PTX(ir::PTXOperand::f32,
				ir::PTXOperand::f32, false, true, false),
			testCvt_INOUT(FP32), testCvt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestCvt-f32-f32-rmi",
			testCvt_REF<float, float, false, false, true, true>,
			testCvt_PTX(ir::PTXOperand::f32,
				ir::PTXOperand::f32, false, false, true, true),
			testCvt_INOUT(FP32), testCvt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestCvt-f32-f32-rzi",
			testCvt_REF<float, float, false, false, true>,
			testCvt_PTX(ir::PTXOperand::f32,
				ir::PTXOperand::f32, false, false, true),
			testCvt_INOUT(FP32), testCvt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestCvt-f32-f64",
			testCvt_REF<float, double, false, false, true>,
			testCvt_PTX(ir::PTXOperand::f32,
				ir::PTXOperand::f64, false, false, true),
			testCvt_INOUT(FP32), testCvt_INOUT(FP64),
			uniformFloat<double, 1>, 1, 1);

		add("TestCvt-f64-u8",
			testCvt_REF<double, uint8_t, false, false, true>,
			testCvt_PTX(ir::PTXOperand::f64,
				ir::PTXOperand::u8, false, false, true),
			testCvt_INOUT(FP64), testCvt_INOUT(I8),
			uniformRandom<uint8_t, 1>, 1, 1);
		add("TestCvt-f64-u16",
			testCvt_REF<double, uint16_t, false, false, true>,
			testCvt_PTX(ir::PTXOperand::f64,
				ir::PTXOperand::u16, false, false, true),
			testCvt_INOUT(FP64), testCvt_INOUT(I16),
			uniformRandom<uint16_t, 1>, 1, 1);
		add("TestCvt-f64-u32",
			testCvt_REF<double, uint32_t, false, false, true>,
			testCvt_PTX(ir::PTXOperand::f64,
				ir::PTXOperand::u32, false, false, true),
			testCvt_INOUT(FP64), testCvt_INOUT(I32),
			uniformRandom<uint32_t, 1>, 1, 1);
		add("TestCvt-f64-u64",
			testCvt_REF<double, uint64_t, false, false, true>,
			testCvt_PTX(ir::PTXOperand::f64,
				ir::PTXOperand::u64, false, false, true),
			testCvt_INOUT(FP64), testCvt_INOUT(I64),
			uniformRandom<uint64_t, 1>, 1, 1);
		add("TestCvt-f64-s8",
			testCvt_REF<double, char, false, false, true>,
			testCvt_PTX(ir::PTXOperand::f64,
				ir::PTXOperand::s8, false, false, true),
			testCvt_INOUT(FP64), testCvt_INOUT(I8),
			uniformRandom<char, 1>, 1, 1);
		add("TestCvt-f64-s16",
			testCvt_REF<double, short, false, false, true>,
			testCvt_PTX(ir::PTXOperand::f64,
				ir::PTXOperand::s16, false, false, true),
			testCvt_INOUT(FP64), testCvt_INOUT(I16),
			uniformRandom<short, 1>, 1, 1);
		add("TestCvt-f64-s32",
			testCvt_REF<double, int, false, false, true>,
			testCvt_PTX(ir::PTXOperand::f64,
				ir::PTXOperand::s32, false, false, true),
			testCvt_INOUT(FP64), testCvt_INOUT(I32),
			uniformRandom<int, 1>, 1, 1);
		add("TestCvt-f64-s64",
			testCvt_REF<double, int64_t,
				false, false, true>,
			testCvt_PTX(ir::PTXOperand::f64,
				ir::PTXOperand::s64, false, false, true),
			testCvt_INOUT(FP64), testCvt_INOUT(I64),
			uniformRandom<int64_t, 1>, 1, 1);
		add("TestCvt-f64-f32",
			testCvt_REF<double, float, false, false, false>,
			testCvt_PTX(ir::PTXOperand::f64,
				ir::PTXOperand::f32, false, false, false),
			testCvt_INOUT(FP64), testCvt_INOUT(FP32),
			uniformFloat<float, 1>, 1, 1);
		add("TestCvt-f64-f64-rzi",
			testCvt_REF<double, double, false, false, true>,
			testCvt_PTX(ir::PTXOperand::f64,
				ir::PTXOperand::f64, false, false, true),
			testCvt_INOUT(FP64), testCvt_INOUT(FP64),
			uniformFloat<double, 1>, 1, 1);
	
		add("TestLocalMemory-u8",
			testLocalMemory_REF<uint8_t>,
			testLocalMemory_PTX(ir::PTXOperand::u8, false, false),
			testLocalMemory_INOUT(I8), testLocalMemory_INOUT(I8),
			uniformFloat<uint8_t, 1>, 1, 1);
		add("TestLocalMemory-u8-global",
			testLocalMemory_REF<uint8_t>,
			testLocalMemory_PTX(ir::PTXOperand::u8, true, false),
			testLocalMemory_INOUT(I8), testLocalMemory_INOUT(I8),
			uniformFloat<uint8_t, 1>, 1, 1);
		add("TestLocalMemory-u8-scoped",
			testLocalMemory_REF<uint8_t>,
			testLocalMemory_PTX(ir::PTXOperand::u8, false, true),
			testLocalMemory_INOUT(I8), testLocalMemory_INOUT(I8),
			uniformFloat<uint8_t, 1>, 1, 1);
		add("TestLocalMemory-u8-global-scoped",
			testLocalMemory_REF<uint8_t>,
			testLocalMemory_PTX(ir::PTXOperand::u8, true, true),
			testLocalMemory_INOUT(I8), testLocalMemory_INOUT(I8),
			uniformFloat<uint8_t, 1>, 1, 1);

		if(sizeof(size_t) == 4)
		{
			add("TestCvta-local-u32",
				testCvta_REF<uint32_t>,
				testCvta_PTX(ir::PTXOperand::u32, ir::PTXInstruction::Local),
				testCvta_INOUT(I32), testCvta_INOUT(I32),
				uniformFloat<uint32_t, 1>, 1, 1);
			add("TestCvta-global-u32",
				testCvta_REF<uint32_t>,
				testCvta_PTX(ir::PTXOperand::u32, ir::PTXInstruction::Global),
				testCvta_INOUT(I32), testCvta_INOUT(I32),
				uniformFloat<uint32_t, 1>, 1, 1);
			add("TestCvta-shared-u32",
				testCvta_REF<uint32_t>,
				testCvta_PTX(ir::PTXOperand::u32, ir::PTXInstruction::Shared),
				testCvta_INOUT(I32), testCvta_INOUT(I32),
				uniformFloat<uint32_t, 1>, 1, 1);
		}
		
		add("TestCvta-local-u64",
			testCvta_REF<uint64_t>,
			testCvta_PTX(ir::PTXOperand::u64, ir::PTXInstruction::Local),
			testCvta_INOUT(I64), testCvta_INOUT(I64),
			uniformFloat<uint64_t, 1>, 1, 1);

		add("TestCvta-global-u64",
			testCvta_REF<uint64_t>,
			testCvta_PTX(ir::PTXOperand::u64, ir::PTXInstruction::Global),
			testCvta_INOUT(I64), testCvta_INOUT(I64),
			uniformFloat<uint64_t, 1>, 1, 1);

		
		add("TestCvta-shared-u64",
			testCvta_REF<uint64_t>,
			testCvta_PTX(ir::PTXOperand::u64, ir::PTXInstruction::Shared),
			testCvta_INOUT(I64), testCvta_INOUT(I64),
			uniformFloat<uint64_t, 1>, 1, 1);

	}

	TestPTXAssembly::TestPTXAssembly(hydrazine::Timer::Second l, 
		uint32_t t) : _tolerableFailures(t), timeLimit(l)
	{
		name = "TestPTXAssembly";
		
		description = "A unit test framework for PTX. Runs random inputs ";
		description += "through unit tests on all available devices until ";
		description += "a timer expires.";		
	}
	
	void TestPTXAssembly::add(const std::string& name, 
		ReferenceFunction function, const std::string& ptx, 
		const TypeVector& out, const TypeVector& in, 
		GeneratorFunction generator, uint32_t threads, uint32_t ctas,
		int epsilon)
	{
		// TODO change this to std::tr1::regex when gcc gets its act together
		if(!regularExpression.empty() &&
			name.find(regularExpression) == std::string::npos) return;
		
		if(enumerate)
		{
			std::cout << name << "\n";
			return;
		}
		
		TestHandle test;
		test.name = name;
		test.reference = function;
		test.generator = generator;
		test.ptx = ptx;
		test.inputTypes = in;
		test.outputTypes = out;
		test.threads = threads;
		test.ctas = ctas;
		test.epsilon = epsilon;
		
		if(print)
		{
			std::cout << "Added test - '" << name << "'\n";
			std::cout << " threads: " << threads << "\n";
			std::cout << " ctas: " << ctas << "\n";
			std::cout << " ptx:\n";
			std::cout << ptx << "\n";
		}
		
		_tests.push_back(std::move(test));
	}	
	
	bool TestPTXAssembly::doTest()
	{

		executive::DeviceProperties properties;
		ocelot::getDeviceProperties(properties);
	
		_loadTests(properties.ISA);
		
		hydrazine::Timer::Second perTestTimeLimit = timeLimit / _tests.size();
		hydrazine::Timer timer;
		
		uint32_t failures = 0;
		
		for(TestVector::iterator test = _tests.begin(); 
			test != _tests.end(); ++test)
		{
			timer.stop();
			timer.start();
			uint32_t i = 0;
			for( ; timer.seconds() < perTestTimeLimit; ++i)
			{
				bool result = _doOneTest(*test, seed + i);
				
				if(!result)
				{
					status << "Test '" << test->name << "' seed '" 
						<< (seed + i) << "' failed.\n";
					if(++failures > _tolerableFailures) return false;
				}
				
				timer.stop();
			}
			status << "Ran '" << test->name << "' for " 
				<< i << " iterations.\n";
		}
		
		return failures == 0;
	}
}

int main(int argc, char** argv)
{
	hydrazine::ArgumentParser parser(argc, argv);
	test::TestPTXAssembly test;
	parser.description(test.testDescription());

	parser.parse("-v", "--verbose", test.verbose, false,
		"Print out status info after the test.");
	parser.parse("-V", "--very-verbose", test.veryVerbose, false,
		"Print out information as the test is running.");
	parser.parse("-p", "--print-ptx", test.print, false,
		"Print test kernels as they are added.");
	parser.parse("-e", "--enumerate", test.enumerate, false,
		"Only enumerate tests, do not run them.");
	parser.parse("-t", "--test", test.regularExpression, "",
		"Only select tests matching this expression.");
	parser.parse("-s", "--seed", test.seed, 0,
		"Random seed for generating input data. 0 implies seed with time.");
	parser.parse("-l", "--time-limit", test.timeLimit, 10, 
		"How many seconds to run tests.");
	parser.parse();

	test.test();
	
	return test.passed();
}

#endif

