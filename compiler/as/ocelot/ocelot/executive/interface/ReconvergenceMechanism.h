/*!
	\file ReconvergenceMechanism.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\date Nov 15, 2010
	\brief extracts the reconvergence mechanism from CooperativeThreadArray
*/

#ifndef OCELOT_EXECUTIVE_RECONVERGENCEMECHANISM_H_INCLUDED
#define OCELOT_EXECUTIVE_RECONVERGENCEMECHANISM_H_INCLUDED

// C++ includes
#include <deque>

// Ocelot includes
#include <ocelot/executive/interface/CTAContext.h>
#include <ocelot/ir/interface/PTXOperand.h>
#include <ocelot/ir/interface/Kernel.h>
#include <ocelot/ir/interface/Texture.h>
#include <ocelot/trace/interface/TraceEvent.h>

////////////////////////////////////////////////////////////////////////////////

namespace executive {

class EmulatedKernel;
class CooperativeThreadArray;

/*!
	\brief base class for abstract reconvergence mechanism within emulator
*/
class ReconvergenceMechanism {
public:
	enum Type {
		Reconverge_IPDOM,
		Reconverge_Barrier,
		Reconverge_TFGen6,
		Reconverge_TFSortedStack,
		Reconverge_TFSoftware,
		Reconverge_unknown
	};
		
public:
	ReconvergenceMechanism(CooperativeThreadArray *cta);
	
public:

	virtual ~ReconvergenceMechanism();

	//! \brief initializes the reconvergence mechanism
	virtual void initialize() = 0;

	//! \brief updates the predicate mask of the active context
	// before instructions execute
	virtual void evalPredicate(CTAContext &context) = 0;
	
	/*! 
		\brief implements branch instruction and updates CTA state
		\return true on divergent branch
	*/
	virtual bool eval_Bra(CTAContext &context, 
		const ir::PTXInstruction &instr, 
		const boost::dynamic_bitset<> & branch, 
		const boost::dynamic_bitset<> & fallthrough) = 0;

	/*! 
		\brief implements a barrier instruction
	*/
	virtual void eval_Bar(CTAContext &context,
		const ir::PTXInstruction &instr) = 0;
	
	/*!
		\brief implements reconverge instruction
	*/
	virtual void eval_Reconverge(CTAContext &context,
		const ir::PTXInstruction &instr) = 0;
	
	/*!
		\brief implements exit instruction
	*/
	virtual void eval_Exit(CTAContext &context,
		const ir::PTXInstruction &instr) = 0;

	/*!
		\brief implements vote instruction
	*/
	virtual void eval_Vote(CTAContext &context,
		const ir::PTXInstruction &instr);

	/*! 
		\brief updates the active context to the next instruction
	*/
	virtual bool nextInstruction(CTAContext &context,
		const ir::PTXInstruction &instr,
		const ir::PTXInstruction::Opcode &) = 0;
	
	//! \brief gets the active context
	virtual CTAContext& getContext() = 0;
	
	//! \brief gets the stack size
	virtual size_t stackSize() const = 0;

	//! \brief push a context
	virtual void push(CTAContext&) = 0;

	//! \brief pop a context
	virtual void pop() = 0;
	
	//! \brief gets the reconvergence mechanism type
	Type getType() const { return type; }

	//! \brief gets a string-representation of the type
	static std::string toString(Type type);

protected:

	//! \brief dynamic type information for convergence mechanism
	Type type;
	
	//! \brief executing CTA
	CooperativeThreadArray *cta;
};

//
//
//

class ReconvergenceIPDOM: public ReconvergenceMechanism {
public:

	ReconvergenceIPDOM(CooperativeThreadArray *cta);
	~ReconvergenceIPDOM();
	
	void initialize();
	void evalPredicate(CTAContext &context);
	bool eval_Bra(CTAContext &context, 
		const ir::PTXInstruction &instr, 
		const boost::dynamic_bitset<> & branch, 
		const boost::dynamic_bitset<> & fallthrough);
	void eval_Bar(CTAContext &context,
		const ir::PTXInstruction &instr);
	void eval_Reconverge(CTAContext &context,
		const ir::PTXInstruction &instr);
	void eval_Exit(CTAContext &context,
		const ir::PTXInstruction &instr);
	bool nextInstruction(CTAContext &context,
		const ir::PTXInstruction &instr, const ir::PTXInstruction::Opcode &);

	CTAContext& getContext();
	size_t stackSize() const;
	void push(CTAContext&);
	void pop();
	
private:
	enum Token
	{
		Call,
		Branch
	};

	typedef std::vector<CTAContext> RuntimeStack;
	typedef std::vector<int>        PCStack;
	typedef std::vector<Token>      TokenStack;

private:
	static std::string toString(Token t);

private:
	//! \brief context stack
	RuntimeStack runtimeStack;
	//! \brief stack of reconvergence points for the current context
	PCStack pcStack;
	TokenStack tokenStack;
	unsigned int reconvergeEvents;
};

class ReconvergenceBarrier: public ReconvergenceMechanism {
public:

	ReconvergenceBarrier(CooperativeThreadArray *cta);
	
	void initialize();
	void evalPredicate(CTAContext &context);
	bool eval_Bra(CTAContext &context, 
		const ir::PTXInstruction &instr, 
		const boost::dynamic_bitset<> & branch, 
		const boost::dynamic_bitset<> & fallthrough);
	void eval_Bar(CTAContext &context,
		const ir::PTXInstruction &instr);
	void eval_Reconverge(CTAContext &context,
		const ir::PTXInstruction &instr);
	void eval_Exit(CTAContext &context,
		const ir::PTXInstruction &instr);
	bool nextInstruction(CTAContext &context,
		const ir::PTXInstruction &instr, const ir::PTXInstruction::Opcode &);

	CTAContext& getContext();
	size_t stackSize() const;
	void push(CTAContext&);
	void pop();
	
private:
	typedef std::vector<CTAContext> RuntimeStack;

private:
	//! \brief context stack
	RuntimeStack runtimeStack;
};


class ReconvergenceTFGen6: public ReconvergenceMechanism {
public:
	typedef std::vector <int> ThreadIdVector;
	
public:
	ReconvergenceTFGen6(CooperativeThreadArray *cta);

	void initialize();
	void evalPredicate(CTAContext &context);
	bool eval_Bra(CTAContext &context, 
		const ir::PTXInstruction &instr, 
		const boost::dynamic_bitset<> & branch, 
		const boost::dynamic_bitset<> & fallthrough);
	void eval_Bar(CTAContext &context,
		const ir::PTXInstruction &instr);
	void eval_Reconverge(CTAContext &context,
		const ir::PTXInstruction &instr);
	void eval_Exit(CTAContext &context,
		const ir::PTXInstruction &instr);
	bool nextInstruction(CTAContext &context,
		const ir::PTXInstruction &instr, const ir::PTXInstruction::Opcode &);

	CTAContext& getContext();
	size_t stackSize() const;
	void push(CTAContext&);
	void pop();
	
private:
	typedef std::vector<CTAContext> RuntimeStack;

private:
	//! \brief context stack
	RuntimeStack runtimeStack;

	//! \brief program counters for each thread
	ThreadIdVector threadPCs;
};

class ReconvergenceTFSortedStack: public ReconvergenceMechanism {
public:
	ReconvergenceTFSortedStack(CooperativeThreadArray *cta);
	~ReconvergenceTFSortedStack();

	void initialize();
	void evalPredicate(CTAContext &context);
	bool eval_Bra(CTAContext &context, 
		const ir::PTXInstruction &instr, 
		const boost::dynamic_bitset<> & branch, 
		const boost::dynamic_bitset<> & fallthrough);
	void eval_Bar(CTAContext &context,
		const ir::PTXInstruction &instr);
	void eval_Reconverge(CTAContext &context,
		const ir::PTXInstruction &instr);
	void eval_Exit(CTAContext &context,
		const ir::PTXInstruction &instr);
	bool nextInstruction(CTAContext &context,
		const ir::PTXInstruction &instr, const ir::PTXInstruction::Opcode &);

	CTAContext& getContext();
	size_t stackSize() const;
	void push(CTAContext&);
	void pop();

private:
	typedef std::map<int, CTAContext> RuntimeStack;
	typedef std::vector<RuntimeStack> StackVector; 
		
public:
	StackVector stack;
	unsigned int reconvergeEvents;
};

class ReconvergenceTFSoftware: public ReconvergenceMechanism {
public:
	ReconvergenceTFSoftware(CooperativeThreadArray *cta);
	~ReconvergenceTFSoftware();

	void initialize();
	void evalPredicate(CTAContext &context);
	bool eval_Bra(CTAContext &context, 
		const ir::PTXInstruction &instr, 
		const boost::dynamic_bitset<> & branch, 
		const boost::dynamic_bitset<> & fallthrough);
	void eval_Bar(CTAContext &context,
		const ir::PTXInstruction &instr);
	void eval_Reconverge(CTAContext &context,
		const ir::PTXInstruction &instr);
	void eval_Exit(CTAContext &context,
		const ir::PTXInstruction &instr);
	void eval_Vote(CTAContext &context,
		const ir::PTXInstruction &instr);
	bool nextInstruction(CTAContext &context,
		const ir::PTXInstruction &instr, const ir::PTXInstruction::Opcode &);

	CTAContext& getContext();
	size_t stackSize() const;
	void push(CTAContext&);
	void pop();

public:
	int warpSize;

private:
	typedef std::vector<CTAContext> ContextStack; 
		
private:
	ContextStack stack;

};
}

////////////////////////////////////////////////////////////////////////////////

#endif

