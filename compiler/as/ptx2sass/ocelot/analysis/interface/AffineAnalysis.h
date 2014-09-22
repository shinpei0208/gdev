/*! \file AffineAnalysis.h
 \date May 21, 2011
 \author Diogo Sampaio <dnsampaio@gmail.com>
 \author Fernando Pereira <fpereira@dcc.ufmg.br>
 \author Sylvain Collange <sylvain.collange@gmail.com>
 \brief The header file for the AffineAnalysis class
 */

#ifndef AFFINE_ANALYSIS_H_
#define AFFINE_ANALYSIS_H_

//#include <ocelot/analysis/interface/SparseAnalysis.h>
#include <map>
#include <queue>
#include <ocelot/analysis/interface/Analysis.h>
#include <ocelot/analysis/interface/DivergenceGraph.h>
#include <ocelot/analysis/interface/PostdominatorTree.h>
#include <ocelot/ir/interface/Module.h>

using namespace std;

namespace analysis {
  typedef int const_t;
  struct ConstantAbstractState {
    public:
      enum State {
        Bottom, Constant, Top
      };

      ConstantAbstractState() :
          type(Top), value(0) {
      }
      ConstantAbstractState(State type, const_t value) :
          type(type), value(value) {
      }
      ConstantAbstractState(const_t value) :
          type(Constant), value(value) {
      }

      ConstantAbstractState Meet(ConstantAbstractState const & other) const;
      bool operator==(ConstantAbstractState const & other) {
        return (type == other.type) && (type == Top || type == Bottom || value == other.value);
      }
      bool operator==(ConstantAbstractState const & other) const {
        return (type == other.type) && (type == Top || type == Bottom || value == other.value);
      }
      bool operator!=(ConstantAbstractState const & other) {
        return !(*this == other);
      }

      bool operator!=(ConstantAbstractState const & other) const {
        return !(*this == other);
      }

      ConstantAbstractState const & operator+=(ConstantAbstractState const & other);
      ConstantAbstractState const & operator-=(ConstantAbstractState const & other);
      ConstantAbstractState const & operator*=(ConstantAbstractState const & other);

      State type;
      const_t value;

      static ConstantAbstractState const top;
      static ConstantAbstractState const bottom;
      static ConstantAbstractState const zero;
      static ConstantAbstractState const one;

  };
  inline ConstantAbstractState operator+(ConstantAbstractState const & a, ConstantAbstractState const & b) {
    ConstantAbstractState r = a;
    return r += b;
  }
  inline ConstantAbstractState operator-(ConstantAbstractState const & a, ConstantAbstractState const & b) {
    ConstantAbstractState r = a;
    return r -= b;
  }
  inline ConstantAbstractState operator*(ConstantAbstractState const & a, ConstantAbstractState const & b) {
    ConstantAbstractState r = a;
    return r *= b;
  }

  std::ostream & operator<<(std::ostream & s, ConstantAbstractState const & as);

  struct AffineAbstractState {
      AffineAbstractState(ConstantAbstractState const & base, ConstantAbstractState const * s) :
          base(base) {
        stride[0] = s[0];
        stride[1] = s[1];
      }
      AffineAbstractState(ConstantAbstractState const & base = ConstantAbstractState::top,
        ConstantAbstractState const s0 = ConstantAbstractState::top, ConstantAbstractState const s1 =
            ConstantAbstractState::top) :
          base(base) {
        stride[0] = s0;
        stride[1] = s1;
      }
      AffineAbstractState(const_t constant) :
          base(constant) {
        stride[0] = ConstantAbstractState::zero;
        stride[1] = ConstantAbstractState::zero;
      }

      bool operator==(AffineAbstractState const & other) const {
        return (base == other.base) && (stride[0] == other.stride[0]) && (stride[1] == other.stride[1]);
      }
      bool operator!=(AffineAbstractState const & other) const {
        return !(*this == other);
      }

      AffineAbstractState const & operator+=(AffineAbstractState const & other);
      AffineAbstractState const & operator-=(AffineAbstractState const & other);
      AffineAbstractState const & operator*=(AffineAbstractState const & other);

      // Non-linear operation: modifies base
      AffineAbstractState const & operator^=(AffineAbstractState const & other);
      AffineAbstractState NonLinear() const;

      AffineAbstractState const & operator&=(AffineAbstractState const & other);
      AffineAbstractState Meet(AffineAbstractState const & other) const;

      ConstantAbstractState base;
      ConstantAbstractState stride[2];

      static AffineAbstractState const top;
      static AffineAbstractState const bottom;
      static AffineAbstractState const uniform;

      bool undefined() const {
        return (*this == AffineAbstractState::top || base == ConstantAbstractState::top
            || stride[0] == ConstantAbstractState::top || stride[1] == ConstantAbstractState::top);
      }

      bool divergent() const {
        return *this == bottom;
      }

      bool isUniform() const {
        return ((!divergent() || undefined()) && (stride[0] == ConstantAbstractState::zero));
      }

      bool constant() const {
        return (isUniform() && (base != ConstantAbstractState::bottom));
      }

      bool affine() const {
        return (!(undefined() || divergent() || isUniform()));
      }

      bool known() const {
        return (!(divergent() || undefined()) && base != ConstantAbstractState::bottom
            && stride[0] != ConstantAbstractState::bottom);
      }

      bool unknown() const {
        return (!(divergent() || undefined() || known()));
      }

      bool hardAffine() const {
        return (affine() && stride[0] == ConstantAbstractState::bottom);
      }

      bool isTidMul() const {
        return (affine() && !(stride[0] == ConstantAbstractState::bottom) && base == ConstantAbstractState::zero);
      }

  };
  std::ostream & operator<<(std::ostream & s, AffineAbstractState const & as);

  inline AffineAbstractState operator+(AffineAbstractState const & a, AffineAbstractState const & b) {
    AffineAbstractState r = a;
    return r += b;
  }
  inline AffineAbstractState operator-(AffineAbstractState const & a, AffineAbstractState const & b) {
    AffineAbstractState r = a;
    return r -= b;
  }
  inline AffineAbstractState operator*(AffineAbstractState const & a, AffineAbstractState const & b) {
    AffineAbstractState r = a;
    return r *= b;
  }
  inline AffineAbstractState operator<<(AffineAbstractState const & a, AffineAbstractState const & b) {
    if((b.stride[0] == ConstantAbstractState::zero) && (b.stride[1] == ConstantAbstractState::zero)
        && (b.base != ConstantAbstractState::bottom)){
      AffineAbstractState r;
      r.stride[0] = ConstantAbstractState::zero;
      r.stride[1] = ConstantAbstractState::zero;
      r.base = ConstantAbstractState::one;
      int count = 0;
      while(count < b.base.value){
        count++;
        r.base.value *= 2;
      }
      return r *= a;
    }

    AffineAbstractState r = a;
    return r ^= b;
  }
  inline AffineAbstractState operator^(AffineAbstractState const & a, AffineAbstractState const & b) {
    AffineAbstractState r = a;
    return r ^= b;
  }
  AffineAbstractState Meet(AffineAbstractState const & a, AffineAbstractState const & b);

  struct TransferFunction {
      virtual bool operator()(DataflowGraph::Instruction const & i) = 0;
      virtual bool operator()(DataflowGraph::PhiInstruction const & i) = 0;
      virtual ~TransferFunction() {
      }
      ;
  };

  struct AffineTransferFunction: TransferFunction {
      static bool _isFunct;
      typedef AffineAbstractState State;
      typedef std::map<unsigned int, State> StateMap;
      AffineTransferFunction(StateMap & m, const DataflowGraph*);

      virtual bool operator()(DataflowGraph::Instruction const & i);
      virtual bool operator()(DataflowGraph::PhiInstruction const & i);
      ~AffineTransferFunction() {
      }
      ;

      void dumpMap(std::ostream & os) const;
    private:
      AffineAbstractState registerState(ir::PTXOperand const & op);
      static AffineAbstractState specialState(ir::PTXOperand const & op);

      AffineAbstractState operandTag(ir::PTXOperand const & op);
      AffineAbstractState operandState(ir::PTXOperand const & op);
      void readInputStates(ir::PTXInstruction const & ptxi, AffineAbstractState types[]);

      bool setState(DataflowGraph::RegisterId id, AffineAbstractState state);
      void setStateDests(DataflowGraph::RegisterPointerVector const & d, AffineAbstractState state, bool & stillMoving);

      StateMap & states;
      const DataflowGraph* _dfg;
  };

//  class SparseAnalysis;

  class PGNode {
    public:
      /*! \brief Constructor. Just to initialize _isActive. */
      PGNode() :
          _isActive(false) {
        nodeId = node_counter++;
      }

      /*! \brief Makes pgn a successor of this node. */
      inline void add_successor(PGNode* pgn) {
        _succs.insert(pgn);
      }

      /*! \brief Returns an iterator to the first successor of this node. */
      inline std::set<PGNode*>::iterator succ_begin() {
        return _succs.begin();
      }

      /*! \brief Returns an iterator to the last successor of this node. */
      inline std::set<PGNode*>::iterator succ_end() {
        return _succs.end();
      }

      /*! \brief True if the node is in the working queue. */
      bool _isActive;

      virtual ~PGNode() {
      }

      virtual std::ostream& toStream(std::ostream& out) const = 0;

      inline unsigned getNodeId() const {
        return nodeId;
      }

    private:
      std::set<PGNode*> _succs;

      static unsigned node_counter;

      unsigned nodeId;
  };

  std::ostream& operator<<(std::ostream& out, const PGNode& pgn);

  class PGPhi: public PGNode {
    public:
      PGPhi(const DataflowGraph::PhiInstruction* phi) :
          PGNode(), _phi(phi) {
      }
      const DataflowGraph::PhiInstruction* _phi;
      virtual ~PGPhi() {
      }
      inline std::ostream& toStream(std::ostream& out) const {
        out << *_phi;
        return out;
      }
  };

  class PGInst: public PGNode {
    public:
      PGInst(const DataflowGraph::Instruction* inst) :
          _inst(inst) {
      }
      virtual ~PGInst() {
      }
      inline std::ostream& toStream(std::ostream& out) const {
        out << *_inst;
        return out;
      }

      const DataflowGraph::Instruction* _inst;
  };

  /*!\brief SparseAnalysis implements a general sparse analysis framework. In our
   context we say that an analysis is sparse when it binds information
   directly to variables, instead of program points.
   */
  class SparseAnalysis {
    public:
      /*! \brief Constructor */
      SparseAnalysis(DataflowGraph* dfg, TransferFunction* tf);

      /*! \brief Solve the sparse analysis using an iterative algorithm. */
      void iterateToFixPoint();

      /*! \brief Generate a dot representation of the propagation graph. */
      ostream& printPropagationGraphInDot(ostream &out) const;

      ostream& operator<<(ostream& out) const;

      inline const TransferFunction* tf() const {
        return _tf;
      }
      ;

    private:
      /*! \brief This structure maps phi-functions to the predicates that
       control their outcomes. */
      std::map<const DataflowGraph::PhiInstruction, std::vector<DataflowGraph::RegisterId> > _gating_predicates;

      /*! \brief A map of variables to nodes in the propagation graph. Each
       variable points to the node where it is defined. */
      std::map<DataflowGraph::RegisterId, PGNode*> _def_map;

      /*! \brief This is the meet operator used in the dataflow analysis. */
      TransferFunction* _tf;

      /*! \brief The propagation graph determines the order in which information
       must be propagated during the analysis. Depending on the way that it is
       implemented, the analysis will be either forward or backward. */
      std::set<PGNode*> _propagation_graph;

      /*!\brief This is the queue of nodes in the propagation graph that still
       need to be processed. */
      std::queue<PGNode*> _workqueue;

      /*!\brief This method builds the propagation graph. */
      void buildPropagationGraph(DataflowGraph* dfg);
  };

  /*!\brief AffineAnalysis implements affine analysis. The affine analysis goes
   over the program dataflow graph and finds all the variables that will always
   hold the same values for every thread.
   */
  class AffineAnalysis: public KernelAnalysis {
    public:
      /* Constructor */
      AffineAnalysis();
      /* inherited from KernelAnalysis */
      virtual void analyze(ir::IRKernel& k);

      const SparseAnalysis * sa() const {
        return _sa;
      }
      const AffineTransferFunction * atf() const {
        return _aft;
      }
      const AffineTransferFunction::StateMap &map() const {
        return *_sm;
      }

      const AffineAbstractState &state(const unsigned int var) const {
        return _sm->find(var)->second;
      }

      AffineAbstractState state(const DataflowGraph::InstructionVector::iterator & ins) const {
        AffineAbstractState out = AffineAbstractState::top;
        if((ins->d.size() == 0) && (ins->s.size() == 0)){
          out = AffineAbstractState::uniform;
          out.base = ConstantAbstractState::zero;
          return out;
        }
        DataflowGraph::RegisterPointerVector::const_iterator start;
        DataflowGraph::RegisterPointerVector::const_iterator end;
        if(ins->d.size() > 0){
          start = ins->d.begin();
          end = ins->d.end();
        }else{
          start = ins->s.begin();
          end = ins->s.end();
        }
        while(start != end){
          out &= _sm->find(*(start->pointer))->second;
          start++;
        }
        return out;
      }

    private:
      /*! The target kernel */
      const ir::Kernel *_kernel;

      SparseAnalysis * _sa;
      AffineTransferFunction *_aft;
      AffineTransferFunction::StateMap *_sm;
  };

}

namespace std {
#ifndef _WIN32
  bool operator==(const analysis::DataflowGraph::const_iterator x, const analysis::DataflowGraph::const_iterator y);
#endif
  bool operator<(const analysis::DataflowGraph::const_iterator x, const analysis::DataflowGraph::const_iterator y);
  bool operator<=(const analysis::DataflowGraph::const_iterator x, const analysis::DataflowGraph::const_iterator y);
  bool operator>(const analysis::DataflowGraph::const_iterator x, const analysis::DataflowGraph::const_iterator y);
  bool operator>=(const analysis::DataflowGraph::const_iterator x, const analysis::DataflowGraph::const_iterator y);
}

#endif /* AFFINE_ANALYSIS_H_ */
