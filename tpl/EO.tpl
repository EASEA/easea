\TEMPLATE_START// -*- mode: c++; c-indent-level: 2; c++-member-init-indent: 8; comment-column: 35; -*-
//
// (The above line is useful in Emacs-like editors)
//
//****************************************
//                                         
//  EASEA.cpp
//                                         
//  C++ file generated by AESAE-EO v0.7b
//                                         
//****************************************
//
//
// Main file for creating a new representation in EO
// =================================================
// 
// This main file includes all other files that have been generated by the
// script create.sh, so it is the only file to compile.
// 
// In case you want to build up a separate library for your new Evolving Object,
// you'll need some work - follow what's done in the src/ga dir, used in the
// main file BitEA in tutorial/Lesson4 dir.
// Or you can wait until we do it :-)

// Miscellany includes and declarations
#include <iostream>
#include <string.h>
using namespace std;

// eo general include
#include "eo"
// real bounds (not yet in general eo include)
#include "utils/eoRealVectorBounds.h"

unsigned  *pCurrentGeneration;
unsigned  *pEZ_NB_GEN;
double EZ_MUT_PROB, EZ_XOVER_PROB, EZ_REPL_PERC=0.0;
int EZ_NB_GEN, EZ_POP_SIZE;
unsigned long EZ_NB_EVALUATIONS=0L;

inline int random(int b1=0, int b2=1){
  return rng.random(b2-b1)+b1;
}
inline double random(double b1=0, double b2=1){
  return rng.uniform(b2-b1)+b1;
}
inline float random(float b1=0, float b2=1){
  return rng.uniform(b2-b1)+b1;
}

\ANALYSE_PARAMETERS
\INSERT_USER_DECLARATIONS


// include here whatever specific files for your representation
// Basically, this should include at least the following

/** definition of representation: 
 * class EASEAGenome MUST derive from EO<FitT> for some fitness
 */
#include "EASEAGenome.h"

// GENOTYPE   EASEAGenome ***MUST*** be templatized over the fitness

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
// START fitness type: double or eoMaximizingFitness if you are maximizing
//                     eoMinimizingFitness if you are minimizing
typedef \MINIMAXI MyFitT ;  // type of fitness 
// END fitness type
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

// Then define your EO objects using that fitness type
typedef EASEAGenome<MyFitT> Indi;      // ***MUST*** derive from EO 

\INSERT_USER_FUNCTIONS
\INSERT_INITIALISATION_FUNCTION 

/** definition of evaluation: 
 * class EASEAEvalFunc MUST derive from eoEvalFunc<EASEAGenome>
 * and should test for validity before doing any computation
 * see tutorial/Templates/evalFunc.tmpl
 */
#include "EASEAEvalFunc.h"

/** definition of initialization: 
 * class EASEAGenomeInit MUST derive from eoInit<EASEAGenome>
 */
#include "EASEAInit.h"

/** include all files defining variation operator classes
 */
#include "EASEAMutation.h"
#include "EASEAQuadCrossover.h"

// Use existing modules to define representation independent routines
// These are parser-based definitions of objects

// how to initialize the population 
// it IS representation independent if an eoInit is given
#include <do/make_pop.h>
eoPop<Indi >&  make_pop(eoParser& _parser, eoState& _state, eoInit<Indi> & _init){
  return do_make_pop(_parser, _state, _init);
}



\INSERT_FINALIZATION_FUNCTION

void EASEAFinal(eoPop<Indi>& pop){
  AESAEFinalFunction(pop);
}



// the stopping criterion
#include "do/make_continue.h"
eoContinue<Indi>& make_continue(eoParser& _parser, eoState& _state, eoEvalFuncCounter<Indi> & _eval){
  return do_make_continue(_parser, _state, _eval);
}

// outputs (stats, population dumps, ...)
#include <do/make_checkpoint.h>
eoCheckPoint<Indi>& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<Indi>& _eval, eoContinue<Indi>& _continue) {
  return do_make_checkpoint(_parser, _state, _eval, _continue);
}

// evolution engine (selection and replacement)
#include <do/make_algo_easea.h>
eoAlgo<Indi>&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<Indi>& _eval, eoContinue<Indi>& _continue, eoGenOp<Indi>& _op){
  return do_make_algo_scalar(_parser, _state, _eval, _continue, _op);
}

// simple call to the algo. stays there for consistency reasons 
// no template for that one
#include <do/make_run.h>
// the instanciating fitnesses
#include <eoScalarFitness.h>
void run_ea(eoAlgo<Indi>& _ga, eoPop<Indi>& _pop){
  do_run(_ga, _pop);
}

// checks for help demand, and writes the status file
// and make_help; in libutils
void make_help(eoParser & _parser);

// now use all of the above, + representation dependent things
int main(int argc, char* argv[]){

try  {
\INSERT_INITIALISATION_FUNCTION

  eoParser parser(argc, argv);  // for user-parameter reading

  eoState state;    // keeps all things allocated

    // The fitness
    //////////////
   EASEAEvalFunc<Indi> plainEval/* (varType  _anyVariable) */;
   // turn that object into an evaluation counter
   eoEvalFuncCounter<Indi> eval(plainEval);

   // a genotype initializer
   EASEAInit<Indi> init;
   // or, if you need some parameters, you might as well 
   // - write a constructor of the eoMyStructInit that uses a parser
   // - call it from here:
   //        eoEASEAInit<Indi> init(parser);
    
    
   // Build the variation operator (any seq/prop construct)
   // here, a simple example with only 1 crossover (2->2, a QuadOp) and
   // one mutation, is given.
   // Hints to have choice among multiple crossovers and mutations are given

   // A (first) crossover (possibly use the parser in its Ctor)
   EASEAQuadCrossover<Indi> cross /* (eoParser parser) */;
    
   // IF MORE THAN ONE:

    // read its relative rate in the combination
// double cross1Rate = parser.createParam(1.0, "cross1Rate", "Relative rate for crossover 1", '1', "Variation Operators").value();

    // create the combined operator with the first one (rename it cross1 !!!)
// eoPropCombinedQuadOp<Indi> cross(cross1, cross1Rate);

    // and as many as you want the following way:
    // 1- write the new class by mimicking eoEASEAQuadCrossover.h
    // 2- include that file here together with eoEASEAQuadCrossover above
    // 3- uncomment and duplicate the following lines:
    //
// eoEASEASecondCrossover<Indi> cross2(eoParser parser); 
// double cross2Rate = parser.createParam(1.0, "cross2Rate", "Relative rate for crossover 2", '2', "Variation Operators").value(); 
// cross.add(cross2, cross2Rate); 

  // NOTE: if you want some gentle output, the last one shoudl be like
  //  cross.add(cross, crossXXXRate, true);

    /////////////// Same thing for MUTATION

   // a (first) mutation   (possibly use the parser in its Ctor)
   EASEAMutation<Indi> mut /* (eoParser parser) */;

    // IF MORE THAN ONE:

    // read its relative rate in the combination
// double mut1Rate = parser.createParam(1.0, "mut1Rate", "Relative rate for mutation 1", '1', "Variation Operators").value();

    // create the combined operator with the first one (rename it cross1 !!!)
// eoPropCombinedMonOp<Indi> mut(mut1, mut1Rate);

    // and as many as you want the following way:
    // 1- write the new class by mimicking eoEASEAMutation.h
    // 2- include that file here together with eoEASEAMutation above
    // 3- uncomment and duplicate the following lines:
    //
// eoEASEASecondMutation<Indi> mut2(eoParser parser); 
// double mut2Rate = parser.createParam(1.0, "mut2Rate", "Relative rate for mutation 2", '2', "Variation Operators").value(); 
// mut.add(mut2, mut2Rate); 

  // NOTE: if you want some gentle output, the last one shoudl be like
  //  mut.add(mut, mutXXXRate, true);

  // now encapsulate your crossover(s) and mutation(s) into an eoGeneralOp
  // so you can fully benefit of the existing evolution engines

  // First read the individual level parameters
    double pCross = parser.createParam(0.6, "pCross", "Probability of Crossover", 'C', "Variation Operators" ).value();
    // minimum check
    if ( (pCross < 0) || (pCross > 1) )
      throw runtime_error("Invalid pCross");

    double pMut = parser.createParam(0.1, "pMut", "Probability of Mutation", 'M', "Variation Operators" ).value();
    // minimum check
    if ( (pMut < 0) || (pMut > 1) )
      throw runtime_error("Invalid pMut");

    // now create the generalOp
    eoSGAGenOp<Indi> op(cross, pCross, mut, pMut);



  //// Now the representation-independent things 
  //
  // YOU SHOULD NOT NEED TO MODIFY ANYTHING BEYOND THIS POINT
  // unless you want to add specific statistics to the checkpoint
  //////////////////////////////////////////////

  // initialize the population
  // yes, this is representation indepedent once you have an eoInit
  eoPop<Indi>& pop   = make_pop(parser, state, init);
  // give popSize to AESAE control
  EZ_POP_SIZE = pop.size();

  // stopping criteria
  eoContinue<Indi> & term = make_continue(parser, state, eval);
  // output
  eoCheckPoint<Indi> & checkpoint = make_checkpoint(parser, state, eval, term);

  // algorithm (need the operator!)
  eoAlgo<Indi>& ga = make_algo_scalar(parser, state, eval, checkpoint, op);

  ///// End of construction of the algorithm

  /////////////////////////////////////////
  // to be called AFTER all parameters have been read!!!
  make_help(parser);

  //// GO
  ///////
  // evaluate intial population AFTER help and status in case it takes time
  apply<Indi>(eval, pop);
  // if you want to print it out
//   cout << "Initial Population\n";
//   pop.sortedPrintOn(cout);
//   cout << endl;

  run_ea(ga, pop); // run the ga

  cout << "Best individual in final population\n";
  cout << pop.best_element() << endl;

  EASEAFinal(pop);

  }
  catch(exception& e)
  {
    cout << e.what() << endl;
  }
  return 0;
}

\START_EO_GENOME_H_TPL// -*- mode: c++; c-indent-level: 2; c++-member-init-indent: 8; comment-column: 35; -*-
//
// (The above line is useful in Emacs-like editors)
//
//****************************************
//                                         
//  EASEAGenome.h
//                                         
//  C++ file generated by AESAE-EO v0.7b
//                                         
//****************************************
//

#ifndef _EASEAGenome_h
#define _EASEAGenome_h

/** 
 *  Always write a comment in this format before class definition
 *  if you want the class to be documented by Doxygen

 * Note that you MUST derive your structure from EO<fitT>
 * but you MAY use some other already prepared class in the hierarchy
 * like eoVector for instance, if you handle a vector of something....

 * If you create a structure from scratch,
 * the only thing you need to provide are 
 *        a default constructor
 *        IO routines printOn and readFrom
 *
 * Note that operator<< and operator>> are defined at EO level
 * using these routines
 */
\ANALYSE_USER_CLASSES
\INSERT_USER_CLASSES

template< class FitT>
class EASEAGenome: public EO<FitT> {
public:
  /** Ctor: you MUST provide a default ctor.
   * though such individuals will generally be processed 
   * by some eoInit object
   */
  EASEAGenome() : EO<FitT>()
  { 
    // START Code of default Ctor of an EASEAGenome object
\GENOME_CTOR 
    // END   Code of default Ctor of an EASEAGenome object
  }

  EASEAGenome(const EASEAGenome & arg) : EO<FitT>()
  { 
\GENOME_CTOR
    copy(arg); 
  }

  virtual ~EASEAGenome()
  {
    // START Code of Destructor of an EASEAGenome object
   \GENOME_DTOR 
    // END   Code of Destructor of an EASEAGenome object
  }

  virtual string className() const { return "EASEAGenome"; }

  EASEAGenome& operator=(const EASEAGenome & arg) { 
    copy(arg); 
    return *this;
  }
  
  void copy(const EASEAGenome& genome) 
  {
    if(&genome != this){
      \GENOME_DTOR
      \COPY_CTOR  
      if (genome.invalid()) {	   // copying an invalid genome
	fitness(FitT());	   // put a valid value (i.e. non NAN)
	invalidate();		   // but INVALIDATE the genome
      }
      else
	fitness(genome.fitness());
    }
  }
  
  bool operator==(const EASEAGenome & genome) const { 
    \EQUAL
      return true;
  }

  bool operator!=(const EASEAGenome & genome) const {
    return !(*this==genome);
  }

  /** printing... */
    void printOn(ostream& os) const
    {
      // First write the fitness
        EO<FitT>::printOn(os);
        os << ' ';
    // START Code of default output 

  /** HINTS
   * in EO we systematically write the sizes of things before the things
   * so readFrom is easier to code (see below)
   */
\INSERT_DISPLAY
\WRITE
    // END   Code of default output
    }

  /** reading... 
   * of course, your readFrom must be able to read what printOn writes!!!
   */
    void readFrom(istream& is)
      {
  // of course you should read the fitness first!
  EO<FitT>::readFrom(is);
    // START Code of input

  /** HINTS
   * remember the EASEAGenome object will come from the default ctor
   * this is why having the sizes written out is useful
   */
\READ
    // END   Code of input
      }

  //private:         // put all data here - no privacy in EASEA
    // START Private data of an EASEAGenome object
\INSERT_GENOME
    // END   Private data of an EASEAGenome object
};
#endif

\START_EO_EVAL_TPL// -*- mode: c++; c-indent-level: 2; c++-member-init-indent: 8; comment-column: 35; -*-
//
// (The above line is useful in Emacs-like editors)
//
//****************************************
//                                         
//  EASEAEvalFunc.h
//                                         
//  C++ file generated by AESAE-EO v0.7b
//                                         
//****************************************
//

/*
Evaluator in EO: a functor that computes the fitness of an EO
=============================================================
*/
 
#ifndef _EASEAEvalFunc_h
#define _EASEAEvalFunc_h

// include whatever general include you need
#include <stdexcept>
#include <fstream>

// include the base definition of eoEvalFunc
#include "eoEvalFunc.h"

/** 
  Always write a comment in this format before class definition
  if you want the class to be documented by Doxygen
*/
template <class EOT>
class EASEAEvalFunc : public eoEvalFunc<EOT>
{
public:
  /// Ctor - no requirement
// START eventually add or modify the anyVariable argument
  EASEAEvalFunc()
  //  EASEAEvalFunc( varType  _anyVariable) : anyVariable(_anyVariable) 
// END eventually add or modify the anyVariable argument
  {
    // START Code of Ctor of an EASEAEvalFunc object
    // END   Code of Ctor of an EASEAEvalFunc object
  }

  /** Actually compute the fitness
   *
   * @param EOT & _eo the EO object to evaluate
   *                  it should stay templatized to be usable 
   *                  with any fitness type
   */
  void operator()(EOT & genome)
  {
    // test for invalid to avoid recomputing fitness of unmodified individuals
    if (genome.invalid())
      {
    // START Code of computation of fitness of the EASEA object
\INSERT_EVALUATOR
    // END   Code of computation of fitness of the EASEA object
      }
  }

private:
// START Private data of an EASEAEvalFunc object
  //  varType anyVariable;       // for example ...
// END   Private data of an EASEAEvalFunc object
};


#endif

\START_EO_INITER_TPL// -*- mode: c++; c-indent-level: 2; c++-member-init-indent: 8; comment-column: 35; -*-
//
// (The above line is useful in Emacs-like editors)
//
//****************************************
//                                         
//  EASEAInit.h
//                                         
//  C++ file generated by AESAE-EO v0.7b
//                                         
//****************************************
//

/*
objects initialization in EO
============================
*/

#ifndef _EASEAInit_h
#define _EASEAInit_h

// include the base definition of eoInit
#include <eoInit.h>

/** 
 *  Always write a comment in this format before class definition
 *  if you want the class to be documented by Doxygen
 *
 * There is NO ASSUMPTION on the class GenoypeT.
 * In particular, it does not need to derive from EO (e.g. to initialize 
 *    atoms of an eoVector you will need an eoInit<AtomType>)
 */
template <class GenotypeT>
class EASEAInit: public eoInit<GenotypeT> {
public:
  /// Ctor - no requirement
// START eventually add or modify the anyVariable argument
  EASEAInit()
  //  EASEAInit( varType & _anyVariable) : anyVariable(_anyVariable) 
// END eventually add or modify the anyVariable argument
  {
    // START Code of Ctor of an EASEAInit object
    // END   Code of Ctor of an EASEAInit object
  }


  /** initialize a genotype
   *
   * @param _genotype  generally a genotype that has been default-constructed
   *                   whatever it contains will be lost
   */
  void operator()(GenotypeT & _genotype)
  {
    // START Code of random initialization of an EASEAGenome object
\INSERT_EO_INITIALISER
    // END   Code of random initialization of an EASEAGenome object
    _genotype.invalidate();    // IMPORTANT in case the _genotype is old
  }

private:
// START Private data of an EASEAInit object
  //  varType & anyVariable;       // for example ...
// END   Private data of an EASEAInit object
};

#endif


\START_EO_MUT_TPL// -*- mode: c++; c-indent-level: 2; c++-member-init-indent: 8; comment-column: 35; -*-
//
// (The above line is useful in Emacs-like editors)
//
//****************************************
//                                         
//  EASEAMutation.h
//                                         
//  C++ file generated by AESAE-EO v0.7b
//                                         
//****************************************
//

/*
simple mutation operators
=========================
*/

#ifndef EASEAMutation_H
#define EASEAMutation_H


#include <eoOp.h>

/** 
 *  Always write a comment in this format before class definition
 *  if you want the class to be documented by Doxygen
 *
 * THere is NO ASSUMPTION on the class GenoypeT.
 * In particular, it does not need to derive from EO
 */
template<class GenotypeT> 
class EASEAMutation: public eoMonOp<GenotypeT>
{
public:
  /**
   * Ctor - no requirement
   */
// START eventually add or modify the anyVariable argument
  EASEAMutation()
  //  EASEAMutation( varType  _anyVariable) : anyVariable(_anyVariable) 
// END eventually add or modify the anyVariable argument
  {
    // START Code of Ctor of an EASEAMutation object
    // END   Code of Ctor of an EASEAMutation object
  }

  /// The class name. Used to display statistics
  string className() const { return "EASEAMutation"; }

  /**
   * modifies the parent
   * @param _genotype The parent genotype (will be modified)
   */
  bool operator()(GenotypeT & _genotype) 
  {
    // START code for mutation of the _genotype object
\INSERT_MUTATOR
    // END code for mutation of the _genotype object

private:
// START Private data of an EASEAMutation object
  //  varType anyVariable;       // for example ...
// END   Private data of an EASEAMutation object
};

#endif

\START_EO_QUAD_XOVER_TPL// -*- mode: c++; c-indent-level: 2; c++-member-init-indent: 8; comment-column: 35; -*-
//
// (The above line is useful in Emacs-like editors)
//
//****************************************
//                                         
//  EASEAQuadCrossover.h
//                                         
//  C++ file generated by AESAE-EO v0.7b
//                                         
//****************************************
//

/*
Template for simple quadratic crossover operators
=================================================

Quadratic crossover operators modify both genotypes
*/

#ifndef EASEAQuadCrossover_H
#define EASEAQuadCrossover_H

#include <eoOp.h>

/** 
 *  Always write a comment in this format before class definition
 *  if you want the class to be documented by Doxygen
 *
 * THere is NO ASSUMPTION on the class GenoypeT.
 * In particular, it does not need to derive from EO
 */
template<class GenotypeT> 
class EASEAQuadCrossover: public eoQuadOp<GenotypeT>
{
public:
  /**
   * Ctor - no requirement
   */
// START eventually add or modify the anyVariable argument
  EASEAQuadCrossover()
  //  EASEAQuadCrossover( varType  _anyVariable) : anyVariable(_anyVariable) 
// END eventually add or modify the anyVariable argument
  {
    // START Code of Ctor of an EASEAQuadCrossover object
    // END   Code of Ctor of an EASEAQuadCrossover object
  }

  /// The class name. Used to display statistics
  string className() const { return "EASEAQuadCrossover"; }

  /**
   * eoQuad crossover - modifies both genotypes
   */
  bool operator()(GenotypeT& child1, GenotypeT & child2) 
  {
      GenotypeT parent1(child1);
      GenotypeT parent2(child2);
    
    // START code for crossover of child1 and child2 objects
\INSERT_CROSSOVER
    return (parent1!=child1)||(parent2!=child2);
    // END code for crossover of child1 and child2 objects
  }

private:
// START Private data of an EASEAQuadCrossover object
  //  varType anyVariable;       // for example ...
// END   Private data of an EASEAQuadCrossover object
};

#endif

\START_EO_CONTINUE_TPL// -*- mode: c++; c-indent-level: 2; c++-member-init-indent: 8; comment-column: 35; -*-
//
// (The above line is useful in Emacs-like editors)
//
//****************************************
//                                         
//  EASEA_make_continue.h
//                                         
//  C++ file generated by AESAE-EO v0.7b
//                                         
//****************************************
//

#ifndef _make_continue_h
#define _make_continue_h

/*
Contains the templatized version of parser-based choice of stopping criterion
It can then be instantiated, and compiled on its own for a given EOType
(see e.g. in dir ga, ga.cpp)
*/

// Continuators - all include eoContinue.h
#include <eoCombinedContinue.h>
#include <eoGenContinue.h>
#include <eoSteadyFitContinue.h>
#include <eoEvalContinue.h>
#include <eoFitContinue.h>
#ifndef _MSC_VER
#include <eoCtrlCContinue.h>  // CtrlC handling (using 2 global variables!)
#endif

  // also need the parser and param includes
#include <utils/eoParser.h>
#include <utils/eoState.h>


/////////////////// the stopping criterion ////////////////
template <class Indi>
eoCombinedContinue<Indi> * make_combinedContinue(eoCombinedContinue<Indi> *_combined, eoContinue<Indi> *_cont)
{
  if (_combined)       // already exists
    _combined->add(*_cont);
  else
    _combined = new eoCombinedContinue<Indi>(*_cont);
  return _combined;
}

template <class Indi>
eoContinue<Indi> & do_make_continue(eoParser& _parser, eoState& _state, eoEvalFuncCounter<Indi> & _eval)
{
  //////////// Stopping criterion ///////////////////
  // the combined continue - to be filled
  eoCombinedContinue<Indi> *continuator = NULL;

  // for each possible criterion, check if wanted, otherwise do nothing

  // First the eoGenContinue - need a default value so you can run blind
  // but we also need to be able to avoid it <--> 0
  eoValueParam<unsigned>& maxGenParam = _parser.createParam(\NB_GEN, "maxGen", "Maximum number of generations () = none)",'G',"Stopping criterion");
  // and give control to EASEA
    EZ_NB_GEN = maxGenParam.value();
    pEZ_NB_GEN = & maxGenParam.value();

    // do not test for positivity in EASEA
    //    if (maxGenParam.value()) // positive: -> define and store
    //      {
  eoGenContinue<Indi> *genCont = new eoGenContinue<Indi>(maxGenParam.value());
  _state.storeFunctor(genCont);
  // and "add" to combined
  continuator = make_combinedContinue<Indi>(continuator, genCont);
  //      }

  // the steadyGen continue - only if user imput
  eoValueParam<unsigned>& steadyGenParam = _parser.createParam(unsigned(100), "steadyGen", "Number of generations with no improvement",'s', "Stopping criterion");
  eoValueParam<unsigned>& minGenParam = _parser.createParam(unsigned(0), "minGen", "Minimum number of generations",'g', "Stopping criterion");
    if (_parser.isItThere(steadyGenParam))
      {
  eoSteadyFitContinue<Indi> *steadyCont = new eoSteadyFitContinue<Indi>
    (minGenParam.value(), steadyGenParam.value());
  // store
  _state.storeFunctor(steadyCont);
  // add to combinedContinue
  continuator = make_combinedContinue<Indi>(continuator, steadyCont);
      }

  // Same thing with Eval - but here default value is 0
  eoValueParam<unsigned long>& maxEvalParam = _parser.createParam((unsigned long)0, "maxEval", "Maximum number of evaluations (0 = none)",'E',"Stopping criterion");

    if (maxEvalParam.value()) // positive: -> define and store
      {
  eoEvalContinue<Indi> *evalCont = new eoEvalContinue<Indi>(_eval, maxEvalParam.value());
  _state.storeFunctor(evalCont);
  // and "add" to combined
  continuator = make_combinedContinue<Indi>(continuator, evalCont);
      }
    /*
  // the steadyEval continue - only if user imput
  eoValueParam<unsigned>& steadyGenParam = _parser.createParam(unsigned(100), "steadyGen", "Number of generations with no improvement",'s', "Stopping criterion");
  eoValueParam<unsigned>& minGenParam = _parser.createParam(unsigned(0), "minGen", "Minimum number of generations",'g', "Stopping criterion");
    if (_parser.isItThere(steadyGenParam))
      {
  eoSteadyGenContinue<Indi> *steadyCont = new eoSteadyFitContinue<Indi>
    (minGenParam.value(), steadyGenParam.value());
  // store
  _state.storeFunctor(steadyCont);
  // add to combinedContinue
  continuator = make_combinedContinue<Indi>(continuator, steadyCont);
      }
    */
    // the target fitness
    eoFitContinue<Indi> *fitCont;
    eoValueParam<double>& targetFitnessParam = _parser.createParam(double(0.0), "targetFitness", "Stop when fitness reaches",'T', "Stopping criterion");
    if (_parser.isItThere(targetFitnessParam))
      {
  fitCont = new eoFitContinue<Indi>
    (targetFitnessParam.value());
  // store
  _state.storeFunctor(fitCont);
  // add to combinedContinue
  continuator = make_combinedContinue<Indi>(continuator, fitCont);
      }

#ifndef _MSC_VER
    // the CtrlC interception (Linux only I'm afraid)
    eoCtrlCContinue<Indi> *ctrlCCont;
    eoValueParam<bool>& ctrlCParam = _parser.createParam(false, "CtrlC", "Terminate current generation upon Ctrl C",'C', "Stopping criterion");
    if (_parser.isItThere(ctrlCParam))
      {
  ctrlCCont = new eoCtrlCContinue<Indi>;
  // store
  _state.storeFunctor(ctrlCCont);
  // add to combinedContinue
  continuator = make_combinedContinue<Indi>(continuator, ctrlCCont);
      }
#endif

    // now check that there is at least one!
    if (!continuator)
      throw runtime_error("You MUST provide a stopping criterion");
  // OK, it's there: store in the eoState
  _state.storeFunctor(continuator);

  // and return
    return *continuator;
}

#endif

\START_EO_PARAM_TPL#****************************************
#                                         
#  EASEA.prm
#                                         
#  Parameter file generated by AESAE-EO v0.7b
#                                         
#****************************************
######    General    ######
# --help=0 # -h : Prints this message
# --stopOnUnknownParam=1 # Stop if unknown param entered
--seed=S   # -S : Random number seed. It is possible to give a specific seed.

######    Evolution Engine    ######
--popSize=\POP_SIZE # -P : Population Size
--selection=\SELECTOR\SELECT_PRM # -S : Selection: Roulette, Ranking(p,e), DetTour(T), StochTour(t) or Sequential(ordered/unordered)
--nbOffspring=\OFF_SIZE # -O : Nb of offspring (percentage or absolute)
--replacement=General # Type of replacement: Generational, ESComma, ESPlus, SSGA(T), EP(T)

######    Evolution Engine / Replacement    ######
--elite=\ELITE_SIZE  # Nb of elite parents (percentage or absolute)
--eliteType=\ELITISM # Strong (true) or weak (false) elitism (set elite to 0 for none)
--surviveParents=\SURV_PAR_SIZE # Nb of surviving parents (percentage or absolute)
--reduceParents=\RED_PAR\RED_PAR_PRM # Parents reducer: Deterministic, EP(T), DetTour(T), StochTour(t), Uniform
--surviveOffspring=\SURV_OFF_SIZE  # Nb of surviving offspring (percentage or absolute)
--reduceOffspring=\RED_OFF\RED_OFF_PRM # Offspring reducer: Deterministic, EP(T), DetTour(T), StochTour(t), Uniform
--reduceFinal=\RED_FINAL\RED_FINAL_PRM # Final reducer: Deterministic, EP(T), DetTour(T), StochTour(t), Uniform

######    Output    ######
# --useEval=1 # Use nb of eval. as counter (vs nb of gen.)
# --useTime=1 # Display time (s) every generation
# --printBestStat=1 # Print Best/avg/stdev every gen.
# --printPop=0 # Print sorted pop. every gen.

######    Output - Disk    ######
# --resDir=Res # Directory to store DISK outputs
# --eraseDir=1 # erase files in dirName if any
# --fileBestStat=0 # Output bes/avg/std to file

######    Output - Graphical    ######
--plotBestStat=1 # Plot Best/avg Stat
--plotHisto=1 # Plot histogram of fitnesses

######    Persistence    ######
# --Load= # -L : A save file to restart from
# --recomputeFitness=0 # -r : Recompute the fitness after re-loading the pop.?
# --saveFrequency=0 # Save every F generation (0 = only final state, absent = never)
# --saveTimeInterval=0 # Save every T seconds (0 or absent = never)
# --status=OneMaxGenomeEA.status # Status file

######    Stopping criterion    ######
--maxGen=\NB_GEN # -G : Maximum number of generations () = none)
# --steadyGen=100 # -s : Number of generations with no improvement
# --minGen=0 # -g : Minimum number of generations
# --maxEval=0 # -E : Maximum number of evaluations (0 = none)
# --targetFitness=0 # -T : Stop when fitness reaches
# --CtrlC=0 # -C : Terminate current generation upon Ctrl C

######    Variation Operators    ######
# --cross1Rate=1 # -1 : Relative rate for crossover 1
# --mut1Rate=1 # -1 : Relative rate for mutation 1
--pCross=\XOVER_PROB # -C : Probability of Crossover
--pMut=\MUT_PROB # -M : Probability of Mutation

\START_EO_MAKEFILE_TPL#****************************************
#                                         
#  EASEA.mak
#                                         
#  Makefile generated by AESAE-EO v0.7b
#                                         
#****************************************

# sample makefile for building an EA evolving a new genotype

DIR_EO = \EO_DIR

.cpp: ; c++  -DPACKAGE=\"eo\" -I. -I$(DIR_EO)/src -Wall -g  -o $@  $*.cpp $(DIR_EO)/src/libeo.a $(DIR_EO)/src/utils/libeoutils.a

.cpp.o: ; c++ -DPACKAGE=\"eo\" -I. -I\EO_DIR/src -Wall -g -c $*.cpp

LIB_EO = $(DIR_EO)/src/utils/libeoutils.a $(DIR_EO)/src/libeo.a

SOURCES = EASEA.cpp \
  EASEAEvalFunc.h \
  EASEAGenome.h \
  EASEAInit.h \
  EASEAMutation.h \
  EASEAQuadCrossover.h \
  $(LIB_EO)

ALL = EASEA EASEA.opt

EASEA : $(SOURCES)
	c++ -g -I. -I$(DIR_EO)/src -o $@ EASEA.cpp $(LIB_EO) -lm -Wno-deprecated -fpermissive

EASEA.opt : $(SOURCES)
	c++ -O4 -I. -I$(DIR_EO)/src -o $@ EASEA.cpp $(LIB_EO) -lm -Wno-deprecated -fpermissive

all : $(ALL)

clean : ; /bin/rm  *.o $(ALL)

\TEMPLATE_END
