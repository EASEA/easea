EASENA -- EAsy Specification of Evolutionary and Neural Algorithms
===============================================

In this beta version, EASENA platform contains two engines within one program: EASEA and EASNA.
- The EASEA compiler (for artificial evolution) is automatically used on files using extension .ez
- The EASNA platform (for neural networks) is automatically used on files using extension .nz

## Requirements for using EASENA 

EASENA requires at least:
For installation EASENA and package dependencies, you will need administrator priviledges, so make sure your login is in the <tt>/etc/sudoers</tt>  or ask a system administrator to do it for you.
 
EASEA requires at least:
* For linux users MUST be installed :
  - g++ 5.0 version (or later)
  - package dependencies : cmake, flex, bison, valgrind, gunzip, wget, r-base
* For MacOSX users MUST be installed :
  - g++ 5.0 version (or later)
  - package dependencies : xcode, Command Tools, llvm, libomp, r
* Optional (Recommended) for linux and MasOSX : 
  - For graph results visualisation of single objective evolutionary runs :
    - java jre 1.6 (or later) is required. You can get it here: https://www.java.com/download/.<br> Without it, an error appears at the start of easea's compiled programs but can be safely ignored.
  - For graph results visualisation of multi-objective algorithms:
    - r-package scatterplot3d is required. To install scatterplot3d in R shell : <br>
       - you can run R shell by type in command line : <tt>$ r</tt>
       - then in R shell, type following : <tt>install.packages("scatterplot3d")</tt>
  - For using EASENA with GPU cards :
    - CUDA SDK > 4.1. Please install CUDA using the following link: https://developer.nvidia.com/cuda-zone. Without it, you can use EASENA without GPU parallel computing. Be carefull about the compability of your CUDA toolkit and your GCC compiler !


## Linux installation
- Open a terminal window and install all required dependencies :<br>
<tt>$ sudo apt-get install flex bison valgrind gunzip unzip wget cmake r-base</tt>
- If you need, install optional dependencies as it was shown above
- Then download zip archive with the latest version from the github Master branch : https://github.com/EASEA/easea 
- Or clone the github repository by the following command in a command line of the terminal :<br> 
<tt>$ git clone https://github.com/EASEA/easea.git</tt>
- If EASENA was downloaded, in the same terminal window, cd to the directory where zip file was downloaded and expand the zip file :<br>
<tt>$ unzip downloaded_file_name.zip</tt> 
- Go to unziped direcory : <tt>cd easea</tt> or <tt>cd easea-master</tt>
- To configure EASENA platform for the architecture of your computer, type in command line (from current directory):<br>
<tt> $ cmake ./</tt> 
- To compile the EASENA platform, type :<br>
<tt>$ make</tt> 
- To move the files around, type :<br>
<tt>$ sudo make install</tt><br> 
Typically, it will create a directory called <tt>/usr/local/easea</tt> that will contain a directory called <tt>bin</tt> and a directory called <tt>tpl</tt> 
- Once this is done, you can modify your personal .bashrc file in your Home for saving EASEA paths for compiler and library. In order to do it, you have to add the two following lines at the end of .bashrc file :<br>
<pre>
export EZ_PATH=/usr/local/easena/
export PATH=$PATH:/usr/local/easena/bin
</pre>



## MacOSX installation 
First, open a terminal window and install dependencies in the command line of the terminal :
- Install Command Tools : <tt>$ xcode-select --install </tt>
- Install gcc by the following command : <tt>$ brew install gcc</tt>
- Install Low Level Virtual Machine library by the following command : <tt>$ brew install llvm</tt>
- Install libomp library by the following command : <tt>$ brew install libomp</tt>
- Install R by the following command : <tt>$ brew install r</tt>
- Install other package dependencies : <tt>$ brew install flex bison gunzip wget cmake</tt>
- If you need, install optional dependencies as it was shown in section Requirements

The latest MacOSX Mojave system is not compliant with valgrind, which is why valgrind tests are limited to linux operating systems.
Then configure, compile and install EASENA.
Make sure that you are in easea or easea-master directory.<br>
From current direcotry run in the terminal command line following commands :
- <tt>$ cmake -DCMAKE_C_COMPILER="/usr/local/opt/gcc@9/bin/gcc-9" -DCMAKE_CXX_COMPILER="/usr/local/opt/gcc@9/bin/g++-9" .</tt>
- <tt>$ make</tt>
- <tt>$ sudo make install</tt>
Once this is done, you can modify your personal .bash_profile in your Home for saving EASEA paths for compiler and library. In order to do it, you have to add the four following lines at the end of .bash_profile file :<br>
<pre>
export EZ_PATH=/usr/local/easena/
export PATH=$PATH:/usr/local/easena/bin
export LDFLAGS="-L/usr/local/opt/llvm/lib -L/usr/local/opt/libomp/lib"
export CXX="/usr/local/opt/gcc@9/bin/g++-9"
</pre>

## Testing you installation

If typing <tt>$ easena</tt> in the command line of terminal does not start the EASENA compiler, please check out the value of these variables and make sure they point to the good directories.
If the <tt>Usage</tt> tool appears, it means the installation was successful.


## ----- Congratulations, you can now use EASENA -----
Thanks for downloading and installing EASEA/EASENA. We hope it will be useful in your work!


# EASEA -- EAsy Specification of Evolutionary Algorithms

## Overview

EASEA and EASEA-CLOUD are Free Open Source Software (under GNU Affero v3 General Public License) developed by the SONIC (Stochastic Optimisation and Nature Inspired Computing) group of the BFO team at Université de Strasbourg. Through the Strasbourg Complex Systems Digital Campus, the platforms are shared with the UNESCO CS-DC UniTwin and E-laboratory on Complex Computational Ecosystems (ECCE).

EASEA (EAsy Specification of Evolutionary Algorithms) is an Artificial Evolution platform that allows scientists with only basic skills in computer science to implement evolutionary algorithms and to exploit the massive parallelism of many-core architectures in order to optimize virtually any real-world problems (continous, discrete, combinatorial, mixed and more (with Genetic Programming)), typically allowing for speedups up to x500 on a $3,000 machine, depending on the complexity of the evaluation function of the inverse problem to be solved.

Then, for very large problems, EASEA can also exploit computational ecosystems as it can parallelize (using an embedded island model) over loosely coupled heterogenous machines (Windows, Linux or Macintosh, with or without GPGPU cards, provided that they have internet access) a grid of computers or a Cloud.

## Changes

- Added new templates for three Multi-Objective Evolutionary Algorithm: NSGA-II, NSGA-III, CDAS, ASREA, IBEA
- Deleted boost
- Added the lightweight C++ command line option parser from opensource https://github.com/jarro2783/cxxopts
- Added event handler and fixed bug when the program is not responding after 1093 evaluations.
- Fixed some bugs in template CUDA_GP.tpl for island model.
- Added in libeasea three performance metrics: HV, GD, IGD 
- Added in libeasea 2-objective tests (ZDT) and 3-objective tests (DTLZ)
- Added in libeasea following crossover operators: SBX, BLX-alpha, BLX-alpha-beta
- Added in libeasea following mutation operators: Polynomial, Gaussian
- Added in libeasea following selector operators: binary tournament (based on dominance comparison and crowding distance comparison), best individual selection
- Added in libeasea dominance estimator and crowdind distance estimator
- Added in libeasea crowding archive archive
- Added in libeasea simple logger

## Implementation of the Evolutionary Multiobjective Optimization

This is the list of defined MOEA templates currently provided by EASEA.

- NSGA-II
Nondominated Sorting genetic algorithm II (tpl/NSGAII.tpl).
NSGA-II is a very popular MOEA in multi-objective optimization area. 
This algorithm makes offspring by using chosen crossover and mutation operators and
selects individuals for a new generation by nondominated-sorting (NDS) and by crowding distance (CD) comparator.  
```
$ easena -nsgaii any_benchmark.ez.
```
- NSGA-III 
Nondominated Sorting genetic algorithm III (tpl/NSGAIII.tpl).
NSGA-III extends NSGA-II to using reference points to handle many-objective problems.
```
$ easena -nsgaiii any_benchmark.ez 
```
- ASREA
Archived-Based Stcochastic Ranking Evolutionary Algorithm (tpl/ASREA.tpl).
This MOEA ranks the population by comparing individuals with members of an archive, that breaks
complexity into O(man) (m being the number of objectives, a the size of the archive and n the population size). 
```
$ easena -asrea any_benchmark.ez 
```
- CDAS
Controlling Dominance Area of Solutions optimization algorithm (tpl/CDAS.tpl).
CDAS controls the degree of expansion or contraction of the dominance area of solutions using a user-defined parameter
in order to induce appropriate ranking of solutions and improve the performance. 
```
$ easena -cdas any_benchmark.ez 
```
- IBEA
Indicator based evolutionary algorithm (tpl/IBEA.tpl).
In current template the IBEA is based on progressively improvement the epsilon indicator function.
```
$ easena -ibea any_benchmark.ez
```

## Benchmark Suite

- Zitzler-Deb-Thiele's Test Problems ZDT(4, 6) : 2-objective tests (examples/zdt)
All problems are continous, n-dimensional and originating from a well thought combination of functions.

- Deb-Thiele-Laumanns-Zitzler's Test Problems DTLZ(1, 2, 3) : 2/3-objective tests (examples/dtlz)
All problems are continuous n-dimensional multi-objective problems, scalable in fitness dimension. 

- Bi-objective COCO 2018 BBOB benchmark problems (examples/coco2018)
The bbob-biobj test suite provides 55 2-objective functions in six dimensions (2, 3, 5, 10, 20, and 40) with a large number of possible instances.
The 55 functions are derived from combining a subset of the 24 well-known single-objective functions of the bbob test suite,
which has been used since 2009 in the BBOB workshop series. 

## A simple example
As a simple example, we will show, how to define the 3-objective DTLZ1 problem using the NSGA-II algorithm.
First of all, we select a test folder and set environment variables EZ_PATH:
```
$ cd examples/dtlz1/
$ export EZ_PATH="your_path_to_easena"
```
The problem has to be defined in dtlz1.ez file as follow:

- In order to define initial settings (normally they are the same) :
```
TRandom m_generator					// random generator
```

- In order to define a number of decision veriable and a number of objectives :
```
#define NB_VARIABLES 10  //  here we set 10 variables
#define NB_OBJECTIVES 3  //  here we set 3 objectives
```
- In order to define a genetic operator parameters :
```
#define XOVER_DIST_ID 20  //  crossover distribution index
#define MUT_DIST_ID 20    //  mutation distribution index
```
- In order to define genetic operators
```
typedef easea::operators::crossover::continuous::sbx::CsbxCrossover<TT, TRandom &> TCrossover;    //Type of crossover
typedef easea::operators::mutation::continuous::pm::CPolynomialMutation<TT, TRandom &> TMutation; //Type of mutation
/*
 * To define crossover operator parameters
 * param[in 1] - random generator
 * param[in 2] - probability
 * param[in 3] - problem boundary
 * param[in 4] - distibution index
 *
 */
TCrossover crossover(m_generator, 1, m_boundary, XOVER_DIST_ID);

/*
 * To define mutation operator parameters
 * param[in 1] - random generator
 * param[in 2] - probability 
 * param[in 3] - problem boundary
 * param[in 4] - distribution index
 */
TMutation m_mutation(m_generator, 1 / m_boundary.size(), m_boundary, MUT_DIST_ID);
```
- In order to define problem 

```
/*
 * param[in 1] - number of objectives
 * param[in 2] - number of decision variables
 * param[in 3] - problem boundary
 *
 */
TP m_problem(NB_OBJECTIVES, NB_VARIABLES, TBoundary(NB_OBJECTIVES - 1 + NB_VARIABLES, std::make_pair<TT, TT>(0, 1)));

```
- In order to define some additional finctions, section \User functions has to be used.

- In order to define problem evaluation function, section \GenomeClass::evaluator has to be used. For example, like below:
```
\GenomeClass::evaluator : 
  // uses Genome to evaluate the quality of the individual

        const size_t pVariables = getNumberOfObjectives() - 1;

        const TT g = (1 + userFunction1<TT>(TI::m_variable.begin() + pVariables, TI::m_variable.end())) * 0.5;
        userFunction2(TI::m_variable.begin(), TI::m_variable.begin() + pVariables, TI::m_objective.begin(), TI::m_objective.end(), g);

        return 1;

\end

```
Important to know:
m_variable - vector of decision variable of every individual ,
m_objective - vector of objective functions for every individual.

After following instructions above, the problem is defined. Now you will be able to compile and run your program:

You select MOEA (possible options: -nsgaii, -nsgaiii, -asrea, -cdas) by changing script compile.sh.

Then run script:
```
$ ./compile.sh
```
If you have successfully compiled you .ez file you can find .cpp, .h, Makefile, executable and .prm files. 
By modifying file .prm, you can set a number of generation (nbGen) and a population size (popSize and nbOffspring must be the same).

Now you can run selected MOEA for resolving your problem :
```
$ ./launch.sh 
 
```
After execution, you can find in the same folder following files: 
- .png - a figure of obtained Pareto Front
- objectives - values of objectives functions

## Performance Metrics

- Hypervolume (HV) maximisation: it provides the volume of the objective space that is dominated by a Pareto Front (PF), therefore, it shows the convergence quality towards the PF and the diversity in the obtained solutions set.
- Generational Distance (GD) minimization: it measures the average Euclidean distance between the optimal solutions,
obtained by algorithm and those in the Pareto Optimal Front.
- Inverted Generational Distance (IGD) minimization: it is an inverted variation of Generational Distance that: i) calculates the minimum Euclidean distance between an obtained solution and the real PF and ii)  measures both the diversity and the convergence towards the PF of the obtained set (if enough members of PF are known).

To use these performance metrics: 

in your ez-file:
```
#define QMETRICS 
#define PARETO_TRUE_FILE "pareto-true.dat"
```
where "pareto-true.dat" is a file with Pareto Otimal Front (which is in the same folder as ez-file). See examples in examples/zdt(4,6).ez and examples/dtlz(1-3).ez.

## References
1. Deb K., Jain H. An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, //
part I: solving problems with box constraints //
IEEE transactions on evolutionary computation. – 2013. – Т. 18. – №. 4. – С. 577-601.

2. Deb K., Jain H. An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, //
part I: solving problems with box constraints //
IEEE transactions on evolutionary computation. – 2013. – Т. 18. – №. 4. – С. 577-601.

3. Sharma D., Collet P. An archived-based stochastic ranking evolutionary algorithm (ASREA) for multi-objective optimization //
Proceedings of the 12th annual conference on Genetic and evolutionary computation. – ACM, 2010. – С. 479-486.

4. Sato H., Aguirre H. E., Tanaka K. Controlling dominance area of solutions and its impact on the performance of MOEAs //
International conference on evolutionary multi-criterion optimization. – Springer, Berlin, Heidelberg, 2007. – С. 5-20.

5. Zitzler, Eckart, Deb K., and Thiele L.. “Comparison of multiobjective evolutionary algorithms: //
Empirical results.” Evolutionary computation 8.2 (2000): 173-195. doi: 10.1.1.30.5848

6. Deb K., Thiele L., Laumanns M., Zitzler E., Scalable test problems for evolutionary multiobjective optimization 

7. COCO: The Bi-objective Black-Box Optimization Benchmarcking Test Suite.
https://numbbo.github.io/coco-doc/bbob-boobj/functions

# EASNA -- EAsy Specification of Neural Algorithms -- Work in progress !

## SYNOPSIS

`./easena` [**--help**] [**--help.activation**] [**--help.cost**] [**--help.examples**] [**--help.randomDistribution**] [**--batch.error** (average|sum)] [**--batch.size** int] [**--compute** string] [**--learn.batch** string] [**--learn.online** string] [**--parse** string] [**--save.architecture** string] [**--save.weights** string] [**--term**] [**--update.weights** string] [**--verbose**]

## COMMAND-LINE PROGRAM IN DETAIL

<dl>
  <dt><strong>--help</strong></dt>
  <dd>Display all possible options of the program.</dd>
  <dt><strong>--help.activation</strong></dt>
  <dd>Display all implemented activation functions.</dd>
  <dt><strong>--help.cost</strong></dt>
  <dd>Display all implemented cost funtions.</dd>
  <dt><strong>--help.examples</strong></dt>
  <dd>Display some classical examples of use of this program.</dd>
  <dt><strong>--help.randomDistribution</strong></dt>
  <dd>Display all implemented distribution function for randomization.</dd>
  <dt><strong>--batch.error</strong> <em>sum or average</em></dt>
  <dd>Use to define the way the error can bu considered in batch retropropagation. Can only be sum or average.</dd>
  <dt><strong>--batch.size</strong> <em>int</em></dt>
  <dd>Use to define the size of batch in batch retropropagation.</dd>
  <dt><strong>--compute</strong> <em>string</em></dt>
  <dd>Compute the given csv file as inputs into the multi-layer perceptron. Modify in place the file to append the result on each line.</dd>
  <dt><strong>--learn.batch</strong> <em>string</em></dt>
  <dd>Start batched learning process only if the perceptron has been initialized that is to say <em>--term or --parse options</em> have been used. A csv file is expected as input and you have to define the following options <em>--batch.error and --batch.size</em>.</dd>
  <dt><strong>--learn.online</strong> <em>string</em></dt>
  <dd>Start online learning process only if the perceptron has been initialized that is to say <em>--term or --parse options</em> have been used. A csv file is expected as input.</dd>
  <dt><strong>--parse</strong> <em>string</em></dt>
  <dd>Parse a json file to initialize the multi-layer perceptron.</dd>
  <dt><strong>--save.architecture</strong> <em>string</em></dt>
  <dd>Store into a json file the architecture of the multi-layer perceptron. Can only be used with <em>--term</em>.</dd>
  <dt><strong>--save.errors</strong> <em>string</em></dt>
  <dd>Store into a csv file the evolution of cost function during learning process.</dd>
  <dt><strong>--save.weights</strong> <em>string</em></dt>
  <dd>Store the 3D weight matrix into a csv file only if at least the multi-layer perceptron has been initialized.</dd>
  <dt><strong>--term</strong></dt>
  <dd>Initialize the multi-layer perceptron through the terminal.</dd>
  <dt><strong>--update.weights</strong> <em>string</em></dt>
  <dd>Update the 3D weight matrix of an initialized perceptron from a csv file.</dd>
  <dt><strong>--verbose</strong></dt>
  <dd>Display more information.</dd>
</dl>

## FURTHER DETAILS

#### LIBRARIES

This program implementa mult-layer perceptron in C++11 language with standard libraries and two header-only ones : RAPIDJson and PCG.

RAPIDJson has been used to define the architecture of the perceptron in a file JSON as presented below. This library was chosen for two main reasons:
 - This library is only composed of C ++ header files firstly implemented for C ++ 03.
 - This library offers very good results according to the bench of tests proposed via the following URL https://github.com/miloyip/nativejson-benchmark.

PCG has been used to manage the pseudo-random generator in this program. You can see the details in http://www.pcg-random.org/.

#### SOFTMAX CROSS-ENTROPY AND MEAN SQUARRED ERROR

These two methods have been developed within the program under certain constraints. For now, the cross-entropy descent is only implemented with a softmax layer as an output layer. As a consequence, the mean squarred error is implemented for the other activation functions, in other words, all except softmax. This means you can for now choose your cost function by choosing the activation function on the output layer.

The learning method uses for now the descent of the error gradient and therefore updates the weights in online or batch mode. The activation functions are common to all neurons of the same layer.

#### ARCHITECTURE DEFINITION CONSTRAINTS

An important remark relates to the first element of the vectors associated with the functions of activation. It is important the first element of these vectors to be **null** because the first configurable layer is in this case the input layer.

Another remark concerned the definition of bias for the last layer of perceptron, which is the output layer. It must always be **false** otherwise it would imply a memory leak according a computing viewpoint and would involve adding a constant according a mathematical viewpoint (which is not desired...).

#### CSV FILE FORMAT

The expected csv file separator must be";". Moreover, your data should be pre-processed.

#### TESTS

A script *easna_regression.sh* is available in the test folder and has been written in bash. Be sure the easena program has been compiled before executing this script. Moreover you will need **wget**, **valgrind** and **gunzip** to run it successfully, so make sure you have both installed. Furthermore, an **internet connection is needed** to download the datasets : do not worry, the script cleans everything at the end.

The principle of this test is to learn on the Lecun's Mnist dataset and to check the evolution of the global accuracy given by the program according the architecture in the **.nz** file.

To run this test, if you are in the root folder of the project :
```
    cd Test
    ./easna_regression.sh
```

*Tips* : If you want to keep the computed files such as the weights ones or the csv dataset, just add an argument, else everything is cleaned.

## EXAMPLES

This program is intended to be executed on the command line or with
the input help received from the terminal. Here is an example of some
orders and their effects:

- ***./easena --help glob.nz***: Displays the help associated with the neural part of the program *easena*. *glob.nz* is only present in order to switch on the easna part.

- ***./easena --term --save.architecture data/architecture.nz***: Allows you to use the terminal to define an architecture for a perceptron and save it to a JSON file.

- ***./easena --parse data/architecture.nz --learn.online data/learn.csv --save.weights data/ weights.csv***: Read a JSON file in order to configure the perceptron, to perform the learning on a  data set and save the updated weights within a CSV file.

- ***./easena --parse data/architecture.nz --learn.batch data/learn.csv --batch.size 32 --batch.error average --save.weights data/weights.csv***: Read a JSON file in order to configure the perceptron, to perform the learning on a data set and save the updated weights within a CSV file.

- ***./easena --parse data/architecture.nz --update.weights data/weights.csv --compute data/inputs.csv***: Read a JSON file to configure the perceptron, to update the weight from a CSV file and calculate the outputs of a data set within the source CSV file.

Finally, you can get here an example of such a json architecture file expected by the program :

```json
{
    "inertiaRate":0.4,
    "learningRate":0.05,
    "numberLayers":3,
    "biasByLayers":[true,true,false],
    "neuronsByLayers":[10,10,1],
    "activationFunctionByLayers":[null,"relu","identity"],
    "seed":0,
    "distribution":"normal",
    "paramOfDist1":0.0,
    "paramOfDist2":0.1
}
```
