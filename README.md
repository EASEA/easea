
EASENA -- EAsy Specification of Evolutionary and Neural Algorithms
===============================================

<div style="text-align:center;" align="center">

| Operating System | Configure | Build | Install | Working examples |
|------------------|:---------:|:-----:|:-------:|:----------------:|
| Windows (msvc)   | ![Configure badge windows](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/configure-windows.svg) | ![Build badge windows](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/build-windows.svg) | ![Install badge windows](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/install-windows.svg) | ![Examples badge windows](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/test-windows.svg) |
| Linux (gcc)   | ![Configure badge linux](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/configure-linux.svg) | ![Build badge linux](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/build-linux.svg) | ![Install badge linux](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/install-linux.svg) | ![Examples badge linux](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/test-linux.svg) |
| MacOS (clang)   | ![Configure badge macos](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/configure-macos.svg) | ![Build badge macos](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/build-macos.svg) | ![Install badge macos](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/install-macos.svg) | ![Examples badge macos](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/test-macos.svg) |

</div>

In this beta version, EASENA platform contains two engines within one program: EASEA and EASNA.
- The EASEA compiler (for artificial evolution) is automatically used on files using extension .ez
- The EASNA platform (for neural networks) is automatically used on files using extension .nz

## How to install

EASENA is tested regularly on Windows, Linux and MacOS and is available on these platforms.

If you're using an ArchLinux based OS you can simply install EASENA via the [easena-git](https://aur.archlinux.org/packages/easena-git) aur package.
Otherwise, you will need to compile this project.

### Dependencies

Compiling EASENA requires you to install `boost`, `flex`, `bison`, `openMP` and `cmake`. 
It is also recommended to install `r` and the r-package _scatterplot3d_, along with `java` and `CUDA` if you would like to use EASENA to its fullest.

#### Linux

If your OS is up-to-date there should be no need to update your C++ compiler.

All the dependencies can be installed on *_Ubuntu_* by typing the following command:

```bash
sudo apt install libboost-all-dev flex bison cmake gcc
```

#### MacOS

All the required dependencies for MacOS can be installed using [brew](https://brew.sh/). Once _brew_ is installed the following command install everything that is needed :

```bash
brew install bison flex libomp coreutils boost cmake clang
```

#### Windows

Windows builds using _msvc_ are supported. The recommended way to install the required packages is to first install [chocolatey package manager](https://chocolatey.org/install).

You need to install [Windows C++ compiler](https://visualstudio.microsoft.com/fr/downloads/). This can be done via _chocolatey_ with using the command below inside a Powershell command prompt, or via the _Visual Studio Community/Entreprise Installer_.

```bash
choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools;includeRecommended"
```

**NOTE:** _if no version of msvc is installed by chocolatey you may need to install it manually, or using_ `choco install visualstudio2022-workload-vctools --package-parameters "--includeRecommended"`

The other dependencies can be installed using chocolatey by opening a Powershell prompt and pasting the following command :

```bash
choco install winflexbison3 boost-msvc-14.3 cmake
```

Some users have reported issues with the latest version of Boost. This software is guaranteed to run on Boost 1.79.0. 
To install this version specifically it is recommended to uninstall the latest version and reinstall Boost 1.79.0 :

```bash
choco uninstall boost-msvc-14.3
choco install boost-msvc-14.3 --version 1.79.0
```

### Building and installing EASEA with CMake

1. Once all dependencies are installed open either a bash shell on Linux/MacOS or a [MSBuild shell](https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170) on Windows

2. Navigate to the directory you [downloaded this repository](https://stevenpcurtis.medium.com/downloading-repos-from-github-13a017951450) at

3. Configure the build by typing the following command into your shell:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=ON .
```

4. Compile EASENA by typing:

```bash
cmake --build build --config Release
```

5. Finally install EASENA with the command:

```bash
cmake --install build
```

#### OpenMP and default MacOS compiler

Apple decided to remove OpenMP from the default C/C++ compiler. You can either choose to disable OpenMP by using `-DUSE_OPENMP=OFF` at step 3. Otherwise, you can either install a real compiler (see accepted answer of [Stack Overflow: "fatal error: 'omp.h' file not found" using clang on Apple M1](https://stackoverflow.com/questions/66663120/fatal-error-omp-h-file-not-found-using-clang-on-apple-m1)). Another solution is to install OpenMP using brew and either add directories to default paths or provide them to CMake as proposed in [a github issue](https://github.com/actions/runner-images/issues/5555#issuecomment-1133906879).

## How to use

**EASENA requires you to set the _EZ_PATH_ environment variable.** *_This variable must include the trailing "/" !_*

* [Set environment variables on Windows](https://docs.oracle.com/en/database/oracle/machine-learning/oml4r/1.5.1/oread/creating-and-modifying-environment-variables-on-windows.html)

* [Set environment variables on MacOS](https://phoenixnap.com/kb/set-environment-variable-mac)

* [Set environment variables on Linux](https://phoenixnap.com/kb/linux-set-environment-variable)

Compiling a .ez file into a binary is a two steps process :

1. Compile the .ez into C++ and generate the CMakeLists.txt.

```bash
$EZ_PATH/bin/easena your_ez_file.ez
```

2. Compile the C++ into a binary

```bash
cmake . && cmake --build .
```

## ----- Congratulations, you can now use EASENA -----
Thanks for downloading and installing EASEA/EASENA. We hope it will be useful in your work!

## Install EASEA GUI

To install the EASEA UI, follow the README instructions in the `easea-ui/` directory.


# EASEA -- EAsy Specification of Evolutionary Algorithms

## Overview

EASEA and EASEA-CLOUD are Free Open Source Software (under GNU Affero v3 General Public License) developed by the SONIC (Stochastic Optimisation and Nature Inspired Computing) group of the BFO team at Université de Strasbourg. Through the Strasbourg Complex Systems Digital Campus, the platforms are shared with the UNESCO CS-DC UniTwin and E-laboratory on Complex Computational Ecosystems (ECCE).

EASEA (EAsy Specification of Evolutionary Algorithms) is an Artificial Evolution platform that allows scientists with only basic skills in computer science to implement evolutionary algorithms and to exploit the massive parallelism of many-core architectures in order to optimize virtually any real-world problems (continous, discrete, combinatorial, mixed and more (with Genetic Programming)), typically allowing for speedups up to x500 on a $3,000 machine, depending on the complexity of the evaluation function of the inverse problem to be solved.

Then, for very large problems, EASEA can also exploit computational ecosystems as it can parallelize (using an embedded island model) over loosely coupled heterogenous machines (Windows, Linux or Macintosh, with or without GPGPU cards, provided that they have internet access) a grid of computers or a Cloud.


## Features

Changes
--------------
- Added new templates for three Multi-Objective Evolutionary Algorithm: NSGA-II, ASREA, FastEMO
- Deleted boost
- Added the lightweight C++ command line option parser from opensource https://github.com/jarro2783/cxxopts
- Added event handler and fixed bug when the program is not responding after 1093 evaluations.
- Fixed some bugs in template CUDA_GP.tpl for island model.
- Added in libeasea three performance metrics: HV, GD, IGD 
- Added in libeasea five 2-objective tests (ZDT) and seven 3-objective tests (DTLZ)
- Added in libeasea three crossover operators: SBX, BLX-alpha, BLX-alpha-beta
- Added in libeasea two mutation operators: Polynomial, Gaussian
- Added in libeasea two selector operators: binary tournament (based on dominance comparison and crowding distance comparison), best individual selection
- Added in libeasea dominance estimator and crowdind distance estimator
- Added in libeasea crowding archive module
- Added in libeasea simple logger


## Changes

- Added new templates for three Multi-Objective Evolutionary Algorithm: NSGA-II, NSGA-III, CDAS, ASREA, IBEA
- Deleted boost
- Added the lightweight C++ command line option parser from opensource https://github.com/jarro2783/cxxopts
- Added a simple logger 
- Added event handler and fixed bug when the program is not responding after 1093 evaluations.
- Fixed some bugs in template CUDA_GP.tpl for island model.
- Added in libeasea three performance metrics: HV, GD, IGD 
- Added in libeasea 2-objective tests (ZDT) and 3-objective tests (DTLZ)
- Added in libeasea following crossover operators: SBX
- Added in libeasea following mutation operators: Polynomial, Gaussian
- Added in libeasea following selector operators: binary tournament (based on dominance comparison and crowding distance comparison)
- Added in libeasea dominance estimator and crowdind distance estimator
- Added in libeasea crowding archive 

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

