
EASEA -- EAsy Specification of Evolutionary Algorithms
===============================================

<div style="text-align:center;" align="center">

| Operating System | Configure | Build | Install | Working examples |
|------------------|:---------:|:-----:|:-------:|:----------------:|
| Windows (msvc)   | ![Configure badge windows](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/configure-windows.svg) | ![Build badge windows](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/build-windows.svg) | ![Install badge windows](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/install-windows.svg) | ![Examples badge windows](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/test-windows.svg) |
| Linux (gcc)   | ![Configure badge linux](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/configure-linux.svg) | ![Build badge linux](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/build-linux.svg) | ![Install badge linux](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/install-linux.svg) | ![Examples badge linux](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/test-linux.svg) |
| MacOS (clang)   | ![Configure badge macos](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/configure-macos.svg) | ![Build badge macos](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/build-macos.svg) | ![Install badge macos](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/install-macos.svg) | ![Examples badge macos](https://raw.githubusercontent.com/EASEA/easea/badges/badges/master/test-macos.svg) |

</div>

## Overview

EASEA (EAsy Specification of Evolutionary Algorithms) is an Artificial Evolution platform that allows scientists with only basic skills in computer science to implement evolutionary algorithms and to exploit the massive parallelism of many-core architectures in order to optimize virtually any real-world problems (continous, discrete, combinatorial, mixed and more (with Genetic Programming)), typically allowing for speedups up to x500 on a $3,000 machine, depending on the complexity of the evaluation function of the inverse problem to be solved.

Then, for very large problems, EASEA can also exploit computational ecosystems as it can parallelize (using an embedded island model) over loosely coupled heterogenous machines (Windows, Linux or Macintosh, with or without GPGPU cards, provided that they have internet access) a grid of computers or a Cloud.


## How to install

EASEA is tested regularly on Windows, Linux and MacOS and is available on these platforms.

If you're using an ArchLinux based OS you can simply install EASEA via the [easena-git](https://aur.archlinux.org/packages/easena-git) aur package.
Otherwise, you will need to compile this project.

### Dependencies

Compiling EASEA requires you to install `boost`, `flex`, `bison`, `openMP` and `cmake`. 
It is also recommanded to install `r` and the r-package _scatterplot3d_, along with `java` and `CUDA` if you would like to use EASEA to its fullest.

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

Some users have reported issues with the latest version of Boost. This software is guaranteed to run on Boost 1.81.0. 
To install this version specifically it is recommended to uninstall the latest version and reinstall Boost 1.81.0 :

```bash
choco uninstall boost-msvc-14.3
choco install boost-msvc-14.3 --version 1.81.0
```

### Building and installing EASEA with CMake

1. Once all dependencies are installed open either a bash shell on Linux/MacOS or a [MSBuild shell](https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170) on Windows

2. Navigate to the directory you [downloaded this repository](https://stevenpcurtis.medium.com/downloading-repos-from-github-13a017951450) at

3. Configure the build by typing the following command into your shell:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=ON .
```

4. Compile EASEA by typing:

```bash
cmake --build build --config Release
```

5. Finally, install EASEA with the command:

```bash
cmake --install build
```

#### OpenMP and default MacOS compiler

Apple decided to remove OpenMP from the default C/C++ compiler. You can either choose to disable OpenMP by using `-DUSE_OPENMP=OFF` at step 3. Otherwise, you can either install a real compiler (see accepted answer of [Stack Overflow: "fatal error: 'omp.h' file not found" using clang on Apple M1](https://stackoverflow.com/questions/66663120/fatal-error-omp-h-file-not-found-using-clang-on-apple-m1)). Another solution is to install OpenMP using brew and either add directories to default paths or provide them to CMake as proposed in [a github issue](https://github.com/actions/runner-images/issues/5555#issuecomment-1133906879).

## How to use

### EZ_PATH
⚠️ IMPORTANT ⚠️ 

**EASEA requires you to set the _EZ_PATH_ environment variable. It should lead to where EASEA was installed (/usr/local/easea on Linux and MacOS, C:/Program Files (x86)/EASEA on Windows)** *_This variable must include the trailing "/" !_*

* [Set environment variables on Windows](https://docs.oracle.com/en/database/oracle/machine-learning/oml4r/1.5.1/oread/creating-and-modifying-environment-variables-on-windows.html)

* [Set environment variables on MacOS](https://phoenixnap.com/kb/set-environment-variable-mac)

* [Set environment variables on Linux](https://phoenixnap.com/kb/linux-set-environment-variable)

### Compiling using EASEA

Compiling a .ez file into a binary is a two steps process :

1. Compile the .ez into C++ and generate the CMakeLists.txt.

```bash
$EZ_PATH/bin/easea your_ez_file.ez
```

2. Compile the C++ into a binary

```bash
cmake . && cmake --build .
```

## Install EASEA GUI

To install the EASEA UI, follow the README instructions in the `GUIDE/` directory.

## License

EASEA and EASEA-CLOUD are Free Open Source Software (under GNU Affero v3 General Public License) developed by the SONIC (Stochastic Optimisation and Nature Inspired Computing) group of the BFO team at Université de Strasbourg. Through the Strasbourg Complex Systems Digital Campus, the platforms are shared with the UNESCO CS-DC UniTwin and E-laboratory on Complex Computational Ecosystems (ECCE).

## Implementation of the Evolutionary Multiobjective Optimization

This is the list of defined MOEA templates currently provided by EASEA.

- NSGA-II
Nondominated Sorting genetic algorithm II (tpl/NSGAII.tpl).
NSGA-II is a very popular MOEA in multi-objective optimization area. 
This algorithm makes offspring by using chosen crossover and mutation operators and
selects individuals for a new generation by nondominated-sorting (NDS) and by crowding distance (CD) comparator.  
```
$ easea -nsgaii any_benchmark.ez.
```
- NSGA-III 
Nondominated Sorting genetic algorithm III (tpl/NSGAIII.tpl).
NSGA-III extends NSGA-II to using reference points to handle many-objective problems.
```
$ easea -nsgaiii any_benchmark.ez 
```
- ASREA
Archived-Based Stcochastic Ranking Evolutionary Algorithm (tpl/ASREA.tpl).
This MOEA ranks the population by comparing individuals with members of an archive, that breaks
complexity into O(man) (m being the number of objectives, a the size of the archive and n the population size). 
```
$ easea -asrea any_benchmark.ez 
```
- CDAS
Controlling Dominance Area of Solutions optimization algorithm (tpl/CDAS.tpl).
CDAS controls the degree of expansion or contraction of the dominance area of solutions using a user-defined parameter
in order to induce appropriate ranking of solutions and improve the performance. 
```
$ easea -cdas any_benchmark.ez 
```
- IBEA
Indicator based evolutionary algorithm (tpl/IBEA.tpl).
In current template the IBEA is based on progressively improvement the epsilon indicator function.
```
$ easea -ibea any_benchmark.ez
```

### Benchmark Suite

- Zitzler-Deb-Thiele's Test Problems ZDT(4, 6) : 2-objective tests (examples/zdt)
All problems are continous, n-dimensional and originating from a well thought combination of functions.

- Deb-Thiele-Laumanns-Zitzler's Test Problems DTLZ(1, 2, 3) : 2/3-objective tests (examples/dtlz)
All problems are continuous n-dimensional multi-objective problems, scalable in fitness dimension. 

- Bi-objective COCO 2018 BBOB benchmark problems (examples/coco2018)
The bbob-biobj test suite provides 55 2-objective functions in six dimensions (2, 3, 5, 10, 20, and 40) with a large number of possible instances.
The 55 functions are derived from combining a subset of the 24 well-known single-objective functions of the bbob test suite,
which has been used since 2009 in the BBOB workshop series. 

### A simple example
As a simple example, we will show, how to define the 3-objective DTLZ1 problem using the NSGA-II algorithm.
First, we select a test folder and set environment variables EZ_PATH:
```
$ cd examples/dtlz1/
$ export EZ_PATH="your_path_to_easea"
```
The problem has to be defined in dtlz1.ez file as follows:

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
m_variable - vector of decision variable of every individual,
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

### Performance Metrics

- Hypervolume (HV) maximisation: it provides the volume of the objective space that is dominated by a Pareto Front (PF), therefore, it shows the convergence quality towards the PF and the diversity in the obtained solutions set.
- Generational Distance (GD) minimization: it measures the average Euclidean distance between the optimal solutions,
obtained by algorithm and those in the Pareto Optimal Front.
- Inverted Generational Distance (IGD) minimization: it is an inverted variation of Generational Distance that: i) calculates the minimum Euclidean distance between an obtained solution and the real PF and ii) measures both the diversity and the convergence towards the PF of the obtained set (if enough members of PF are known).

To use these performance metrics: 

in your ez-file:
```
#define QMETRICS 
#define PARETO_TRUE_FILE "pareto-true.dat"
```
Where "pareto-true.dat" is a file with Pareto Otimal Front (which is in the same folder as ez-file). See examples in examples/zdt(4,6).ez and examples/dtlz(1-3).ez.

### References
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
