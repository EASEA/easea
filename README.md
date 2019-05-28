EASENA -- EAsy Specification of Evolutionnary and Neural Algorithms
===============================================

Below, EASEA and EASNA are presented as two platforms in development within one program. For further details, go in the corresponding part of this README.

To use one of these platforms, you just have to use files with the correct file extension :
- **.ez**, in order to use EASEA ;
- **.nz**, in order to use EASNA.

The program will detect them and basculate in the correct mode.

#### Requirements

This project required you to have cmake, flex, bison, valgrind, gunzip and wget installed:
```
$ sudo apt-get install flex bison valgrind gunzip wget cmake
```
C++ compiler that supports at least C++14.

#### Quick start

- cmake **.**
- make
- (*Optionnal*) make install : sudo could be necessary
- Export and setting EZ_PATH and PATH environment variables : for example, set in your .bashrc file EZ_PATH="/usr/local/easena/", PATH="$PATH:/usr/local/easena/bin"

# EASEA -- EAsy Specification of Evolutionnary Algorithms

## Overview

EASEA and EASEA-CLOUD are Free Open Source Software (under GNU Affero v3 General Public License) developed by the SONIC (Stochastic Optimisation and Nature Inspired Computing) group of the BFO team at Université de Strasbourg. Through the Strasbourg Complex Systems Digital Campus, the platforms are shared with the UNESCO CS-DC UniTwin and E-laboratory on Complex Computational Ecosystems (ECCE).

EASEA (EAsy Specification of Evolutionary Algorithms) is an Artificial Evolution platform that allows scientists with only basic skills in computer science to implement evolutionary algorithms and to exploit the massive parallelism of many-core architectures in order to optimize virtually any real-world problems (continous, discrete, combinatorial, mixed and more (with Genetic Programming)), typically allowing for speedups up to x500 on a $3,000 machine, depending on the complexity of the evaluation function of the inverse problem to be solved.

Then, for very large problems, EASEA can also exploit computational ecosystems as it can parallelize (using an embedded island model) over loosely coupled heterogenous machines (Windows, Linux or Macintosh, with or without GPGPU cards, provided that they have internet access) a grid of computers or a Cloud.

## Changes

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

## New templates (MOEA)

- NSGA-II 
Nondominated Sorting genetic algorithm II (tpl/NSGAII.tpl).
```
$ easea -nsgaii any_benchmark.ez 
```
- ASREA
Archived-Based Stcochastic Ranking Evolutionary Algorithm. (tpl/ASREA.tpl)
```
$ easea -asrea any_benchmark.ez 
```
- FastEMO
Fast Evolutionary Multi-objective Optimization Algorithm. (tpl/FastEMO.tpl)
```
$ easea -fastemo any_benchmark.ez 
```

## Benchmark Suite

- Zitzler-Deb-Thiele's Test Problems ZDT(1, 2, 3, 4, 6) : 2-objective tests (see examples/zdt)
- Deb-Thiele-Laumanns-Zitzler's Test Problems DTLZ(1, 2, 3, 4, 5, 6, 7) : 3-objective tests (see examples/dtlz)

An example to compile zdt1 test with algorithm FastEMO:
```
$ cd examples/zdt1/
$ easea -fastemo zdt1.ez 
```
## Performance Metrics

- Hypervolume (HV) maximisation: it provides the volume of the objective space that is dominated by a Pareto Front (PF), therefore, it shows the convergence quality towards the PF and the diversity in the obtained solutions set.
- Generational Distance (GD) minimization: it measures the average Euclidean distance between the optimal solutions,
obtained by algorithm and those in the Pareto Optimal Front.
- Inverted Generational Distance (IGD) minimization: it is an inverted variation of Generational Distance that: i) calculates the minimum Euclidean distance between an obtained solution and the real PF and ii)  measures both the diversity and the convergence towards the PF of the obtained set (if enough members of PF are known).

To use these performance metrics: 

in your ez-file
```
 #define QMETRICS 
 #define PARETO_TRUE_FILE "pareto-true.dat"
```
where "pareto-true.dat" is a file with Pareto Otimal Front (which is in the same folder as ez-file). See examples in examples/zdt(1-6).ez and examples/dtlz(1-7).ez.

## Features

- Runs can be distributed over cluster of homogeneous AND heterogeneous machines.
- Distribution can be done locally on the same machine or over the internet (using a embedded island model).
- Parallelization over GPGPU cards leading to massive speedup (x100 to x1000).
- C++ description language.

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
## AUTHOR

EASNA : Romain ORHAND : <rorhand@unistra.fr>