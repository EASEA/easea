
The EASEA-CLOUD platform
========================

Overview
--------------
EASEA and EASEA-CLOUD are Free Open Source Software (under GNU Affero v3 General Public License) developed by the SONIC (Stochastic Optimisation and Nature Inspired Computing) group of the BFO team at Université de Strasbourg. Through the Strasbourg Complex Systems Digital Campus, the platforms are shared with the UNESCO CS-DC UniTwin and E-laboratory on Complex Computational Ecosystems (ECCE).

EASEA (EAsy Specification of Evolutionary Algorithms) is an Artificial Evolution platform that allows scientists with only basic skills in computer science to implement evolutionary algorithms and to exploit the massive parallelism of many-core architectures in order to optimize virtually any real-world problems (continous, discrete, combinatorial, mixed and more (with Genetic Programming)), typically allowing for speedups up to x500 on a $3,000 machine, depending on the complexity of the evaluation function of the inverse problem to be solved.

Then, for very large problems, EASEA can also exploit computational ecosystems as it can parallelize (using an embedded island model) over loosely coupled heterogenous machines (Windows, Linux or Macintosh, with or without GPGPU cards, provided that they have internet access) a grid of computers or a Cloud.

Release versions
--------------
easea_v2.0 is generally a work in progress.


Requirements
--------------
This project required you to have flex and bison installed:
```
$ sudo apt-get install flex bison
```
C++ compiler that supports C++14.

Quick start
-------------
- cmake ./
- make
- make install

New templates (MOEA)
-------------
- NSGA-II

- ASREA

- FastEMO

Benchmark Suite
-------------
- ZDT(1, 2, 3, 4, 6)
- DTLZ(1, 2, 3, 4, 5, 6, 7)


Features
--------------

- Runs can be distributed over cluster of homogeneous AND heterogeneous machines.
- Distribution can be done locally on the same machine or over the internet (using a embedded island model).
- Parallelization over GPGPU cards leading to massive speedup (x100 to x1000).
- C++ description language.



