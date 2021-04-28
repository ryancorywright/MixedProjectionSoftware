# MixedProjectionSoftware
This supplement is for reproducing the computational results in section 5 of the paper "Mixed-Projection Conic Optimization: A New Paradigm for Modeling Rank Constraints" by Dimitris Bertsimas, Ryan Cory-Wright and Jean Pauphilet. Note that a preprint is available [here](http://www.optimization-online.org/DB_HTML/2020/09/8031.html).

# Introduction
The software in this package is designed to provide certifiably optimal or near-optimal solutions to low-rank problems using branch and cut, relax-and-round and alternating minimization strategies among others. The algorithms implemented here are described in the paper "Mixed-Projection Conic Optimization: A New Paradigm for Modeling Rank Constraints" by Bertsimas, Cory-Wright and Pauphilet.



## Setup and Installation
In order to run this software, you must install a recent version of Julia from http://julialang.org/downloads/. The most recent version of Julia at the time this code was last tested was Julia 1.5.3; the code should work on any version of Julia beginning in 1.x

You must also have a valid installation of Mosek (>=9.0) and Gurobi (>=9.0) for this software to run (academic licenses are freely available at https://www.mosek.com/products/academic-licenses/ and https://www.gurobi.com/downloads/end-user-license-agreement-academic/ respectively). This software was tested on Mosek version 9.1 and Gurobi version 9.0.3, but should also work on more recent versions of these solvers. Since Gurobi only released the non-convex branch-and-bound solver used to solve the master problems here in version 9.0 (and are actively improving it) we recommend using the most recent version of Gurobi.

A number of packages must be installed in Julia before the code can be run. They can be added by running:

```
using Pkg; Pkg.add("JuMP, Gurobi, Mosek, MosekTools, Random, LinearAlgebra, DataFrames, Test, Suppressor, DelimitedFiles, CSV, StatsBase, Compat")
```

You will also need to ensure that the Julia packages are of the correct version, to guarentee that the code will run (any configuration where the JuMP version is >=0.19 "should" work, but using a different configuration is at your own risk). The versions of the packages which we benchmarked the code on are:


```
- JuMP  0.21.3
- Gurobi 0.9.3
- MathOptInterface 0.9.19
- Mosek 1.13
- Ipopt 0.6.5
- MosekTools 0.9.4
- StatsBase 0.33.2
- Suppressor 0.2.0
```



At this point, the files "exact_approaches.jl" and "exact_edm.jl"(under Matrix Completion and Euclidean Distance Matrix Completion respectively) should run successfully.

If you are interested in using this software to solve different matrix completion instances exactly (as opposed to reproducing the output in the paper), we recommend (a) using the multi-tree implementation and (b) tuning the following parameters:

```
- FuncPieceError, FuncPieceLength and MIPGap: set these to the largest values which give ``reasonable'' results.
- TimeLimit.
- maxOAIters: larger instances will warrant a larger limit on the number of cut passes. 
```


## Development Notes
This package should be used at your own risk.
If you run into any issues, please report them via the [Github issue tracker](https://github.com/ryancorywright/MixedProjectionSoftware/issues).

## Reproducing output from the paper
The script files used to generate output in the paper are currently designed to be run on MITs engaging cluster, and therefore require a small amount of editing before they can be run on a personal computer (in particular, modifying the line "for ARG in ARGS" to "for ARG in ["1", "2", ...]"). You can reproduce the output used to create a figure or table by running the files "createFigurex.jl" or "createTabley.jl", or view it by looking at the associated Excel notebooks, where available. 

The key files in the matrix completion folder are the "exact_approaches.jl" file, which contains a number of functions for single-tree and multi-tree methods for solving matrix completion problems exactly, convex_penalties.jl which contains the functions used to optimize over various convex relaxations
, and alt_min.jl which contains the code for the alternating minimization method (the Euclidean Distance Matrix Completion folder is similar). 

The remaining files are script files which can be used to reproduce the numerical experiments in the paper.

## A High-Level Overview
I gave a high-level talk on mixed-projection optimization at INFORMS 2020. The slides are available [here](https://ryancorywright.github.io/pdf/MixedProjection_INFORMS.pdf) and a recording of the talk is available [here](https://drive.google.com/file/d/179UW6-XTkrHkQ2QTxPhZgXP079Nt8fWz/view?usp=sharing)

## Citing MixedProjectionSoftware

If you find MixedProjectionSoftware useful in your research, we encourage you to (1) star this repository and (2) cite the following [paper](http://www.optimization-online.org/DB_HTML/2020/09/8031.html):
```
@article{bertsimas2020mixed,
  title={Mixed-Projection Conic Optimization: A New Paradigm for Modeling Rank Constraints},
  author={Bertsimas, Dimitris and Cory-Wright, Ryan and Pauphilet, Jean},
  journal={arXiv preprint arXiv:2009.10395},
  year={2020}
}
```
## Thank you

Thank you for your interest in MixedProjectionSoftware. Please let us know if you encounter any issues using this code, or have comments or questions.  Feel free to email us anytime.

Dimitris Bertsimas
dbertsim@mit.edu

Ryan Cory-Wright
ryancw@mit.edu

Jean Pauphilet
jpauphilet@london.edu
