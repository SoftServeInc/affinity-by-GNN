# affinity-sampling

## Introduction

This repository contains the code for the paper 

focused on two aspects of designing AI-driven tools for solving the early-stage drug discovery problems. 
First, it reports a new GNN architecture for predicting affinity of small molecule ligand to protein target using 
a novel graph-based neural network architecture.
Second, it showcases how naive application of commonly used performance evaluation strategies can yield overly optimistic
performance metrics for a given ML model.


## Installation

The code depends on a number of packages (moreover, specific combination of their versions facilitates
good performance in terms of speed) and we recommend using [conda](https://conda.io/)
package manager to automatically install these dependencies. To do so, ensure you have `conda` (actually, 
[conda](https://docs.conda.io/en/latest/miniconda.html#installing) distribution should be pretty enough) installed,
clone the repository and run

    conda env create -f environment.yml

this will create an environment called `affgnn` and installed all neccessary packages there.
After that, use

    conda activate affgnn

to make the new envoronment usable.
Note, however, that the presented configuration also requires CUDA to be usable on the syetem
(the packages to be installed use CUDA 10.2).


## Configuration and use

The code can be run as

    python train.py

which will first train the model using the folds marked `0`,`1`,`2`,`3`, then make prediction on fold `4` and save the
predicted affinities as text files.

The input data needs to be supplied in two ways: a `.csv` defining the structure of ligands, PDB codes and UniProtIDs 
of receptors and the distrubution of the data over five folds, and `.dssp` files defining the secondary structure
elements of target proteins.

Location of the `input.csv` containing ligand SMILES strings and other into, as well as the location
of the folder containing .dssp files for receptors, should then be set by editing `affinity_module/config.py`
(see `master_data_table` and `dssp_files_path` keys respectively).

To prepare receptor data in .dssp format, the underlying receptor structures, eitehr in PDB or CIF format,
should be processed with DSSP program.

