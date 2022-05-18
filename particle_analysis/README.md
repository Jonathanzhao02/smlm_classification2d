# Setup Instructions
This README is specific to the analysis for our project, and hence is nested in a folder called "particle_analysis".

If you are running on the Thelio Mira, all dependencies should already be set up.
The conda environment can be activated with `conda activate dna_paint`.
Otherwise, you will need to clone and follow the set-up instructions at these repositories:
* https://github.com/DIPlib/diplib
* https://github.com/Jonathanzhao02/smlm_datafusion2d
* https://github.com/Jonathanzhao02/smlm_classification2d

To install the conda environment, run `conda env create --file <path to dna_paint.yml>`.
This will create a new conda environment called "dna_paint" with all the required dependencies.

The software follows this pipeline:
* Processed Picasso data is converted into a compatible (+ labeled) format using the code in picasso_conversion
* Clustering results are created using the MATLAB code from smlm_datafusion2d
* Individual readouts are conducted using `process_particles.py` and `process_particles_mle.py`
* Individual readout analysis is conducted using `process_reads.py`
* Superparticle analysis is conducted using `process_superparticles.py` and `process_superparticles_mle.py`

More details about each of these steps are found below.

# Picasso Data Formatting
The data should be processed by Picasso into the .hdf5 format.
Picks should already have been created.

In order to format the data for use in clustering and readout analysis, utilize the scripts in the `picasso_conversion` folder.

## `convert.py`
This script converts the .hdf5 into a .mat format.

All other scripts utilize the data in a .mat format, so this step must be done for all .hdf5 files.

To use this script, run: `python3 picasso_conversion/convert.py <path to .hdf5 file> [-o <output folder name>] [-t <label name>] [-p <# of picks>]`. Arguments in square brackets are optional, arguments in angled brackets are required.

The arguments work as follows:
* -o / --output: Name of the output folder to save the .mat to. This folder is saved into a folder called `data` in the parent directory of picasso_conversion.
* -t / --tag: Name of the label to associate with each pick in the data. For example, for NSF, the tag is 'N' for N, 'S' for S, and 'F' for F. This enables processing the readouts with respect to their ideal readouts.
* -p / --picks: The number of picks to select from the file. <= 0 means select all picks.

The output is a folder containing a file named `subParticles.mat`.

## `merge.py`
This script combines multiple .mat files into a single file.

Only a single .mat file is accepted by the other scripts, so this script allows the mixture of multiple classes into a single file.

To use this script, run: `python3 picasso_conversion/merge.py <path to folder of .mat files> <output folder name> [-p <# of picks>]`. Arguments in square brackets are optional, arguments in angled brackets are required.

The arguments work as follows:
* -p / --picks: The number of picks to select from each file. <= 0 means select all picks.

The output is a folder containing a file named `subParticles.mat`.

# Spectral Clustering
All scripts related to spectral clustering (method from "Detecting structural heterogeneity in single-moleculue localization microscopy data") run in MATLAB.

Open up MATLAB and navigate to the parent directory of all .m scripts.

## `demo_classification.m`
This script performs spectral clustering, including the all2all registration step.
It requires a folder in `data` which contains a single `subParticles.mat` file.

Parameters are at the top of the file. These include:
* dataset: The name of the dataset in `data` to run over.
* K: The number of clusters in the dataset.
* C: The number of clusters to pare down to during the eigen-image clustering step. This is unused.
* width: Affects the width of visualizations.
* furtherClustering: Whether to perform the eigen-image clustering step. This is unused, so it is kept at false.
* scale: A measure of uncertainty in localization position in pixel values. Affects the alignment of the particles to one another and the dissimilarity measurement.
* nAngles: The number of angles to test with the Bhattacharya cost function. Affects accuracy of alignment and the dissimilarity measurement.

The most significant output is an all2all_matrix folder, figures of the fusion of each class, a figure of the MDS space in 3D, `classes.mat`, which contains the picks separated into clusteres, `clusters.mat`, which contains the identifications of each pick with each class, `superParticle.mat`, which is the superparticle rendering of each cluster, and miscellaneous other information.

## `demo_classification_withoutAll2All.m`
This script performs spectral clustering, assuming the all2all registration step has already been performed.
This helps save time in the case that classification must be rerun.

Parameters are at the top of the file. These include:
* dataset: The name of the dataset in `data` to run over.
* K: The number of clusters to use during K-Means. Note the final number of clusters may be different.
* C: The number of clusters to pare down to during the eigen-image clustering step. This is unused.
* width: Affects the width of visualizations.
* furtherClustering: Whether to perform the eigen-image clustering step. This is unused, so it is kept at false.
* scale: A measure of uncertainty in localization position in pixel values. Affects the alignment of the particles to one another and the dissimilarity measurement.
* nAngles: The number of angles to test with the Bhattacharya cost function. Affects accuracy of alignment and the dissimilarity measurement.

The most significant output is the figures of the fusion of each class, a figure of the MDS space in 3D, `classes.mat`, which contains the picks separated into clusteres, `clusters.mat`, which contains the identifications of each pick with each class, `superParticle.mat`, which is the superparticle rendering of each cluster, and miscellaneous other information.

## `fig2img.m`
This script takes the figures generated by the previous scripts, and turns them into .png images.
It requires that classification has been fully performed.

Parameters are at the top of the file. These include:
* dataset: The name of the dataset in `data` to run over.
* K: The number of clusters in the dataset.
* C: The number of clusters the dataset was pared down to during the eigen-image clustering step. This is unused.

The output is multiple .png images of each cluster, the MDS space in 3D, the MDS space in 3D with colored clusters, and a random particle.

## `all_fusion.m`
This script creates a superparticle rendering of all particles layered on one another.
It requires that all2all registration has already been performed in a separate step.

Parameters are at the top of the file. These include:
* dataset: The name of the dataset in `data` to run over.
* width: Affects the width of the visualization.
* scale: A measure of uncertainty in localization position in pixel values. Affects the alignment of the particles to one another.
* nAngles: The number of angles to test with the Bhattacharya cost function. Affects accuracy of alignment.

The output is a file called `superParticle.fig` in the dataset folder.

## `cluster_analysis.m`
This script analyzes the clusters resulting from spectral clustering.
It requires that classification has been fully performed.

Parameters are at the top of the file. These include:
* dataset: The name of the dataset in `data` to run over.
* K: The number of clusters in the dataset.
* tags: The names of the tags to look for in the dataset. Depends on how you decided to tag data during the data formatting step.

The output is printed directly to standard output (the terminal).

## `kmeans_elbow.m`
This script performs K-Means clustering repeatedly over the MDS space and displays the inertia curve.
It requires that all2all registration has already been performed in a separate step.

Parameters are at the top of the file. These include:
* dataset: The name of the dataset in `data` to run over.
* K: The maximum number of clusters to create in the dataset.

The output is a figure window which contains the inertia curve.

## `scale_sweep.m`
This script performs the scale sweep mentioned in the paper.
Although it matches the procedure outlined by the authors, we have found that using a scale value of 0.03 works better than the value found by this method.

Parameters are at the top of the file. These include:
* dataset: The name of the dataset in `data` to run over.
* nAngles: The number of angles to test with the Bhattacharya cost function. Affects accuracy of alignment.
* scales: The scale values to iterate over and test.

The output is a figure window which contains the scale sweep results.

# Individual Particle Readout
## Valid methods/alignments/templates
VALID_CLUSTER_METHODS = ['kmeans', 'dbscan', 'meanshift', 'mle']
VALID_ALIGNMENT_METHODS = ['differential_evolution', 'shgo', 'dual_annealing', 'rough']
VALID_TEMPLATE_NAMES = ['nsf', 'asu_2', 'asu_3']

## Description
The below scripts perform particle readout and analysis over individual picks.
It assumes picks have already been assigned to clusters, although these scripts could be augmented to work directly on the `subParticles.mat` file.

Within both files are various boolean parameters at the top of the file, which control interactive visualizations. Turning them on/off is mainly a matter of convenience by speeding up certain scripts.

## `process_particles.py`
This script performs and saves readouts over each individual pick.
It assumes spectral clustering has already been performed.

To use this script, run: `python3 particle_analysis/process_particles.py <path to dataset folder> [-o <output file name>] [-t <template name>] [-c <clustering method>] [-a <alignment method>] [-j <path to config file>]`. Arguments in square brackets are optional, arguments in angled brackets are required.

The arguments work as follows:
* -o / --output: The name of the output .mat file which contains all readouts.
* -t / --template: The name of the template to use. Templates are in the `templates` folder.
* -c / --cluster: The name of the clustering method to use, or MLE.
* -a / --alignment: The name of the alignment optimization method to use. Unused for MLE.
* -j / --config: The path to the config file for the methods.

The output is a single file containing the readout results, including centroid positions, cost function values, aligned grid positions, binary readout, correctness, etc.

## `process_reads.py`
This script analyzes the readout results from `process_particles.py`.

To use this script, run: `python3 particle_analysis/process_reads.py <path to dataset folder> [-i <readout file>] [-o <output file name>] [-t <template name>]`. Arguments in square brackets are optional, arguments in angled brackets are required.

The arguments work as follows:
* -i / --infile: The name of the file containing individual readouts located within the dataset folder.
* -o / --output: The name of the file to save analysis results to. These consist currently only of misclassifications.
* -t / --template: The name of the template to use. Templates are in the `templates` folder.

The output is a folder containing the misclassifications visualizations. Various figures and standard output consisting of analysis results are also displayed.

# Superparticle Readout
## Valid methods/alignments/templates
VALID_CLUSTER_METHODS = ['kmeans', 'dbscan', 'meanshift', 'mle']
VALID_ALIGNMENT_METHODS = ['differential_evolution', 'shgo', 'dual_annealing', 'rough']
VALID_TEMPLATE_NAMES = ['nsf', 'asu_2', 'asu_3']

## Description
The below script performs particle readout over the superparticles of each cluster.
It assumes spectral clustering has already been performed.

## `process_superparticles.py`
This script performs readout over the superparticle results.
Within this file are various boolean parameters at the top of the file, which control interactive visualizations. Turning them on/off is mainly a matter of convenience.

To use this script, run: `python3 particle_analysis/process_superparticles.py <path to dataset folder> [-t <template name>] [-c <clustering method>] [-a <alignment method>] [-j <path to config file>]`

The arguments work as follows:
* -t / --template: The name of the template to use. Templates are in the `templates` folder.
* -c / --cluster: The name of the clustering method to use, or MLE.
* -a / --alignment: The name of the alignment optimization method to use. Unused for MLE.
* -j / --config: The path to the config file for the methods.

Various figures and standard output consisting of analysis results are displayed.
