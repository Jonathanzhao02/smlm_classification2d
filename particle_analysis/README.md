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

# Spectral Clustering

# Individual Particle Readout

## Individual Particle Readout Analysis

# Superparticle Readout
