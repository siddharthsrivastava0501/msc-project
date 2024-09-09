#!/bin/bash
#PBS -l walltime=05:00:00
#PBS -lselect=1:ncpus=1:mem=1gb
#PBS -J 1-75

module load tools/prod
module load SciPy-bundle/2022.05-foss-2022a
source ~/venv/bin/activate

cd $PBS_O_WORKDIR

python3 generate-plots.py $PBS_ARRAY_INDEX
# python3 generate_plots_hopf.py $PBS_ARRAY_INDEX
