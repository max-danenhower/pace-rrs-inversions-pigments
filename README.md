Using the environment.yml file in the git repository, create a new conda environment using the terminal and open Jupyter Lab:

conda env create --file environment.yml
conda activate pace
python -m ipykernel install --user --name pace
jupyter lab
