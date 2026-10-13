## Setup
Make sure your machine is set up with the assignment dependencies.

**[Option 1] Use Google Colab (Recommended):**
The preferred approach to do this assignment is to use [Google Colab](https://colab.research.google.com/).


**[Option 2] Use a local Conda environment:**
Another approach for installing all the assignment dependencies is to use
[Anaconda](https://www.continuum.io/downloads), which is a Python distribution
that includes many of the most popular Python packages for science, math,
engineering and data analysis. Once you install it you can skip all mentions of
requirements and you are ready to go directly to working on the assignment.

```bash
conda create -n cs182hw2 python=3.8 jupyter
conda activate cs182hw2
python -m pip install numpy==1.21.6 imageio==2.9.0 matplotlib==3.2.2
```

**Download data:**
Once you have the starter code, you will need to download the CIFAR-10 dataset.

```bash
cd deeplearning/datasets
./get_datasets.sh
cd ../..
```

If you are on Mac, this script may not work if you do not have the wget command
installed, but you can use curl instead with the alternative script.
```bash
cd deeplearning/datasets
./get_datasets_curl.sh
cd ../..
```

**Start Jupyter:**
After you have the CIFAR-10 data, you should start the IPython notebook server
from this directory.
```bash
jupyter notebook
```

Once you get started, you can open the Jupyter Notebook `hw2_optimizer_init.ipynb` for further instructions.
