# Cognitive Radio Machine Learning
Using Machine Learning for Spectrum Awareness

# Dependencies
Most Notebooks will be written in Python 3. The project uses the following tools:

* Python Version: 3.5.3
* Pandas Version: 0.20.1
* matplotlib Version: 2.0.2
* NumPy Version: 1.12.1
* SciPy Version: 0.19.0
* IPython Version: 6.0.0
* Scikit-learn Version: 0.18.1
* graphviz Version: 0.7.1
* Mglearn Version: 0.1.5
* Jupyter Notebook

You can install all this tools via pip by running:

    $ sudo pip install numpy scipy scikit-learn matplotlib pandas graphviz mglearn
    $ sudo pip install jupyter

It is also recommended that you install the dependencies in a Python Virtual
Env. The procedure to set up one is very straight forward:

## Usage with Conda
In the repository you will find an updated version of the frozen conda environment
that you can use to set up your own. Just run

    $ conda env create -f env.yaml -n <env_name>

where env_name is the name that you want to give to your environment. The whole
installation should run automatically, and you should just activate the environment

    $ source activate <env_name>

If you then want to remove the env, just run:

    $ conda env remove -n <env_name>

## Other installation methods

### Install virtualenv

    $ pip3 install virtualenv

In case this rises, then try using the packaged version. For example, in Ubuntu:

    $ sudo apt install python3-env

### Create a new virtualenv

    $ python3 -m venv <envname>

### Activate the environment

    $ . <envname>/bin/activate

### Install the requirements
Provided there is a list with the requirements of this project. You can install
them using the list and pip:

    $ pip3 install -r requirements.txt

### Exit de Environment when done

    $ deactivate

### Updating the requirements

When new requirements are added for the project, or you add custom reqs
yourself, you can update the project using pip as well and updating the list in
the requirements file:

    $ pip3 freeze > requirements.txt

# The Datasets

The datasets are extracted from the DySpan Spectrum Challenge 2017 Setup [1] and
the procedures for the feature extraction are under development. I plan to add
the recorded data in some sort of online host. Ie. I haven't done this just yet.

# "It ain't working!" #
    ¯\_(ツ)_/¯ It works on my machine!

However please feel free to report any problems or suggest improvements!

