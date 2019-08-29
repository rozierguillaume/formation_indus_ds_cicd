What is this?
-------------

This is a practical work to illustrate Data Science industrialization.

How do I install it?
-------------------
First, make sure you have miniconda or anaconda installed. If not, install it!

Create a conda env
```
conda create -n python_indus python=3.7
conda activate python_indus
```

Install ipykernel and add kernel to jupyter notebook
```
conda install ipykernel
python -m ipykernel install --user --name python_indus --display-name "Python indus"
```

Retrieve project from gitlab
```
git clone git@gitlab.com:etoulemonde/formation_indus_ds.git
```

Start a jupyter notebook in the folder
```
cd formation_indus_ds
jupyter-notebook
```

How to follow it?
------------------

It is highly linked to the presentation of the formation.

To navigate between steps change branch. The initial state of the project is "v0"

For example, the first branch is "v1":
```
git checkout v1
```
