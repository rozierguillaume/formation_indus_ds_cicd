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

Retrieve project from gitlab
```
git clone git@gitlab.com:etoulemonde/formation_indus_ds.git
```

Start a jupyter notebook in the folder
```
cd formation_indus_ds
jupyter-notebook
```

If it's not working, make your jupyter-notebook is installed. To install it :
```
pip3 install jupyter
```

How to follow it?
------------------

It is highly linked to the presentation of the formation.

To navigate between steps change branch. 

To see all branches

```
git branch -a
```

To start the practical work you should checkout branch `0_initial_state`

```
git checkout 0_initial_state
```

For windows users :
-------------------

You will need a `git bash` terminal and a conda terminal : 
- All `git` command should be executed in the `git bash` terminal.
- All `python` and `conda` related command should be executed in the conda terminal. 

For linux users :
-----------------

Every command can be executed in your terminal.
