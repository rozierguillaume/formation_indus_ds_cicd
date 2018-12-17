from setuptools import setup

with open("requierments.txt", "r") as requirements_file:
    requirements = requirements_file.read().split()

setup(
    name='formation_indus_ds',
    version='0.1.0',
    packages=['src'],
    url='',
    license='',
    author='Octo-TOUL',
    author_email='',
    description='Demonstration pour la formation indus de la data science',
    install_requires=requirements
)
