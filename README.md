# Clinical and Epidemiological Profile of Parkinson's Disease in India

This package serves as companion to a research paper on Parkinson's disease in India. It consists of two parts, a command line interface and a set of notebooks.

## Command Line Interface

The CLI of the package is tailored to transform the raw data coming from the RedCap database and compute new features needed for subsequent analysis. The usage is quite simple once installed on the system or on a virtual environment. The input file most be a STATA file and the output file will be a CSV file.

```
python3 luxgiant_clinical --input-file <path to input file> --output-folder <path to output folder>
```

## Notebooks

There are two sets of notebooks: `notebooks_aux` and `notebooks_finals`.

The first group contains the notebooks needed to build intermediate tables about the clinical and epidemiological profile of PD. The resultant tables are stored in the folder `data/auxiliar`.

The second group contains the notebooks with the tables as shown in the research paper. The resultant tables are stored in the folder `data/final`.
