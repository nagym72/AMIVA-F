
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# **<p align="center">The web-based version of AMIVA-F can be found here:</p>** 
# <p align="center">http://amiva.msp.univie.ac.at/</p>
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


### AMIVA-F-test package


AMIVA-F is a machine learning based algorithm, trained to correlate single point mutations
with disease in FLNc.

## General Information:

AMIVA-F requires the following packages installed:

1) Pymol open source can be installed through conda e.g ``` conda install conda-forge::pymol-open-source```
2) Scikit-learn 1.3.2 can be installed through pypi e.g ``` pip install scikit-learn==1.3.2```
3) Biopython can be installed through pypi e.g ``` pip install biopython```
4) Pandas can be installed through pypi e.g ``` pip install pandas```
5) Numpy can be installed through pypi e.g ``` pip install numpy```
6) Freesasa can be installed through pypi e.g ``` pip install freesasa```
7) Joblib can be installed through pypi e.g ``` pip install joblib```
8) Shap can be installed through pypi e.g ``` pip install shap```
9) Matplotlib can be installed through pypi e.g ``` pip install -U matplotlib```

This was tested for Python 3.8 and its important to use Scikit-learn 1.3.2 because AMIVA-F.joblib is otherwise not readable!


It is recommended to run AMIVA-F in a virtual environment inside Anaconda e.g <br>

----------------------------------------------------------------------------------------------------------------


AMIVA-F requires korp6Dv1.bin for its ddG paramerter, which can be downloaded from:
[korp6Dv1.bin Download for Linux](https://chaconlab.org/modeling/korp/down-korp/item/korp-linux)

After downloading, put the korp6Dv1.bin file (size 331MB) in ./AMIVA_F/Korpm/pot

# Workflow that I used to setup AMIVA-F in python3.8

```conda create --name amiva_test_empty python=3.8```
```conda activate amiva_test_empty```
```mkdir amiva_test```
```cd amiva_test/```
```git clone git@github.com:nagym72/AMIVA-F.git```
```cd AMIVA-F/```
```conda install conda-forge::pymol-open-source```
```pip install scikit-learn==1.3.2```
```pip install Biopython```
```pip install pandas```
```pip install numpy```
```pip install freesasa```
```pip install joblib #already installed with scikit-learn```
```pip install shap```
```pip install matplotlib```

```./AMIVA-F Q 2058 S``` 
This will work but complain about a missing  #dependency. korp6Dv1.bin is missing. We need to download this  #file.
https://chaconlab.org/modeling/korp/down-korp/item/korp-linux - > Download unzip and search for korp6Dv1.bin
mv korp6Dv1 to ./amiva_test/AMIVA-F/Korpm/pot so it ends up as:
./amiva_test/AMIVA-F/Korpm/pot/korp6Dv1.bin

#rerun again AMIVA-F Q 2058 S

./AMIVA-F Q 2058 S  
#this will work now but still might throw a #warning regarding #XDG_SESSION_TYPE graphical output (can be #ignored)

Output should look like this:

Input was converted to standard MET, LYS
PyMOL>refresh_wizard
 ExecutiveRMSPairs: RMSD =    0.055 (4 to 4 atoms)
AMIVA_F predicts: pathological with associated probability: 0.91
Warning: Ignoring XDG_SESSION_TYPE=wayland on Gnome. Use QT_QPA_PLATFORM=wayland to run on Wayland anyway.

Additionally there should be a new file in ./AMIVA-F/Predictions

ls ./Predictions 
There are some files already as example and there should be a new one called: 
	'Shap_plot_GLN 2058 SER.png'



