
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# **<p align="center">The web-based version of AMIVA-F can be found here:</p>** 
# <p align="center">http://amiva.msp.univie.ac.at/</p>
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


### AMIVA-F-Analysis of pathogenic missense mutations in human FLNC

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

# Explanation of features used by AMIVA-F:

### Feature 1: Clashcounter (Numerical) 

+ Type: Integer 

+ Description: Represents the number of introduced clashes that occurs when the 	mutated amino acid is inserted into the wild type position. 	 

+ Computation and Transformation: We utilised the PyMOL*Schrödinger, L., & DeLano, W. (2020). PyMOL. Retrieved from http://www.pymol.org/pymol* mutagenesis tool, which can be found under https://github.com/Pymol-Scripts/Pymol-script-repo/blob/master/rotkit.py. It allows to introduce the most commonly observed side-chain configuration for a given amino acid and each probability based on introduced clashes and contacts. We selected the most probable side chain orientation, given that this corresponds to the most commonly tolerated rotamer configuration and used it to insert the new mutated amino acid accordingly. Pertinent technical details about Rotkit can be found in the GitHub repository. 
We then further selected all side-chain atoms (excluding N, C, CA, O of the main chain) and computed the number of clashes for the newly introduced side-chain with surrounding atoms (excluding its own main chain atoms). 
Atoms found within 3 Å or less distance to the side-chain were considered as clashes, and each counted as 1 clash. There were no transformations to this feature applied. 

+ Feature Importance: 0.09 

### Feature 2: SAP-score (Spatial Aggregation Propensity) (Numerical) 

+ Type: Float 
Description: A scoring parameter that combines the solvent exposure, and the Hydrophobicity of surrounding residues within a specified volume. 	 
This parameter describes the local chemical neighbourhood and was shown to correlate well with aggregation propensity (Vladimir 	Voynov 	et al 2009). 	 

+ Computation and Transformation: For the Hydrophobicity scale used in this computation, we normalised values so that glycine has a value of 0 (identical to normalisation done in Vladimir Voynov et al 2009), and used FreeSASA (version 2.1.2, https://github.com/mittinatten/freesasa) to compute solvent accessibility. In order to compute the solvent accessibility for each residue, the surface of the protein is represented as a collection of atoms, each represented as a sphere with fixed radius according to its Van der Waals radius of the atom plus the radius of a water molecule. The algorithm then places many points on the surface of these spheres and checks which of the points comes into contact with water. If a point is not blocked by other parts of the protein, it’s considered accessible to the solvent. The algorithm then calculates the total area of these accessible points and returns the area of each amino acid and its exposure to the solvent. For each amino acid, there is a library available corresponding to the fully exposed amino acid in a tri-peptide configuration, where each amino acid is flanked by alanine to simulate full exposure. These values are then taken to compute relative accessibilities based on observed solvent exposure normalized by the “fully exposed” tri-peptide exposition. It should be noted that certain loop-configurations can be “more exposed” (mostly glycine) than the amino acids tri-peptide configuration, resulting in values greater than 1.  
In order to compute solvent accessibilities we chose the following parameters: 	 
 

	+ Algorithm = ShrakeRupley 
	+ probe_radius = 1.4
 	+ n_points = 20 

+ For each atom of a residue, we computed SAP-scores with a cutoff of 8 Å as described in Vladimir Voynov et al 2009. This means for each target atom, we included all side chain atoms within 8 Å of the chosen atom and computed their solvent accessibility. For all side chain atoms and their computed solvent accessibility, we normalized through the fully accessible solvent area that the corresponding residue would have in its tri-peptide configuration explained above. This results in the relative accessible surface area of each side-chain atom. 
We then assigned hydrophobicity values based on the residue the surrounding atoms belongs to e.g. a carbon side-chain atom belonging to valine will be assigned the hydrophobicity value of valine. 
We then computed the product of relative solvent accessibility and hydrophobicity for each atom and assigned this value to the target atom. For a target residue, we took the mean value of the SAP score of its atoms and assigned this mean value to the residue. 

+ Feature Importance: 0.19 

### Feature 3: ddG (change in Gibbs-Free energy) (Numerical) 

+ Type: Float 

+ Description: Predicted change in Gibbs-Free energy upon mutation. 

+ Computation and Transformation: We utilised Korpm (https://doi.org/10.1093/bioinformatics/btad011).  
We computed the absolute ddG value for each predicted mutation to capture the effect that mutations which increase stability are equally associated with pathological disease outcomes as those that decrease stability. 

+ Feature Importance: 0.21 

### Feature 4: Conservation (Numerical) 

+ Type: Float 	 

+ Description: The Lockless-conservation of each residue for residues included in the FLNC Ig domains against all other Ig domains. 

+ Computation and Transformation: For computation of Conservation scores, we used PyCanal (Gado, Japheth E., 2021, "Machine learning and bioinformatic insights into key enzymes for a bio-based circular economy". Theses and Dissertations: Chemical and Materials Engineering, 129. DOI: 10.13023/etd.2021.009.) Input sequence alignment was generated for Ig domains Ig1 - Ig24 with Clustal Omega (https://doi.org/10.1093/nar/gkac240). For mutations introduced in regions outside of the Ig domains (ABD and linker regions), we imputed the average conservation of all other residues. 

+ Feature Importance: 0.19 

### Feature 5: Hydrophobicity-Weight-Number_atoms (Numerical) 

+ Type: Float 

+ Description: A combined feature that consists of the introduced change in hydrophobicity, atomic weight, and number of atoms upon mutation.	 

+ Computation and Transformation: Product of Changes in Hydrophobicity (Kyte Doolittle), atomic weight, and number of atoms (excluding hydrogens) upon mutation. 

+ Feature Importance: 0.10 

### Feature 6: Cost (Numerical) 

+ Type: Float 

+ Description: A cost function that assigns a value based on observed mutational transitions and their associated clinical outcome to each potential mutational transition. Benign mutations are given more importance and for unobserved transitions we assign a cost of 0. The cost feature aims to incorporate observed mutational transitions of real clinical data but care shall be taken with care for transitions that lack sufficient clinical data. 

+ Preprocessing, computation and transformation: 0 if no observed mutational transition found, else cost = number of pathological transitions - 2.5 * number of benign transitions.
Example: Arginine is found to be either mutated or being inserted 4 times (3 times involved in pathological mutations and once in a benign mutation). Cost would be in this case: num_pathogenic_transitions - 2.5*num_benign_transitions = 3*1 - 2.5 * 1 = 0.5. In the absence of any transitions cost takes the default value of 0. The value of 2.5 was chosen to put more weight towards benign mutations, based on the rationale that there is an intrinsic bias towards pathogenic mutations in the databases. This is in agreement with classifications made by other predictors (https://www.science.org/doi/10.1126/science.adg7492 , https://www.nature.com/articles/s41586-021-04043-8), which claim a larger proportion of mutations to be benign. For future versions of AMIVA-F, this parameter and its computation can be fine-tuned based on additional available clinical information and might be useful for better generalization. 

+ Feature Importance: 0.22  
