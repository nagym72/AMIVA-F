
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# **<p align="center">The web-based version of AMIVA-F can be found here:</p>** 
# <p align="center">http://amiva.msp.univie.ac.at/</p>
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


### AMIVA-F-test package


AMIVA-F is a machine learning based algorithm, trained to correlate single point mutations
with disease in FLNc.

## General Information:

AMIVA-F requires the following packages installed:

1) Pymol open source (can be installed through conda e.g ``` conda install conda-forge::pymol-open-source```)
2) Scikit-learn 1.3.2 (can be installed through pypi e.g ``` pip install -U scikit-learn```)
3) Biopython (can be installed through pypi e.g ``` pip install biopython```)
4) Pandas (can be installed through pypi e.g ``` pip install pandas```)
5) Numpy (can be installed through pypi e.g ``` pip install numpy```)
6) Freesasa (can be installed through pypi e.g ``` pip install freesasa```)
7) joblib (can be installed through pypi e.g ``` pip install joblib```)
8) Shap (can be installed through pypi e.g ``` pip install shap```)
9) Matplotlib (can be installed through pypi e.g ``` pip install -U matplotlib```)

This was tested for Python 3.8 and its important to use Scikit-learn 1.3.2 because AMIVA-F.joblib is otherwise not readable!


It is recommended to run AMIVA-F in a virtual environment inside Anaconda e.g <br>

----------------------------------------------------------------------------------------------------------------

Anaconda is a distribution of python programming language that helps with package management and deployment.<br> 
**Its available for Windows, Linux and macOS**.<br> 
Package versions in Anaconda are managed by an internal package managment system which does not mess with your local computer package depository.<br>
It further simplifies Path dependency problems and is outlined below for a full installation of AMIVA-F tested on Windows in Anaconda3.

# Installation for Windows10(Anaconda3):
If you already installed anaconda, you can skip the next step.

## Installation of Anaconda

1. Download Anaconda from https://www.anaconda.com/products/individual
2. Click on the downloaded .exe data.
3. Close all other applications in the background and click **Next>**
4. Accept by clicking **I Agree**
5. Select **Just Me (recommended)** and then **Next>**
6. Accept the default destination folder and click **Next>** 
7. Select **Register Anaconda3 as my default Python 3.8** and click **Install**


# Workflow inside Anaconda3

## Setup of the virtual environment

1. Open the anaconda prompt (anaconda3) which you find by entering **anaconda prompt** into the windows search bar (Found at the left bottom of your screen). 
2. This should open a black command line interface where you now need to enter the following command
 
		conda create -n amivaenv python=3.8
      	
2. This creates a new virtual environment with python 3.8 named amivaenv which will be used to install AMIVA and its dependencies without polluting your local pythonspace.
2. You will get a message telling you what is going to be installed. Enter *y* and press enter.    
	 
3. After creation enter: 
	
		conda activate amivaenv

3. Which will then activate the new environment.


# Getting Java required dependencies

1. Download javabridge from https://www.lfd.uci.edu/~gohlke/pythonlibs/#javabridge<br>
The site will offer you different versions of javabridge but you need the correct version corresponding to your systems bitness.<br>
You can check the bitness of your PC by pressing **windows key + i** together, then navigate to **System** and then chose **About**.<br>
Under **Device specifications** you will find informations about your system type<br>
e.g _64 bit operating system, x64-based processor_<br>
If you work on a **64-bit operating system** you simply need to download<br> 
**javabridge‑1.0.19‑cp38‑cp38‑win_amd64.whl**.<br>
If you work on a **32-bit operating system** you simply need to download<br> 
**javabridge‑1.0.19‑cp38‑cp38‑win32.whl**.<br>    
Note: The cp38 part specified the cpython version we use<br>(3.8 as we created a virtual environment with python 3.8 before, 
In case that you want to use another python version you need to select cp37 version for 3.7 and so on).<br>	  	
2. Open https://adoptopenjdk.net/<br>
Under **Choose a Version** select **OpenJDK11 (LTS)** and for **Choose a JVM** select **HotSpot**<br>
Click the blue button **Latest Release** and wait for the download to be completed.<br>
This will download a .msi file which you should open by left clicking on it (Left bottom of the screen, alternatively found in the Download directory).  
Click **Next**<br>
Check the **I accept the terms in the License Agreement** and click **Next**<br>
**Attention!** Click now the **Set JAVA_HOME variable** and select **Will be installed on local hard drive** _This is the 3rd row in the directory structure_.<br> 
Click **Next**<br>
Click **Install**
If everything proceeded normally you managed to install AdoptOpenJDK. Click **Finish**<br> 		
3. Open http://downloads.gvsig.org/download/gvsig-desktop/runtimes/java/<br>
Select the appropriate version corresponding to your bitness (if you found before that your bitness is 64-bit you would select _For Windows 64 bits (jre-8u72)_<br>
After Downloading the file, leftclick on the .exe file.<br>
Without altering anything, click **Install**<br>
Finish by clicking **Close**<br>
This procedure might be tedious but is essential in order to make sure<br>WEKA(the machine learning platform which is written in JAVA) can be utitlized from within our python script)<br>
Congratulations on installing all required JAVA dependencies.<br>


# Installing everything required for AMIVA-F
If you followed the previous steps, you should now have everything required to make a full installation of AMIVA-F.<br> 
Open the Anaconda3 prompt (enter anaconda3 prompt into the windows search bar, found in the lower left corner of the screen).<br>
This will open again the black cmd line.<br>
If you closed this window during the installation process before, you might need to activate again the virtual environment.<br>
Take a look at the beginning of the command line.<br>
If you find **(base)** at the beginning you need to reactivate your virtual environment.<br>

	conda activate amivaenv

Now you should either find **(amivaenv)** or it was already there because you did not close the terminal before.<br>

1. Open the *File explorer* and navigate to the Download directory _(default directory for downloads)_.<br>
Search for the downloaded **javabridge-1.0.19** file<br>
**Right click** on the **javabridge-1.0.19** file and select **Properties**<br>
In order to access this downloaded file, we need to specify the path where the file can be found.<br>
For that reason we could open an editor and simply assemble the path together there which we will then copy/paste afterwards into anaconda.<br>
You could now open an editor of your choice and do the following:<br>	
Copy the path from the **Location** Point e.g in my case _C:\Users\adm2\Downloads_.<br>
In you case this could look similar like _C:\Users\yourname\Downloads_.<br>
Copy this into the editor and add **\\** and _**javabridge-1.0.19-cp38-cp38-win_amd64.whl**__(in my case it was this version) which can be found at the top of the window.<br>
In your editor, the line should now finally look like this:     
**C:\Users\adm2\Downloads\javabridge-1.0.19-cp38-cp38-win_amd64.whl**<br>
_(again in your case the **adm2** name is different)_<br><br>     
2. In the anaconda prompt you now need to enter the following:<br>
_you need to enter pip install and then copy the path you assembled before in the editor_

		pip install C:\Users\adm2\Downloads\javabridge-1.0.19-cp38-cp38-win_amd64.whl 
If everything worked you will get a message telling you:<br>
**Successfully installed javabridge-1.0.19 numpy-1.20.3**<br><br>
3. Now we install AMIVA-F:

	pip install AMIVA-F

If everything worked you will get a message telling you:<br>
Successfully installed AMIVA-F-0.0.6 biopython-1.79 freesasa-2.1.0 python-weka-wrapper3-0.2.3<br><br>	
4. We require Pymol:

	conda install -c schrodinger pymol

This will ask your permission to install a bunch of files which you accept by **entering y** and pressing **enter**.<br><br>		
5. Open AMIVA-F

Open the **File explorer** and search for your location where you installed **Anaconda3**.<br>
Usually you will find this under **your username**<br>
e.g in my case this was _adm2_ , then select **anaconda3**, **envs**, **amivaenv**,**Lib**, **site-packages**, **AMIVA-F**.<br> 		
If everything worked well you should see a directory containing the **AMIVA-F.py** file.<br>
**Right click** on this file, select **Properties** and **copy the location**<br>
e.g _C:\Users\adm2\anaconda3\envs\amivaenv\Lib\site-packages\AMIVA-F_<br>
Now enter in the command line in anaconda:<br>
_this will look slightly different in your case you need to change **adm2** to **your user name**_
 
	cd C:\Users\adm2\anaconda3\envs\amivaenv\Lib\site-packages\AMIVA-F	 	  	
and then enter:

	python AMIVA-F.py			

This should open now a GUI and you can use AMIVA-F.<br>
For instructions on how to use AMIVA-F see at the bottom of this page.<br>




# Linux installation of AMIVA-F (Ubuntu tested, preliminary documentation):


1. Open the linux bash (cmd). 
2. Install pip _(optional you might have it already)_
 
		sudo apt-get install python3-pip
      	
3. Install Jdk (Java development kit) 
	
		sudo apt-get install default-jdk

4. Install tkinter (GUI) 
	
		sudo apt-get install python3-tk

5. Install PyMol
	
		sudo apt-get install pymol

6. Install AMIVA-F
	
		pip install AMIVA-F

7. Opening AMIVA-F (for me this was the location where AMIVA-F was installed , default)<br>
In your case you need to switch /user/ and /pythonx.xx/ to your respective name and python interpreter version.
	
		cd /home/user/.local/lib/pythonx.xx/site-packages/AMIVA-F/

Afterwards open AMIVA-F via

		python3.8 AMIVA-F.py

**I used python3.8**


# AMIVA-F Tutorial 

Hello, thanks for using the AMIVA-F package in order to predict the pathogenicity for your mutation of interest.
AMIVA-F works fully automated and is easy to use, even in the absence of knowledge about the underlying parameters which are used as input for the neural network.
In order to make a prediction on your mutation of interest, follow the steps below.

1) AMIVA-F works at the protein annotation level, therefore if you have mutations of interest in the c notation (DNA), look up the corresponding p.notation.<br>
Once you have your mutation of interest in protein notation, enter it in the entry field location directly above the blue button **AMIVA-F Analysis**.<br>
The required input should look like this:

		M82K             

This input would correspond to the single point mutation at position 82 in FLNc, where the wildtype amino acid (M, Methionine) is substituted
by the mutated amino acid (K, Lysine).<br>
If you by any chance submit a wrong amino acid (the amino acid you specified for the wildtype position is in fact not what you submitted, e.g FLNc position 82
corresponds to methionine, but you wrote S82K, which would correspond to serine), AMIVA-F automatically corrects you and offers you to proceed calculations with the correct amino acid in in place.<br>

2) Click on the button **AMIVA-F Analysis**.<br>
This will take a couple of seconds and give you afterwards your prediction.<br> 
After you received your prediction, you can save the prediction and its associated parameters<br>used as input for AMIVA-F in simple text format by clicking "Save Mutation prediction".<br>
This will prompt you in order to choose a directory where the outputfile will be saved.<br>
In this outputfile you will not only find the prediction together with its input parameters,<br>but furthermore also details about the associated training set accuracy and many more statistically interesting things.<br>

3) (Optional)
If you want to make another mutation, simply delete (or click **clear**) and enter your new mutation into the entry field at the beginning.<br>
It is sufficient to simply delete the old mutation and enter the new mutation and then to click **AMIVA-F Analysis**.<br>The old parameters will be deleted and everything will proceed as if you just started AMIVA-F.<br>
When you are done, simply leave AMIVA-F by clicking **Exit AMIVA-F** which will turn off the Java Virtual Machine in the background.

Additional Information:
For additional information about the posttranslational sites/ binding partners which are used to feed AMIVA-F, there are 2 buttons where you can take a look into them.<br>

Regarding performance prediction accuracy:<br>
Currently this seems to be a bit buggy (Given that it only gives you only the summary of 1 out of the 10 crossfold validated runs (Instead the average of 10) and can take values a bit lower or higher than reported in the paper).<br>
If you take the trainingset to WEKA (a desktop installed version) and do the cross validation there, you will find the exact numbers specified in the publication of 78.6% accuracy.<br>
For your prediction it can happen that you will find prediction values of 76 - 80%,<br>but these are not true values but instead single models from the 10x cross validation procedure.<br>
We are working to fix this issue in order to provide the average accuracy of 78.6 %, which results after averaging all 10 models generated in cross validation.<br>



**More information can be found at the full tutorial inside the package.**



AMIVA-F requires also Pymol which can be download through conda  

In your active conda environment enter:
``` conda install conda-forge::pymol-open-source```

AMIVA-F requires korp6Dv1.bin for its ddG paramerter, which can be downloaded from:
[korp6Dv1.bin Download for Linux](https://chaconlab.org/modeling/korp/down-korp/item/korp-linux)

After downloading, put the korp6Dv1.bin file (size 331MB) in ./AMIVA_F/Korpm/pot
