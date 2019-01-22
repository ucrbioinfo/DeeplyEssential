# DeeplyEssential

DeeplyEssential is a Convolutional neural network for the identification of Essential genes in bacteria species. The dataset used for the learning of the nework contains 30 bacterial species collected from DEG. 

<h3>Dependency </h3>

1. Python 2.7
2. keras==2.1.5
3. numpy==1.14.2
4. pandas==0.22.0
5. scikit-learn==0.19.1
6. tensorflow==1.6.0

<h3>Parameters</h3>

DeeplyEssential takes 6 parameters
1. Essential gene directory path
2. Non Essential gene directory path
3. Clustered gene file path clusted by OrthoMCL
4. Text file containing bacteria species information
5. Experiment option
	- '-gp' for Gram Positive (GP) Dataset
	- '-gn' for Gram Negative (GN) Dataset
	- '-c' for GP + GN dataset 
6. Name of the experiment

<h4>Run code</h4>

```
$ python 
```