# DeeplyEssential

DeeplyEssential is a Deep neural network for the identification of Essential genes in bacteria species. The dataset used for the learning of the nework contains 30 bacterial species collected from DEG. 

<h3>Dependency </h3>

1. Python 2.7
2. keras==2.1.5
3. numpy==1.14.2
4. pandas==0.22.0
5. scikit-learn==0.19.1
6. tensorflow==1.6.0

<h3>GPU </h3>
Titan GTX 1080 Ti

<h3>Parameters</h3>

DeeplyEssential takes 6 parameters
1. Essential gene directory path. The directory contains
	- A essential gene sequence file
	- A essential protein sequence file
	- An gene annotation file
2. Non Essential gene directory path. This directory contains
	- A essential gene sequence file
	- A essential protein sequence file
	- An gene annotation file
3. Clustered gene file path clusted by OrthoMCL (sample given, `orthoMCL.txt`)
4. Text file containing bacteria species information (sample given, `dataset.txt`)
5. Experiment option
	- '-gp' for Gram Positive (GP) Dataset
	- '-gn' for Gram Negative (GN) Dataset
	- '-c' for GP + GN Dataset 
6. Name of the experiment

<h4>Run code</h4>

```
$ python main.py <essential gene dir> <non-essential gene dir> <cluster gene file> <dataset> -c <experiment name>
```

The dataset are collected from [DEG](http://www.essentialgene.org/). 

<h4>Output</h4>

DeeplyEssential generates a report containing experiment name, basic statistics about the dataset and evaluation metics for each iteration of experiment. A sample (`sample_output.tab`) is provided.
