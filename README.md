# CrossMatch
Implementation of the paper "Context Consistency Regularization for Label Sparsity in Time Series" (ICML'23)


This folder contains source codes and a part of datasets for ICML 2023 submitted paper whose title is "Context Consistency Regularization for Label Sparsity in Time Series".

To run the source codes, please follow the instructions below.

1. We require following packages to run the code. Please download all requirements in your python environment. Datasets are already preprocessed in dataset folder.
	- python 3.9.12
	- tensorflow 2.9.1
	- numpy 1.21.5
	- pandas 1.4.2
	- matplotlib 3.5.1
	- tqdm 4.64.6

2. At current directory which has all source codes, run crossmatch.py (using context-attached augmentation) or fixmatch.py (using jittering and scaling) with the parameters as follows. 
	- dataset: {mHealth, HAPT, opportunity}   # designate which dataset to use.
	- seed: {0, 1, 2, 3, 4}	# seed for 5-fold cross validation.
	- gpu: an integer for gpu id
	- pltest: {0,1,2,3} # representing {CrossMatch, FixMatch, FlexMatch, PropReg} for crossmatch.py
			  {0,1,2} # representing {FixMatch, FlexMatch, PropReg} for fixmatch.py
	- overlap: an integer in [1, infinity]  # representing the number for 2o, the length of a target instance. If not designated, it uses the default value described in Table 4 of the paper.
	- window: an integer in [1, infinity] # representing overlap + context length. If not designated, it uses the default value described in Table 4 of the paper.
	- lambda1: a float in [0, infinity] # representing \lambda in the paper, the weight of unlabeled loss.
	- mul_label_per_class: a float in [1.0, infinity] # representing a multiplier for the number of initial labels.

e.g.) python3 crossmatch.py --dataset HAPT --window 1280 --overlap 1024 --lambda 1 --mul_label_per_class 1.0 --pltest 1 --gpu 0 --seed 0 
e.g.) python3 fixmatch.py --dataset opportunity --window 1088 --overlap 1024 --lambda 1 --mul_label_per_class 10.0 --pltest 0 --gpu 2 --seed 1 

3. Results are saved in ./metadata as in .npy format.
