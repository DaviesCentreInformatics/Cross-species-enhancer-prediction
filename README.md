# Cross-species-enhancer-prediction
Repo for code relating to our cross-species enhancer predicition work. Insert citation

## Download the high confidence enhancers here:  
[**Cattle**](https://github.com/DaviesCentreInformatics/Cross-species-enhancer-prediction/blob/main/ARS-UCD1.2.200bp.highConfidence.Enhancer.predictions.bed.gz)  
[**Pig**](https://github.com/DaviesCentreInformatics/Cross-species-enhancer-prediction/blob/main/SSCROFA11.1.200bp.highConfidence.Enhancer.predictions.bed.gz)  
[**Dog**](https://github.com/DaviesCentreInformatics/Cross-species-enhancer-prediction/blob/main/DOG_10K.BOXER.200bp.highConfidence.Enhancer.predictions.bed.gz)  

## Methods Overview
![](fig1.png)

## Training Data
Screenshots from the VISTA database showing how the positive examples were retrieved from the database.
**Human**
![](Screen%20Shot%202022-07-14%20at%202.13.59%20pm.png)


**Mouse**
![](Screen%20Shot%202022-07-14%20at%202.14.17%20pm.png)

Code relating to generating the training data from VISTA can be found in these notebooks.
- Processing VISTA fasta file into ML data. [Process VISTA](data_process_Vista_clean.ipynb)
- Convert fasta to one-hot encoding. [seq2one-hot](fasta_process_Seq2onehot.ipynb)
- Convert fasta to image. [Seq2Img](fasta_process_DNA2Img.ipynb)
- Convert fasta to word-vector. [Seq2Word-vector](fasta_process_word_vector.ipynb)

## Model Permutations
Code relating to model permutations can be found here:
- [Permute model](clean_permutations.ipynb)

