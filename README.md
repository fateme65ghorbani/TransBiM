# TransBiM
TransBiM: Antimicrobial Peptide Prediction through a Transformer based Method Enhanced by Bi-LSTM layer and Protein Pre-trained Network

TransBiM is an advanced deep learning framework designed for the accurate prediction of antimicrobial peptides (AMPs). By integrating large language model embeddings, bidirectional sequence modeling, and transformer-based attention mechanisms, TransBiM achieves state-of-the-art performance on multiple benchmark datasets.

## Methodology
TransBiM employs a three-stage hybrid architecture:

# Feature Extraction with T5-XL
Peptide sequences are embedded using the T5-XL model, a powerful pre-trained Transformer optimized for biological sequence representation. It generates 1024-dimensional embeddings that capture both structural and functional features.

# Sequence Modeling with Bi-LSTM and Transformer Encoder
The embeddings are processed by a Bidirectional LSTM (Bi-LSTM) layer to model local and global dependencies in both forward and backward directions. A Transformer Encoder with Self-Attention further captures complex, long-range interactions between amino acid residues.

# Classification via Multi-Layer Perceptron (MLP)
The final sequence representations are fed into a fully connected MLP consisting of two dense layers with ReLU activations, Dropout for regularization, and a SoftMax output layer to predict AMP activity.

This hybrid design effectively leverages the strengths of LLM-based embeddings, sequential pattern learning, and deep feature abstraction.
# Model Architecture Illustration
![Fig1 ](https://github.com/user-attachments/assets/b9d74094-f135-4a6a-a115-73968a61df88)
Figure 1 visualizes the complete workflow of TransBiM:

(A) Input peptide sequences
(B) Embedding generation using T5-XL
(C) Feature extraction via Bi-LSTM
(D) Enhancement through Transformer Encoder with Attention
(E) Final classification using fully connected layers (MLP with ReLU, Dropout, and SoftMax)
# Datasets
TransBiM is evaluated on six publicly available datasets:

Dataset	Source Link
DS1 	           LMPred_AMP_Predictionhttps://github.com/williamdee1/LMPred_AMP_Prediction
DS2	               https://github.com/Wangsicheng52/E-CLEAP
DS3	               https://github.com/harkic/UniproLcad
DS4	               https://github.com/wrab12/diff-amp
DS5                https://www.google.com/url?q=https%3A%2F%2Ffigshare.com%2Farticles%2Fdataset%2FSupporting_Data_for_manuscript_entitled_sAMP-VGG16_Force- 
                   field_assisted_image-based_deep_neural_network_prediction_model_for_short_antimicrobial_peptides_%2F25403680
DS6	               https://github.com/Shazzad-Shaon3404/Antimicrobials_
For robust model evaluation, all datasets were stratified into training (70%), validation (20%), and testing (10%) subsets to preserve AMP/non-AMP class distributions.
# Project Structure
This repository is organized into two main parts:

# TransBiM1: Training and Evaluation on DS1
01_Word_Embeddings_Creation.ipynb
Generates T5-XL based embeddings for DS1.

02_TransBiM_Model_Architecture.ipynb
Defines the complete TransBiM architecture and prepares resized input data.

03_TransBiM_Training_and_Testing.ipynb
Trains and validates the model on DS1.

best0021_model.pth
The best-performing trained model checkpoint.

TransBiM_Complete_Implementation.py
End-to-end implementation script for TransBiM training and inference.
# TransBiM2: Testing Across DS2-DS6
Dataset2_Clone_and_Split.ipynb to Dataset6_Clone_and_Split.ipynb
Scripts to preprocess and split datasets DS2 to DS6.

Final_Testing_Multiple_Datasets.ipynb
Performs final evaluation across all datasets using the trained model.

Dataset4_CV_Training_and_Final_Testing.ipynb
Implements cross-validation (CV) training and reports averaged metrics for DS4.

# How to Run
# Train TransBiM on DS1 from scratch:

Execute 01_Word_Embeddings_Creation.ipynb
Then 02_TransBiM_Model_Architecture.ipynb
Finally 03_TransBiM_Training_and_Testing.ipynb
# Evaluate TransBiM on DS2-DS6:

Preprocess each dataset using the corresponding DatasetX_Clone_and_Split.ipynb
Run Final_Testing_Multiple_Datasets.ipynb to evaluate.
In datasets 3 and 6, we had independent test sets, as mentioned in the related paper from which the data was obtained. Therefore, in this code, we did not perform a 70-20-10 split, and we only had training and test data.
# Requirements
Python: 3.6.12
TensorFlow: 2.1
PyTorch: 1.7.1
Transformers: 4.5.0
NumPy: 1.19
Pandas: 1.1.3
Matplotlib: 3.3.2
Seaborn: 0.11.0
scikit-learn: 0.24.2
