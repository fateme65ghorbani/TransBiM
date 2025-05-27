# TransBiM
TransBiM: Antimicrobial Peptide Prediction through a Transformer based Method Enhanced by Bi-LSTM layer and Protein Pre-trained Network

TransBiM is an advanced deep learning framework designed for the accurate prediction of antimicrobial peptides (AMPs). By integrating large language model embeddings, bidirectional sequence modeling, and transformer-based attention mechanisms, TransBiM achieves state-of-the-art performance on multiple benchmark datasets.

## Methodology
TransBiM employs a three-stage hybrid architecture:

### Feature Extraction with T5-XL
Peptide sequences are embedded using the T5-XL model, a powerful pre-trained Transformer optimized for biological sequence representation. It generates 1024-dimensional embeddings that capture both structural and functional features.
### Sequence Modeling with Bi-LSTM and Transformer Encoder
The embeddings are processed by a Bidirectional LSTM (Bi-LSTM) layer to model local and global dependencies in both forward and backward directions. A Transformer Encoder with Self-Attention further captures complex, long-range interactions between amino acid residues.

### Classification via Multi-Layer Perceptron (MLP)
The final sequence representations are fed into a fully connected MLP consisting of two dense layers with ReLU activations, Dropout for regularization, and a SoftMax output layer to predict AMP activity.
This hybrid design effectively leverages the strengths of LLM-based embeddings, sequential pattern learning, and deep feature abstraction.
## Visualization architectural model
![Fig1 ](https://github.com/user-attachments/assets/b9d74094-f135-4a6a-a115-73968a61df88)
Figure 1 shows the complete workflow of TransBiM:

(A) Input peptide sequences
(B) Embedding generation using T5-XL
(C) Feature extraction via Bi-LSTM
(D) Enhancement through Transformer Encoder with Attention
(E) Final classification using fully connected layers (MLP with ReLU, Dropout, and SoftMax)
## Datasets

TransBiM is evaluated on six publicly available datasets:

| Dataset | Source Link |
|--------|-------------|
| DS1    | [LMPred_AMP_Prediction](https://github.com/williamdee1/LMPred_AMP_Prediction) |
| DS2    | [E-CLEAP](https://github.com/Wangsicheng52/E-CLEAP) |
| DS3    | [UniproLcad](https://github.com/harkic/UniproLcad) |
| DS4    | [diff-amp](https://github.com/wrab12/diff-amp) |
| DS5    | [sAMP-VGG16 Dataset](https://figshare.com/articles/dataset/Supporting_Data_for_manuscript_entitled_sAMP-VGG16_Force-field_assisted_image-based_deep_neural_network_prediction_model_for_short_antimicrobial_peptides_/25403680) |
| DS6    | [Antimicrobials_](https://github.com/Shazzad-Shaon3404/Antimicrobials_) |



## Project Structure
This repository is organized into two main parts:

### TransBiM1: Training and Evaluation on DS1
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
### TransBiM2: Testing Across DS2-DS6
Dataset2_Clone_and_Split.ipynb to Dataset6_Clone_and_Split.ipynb
Scripts to preprocess and split datasets DS2 to DS6.

Final_Testing_Multiple_Datasets.ipynb
Performs final evaluation across all datasets using the trained model.

Dataset4_CV_Training_and_Final_Testing.ipynb
Implements cross-validation (CV) training and reports averaged metrics for DS4.

## How to Run

### 🏋️ Train TransBiM on DS1 from scratch:

1. Run `01_Word_Embeddings_Creation.ipynb` to generate T5-XL embeddings.
2. Execute `02_TransBiM_Model_Architecture.ipynb` to prepare the model.
3. Run `03_TransBiM_Training_and_Testing.ipynb` to train and validate.

### 📊 Evaluate TransBiM on DS2–DS6:

1. Preprocess each dataset with `DatasetX_Clone_and_Split.ipynb`.
2. Run `Final_Testing_Multiple_Datasets.ipynb` for final evaluation.

ℹ️ **Note:** For DS3 and DS6, only training and test data are used (no 70-20-10 split).

## Requirements
Python: 3.6.12
TensorFlow: 2.1
PyTorch: 1.7.1
Transformers: 4.5.0
NumPy: 1.19
Pandas: 1.1.3
Matplotlib: 3.3.2
Seaborn: 0.11.0
scikit-learn: 0.24.2
