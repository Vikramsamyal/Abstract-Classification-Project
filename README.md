# Abstract Classification Using SVM and TF-IDF

## Problem Statement
Given an abstract of a paper, the objective of this task is to build a reliable ML model that takes as input that abstract and classifies it into one of the 7 predefined domains.
The names of the pre-defined domains are as follows:

- Computation and Language (CL)
- Cryptography and Security (CR)
- Distributed and Cluster Computing (DC)
- Data Structures and Algorithms (DS)
- Logic in Computer Science (LO)
- Networking and Internet Architecture (NI)
- Software Engineering (SE)

## Dataset Access
The dataset can be found at the following URL: 
Dataset credits: [Mendeley Data](https://data.mendeley.com/datasets/njb74czv49/1)

## Dataset Description
The dataset consists of two files in CSV format: one for training and one for validation.

**Training Dataset**:
- Number of rows: 16,800
- Total number of words: 2,607,827
- Average number of words per abstract: 155.23

**Validation Dataset**:
- Number of rows: 11,200
- Total number of words: 1,734,021
- Average number of words per abstract: 154.82

I have set the column headers in the datasets as abstract (the text of the abstract) and domain (the target classification domain).

## Model Selection
I tested building the algorithm based on the following two models and analyzed the results:
- Logistic Regression with TF-IDF Vectorization
- Support Vector Machine (SVM) with TF-IDF Vectorization

## Approach Walkthrough
The best weighted F1 score of 91.51 was achieved using SVM with TF-IDF vectorization for building the model. Below is a walkthrough of the approach:

**Data Loading**: The training and validation datasets are loaded into data frames with columns named abstract and domain (Datasets originally didn’t have any headers).

**Label Encoding**: Then I encoded the target domain labels into numerical values using LabelEncoder. This is because ML algorithms require numerical input.

**Text Preprocessing**: A simple preprocessing function is applied to convert the text to lowercase, which helps in standardizing the text data. 
I tested advanced text preprocessing techniques such as removing stop words, stemming, and lemmatization but they ended up reducing the model’s quality.

**Feature Extraction**: I made use of TF-IDF vectorization to transform the text data into numerical features as it is quite simple yet robust. 
I experimented with multiple values of max_features and did hyperparameter tuning (10000, 12500, 14000, and 13000). The best value came out to be 12500 maximum features.

**Model Initialization**: An SVM model with a linear kernel and a regularization parameter C=1.0 is used. 
I chose a linear kernel for its effectiveness in text data, and C=1.0 because it provides a good trade-off between margin maximization and classification error minimization.

**Model Training**: Finally, I trained the SVM model on the TF-IDF transformed training data.

**Prediction and Evaluation**: Then, predictions are made on the validation data, and the weighted F1 score is calculated to evaluate the model's performance.

## Final Output
Weighted F1 Score: 0.9151

| Domain | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| CL     | 0.98      | 0.98   | 0.98     | 1866    |
| CR     | 0.91      | 0.91   | 0.91     | 1835    |
| DC     | 0.85      | 0.82   | 0.83     | 1355    |
| DS     | 0.92      | 0.94   | 0.93     | 1774    |
| LO     | 0.92      | 0.93   | 0.92     | 1217    |
| NI     | 0.92      | 0.90   | 0.91     | 1826    |
| SE     | 0.89      | 0.91   | 0.90     | 1327    |

**Accuracy**: 0.92 (11200)

**Macro avg**: 0.91 (11200)

**Weighted avg**: 0.92 (11200)

The final result of the algorithm shows a weighted F1 score of 0.9151, indicating strong overall performance in classifying abstracts into predefined domains.
The precision, recall, and F1-score values for each class are consistently high, with an overall accuracy of 92%. 
This suggests that the model is well-balanced and effective across all classes, making it a robust solution for the classification task.

## Future Scope
To further improve the model performance, the following steps can be considered:

- **Preventing Overfitting**: Another possible approach could be to split the testing dataset as it has more than 11000 rows, and use one part of it as an actual validation dataset which
can be used while training the model to prevent the model from memorizing the features of the training set by stopping the model training when its accuracy starts decreasing on the validation dataset and
then finally testing it on the remaining half of the testing dataset. I didn’t go with this as the existing weighted F1 score value is quite good itself.

- **Hyperparameter Tuning**: Experiment with different values of hyperparameters such as C, kernel types, and TF-IDF parameters.

- **Ensemble Methods**: Combining multiple models to leverage their strengths and achieve better overall performance.

- **Deep Learning Models**: Utilizing advanced deep learning models like BERT, which can capture contextual relationships between words better than traditional methods.


## Assumptions
- The abstracts provided in the dataset are representative of the real-world data the model will encounter.
- The domains are mutually exclusive, and each abstract belongs to only one domain.
