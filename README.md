
## Telco Customer Churn EDA & Prediction
In this project, I have done Telco Customer Churn EDA & Prediction. I have first used the concept of eexploratory data analysis, and then I have used various methods of Machine Learning.

## Work plan
EDA
Building a Machine Learning Model

## Linear regression

Linear regression is a fundamental statistical method used to model the relationship between a dependent variable and one or more independent variables. In simple linear regression, the relationship between one dependent variable and one independent variable is modeled with a straight line, represented by the equation 
The process involves collecting and preprocessing data, fitting the model using software like statsmodels or scikit-learn, checking assumptions (linearity, independence, homoscedasticity, normality, and no multicollinearity), and evaluating the model's performance with metrics such as R-squared and Mean Squared Error (MSE). Once validated, the model can be used for predictions, providing valuable insights and forecasting capabilities.







## Logistic regression

Logistic regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables, particularly when the dependent variable is binary (i.e., it has two possible outcomes such as "yes" or "no", "success" or "failure"). Unlike linear regression, which predicts continuous outcomes, logistic regression predicts the probability that a given input point belongs to a certain class.

In logistic regression, the output is transformed using a logistic function (also known as the sigmoid function), which produces a probability value between 0 and 1. This makes it suitable for classification problems. The model estimates the probability that a given observation belongs to the default class (usually coded as 1).

## Decision tree classifier

A decision tree classifier is a type of supervised machine learning algorithm used for classification tasks. It models decisions and their possible consequences as a tree-like structure of choices. Each internal node represents a "test" or decision on an attribute, each branch represents the outcome of that decision, and each leaf node represents a class label (the decision taken after computing all attributes).

Key Concepts
Nodes:

Root Node: The top node of the tree representing the entire dataset, which is then split into two or more homogeneous sets.
Decision Nodes: Nodes that split into further nodes based on certain conditions.
Leaf Nodes (Terminal Nodes): Nodes that do not split further and represent the final classification outcome.
Splitting: The process of dividing a node into two or more sub-nodes based on certain conditions or criteria.

Attribute Selection Measures: Metrics like Gini Index, Information Gain, or Chi-Square are used to select the attribute that best splits the data into distinct classes.

Pruning: The process of removing parts of the tree that do not provide additional power to classify instances, aimed at reducing overfitting and improving the model's generalization.

Steps in Building a Decision Tree Classifier
Data Collection: Gather the data with the target variable and predictor variables.
Data Preprocessing: Clean the data, handle missing values, and convert categorical variables into numerical values if necessary.
Choosing the Best Attribute: Use attribute selection measures to determine the best attribute to split the data.
Splitting: Divide the data into subsets based on the best attribute, creating branches of the tree.
Repeat Steps 3-4: Continue splitting each subset recursively, using the best attribute at each step, until all data is classified or a stopping criterion (like maximum depth of the tree) is met.
Pruning: Remove unnecessary branches from the tree to avoid overfitting and improve the tree's performance on unseen data.

## Support vector machine

Support Vector Machine (SVM) is a robust supervised machine learning algorithm primarily used for classification tasks but also applicable to regression. It works by finding the optimal hyperplane that separates data points of different classes with the maximum margin. The closest data points to this hyperplane are called support vectors, which are crucial in defining the boundary. SVM is particularly effective in high-dimensional spaces and can handle both linear and non-linear classification problems through the use of kernel functions, such as linear, polynomial, and radial basis function (RBF) kernels. Despite its strengths, including robustness to overfitting and versatility, SVM can be computationally intensive, especially with large datasets, and selecting the appropriate kernel and tuning its parameters can be complex. SVMs are widely used in various fields like bioinformatics for gene classification, text categorization, and image recognition, due to their accuracy and effectiveness in handling complex datasets.



## Naive Byes
Naive Bayes is a simple yet powerful classification algorithm based on Bayes' Theorem. It assumes independence between predictors, making it "naive." It's particularly effective for large datasets and commonly used in text classification and spam filtering.

## Libraries and Usage

```
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import f_oneway
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency

import statsmodels.api as sm

from imblearn.over_sampling import SMOTE

from sklearn import preprocessing

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer






## Accuracy
These models were selected and evaluated:¶
linear_model


1 - LogisticRegression

Training data

Accuracy : 0.8435265104808878
Recall Score : 0.8469891411648569
Precision Score : 0.8409703504043127
F1 Score : 0.8439690151235706
ROC-AUC Score : 0.9267266722844519

Recall scores for each fold: [0.82860666 0.8323058  0.8617284  0.8617284  0.85185185]
Average recall score: 0.8472442191472196


Test data

Accuracy : 0.8456607495069034
Recall Score : 0.8446411012782694 Precision Score : 0.8471400394477318
F1 Score : 0.845888724766125
ROC-AUC Score : 0.9232411030289238

2 - K nearest neighbour

Train data

Accuracy : 0.8723797780517879
 Recall Score : 0.8820335636722606
 Precision Score : 0.8651658194141855
 F1 Score : 0.8735182695832825
 ROC-AUC Score : 0.9529814968251566

 Recall scores for each fold: [0.80764488 0.80517879 0.8345679  0.83703704 0.83580247]
 Average recall score: 0.8240462163766725

 Test data
 Accuracy : 0.8195266272189349
 Recall Score : 0.8298918387413963
 Precision Score : 0.8138862102217936
 F1 Score : 0.821811100292113
 ROC-AUC Score : 0.8923979781887925

3 - Decision Tree

Train data

Accuracy : 0.9959309494451295
 Recall Score : 0.9970384995064165
 Precision Score : 0.9948288598867274
 F1 Score : 0.9959324540860347
 ROC-AUC Score : 0.9999474853031828

Test data

 Accuracy : 0.796844181459566
 Recall Score : 0.8112094395280236
 Precision Score : 0.7894736842105263
 F1 Score : 0.8001939864209505
 ROC-AUC Score : 0.7966357287147182

4 - Random Forest

Train data
 Accuracy : 0.9959309494451295
 Recall Score : 0.9997532082922014
 Precision Score : 0.9921626255204506
 F1 Score : 0.9959434542102029
 ROC-AUC Score : 0.9995641006491752

 Recall scores for each fold: [0.82120838 0.84340321 0.83333333 0.85308642 0.85432099]
 Average recall score: 0.8410704662739189


Test data

Accuracy : 0.8476331360946746
 Recall Score : 0.8456243854473943
 Precision Score : 0.849802371541502
 F1 Score : 0.8477082306554952
 ROC-AUC Score : 0.9245215121373835


5 - Naive Byes

Train data

Accuracy : 0.8141800246609124
 Recall Score : 0.8247778874629812
 Precision Score : 0.807441410968833
 F1 Score : 0.8160175802710291
 ROC-AUC Score : 0.8875685883903537

 Recall scores for each fold: [0.81011097 0.81011097 0.84320988 0.83333333 0.81975309]
 Average recall score: 0.8233036489016762

 Test data
 Accuracy : 0.8081854043392505
 Recall Score : 0.8171091445427728
 Precision Score : 0.8036750483558994
 F1 Score : 0.810336421257923
 ROC-AUC Score : 0.8765890835032926
## Run Locally

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  npm install
```

Start the server

```bash
  npm run start
```


## Used By
In the real world, this project is used in telecommmunication industry exclusively.
## Appendix

A very crucial project in the realm of data science and new age predictions domain using visualization techniques as well as machine learning modelling.

## Tech Stack

**Client:** Python, Naive byes classifier, gaussian naive byes, suppport vector machine, stack model, linear regression, decision tree classifier, logistic regression model, EDA analysis, machine learning, sequential model of ML,, data visualization libraries of python.



## Feedback

If you have any feedback, please reach out to us at chawlapc.619@gmail.com

