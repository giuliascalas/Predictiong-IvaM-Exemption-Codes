# Project-772601

Predicting IvaM exemption code

Giulia Scalas - 772601, Contributor 2 - 77623, Contributor 3- 785051

# Introduction

This project aims to forecast the exemption VAT codes listed in the **"IvaM"** column of a dataset containing several invoice lines and information about each row indicating a particular transaction provided by CompanyName (For privacy reason, the name cannot be disclosed). 
Therefore, it was necessary to determine which attributes within the dataset have meaningful correlations in predicting the exemption VAT code (“IvaM” column) for each invoice line. 
It was mentioned that the Nature code ("Iva" column) and the exemption VAT code have similar meanings, but given that the Nature code contains only 21 values and the exemption VAT code contains over 60 values, direct mapping of the two variables was determined to be not useful without additional consideration of the other variables present in the dataset. 
As such, we took several steps before implementing the multiclass classification models. 

In the next section, the reader will be able to follow the decisions taken throughout the entirety of our code step-by-step, which leads to our concluding remarks on the best-performing model and possible industrial implementation. 

# Methods

The code was entirely executed on Google Colab and a requirement.txt is available in the GitHub folder. 

The code is divided into several sections and presents the application of 7 different classification models: 
  * **KNN;** 
  * **Gradient Boost;**
  * **Artificial Neural Network;**
  * **Support Vector Machine;**
  * **Decision Tree;**
  * **Distributed Random Forest with H20;**
  * **Random Forest Classifier with Scikit Learn.**

A general overview of our methods is shown in Figure 1 below. 

![PHOTO-2024-05-20-15-13-08](https://github.com/aidaisayas/BIPproject-772601/assets/149505119/635c70a9-01ca-4cb2-a2f2-a0d4812c9a73)
_Figure 1. Flowchart detailing the process of the multiclass classification of IvaM_

Before explaining the final method selected, EDA and Data Pre-Processing will be expanded upon. 

# Exploratory Data Analysis
The first step was to check the dataset shape and data types, here we found out it has a total of:

* **134437 entries**
* **44 columns**

Since many variables contained null values, we plotted a heatmap to understand how the null values were distributed, as shown in Figure 2. 

<img width="853" alt="Screenshot 2024-05-20 at 9 40 50 AM" src="https://github.com/aidaisayas/BIPproject-772601/assets/149505119/212fb7b6-e09e-41b0-863b-4224a5afe216">

_Figure 2. Heatmap of missing values in the uncleaned, unprocessed dataset, with present values indicated with light blue and missing values indicated with dark blue_

Before moving to the next steps, we decided to plot the class distribution of the target variable, IvaM, as shown in Figure 3. 

We observed that there was a high-class imbalance, with some classes having just one value. The uneven class distribution is important to address as it could impact the performance of our models. In the next step, we handled the missing values and the class imbalance.

<img width="576" alt="Screenshot 2024-05-20 at 9 44 12 AM" src="https://github.com/aidaisayas/BIPproject-772601/assets/149505119/a11c67b5-1a07-4675-ae1d-fca10ec64501">

_Figure 3. Distribution plot of classes in IvaM variable_


# Data Cleaning and Pre-Processing 

The list of the columns we dropped in the ‘Handle missing values’ section were:

 * _[ 'C','D','E', 'F', 'G', 'H', 'CE', 'Comp', 'Iva11', '%Forf', 'Nomenclatura', 'Ritac', 'RF', 'RifNormativo', 'Rifamm', 'Art2', 'Valore2','Art3', 'Valore3', 'Conto', 'DataDoc', 'Rev', 'X', 'CTra', 'CVia', '%RIT1', '%RIT2', 'CoDitta', 'CMar', 'Art1', 'Valore1', 'Caus', 'Importo','DescrizioneRiga']_

From the previous list, the following columns did not have missing values: 

 * _[‘Datadoc’, ‘D’,  ‘Descrizione Riga’, ‘Conto’, ‘Importo’, ‘Art1’, ‘Valore 1’, ‘RifNormativo’, ‘%Rit1’, ‘%Rit2’, ‘CoDitta’, ‘Cmar’, ‘Cvia’, ‘Ctra’, ‘Rev’,’ X’, ‘Caus’]_ 

However, after carefully reviewing their meanings to understand what they meant and how they related to each other, we noticed that many columns were irrelevant to predicting the IvaM code. 

For instance, the variable _**‘CoDitta’**_ is the number assigned to a company in the accountant's personal profile, and in this case, it is irrelevant because it is an arbitrary number assigned by the user.  We also thought that variable _**‘RifNormativo’**_ could have provided us with more information for classification since it refers to each exemption code according to a specific Italian law, however, it would have been difficult with the selected classification models to encode the variable properly. 

Based on these observations, we decided to remove the unhelpful columns, and this process left us with our preliminary columns:

 * _[‘A’, ‘Ateco’, ‘B’, ‘Tdoc’, ‘VA’, ‘Iva’,  ‘ContoStd’, ‘IvaM’, ‘TM’, ‘TIva’]_ 

These variables were the ones utilized as features in our models. 

 * **"A"** signifies the business type involved.
 * **"B"** indicates if the business has deferred IvaM payments.
 * **"Tdoc"** denotes the document type (sales or purchases)
 * **"VA"** specifies if the document pertains to sales or purchases
 * **"Iva"** describes the nature or applied VAT rate
 * **"TM"** denotes the mapping type
 * **"TIva"** represents the VAT type associated with the transaction reason
 * **"Ateco"** provides the ATECO code indicating the economic activity category
 * **"ContoStd"** refers to the standard account linked to the transaction.

These variables collectively capture crucial information about transactions, including business characteristics, VAT treatment, document type, industry classification, and accounting details, essential for analysis and IvaM classification. 

As part of the preprocessing, we checked the unique values for all the variables as we did for the **‘IvaM’** variable, and we observed that some variables contained unmapped values. This is to say that in the data itself, we had classes that the dataset description provided by CompanyName did not include or explain. 

As variables _‘A’_ and _‘B’_ contained a small number of unmapped values, we decided to drop the rows that contained unmapped values for these variables. However, we observed that the quality of the dataset was compromised by unmapped values in both variables _‘Iva’_ and _‘IvaM’_.  ‘Iva’ contained values N2 and N3 that were not present in the dataset description. Therefore we chose to manually map _N2_ to values _N2.1_ or _N2.2_, and map _N3_ to _N3.1_, _N3.2_, _N3.4_, or _N3.6_, based on the frequency of corresponding IvaM values, further explained in the program notes.

We also noticed unmapped values in the IvaM column, however, we decided to keep them to avoid biases in the model. 


# Balancing the Dataset

The imbalance of classes in the IvaM variable poses a challenge for model training, as classes with more values tend to receive greater emphasis than the classes with fewer values, potentially leading to biased predictions favoring those classes and overall poor predictive performance of the models.

We first considered using **SMOTE**, an oversampling technique in which the synthetic samples are generated for the minority class.

However, we chose instead to introduce a new class labeled with an arbitrary value, **"99"**, to group invoices with an ‘IvaM’ class with fewer than 600 variables together. By doing so, we aimed to create a more balanced representation of classes, thereby improving model accuracy across all categories. 

For comparison, we developed two models for the Decision Tree, KNN, Distributed Random Forest, and Random Forest classification methods: one model trained on the original **imbalanced dataset** and a second model trained on the **balanced class** distribution. 

By comparing the results of the two models for each classification method, we sought to confirm the effectiveness of our approach in enhancing prediction accuracy for all classes, particularly those with fewer variables. Next, we will explain the specific steps in the creation of this new class.


# Manual Mapping

We first removed classes in the IvaM variable that contained fewer than 10 values, as they contained an insufficient amount of values to be useful for reliable prediction. 

We then created a second dataset, labelled as **"data_600"** in the program, and performed manual mapping of the remaining classes in the IvaM variable. We aggregated all classes falling below the 600 value threshold to the new class **‘99’**. 

This process aimed to establish a more balanced distribution of classes, facilitating more accurate predictions across the dataset.

# Encoding

As the dataset contained mostly categorical variables, we evaluated which method for performing encoding would be the most optimal. We first considered to encode the columns using the one hot encoding method:
 * _['A', 'B', 'Tdoc', 'Iva', 'TM', 'VA']_ 
 

To document our other encoding trials, we attempted to encode *‘Contostd’* and *‘Ateco’* using one hot encoding method, however, due to the large number of unique classes within these variables, the encoded data frame had over 800 columns, compared to the 68 with just the previous encoding. This data frame was not only much larger and led to longer model training time, but it was also a sparse dataset. 

Another attempt made was to perform one hot encoding with the following columns, as they are categorical variables comprised of few classes: 

 * _['A', 'B', 'Tdoc', 'Iva', 'TM', 'VA', ‘TIva’]_ 

Then we performed **target encoding** to the *‘Contostd’* and *‘Ateco’* variables, as they are comprised of 300 to 400 unique classes. However, the results following this encoding attempt were not consistent and after some trials, we realized that the variables encoded with target encoding were contributing to model overfitting. As such, we decided to continue with the first one hot encoder method for the training of our models.

In addition, the SVM does not natively support categorical features, as such target encoder was considered instead of standard scaling. However, although Ateco, ContoStd, and TIva are categorical variables represented as integers, we did not apply standard scaling to these variables. 

# Models

## First Model - K-Nearest Neighbor (KNN): Baseline Model

The KNN model is used as a baseline to compare the performance of other models. It is straightforward to implement and does not make assumptions about the underlying data distribution. The KNN algorithm classifies data points based on the majority label of their k-nearest neighbors in the feature space. It is sensitive to the choice of k, the distance metric used, and the utilization of a weight function.

**Balanced/Imbalanced Attempt:**  Both balanced and imbalanced attempts were tested to evaluate the impact on performance.

**Parameters:** 
 * **n_neighbors:** The value of k, and the number of neighbors closest to the data point. We tested with values between 2 and 5, with 5 being the most optimal value of k in terms of classification accuracy.
 * **weights:** The weight function used in prediction, tested with 'uniform' and 'distance'. ‘Uniform’ indicates that all neighbors are weighed the same, while ‘Distance’ indicates 
that neighbors closer to the data point will have more influence than neighbors farther from the data point. ‘Distance’ was determined to be the most optimal hyperparameter in terms of classification accuracy.
* **metric:** The algorithm used to compute the distance metrics, tested with ‘minkowski' and ‘Euclidean’. Training the KNN model using Minkowski distance was determined to return the highest accuracy. 

**Evaluation:**
Accuracy, precision, recall, F1-score, and confusion matrix were used to evaluate the model's performance. The baseline provided a point of reference to determine improvements with more complex models.

## Second Model - Support Vector Machine (SVM)

The SVM model is a supervised learning method that aims to find the optimal hyperplane that maximizes the margin between different classes in the feature space. SVM works by finding the hyperplane that best divides a dataset into classes with the largest margin between the nearest data points of any class (support vectors). Although SVM is typically more well-suited for binary classification problems, we chose to implement this method to compare to the baseline model.

**Balanced/Imbalanced Attempt:**  We trained this model on the imbalanced dataset only.

**Parameters:**
 * **kernel:** Different types of kernels (linear, rbf, poly, sigmoid) to find the best decision boundary for classification. We chose the radial basis function (RBF) kernel to measure the similarity between data points because of its flexibility to be applied to many different datasets. 

**Evaluation:**
Accuracy, precision, recall, F1-score, and confusion matrix were used for evaluation.


## Third Model - Gradient Boost Machine (GBM)

GBM builds a strong predictive model by combining multiple weak models, such as decision trees, by iteratively adding trees, each correcting errors from previous trees using gradient descent on the loss function, and iteratively correcting errors to enhance performance.

**Balanced/Imbalanced Attempt:**  We trained this model on the imbalanced dataset only.

**Parameters:**
 * **n_estimators:** The number of boosting stages to be run, tested with value of 150.
 * **learning_rate:** Shrinks the contribution of each tree, tested with values of 0.1.
 * **max_depth:** The maximum depth of the individual trees, tested with values of 3.

**Evaluation:**
Accuracy and classification reports were the primary evaluation metrics.


## Fourth Model - Artificial Neural Network (ANN)

ANN consists of multiple layers of interconnected neurons that process input data through nonlinear transformations. The model learns to map input features to output target variable classes by adjusting weights through backpropagation, learning patterns across data.

**Balanced/Imbalanced Attempt:** The attempt was made with the balanced dataset, as the model is slow to train.

**Parameters:**
 * **hidden_layer_sizes:** The number of neurons in the hidden layers, tested with values of (50,), (100,), and (100, 50).
 * **activation:** The activation function for the hidden layer, tested with 'relu' and 'softmax'.
 * **solver:** The optimization algorithm, tested with 'adam'
 * **Alpha:** L2 regularization term, tested with value of 0.001.
 
**Evaluation:**
Accuracy and validation loss were the primary evaluation metrics.

## Fifth Model - Decision Tree

The Decision Tree model is used to create a simple and interpretable model that makes decisions based on a series of binary splits in the data. Each branch is a decision rule, and the leaves of the tree represent the final classification of the data point. This process is repeated recursively, resulting in a tree-like model of decisions.

**Balanced/Imbalanced Attempt:** Both balanced and imbalanced attempts were tested to evaluate the impact on performance.

**Parameters:**
 * **max_depth:** The maximum depth of the tree, tested with value of 30.
 * **min_samples_split:** The minimum number of samples required to split an internal node, tested with value of 2.
 * **min_samples_leaf:** The minimum number of samples required to be at a leaf node, tested with value of 1.

**Evaluation:**
Accuracy, precision, recall, F1-score, and confusion matrix were used to evaluate the model's performance.

## Sixth Model - Distributed Random Forest with H2O

The Distributed Random Forest model with H2O is designed to handle large-scale data by distributing the computation across multiple nodes, improving scalability and performance. H2O's Distributed Random Forest constructs multiple decision trees across a distributed computing environment. It aggregates the results to make final predictions, leveraging the power of distributed computing for enhanced performance. This model was chosen due to its ability to handle datasets with imbalanced classes, as well as not needing to encode categorical variables before training the model. 

**Balanced/Imbalanced Attempt:** Both balanced and imbalanced attempts were tested to evaluate the impact on performance.

**Parameters:**
 * **ntrees:** The number of trees built in the model, the optimal number being 30 to minimize the log loss metric.
 * **max_depth:** The maximum depth of each tree, tested with value of 16 being the optimal depth of each tree to minimize log loss score.
 * **min_rows:** The minimum number of observations for a node to be split, tested with value 2.
 * **Nbins_cats:** A method to control categorical feature binning to group similar categories, with 1024 bins being the default and an optimal number of bins.

**Evaluation:**
Log loss score and confusion matrix were used to evaluate the model's performance, as the distributed random forest does not return an accuracy score for multiclass classification models. 


## Seventh Model - Random Forest with Scikit-learn

The Random Forest model with Scikit-learn is a versatile and widely used ensemble learning method for classification tasks, leveraging multiple decision trees to improve accuracy and prevent overfitting. Scikit-learn's Random Forest constructs multiple decision trees during training and outputs the mode of the classes (for classification) of the individual trees. Each tree is trained on a random subset of the training data and selects a random subset of features at each split.

**Balanced/Imbalanced Attempt:** Both balanced and Imbalanced attempts were tested to evaluate the impact on performance.

**Parameters:**
 * **n_estimators:** The number of trees in the forest, tested with values of 50, 100, 150.
 * **max_depth:** The maximum depth of each tree, explored with values of 10, 20, and None.
 * **min_samples_split:** The minimum number of samples required to split a node, tested with values of 2, 5, and 10.
 * **min_samples_leaf:** The minimum number of samples required in a leaf node, explored with values of 1, 5, and 10.

**Evaluation:**
Accuracy, precision, recall, F1-score, and confusion matrix were used to evaluate the model's performance.

# Experimental design

## Experiment 1

**Main Purpose:** Find the model with the highest accuracy that successfully predicts IvaM. 

As mentioned earlier, we have selected seven models. We have used KNN as our baseline model to compare the other models to determine which produces the best accuracy score. For each model, the test splits were 80%- 20% as the standard recommended method. The following evaluation metrics were considered: 

**Evaluation Metric 1: Accuracy**
Measures the proportion of correctly classified instances out of the total instances. It is calculated as (True Positives + True Negatives) / (Total Instances). Used for KNN, SVM, GBM, ANN, Decision Tree, Random Forest (Scikit-learn).

**Evaluation Metric 2: Precision**

Measures the proportion of true positive instances out of the total instances predicted as positive. It is calculated as True Positives / (True Positives + False Positives). Used for KNN, SVM, ANN, Decision Tree, and Random Forest (Scikit-learn).

**Evaluation Metric 3: Recall**

Measures the proportion of true positive instances out of the total actual positives. It is calculated as True Positives / (True Positives + False Negatives). Used for KNN, SVM, ANN, Decision Tree, Random Forest (Scikit-learn).

**Evaluation Metric 4: F1- Score**

The harmonic mean of precision and recall. It balances the two metrics, providing a single score to evaluate the model's performance. It is calculated as 2 * (Precision * Recall) / (Precision + Recall). Usage: Provides a balance between precision and recall. Used for KNN, SVM, ANN, Decision Tree, Random Forest (Scikit-learn).

**Evaluation Metric 5: Confusion Matrix**

A table that is often used to describe the performance of a classification model. It displays the true positives, true negatives, false positives, and false negatives. Usage: Gives detailed insight into how the model is performing in different classes. Used for **KNN, SVM, ANN, Decision Tree, Distributed Random Forest (H2O), Random Forest (Scikit-learn).**

**Evaluation Metric 6: Log loss**

Measures the performance of a classification model where the prediction is a probability value between 0 and 1. Lower log loss indicates better performance. Used for models that output probabilities, providing a nuanced view of model performance. Used for H2O Distributed Random Forest.

**Experiment 2**

**Main Purpose:** To benchmark the performance of the proposed models against the **baseline model (KNN).** 

**Baseline:** The baseline model predicts an accuracy of **96.18%** when trained with the imbalanced dataset, and **96.24%** when trained with the balanced dataset. 

**Evaluation Metric:** Same metrics as the previous experiment (if applicable).

# Results

Table 1 below provides a summary of the metric score of all seven model trials.
<img width="660" alt="Screenshot 2024-05-20 at 4 14 56 PM" src="https://github.com/aidaisayas/BIPproject-772601/assets/149505119/d1727bec-ba93-411c-a616-2501ae5ee8e7">

_Table 1. Metrics (Accuracy, Precision, Recall, F1 Score, Log loss score (DRF with H2O) for models trained on imbalanced and balanced datasets, where applicable. The third column Class 99 displays the model scoring metrics for the aggregated class 99 in the model trained on the balanced dataset. (Created on GoogleDoc and Uploaded as a picture)_

The two precision-recall curves below in _Figure 4. and 5. KNN Precision and Recall Curve in balanced and imbalanced datasets_ are a result of KNN model trained on the imbalanced dataset (left) and balanced dataset (right). There is clearly an improvement when balancing the classes. 
<p align="center">
  <img src="./images/knn1.png" height="300"/>
  <img src="./images/knn1.png" height="300"/>
</p>

 _Figure 4. and 5. Precision-Recall Curve for the KNN_



As visible, **Random Forest was our best-performing model**. We used the Random Forest with Scikit-learn to gain precision metrics and compare our model's performance to the other models. We began with our non-balanced dataset and used the following parameters to create the model _(n_estimators=150, max_depth=None, min_samples_split=2, min_samples_leaf=1 random_state=42)._ The overall accuracy of this model was **96.89% without balancing classes.** We were curious to see how this accuracy was reflected in both larger and smaller classes. The accuracy for the smaller classes with less than 600 counts was **89.29%**, which was not too low but could be improved. In contrast, the accuracy for classes with 600 or more counts was **97.13%,** which was one of the best scores across all models.

We then replicated this model with the balanced classes as previously discussed. By doing so, the **overall accuracy was 97.05%,** while the accuracy for the _new class "99"_ after balancing was **92%.** Balancing classes did not significantly change the overall accuracy but improved the accuracy for the low-frequency classes.

Although H20 did not provide accuracy, an interesting aspect of this model is the log loss plot in Figure 6, which shows the value of log loss as the number of trees in the classifier increases. We observed a plateau of 30 trees, which was determined to be the ideal number of trees for the model. In addition, we were able to see the most important features, as shown below in Figure 7. 
<img width="613" alt="Screenshot 2024-05-20 at 2 22 10 PM" src="https://github.com/aidaisayas/BIPproject-772601/assets/149505119/2ab95335-05d0-485b-a84e-53ed1ccc31ff">

_Figure 6. Plot of log loss score per number of trees built, obtained from H2O model_

<img width="608" alt="Screenshot 2024-05-20 at 2 19 31 PM" src="https://github.com/aidaisayas/BIPproject-772601/assets/149505119/dc64a6b8-1ed8-4575-b2d4-4f52c53154cd">

_Figure 7. Plot of relative variable importance obtained from H2O model_


Considering that KNN was our baseline model, we excluded the other models for the following reasons:
 * **SVM, GBM, and ANN:** These models were excluded because they were too slow to train and evaluate (as suggested by TA Davide Torre).
 * **Decision Tree:** This model was excluded because it tended to overfit, especially with complex data structures like medical images.
 * **Logistic Regression:** This model was not tried because it is too simple and does not handle non-linear relationships well.
 * **H2O Random Forest:** This model was excluded from final consideration because it did not provide accuracy metrics directly.

While KNN served as our baseline model, the **best-performing model was the Random Forest with Scikit-learn**. It provided the highest accuracy and balanced performance across different class frequencies, making it the most successful classification method for predicting IvaM.

# Conclusions
In our work, we aimed to find a model that could successfully predict **IvaM (the VAT code)** using various machine learning algorithms. After comparing several models, we found that the **Random Forest model**, particularly with the Scikit-learn implementation and balanced classes, provided the **best performance**. This model achieved high accuracy while mitigating the risks of overfitting, which were prevalent in simpler models like Decision Trees. The use of standard evaluation metrics, including accuracy, precision, recall, F1-score, and confusion matrix, allowed us to comprehensively assess the models and select the most suitable one for our classification task.

However, some questions remain unanswered and areas for future work. One significant challenge was handling data quality and preprocessing. We had to drop columns with high NaN values, which might have led to the loss of potentially useful information. Additionally, the target class **'IvaM' was very imbalanced**, and handling unmapped values in the 'Iva' and 'IvaM' columns through manual mapping could have introduced biases or errors. For future work, it would be interesting to explore the implementation of a cascade model. This approach would involve using a second model to predict the original classes for the instances that were classified as the merged class (99) by the primary model. This could provide a more nuanced understanding and improve the model's performance in rare classes.

In addition, we have created a **user interface** that can be found at the end of the code as a showcase of the practical application. This intuitive and user-friendly interface would allow professionals in accounting and auditing to leverage the model for automating part of their job. This could streamline processes and improve efficiency in handling IvaM codes.

Overall, our model can significantly aid professionals in accounting and auditing by automating the classification of IvaM codes. The detailed exemption codes provided by TeamSystem can help delve into specific fields of IvaM exemptions, offering more granularity than the general categories provided by Agenzie delle Entrate. Finally, expanding the number of classes in the imbalanced IvaM exemption category could improve future performance metrics, as each exemption code is associated with a distinct exemption category, reducing biases from merged classes.
