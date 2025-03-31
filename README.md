# Machine-Learning-Algorithms-for-Interview
# Foundations of Machine Learning
## Supervised vs. Unsupervised Learning (classification, regression, clustering)
### **Supervised vs. Unsupervised Learning**  

| Feature                | **Supervised Learning**                                       | **Unsupervised Learning**                                 |
|------------------------|-------------------------------------------------|-------------------------------------------|
| **Definition**         | Learning from labeled data where the model is trained with input-output pairs. | Learning from unlabeled data to identify patterns and structure. |
| **Goal**              | Predict outcomes (classification or regression). | Discover hidden structures and relationships in data. |
| **Training Data**      | Labeled (contains both input and correct output). | Unlabeled (only input data, no predefined output). |
| **Example Algorithms**| Linear Regression, Decision Trees, Neural Networks, Support Vector Machines (SVM). | K-Means, DBSCAN, Hierarchical Clustering, Principal Component Analysis (PCA). |
| **Common Applications** | Spam detection, fraud detection, image recognition, speech recognition. | Customer segmentation, anomaly detection, dimensionality reduction. |

---

### **Classification vs. Regression vs. Clustering**  

| Feature                | **Classification**                         | **Regression**                           | **Clustering**                           |
|------------------------|--------------------------------|--------------------------------|--------------------------------|
| **Type**              | Supervised Learning             | Supervised Learning             | Unsupervised Learning         |
| **Output**            | Categorical (labels/classes)   | Continuous (numerical values)  | Groups or clusters            |
| **Goal**              | Assigns input to predefined categories. | Predicts a numerical value.   | Finds natural groupings in data. |
| **Example Algorithms**| Logistic Regression, Decision Trees, Random Forest, SVM, Neural Networks. | Linear Regression, Ridge Regression, Lasso, Neural Networks. | K-Means, Hierarchical Clustering, DBSCAN. |
| **Example Use Cases** | Email spam detection (spam/ham), image classification (dog/cat). | Predicting house prices, stock price forecasting. | Customer segmentation, document clustering. |

## Bias-Variance Tradeoff (overfitting vs. underfitting)
### **Bias-Variance Tradeoff** 🎯  

The **bias-variance tradeoff** describes the balance between two types of errors that affect a model’s performance:  

- **Bias (Underfitting)** → Error due to oversimplification  
- **Variance (Overfitting)** → Error due to sensitivity to noise  

🔹 **Bias**:  
- Measures how much the predicted values deviate from the true values.  
- High bias occurs when a model is too simple and ignores patterns in the data.  
- **Example**: Linear regression on a highly nonlinear dataset.  

🔹 **Variance**:  
- Measures how much the predictions fluctuate for different training sets.  
- High variance occurs when a model is too complex and captures noise as patterns.  
- **Example**: Deep neural network with too many layers trained on a small dataset.  

---

### **Overfitting vs. Underfitting**  

| **Feature**        | **Overfitting** 🏋️‍♂️ (High Variance) | **Underfitting** 🏃‍♂️ (High Bias) |
|-------------------|--------------------------------|--------------------------------|
| **Definition**   | Model learns too much from training data, including noise. | Model is too simple to capture underlying patterns. |
| **Error Type**   | Low training error, high test error. | High training and test error. |
| **Model Complexity** | Too complex (deep neural networks, high-degree polynomials). | Too simple (linear regression on nonlinear data). |
| **Generalization** | Poor – does not perform well on unseen data. | Poor – fails to learn from training data. |
| **Solution**      | Reduce model complexity, regularization (L1/L2), more training data. | Increase model complexity, add more features, reduce regularization. |

---

### **How to Find the Right Balance?**  

1️⃣ **Train-Validation Split**: Use a validation set to monitor performance.  
2️⃣ **Cross-Validation**: Helps assess how the model generalizes.  
3️⃣ **Regularization**: L1/L2 regularization (Ridge/Lasso) helps reduce overfitting.  
4️⃣ **Feature Engineering**: Improve model inputs instead of adding complexity.  
5️⃣ **More Data**: Helps combat overfitting by exposing the model to more variations.  

**Note**
* Bias: Error of the training data
* Variance: Error of the test data
* **Underfitting**: For training data ,Model's accuracy is low and also for test data,Model's accuracy is low (i.e. High bias and High variance)
* **Overfitting** : For the training data,Model's accuracy is high but for the test data,Model's accuracy is going down(i.e. low bias and High variance)

## Feature Engineering (scaling, encoding, selection, transformation)
### **Feature Engineering: Enhancing Model Performance 🚀**  
Feature engineering improves model accuracy by transforming raw data into meaningful inputs. It includes **scaling, encoding, selection, and transformation** techniques.

---

## 🔹 **1. Feature Scaling (Normalization & Standardization)**  
Feature scaling ensures that features have comparable scales, improving model stability and convergence (especially for gradient-based models).  

✅ **Methods:**  
| Method            | Formula | Use Case |
|------------------|---------|----------|
| **Min-Max Scaling (Normalization)** | \( X' = \frac{X - X_{\min}}{X_{\max} - X_{\min}} \) | Rescales data to [0,1]; useful for deep learning & distance-based models (KNN, SVM). |
| **Z-Score Scaling (Standardization)** | \( X' = \frac{X - \mu}{\sigma} \) | Centers data around 0 with unit variance; used in linear regression, PCA, neural networks. |
| **Robust Scaling** | Uses median & IQR instead of mean/variance | Handles outliers better than standardization. |

📌 **Example (Using `sklearn`)**  
```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()  # or StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## 🔹 **2. Feature Encoding (Handling Categorical Data)**  
Encoding converts categorical variables into numerical format for model compatibility.

✅ **Methods:**  
| Encoding Type | Description | Use Case |
|--------------|-------------|----------|
| **One-Hot Encoding** | Creates binary columns for each category. | Categorical features with no ordinal relationship (e.g., colors: red, blue, green). |
| **Label Encoding** | Assigns unique integer values to categories. | Ordinal categorical data (e.g., small=0, medium=1, large=2). |
| **Target Encoding** | Replaces categories with mean target value. | Useful for high-cardinality categorical data. |
| **Binary Encoding** | Converts categories into binary form. | Reduces dimensionality for high-cardinality features. |

📌 **Example (One-Hot Encoding with `pandas`)**  
```python
import pandas as pd
df = pd.get_dummies(df, columns=['Category'], drop_first=True)  # Avoids multicollinearity
```

---

## 🔹 **3. Feature Selection (Choosing the Most Relevant Features)**  
Feature selection reduces dimensionality, improving efficiency and avoiding overfitting.

✅ **Techniques:**  
| Type | Method | Description |
|------|--------|-------------|
| **Filter Methods** | Correlation, Mutual Information | Select features based on statistical properties. |
| **Wrapper Methods** | Recursive Feature Elimination (RFE) | Iteratively removes less important features. |
| **Embedded Methods** | Lasso Regression, Decision Trees | Feature selection occurs during model training. |

📌 **Example (Using `SelectKBest`)**  
```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=5)  # Select top 5 features
X_selected = selector.fit_transform(X, y)
```

---

## 🔹 **4. Feature Transformation (Modifying Feature Representation)**  
Transformation improves model performance by reshaping data distributions.

✅ **Methods:**  
| Transformation | Description | Use Case |
|---------------|-------------|----------|
| **Log Transformation** | Reduces right-skewed data. | Salary, population, prices. |
| **Box-Cox Transformation** | Stabilizes variance in non-normal data. | Time series & regression models. |
| **Polynomial Features** | Creates interaction terms & higher-order features. | Capturing nonlinear relationships. |
| **Principal Component Analysis (PCA)** | Reduces dimensionality by projecting data onto principal components. | Large datasets with collinear features. |

📌 **Example (Applying PCA in `sklearn`)**  
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X)
```

---

### **Key Takeaways 🏆**  
✔ **Scaling** ensures consistent feature magnitudes.  
✔ **Encoding** converts categorical data into numerical form.  
✔ **Selection** removes irrelevant or redundant features.  
✔ **Transformation** reshapes data for better model fit.  

## Optimization Techniques (gradient descent, stochastic gradient descent, Adam)
### **Optimization Techniques 🚀**  
Optimization algorithms minimize the loss function to improve model performance.
#### **1. Gradient Descent (GD) 🔽**  
- **Goal:** Minimize loss by updating parameters in the direction of the negative gradient.  
- **Types:**  
  - **Batch GD** – Uses the entire dataset (slow but stable).  
  - **SGD** – Updates after each sample (fast but noisy).  
  - **Mini-Batch GD** – Uses small batches (balance of speed & stability).  

📌 **Formula:**  
$$
\theta = \theta - \alpha \cdot \nabla J(\theta)
$$


---

#### **2. Stochastic Gradient Descent (SGD) ⚡**  
- **Fast updates**, great for **large datasets**, but has **high variance**.  
- **Fixes:** Use **Momentum** or **Mini-Batches** to stabilize.  

---

#### **3. Adam Optimizer 🏆**  
- **Combines Momentum & RMSprop** for adaptive learning rates.  
- **Pros:** Fast, stable, works well for deep learning & sparse data.  

📌 **Formula (Update Rule):**  
\[
\theta = \theta - \alpha \cdot \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon}
\]

---

### **Comparison Table 📊**  

| Optimizer  | Speed | Stability | Best For |
|------------|-------|-----------|----------|
| **GD**     | Slow  | High      | Small datasets |
| **SGD**    | Fast  | Noisy     | Large datasets |
| **Mini-Batch GD** | Balanced | Good | Most ML tasks |
| **Adam**   | Fastest | Very Stable | Deep learning |

🔹 **Best Choice?** **Adam** for deep learning, **Mini-Batch GD** for general ML.  
## Evaluation Metrics (accuracy, precision, recall, F1-score, AUC-ROC, RMSE) 

#### **1. Accuracy 📊**  
- **Formula:**  
  Accuracy = (TP + TN) / (TP + TN + FP + FN)
- **Best for:** Balanced datasets.  
- **Issue:** Misleading for imbalanced data.  

---

#### **2. Precision 🎯 (Positive Predictive Value)**  
- **Formula:**  
  Precision = TP / (TP + FP)
- **Best for:** Avoiding false positives (e.g., spam detection).  

---

#### **3. Recall (Sensitivity) 🔥**  
- **Formula:**  
 Recall = TP / (TP + FN)
- **Best for:** Catching all positives (e.g., medical diagnoses).  

---

#### **4. F1-Score ⚖️ (Balance Between Precision & Recall)**  
- **Formula:**  
 F1 = 2 × (Precision × Recall) / (Precision + Recall)
- **Best for:** Imbalanced datasets.  

---

#### **5. AUC-ROC (Area Under Curve - Receiver Operating Characteristic) 📈**  
- **Measures:** Model’s ability to distinguish between classes.  
- **Best for:** Evaluating classification performance at different thresholds.  

---

#### **6. RMSE (Root Mean Squared Error) 📉**  
- **Formula:**  
 RMSE = √((1/n) ∑ (y_true - y_pred)²)
- **Best for:** Regression tasks (penalizes large errors more).  

---

### **Metric Selection Guide 🏆**  
| Task | Best Metric |
|------|------------|
| **Balanced Classification** | Accuracy, F1-score |
| **Imbalanced Classification** | Precision (low FP) / Recall (low FN), AUC-ROC |
| **Regression** | RMSE |

# Algorithms and Models
## Linear Models
### linear regression : 
* Linear regression aims to find the best fit line for a dataset, represented by the equation y = MX + C.
* In this equation, M is the slope, indicating the change in the dependent variable (y, e.g., price) for a unit change in the independent variable (x, e.g., size).
* C is the intercept, representing the value of y when x is zero.
* The best fit line is determined by minimising the error between the predicted values (Y hat) from the line and the actual data points (Y).
* This error is quantified using a cost function: 1/(2m) * Σ(Y hat - Y)^2, where m is the number of data points.
* Y hat is calculated using the linear equation: Y hat = MX + C.
* The goal is to find the values of M and C that minimise the cost function.
* Gradient descent is an iterative optimisation algorithm used to find these optimal values.
* It involves starting with initial values for M and C and then iteratively updating them based on the convergence theorem: M_new = M_old - α * d(Cost)/dM.
* α is the learning rate, a small positive value that controls the step size of each iteration.
* d(Cost)/dM is the derivative of the cost function with respect to M, representing the slope of the cost function curve at the current M value.
* If the slope of the cost function is negative, M is increased; if it's positive, M is decreased, to move towards the global minima (the minimum point of the cost function).
* A small learning rate is crucial for avoiding overshooting the global minima.
* The algorithm stops when the slope of the cost function approaches zero, indicating that the optimal values for M and C have been found.
* For multiple linear regression (with more than one independent variable), the concept extends to higher dimensions, with gradient descent optimising the coefficients for each feature to minimise a similar cost function
# How Random Forest Works:
• Random Forest is an ensemble learning technique that combines learning from multiple models. 

• It is based on decision trees.

• Multiple decision trees are built using the bagging technique or bootstrap aggregation.

• For each tree, multiple bags of data are created by sampling the original data with replacement.

• Each bag involves row sampling (randomly selecting a subset of rows, typically two-thirds of the original data) and column sampling (randomly selecting a subset of features, a starting point is the square root of the total number of columns).

• One decision tree is fitted to each bag of data.

• When making a prediction for a classification problem, majority voting is used across the predictions of all the trees.

• For a regression problem, the prediction is typically the mean or average of the predictions from all the decision trees.

• Random Forest performs well because each tree is trained on a different subset of data and features, allowing the algorithm to capture patterns from various angles and reducing the dominance of very strong predictors like salary in the example given.

## Pros (Advantages) of Random Forest:
• It reduces high variance and the tendency to overfit compared to individual decision trees.

• It tends to give good results.

• It is easy to implement using libraries in Python and R.
## Cons (Disadvantages) of Random Forest:

• It can be considered a black box model, making it difficult to explain the model's mathematical workings in detail.

• Training can be computationally expensive in terms of space and time if the data size (number of rows) and feature size (number of columns) are very large because multiple decision trees are created

# AdaBoost and how it works:
* AdaBoost is a boosting technique and an implementation of ensemble learning.
* It is a sequential learning process where one model (or weak learner) is dependent on the previous one. Model 2 depends on the output of Model 1, and so on.
* Unlike bagging techniques like Random Forest which use parallel learning, AdaBoost's models learn in sequence.
* The individual models in AdaBoost, also known as weak learners, do not have an equal say or weight in the final model. Some models will have more influence than others.
* The weak learners in AdaBoost are typically stumps, which are decision trees with just one root node and two leaf nodes. This is different from bagging methods like Random Forest which use fully grown decision trees.
* The AdaBoost algorithm starts by assigning equal initial weights to all the data records. If there are five records, each will have an initial weight of 1/5. This signifies that initially, all records are equally important for the model.
* The first weak learner (stump) is fit on this data.
* After the first weak learner is created, the data is tested for accuracy on this stump, and some classifications might be incorrect.
* In the next iteration, the weights of the misclassified records are increased, making them more important for the subsequent weak learner.
* To normalise the overall weight, the weights of the correctly classified records are decreased.
* This process ensures that the next weak learner focuses more on the records that were misclassified by the previous learner.
* Subsequent weak learners are built sequentially, with each one trying to correct the errors of the previous ones by giving more weight to the previously misclassified instances.
* The name "adaptive boosting" comes from the fact that the algorithm adapts to the previous model's performance. It focuses on the examples that were difficult to classify correctly.
* The final model is a combination of all the weak learners created during the process.
* The key differences in AdaBoost compared to bagging are the sequential learning, the unequal weighting of weak learners, and the use of stumps as weak learners.
* The internal methods for creating the stumps, such as using Gini index or entropy, remain the same as in standard decision tree creation.

# How Gradient Boost works:
* Gradient Boost is a boosting technique that combines multiple models sequentially.
* Unlike AdaBoost, which adjusts the weights of misclassified records, Gradient Boost learns by optimising a loss function.
* The process starts with a base value, which, in the case of regression, is the average of the target variable.
* Residuals are then calculated as the difference between the actual target values and the initial predicted values (the average).
* A new model, often a decision tree with leaf nodes typically ranging from 8 to 32, is trained to predict these residuals, using the original independent variables. This residual becomes the target column for this new model.
* The predictions from this residual model are then used to update the initial prediction. This update is done by adding the predicted residuals, multiplied by a learning rate, to the previous prediction. The learning rate controls the step size of the update.
* This process is iterative; after the first iteration, new residuals are calculated based on the updated predictions, and another model is trained to predict these new residuals.
* The final prediction of the Gradient Boost model is an additive combination of the initial base value and the predictions from each of the subsequent residual models, each scaled by the learning rate. This continues for a predefined number of trees or until a certain criterion is met.

# overview of XGBoost:
* XGBoost is a popular and widely used boosting algorithm favoured by many data scientists.
* It offers multi-language support, allowing you to run it with Python, R, Java, Scala, and Julia.
* XGBoost is platform-free, enabling its use across different operating systems like Windows, macOS, and Linux.
* It boasts easy installation and compatibility with various systems and integration with many platforms.
* XGBoost has gained positive recognition due to its fast processing speed and fast performance leading to quick results.
* It frequently outperforms other boosting algorithms and ensemble learning methods.
* Speed and performance are key advantages of XGBoost.
* XGBoost is fast to train and fast to predict.
* It supports parallel processing by using all CPU cores, which contributes to its speed.
* XGBoost can be run in a distributed manner, leveraging the maximum computational power of distributed systems.
* It utilizes cache awareness, optimising data access from memory for faster computations.
* XGBoost employs memory cache access to store frequently accessed information for quicker retrieval.
* It is highly scalable and can efficiently handle datasets of various sizes.
* **XGBoost incorporates automatic regularisation to prevent model overfitting.**
* **It internally handles missing values in the data, making it more robust.**
* **XGBoost is considered one of the best models in terms of performance and accuracy.**
* **The algorithm has internal cross-validation capabilities.**

# How XGBoost works:
* XGBoost is a boosting algorithm, which is an ensemble technique based on sequential learning. Unlike bagging (parallel), boosting trains models one after another.
* XGBoost is considered an extension of gradient boosting.
* The process begins by creating a base model. A simple initial assumption for prediction can be the average of the target variable. This initial model will have errors or residuals.
* The next model is then fitted on these residuals with the objective of minimising them. For example, a decision tree might be fitted to these residual values using the original independent features (age in the example) as input and the residuals as the target.
* XGBoost calculates a similarity score of residuals at each node of the tree. This score is determined by the sum of squared residuals divided by the number of residuals plus a regularisation parameter lambda.
* A tree splitting criterion is defined (e.g., age greater than 10) to divide the data in the residual tree.
* The gain from a split is calculated as the difference between the similarity score after the split and the similarity score before the split.
* A parameter called gamma acts as a threshold for splitting. A split will only occur if the calculated gain is greater than the given gamma value. This mechanism facilitates auto pruning of the tree, helping to control overfitting. A higher gamma leads to more aggressive pruning.
* The lambda parameter is a regularisation parameter that helps control overfitting. Increasing lambda reduces the similarity score and consequently the gain, potentially preventing splits and pruning the tree. Lambda also helps to reduce the impact of outliers on predictions.
* For prediction with a new data point, the output of the residual tree (sum of residuals divided by the number of residuals plus lambda for the leaf node it falls into) is combined with the previous prediction using a learning rate (eta).
* The new prediction is calculated as: **Previous Prediction + Learning Rate * Output of the Residual Tree.** The learning rate (eta) controls how quickly the model converges to the next value.
* After obtaining the new prediction, the residual is updated (original target value minus the new prediction).
* Subsequent models are trained on these new, reduced residuals, and this process is repeated iteratively. The goal is to progressively reduce the residuals and create a final ensemble model that provides accurate predictions.

# CatBoost:
* CatBoost is a machine learning algorithm that can directly process text and categorical features alongside numerical features without requiring explicit pre-processing like TF-IDF, bag of words, one-hot encoding, or label encoding.
* It can effectively train models on a limited amount of data by deriving maximum information.
* CatBoost tends to perform well when dealing with datasets having many categorical columns with numerous categories.
* The algorithm is known for its fast training and prediction times, and it can utilise both CPU and GPU resources.
* CatBoost employs a special method for handling categorical features called ordered target-based encoding to avoid data leakage. This involves shuffling the data and encoding categorical values based on the target variable from preceding rows according to a specific formula (current count + prior) / (max count + 1).
* For sampling, CatBoost uses a technique called minimal variance sampling (MVS) at the tree level, which performs weighted sampling to maximise accuracy at each split.
* Important parameters in CatBoost include the loss function, pool (an internal data structure for efficiency), GPU/CPU settings, the concept of a golden parameter (assigning higher importance to specific features), and different bootstrapping methods like MVS, uniform, and random.
