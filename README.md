# Machine-Learning-Algorithms-for-Interview
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
