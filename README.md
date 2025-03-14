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

