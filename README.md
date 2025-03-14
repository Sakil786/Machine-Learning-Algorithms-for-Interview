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

# Pros (Advantages) of Random Forest:
• It reduces high variance and the tendency to overfit compared to individual decision trees.

• It tends to give good results.

• It is easy to implement using libraries in Python and R.
# Cons (Disadvantages) of Random Forest:

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

