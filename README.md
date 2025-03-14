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
