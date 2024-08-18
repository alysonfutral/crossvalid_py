# Cross-validation is simply a way of rotating our data to make sure our algorithm is accurate. When using our model (such as with rock paper scissors), we don't entirely know what the correct prediction is, so we can't test if our result was accurate or not. Most of the data we do know the results for is in our input and output data sections.


# We can do this by using cross-validation. Cross-validation is a way of splitting our data into training and testing data. We can then use our model to make predictions on the testing data and compare those to the actual results


# To do this manually, we would simply remove a portion of our original input data (as well as the paired output data) and use that as our test data (what we are predicting). Since we know what the answer should be, we can gauge the accuracy of our model.

from sklearn import svm

# To do this manually, we would simply remove a portion of our original input data (as well as the paired output data) and use that as our test data (what we are predicting). Since we know what the answer should be, we can gauge the accuracy of our model
input_data = [
    [1,1],
    [1,2],
    [1,3],
    [2,1],
    [2,2],
    [2,3]
    # [3,1], test data
    # [3,2], test data
    # [3,3]  test data
    # these will be our validation data!
]

output_data = [1,2,3,1,2,3]
# output_data used to be [1,2,3,1,2,3,1,2,3]

model = svm.SVC()
model.fit(input_data, output_data)

# predict against our validation data
print(model.predict([[3,2], [3,1], [3,3]]))

#Since our output of our prediction is the same as the original output data, 3,1 predicts 1, 3,2 predicts 2, and 3,3 predicts 3, we know our program is decently accurate.
# [2 1 3]



##################################################################
# While this is a strong result, one test is not enough evidence to conclude perfect accuracy. Doing this for larger amounts of data or more complicated problems will be, however.

# Luckily, sklearn provides a function for when we need to figure which algorithm/model is better for our problem called cross_val_score. This allows us to not have to modify our data and we can retain most of our original program.

from sklearn.model_selection import cross_val_score

input_data = [
    [1,1],
    [1,2],
    [1,3],
    [2,1],
    [2,2],
    [2,3],
    [3,1],
    [3,2],
    [3,3]
]

output_data = [1,2,3,1,2,3,1,2,3]

model = svm.SVC()
model.fit(input_data, output_data)

score = cross_val_score(model, input_data, output_data,cv = 3)
print(score)

#[1. 1. 1.]

# The output 1. 1. 1. indicates the cross-validation scores for each fold when you use cross_val_score. In this case, we used a 3-fold cross-validation cv=3, so the cross_val_score function returns an array of three accuracy scores.

# An accuracy score of 1.0 means that the model predicted the correct output for all samples in that particular fold of the cross-validation. Since we're getting [1. 1. 1.], it suggests that the model is performing perfectly on each fold of the cross-validation.

# QUESTIONS ON WORD
