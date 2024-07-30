# Import accuracy score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# # Function to predict and evaluate
def evaluate_model(model, X_test_scaled, y_test,X_train_scaled, y_train):
    # Predict the loan eligibility on the testing set
    y_pred = model.predict(X_test_scaled)

    # Calculate the accuracy score
    accuracy = accuracy_score(y_pred, y_test)

    # Calculate the confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred)
    
    kfold = KFold(n_splits=5)
    lr_scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold)

    return accuracy, confusion_mat,lr_scores

def rf_evaluate_model(rfmodel, xtest,ytest,xtrain_scaled, ytrain):
    
    ypred = rfmodel.predict(xtest)

    # Calculate the accuracy score
    accuracy_rf = accuracy_score(ypred, ytest)

    # Calculate the confusion matrix
    confusion_mat_rf = confusion_matrix(ytest, ypred)
    
    kfold = KFold(n_splits=5)
    rf_scores = cross_val_score(rfmodel, xtrain_scaled, ytrain, cv=kfold)

    return accuracy_rf, confusion_mat_rf,rf_scores