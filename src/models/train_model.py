from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle


# Function to train the model
def train_logistic_regression(X, y):
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the logistic regression model
    model = LogisticRegression().fit(X_train_scaled, y_train)
    
    # Save the trained model
    with open('models/logistic_regression.pkl', 'wb') as f:
        pickle.dump(model, f)
    
   

    return model, X_test_scaled, y_test, X_train_scaled, y_train

def random_forest(X,y):
    xtrain, xtest, ytrain, ytest = train_test_split(X,y, test_size=0.2, random_state=123)

   
    scale = MinMaxScaler()
    xtrain_scaled = scale.fit_transform(xtrain)
    xtest_scaled = scale.transform(xtest)
    
    rfmodel = RandomForestClassifier(n_estimators=100, 
                                 min_samples_leaf=5, 
                                 max_features='sqrt')
    rfmodel.fit(xtrain, ytrain)
    
    # Save the trained model
    with open('models/random_forest.pkl', 'wb') as f:
        pickle.dump(rfmodel, f)

    return rfmodel, xtest, ytest,xtrain_scaled, ytrain