from src.data.make_dataset import load_and_preprocess_data
from src.visualization.visualize import plot_correlation_heatmap, plot_feature_importance, plot_confusion_matrix
from src.features.build_features import create_dummy_vars
from src.models.train_model import train_logistic_regression,random_forest
from src.models.predict_model import evaluate_model,rf_evaluate_model

if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/credit.csv"
    df = load_and_preprocess_data(data_path)

    # Create dummy variables and separate features and target
    X, y = create_dummy_vars(df)

    # Train the logistic regression model
    model, X_test_scaled, y_test,X_train_scaled, y_train = train_logistic_regression(X, y)

    # Evaluate the model
    accuracy, confusion_mat,lr_scores = evaluate_model(model, X_test_scaled, y_test, X_train_scaled, y_train)
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{confusion_mat}")
    print("Accuracy scores:", lr_scores)
    print("Mean accuracy:", lr_scores.mean())
    print("Standard deviation:", lr_scores.std())
    
    rfmodel, xtest, ytest,xtrain_scaled, ytrain = random_forest(X, y)

   
    accuracy_rf, confusion_mat_rf,rf_scores= rf_evaluate_model (rfmodel, xtest,ytest,xtrain_scaled, ytrain)
    print(f"Accuracy in Random Forest: {accuracy_rf}")
    print(f"Confusion Matrix:\n{confusion_mat_rf}")
    print("Accuracy scores:", rf_scores)
    print("Mean accuracy:", rf_scores.mean())
    print("Standard deviation:", rf_scores.std())