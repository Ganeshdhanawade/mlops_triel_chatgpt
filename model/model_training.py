import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

## load dataset
df = pd.read_csv('I:\Common\Ganesh\mlops_github\data\iris_data.csv')

# split dataset
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define the objective function for hyperparameter optimization
def objective(params):
    params['max_depth'] = int(params['max_depth'])  # Ensure max_depth is an integer
    model = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return -accuracy  # minimize the negative accuracy

# define the search space for hyperparameters
space = {
    'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
    'max_depth': hp.choice('max_depth', [5, 10, 15])  # Choose from a fixed list of integer values
}

# start MLflow experiment
mlflow.start_run()

# hyperopt optimization
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials)

# log the best parameters found
mlflow.log_param("best_n_estimators", best['n_estimators'])
mlflow.log_param("best_max_depth", best['max_depth'])

# Train the final model with the best hyperparameters
best_model = RandomForestClassifier(n_estimators=best['n_estimators'], max_depth=best['max_depth'])
best_model.fit(X_train, y_train)

# Log the model with MLflow
mlflow.sklearn.log_model(best_model, "best_random_forest_model")

# Evaluate the model performance
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
mlflow.log_metric("accuracy", accuracy)

# Save the model with DVC
joblib.dump(best_model, 'I:\Common\Ganesh\mlops_github\model\model.pkl')

# End MLflow run
mlflow.end_run()
