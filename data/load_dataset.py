from sklearn.datasets import load_iris
import pandas as pd

#load the dataset
iris = load_iris()

## convert the padas dataframe
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target


if __name__ == "__main__":
    iris_df.to_csv('I:\Common\Ganesh\mlops_github\data\iris_data.csv',index=False)

