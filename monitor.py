import evidently
import joblib
from evidently.dashboard import Dashboard
from evidently.dashbord.tabs import DataDriftTab
import pandas as pd

# Load your model and data
model = joblib.load('I:\Common\Ganesh\mlops_github\model\model.pkl')
df = pd.read_csv('I:\Common\Ganesh\mlops_github\data\iris_data.csv')

#create a dashboard for monitoring drift
dashbord = Dashboard(tabs=[DataDriftTab()])
dashbord.calculate(df, df)

#Save the dashboard to HTML
dashbord.save("model_monitoring.html")