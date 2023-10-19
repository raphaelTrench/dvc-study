from dvclive import Live
from dvclive.lgbm import DVCLiveCallback
import lightgbm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from mlem.api import save

X,y = load_iris(as_frame=True,return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

training = lightgbm.Dataset(X_train,y_train)
test= lightgbm.Dataset(X_test,y_test)


with Live("custom_dir") as live:
    lightgbm.train(
        params={},
        train_set=training,
        valid_sets=[test],
        callbacks=[DVCLiveCallback(live=live)])

    # Log additional metrics after training
    live.log_metric("summary_metric", 1.0, plot=False)
    

# instead of joblib.dump(rf, "models/rf")
save(lightgbm, "models/lgbm", sample_data=X_train)