import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

DATA_PATH = "/home/jovyan/white_matter.csv"
FEATURES = ["Genu of corpus callosum", "Body of corpus callosum", "Splenium of corpus callosum", "Fornix", "Superior corona radiata R","Superior corona radiata L", "Superior longitudinal fasciculus R", "Superior longitudinal fasciculus L"]
TARGET = "Group"
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
auc_scores = []

coef_list = []

mean_fpr = np.linspace(0,1,100)
tprs = []

def load_data():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    y = y.map({
        "LS": 0,
        "HS": 1
    })

    return X, y

def run_cross_validation(X, y):
    cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
    )

    for fold,(train_idx,test_idx) in enumerate(cv.split(X,y),1):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
    
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
    
        pipeline = Pipeline([
            ("scaler",StandardScaler()),
            ("model",LogisticRegression(max_iter=1000))
        ])
    
        pipeline.fit(X_train,y_train)
    
        predictions = pipeline.predict(X_test)
    
        probabilities = pipeline.predict_proba(X_test)[:,1]
    
        accuracy_scores.append(
            accuracy_score(y_test,predictions)
        )
    
        precision_scores.append(
            precision_score(y_test,predictions)
        )
    
        recall_scores.append(
            recall_score(y_test,predictions)
        )
    
        f1_scores.append(
            f1_score(y_test,predictions)
        )
    
        auc_scores.append(
            roc_auc_score(y_test,probabilities)
        )
    

def plot_roc():
    fpr, tpr, _ = roc_curve(y_test, probabilities)

    interp = np.interp(mean_fpr, fpr, tpr)
    interp[0] = 0
    
    tprs.append(interp)
    mean_tpr = np.mean(tprs,axis=0)
    mean_tpr[-1]=1
    
    plt.figure(figsize=(6,6))
    
    plt.plot(
        mean_fpr,
        mean_tpr,
        label=f"Mean ROC (AUC={np.mean(auc_scores):.3f})"
    )
    
    plt.plot([0,1],[0,1],'--')
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    
    plt.legend()
    
    plt.tight_layout()
    
    plt.savefig("figures/roc_curve.png")

def plot_coefficients():
    coef_df = pd.DataFrame(
    coef_list,
    columns=FEATURES
    )
    
    mean_coef = coef_df.mean()
    
    std_coef = coef_df.std()
    
    coef_summary = pd.DataFrame({
    
        "Feature":FEATURES,
    
        "Coefficient":mean_coef,
    
        "Std":std_coef
    
    })
    
    coef_summary.sort_values(
        "Coefficient",
        inplace=True
    )
    
    coef_summary.to_csv(
        "results/logistic_coefficients.csv",
        index=False
    )
    
    coef = pipeline.named_steps["model"].coef_[0]
    coef_list.append(coef)
    
    plt.figure(figsize=(8,5))

    plt.barh(
        coef_summary["Feature"],
        coef_summary["Coefficient"]
    )
    
    plt.xlabel("Mean Logistic Regression Coefficient")
    
    plt.tight_layout()
    
    plt.savefig("figures/logistic_coefficients.png")
    
    print("\nFinished.")
    


def main():
    X, y = load_data()

    run_cross_validation(X, y)

    plot_roc()

    plot_coefficients()

    print("Finished!")
    
if __name__ == "__main__":
    main()