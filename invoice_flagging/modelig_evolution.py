from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

def train_random_forest(x_train, y_train):
    rf  = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )
    param_grid = {
        "n_estimators" : [100, 200, 300],
        "max_depth" : [None, 4, 5, 6],
        "min_samples_split" : [2, 3, 5],
        "min_samples_leaf" : [1, 2 , 5 ],
        "criterion" : ['gini', 'entropy']
    }

    scorer = make_scorer(f1_score)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid= param_grid,
        scoring=scorer,
        cv=5,
        verbose=0,
        n_jobs=-1
    )

    grid_search.fit(x_train, y_train)
    return grid_search

def evaluate_classifier(model, x_test, y_test, model_name):
    preds = model.predict(x_test)

    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    print(f"\n{model_name} Performance")
    print(f"Accuracy: {accuracy:.2f}")
    print(report)