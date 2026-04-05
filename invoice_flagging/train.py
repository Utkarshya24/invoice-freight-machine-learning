import joblib 
from data_preprocessing import load_invoice_data, split_data, scale_features, apply_labels
from modelig_evolution import train_random_forest, evaluate_classifier

FEATURES =  [
    'invoice_quantity', 
    'invoice_dollars', 
    'Freight', 
    'total_item_quantity', 
    'days_po_to_invoice' , 
    'total_item_dollars'
]

TARGET = "flag_invoice"
def main():
    
    # load data
    df = load_invoice_data()
    df = apply_labels(df)

    # prepare data
    x_train, x_test, y_train, y_test = split_data(df, FEATURES, TARGET)
    x_train_scaled, x_test_scaled = scale_features(
        x_train, x_test, "models/scaler.pkl"
    )
    
    # train and evaluate model
    grid_search = train_random_forest(x_train_scaled, y_train)

    evaluate_classifier(
        grid_search.best_estimator_,
        x_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    #save best model
    joblib.dump(grid_search.best_estimator_, "models/predict_flag_invoice.pkl")


if __name__ == "__main__":
    main()