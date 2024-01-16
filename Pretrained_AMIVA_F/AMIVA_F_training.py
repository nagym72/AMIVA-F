from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, fbeta_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from joblib import dump, load
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

# Parallelized Random Forest Bootstrap

# This file can be found under Supplementary_files 
df_amiva = pd.read_csv("./Full_Dataset_AMIVA_F.csv")

X = df_amiva[["cost",'hydro_weight_num_abs_dhydro',"lockless_cons", "ddg_abs", "clashes_introduced","SAP_scores"]]
y = df_amiva['effect']

# Upsample the minority class
ros = RandomOverSampler(sampling_strategy='auto', random_state=7)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Extensive GridSearchCV 
param_grid = {
    'n_estimators': [40,45,50,55],
    'max_depth': [25,30,35,40], 
    'min_samples_split': [2, 3,4,6],
    'min_samples_leaf': [1,2,3]
}

#Best Parameters from Nested CV: {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 45} 

# Initialize the model
model = RandomForestClassifier()

# Outer cross-validation for model evaluation
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

# Inner cross-validation for hyperparameter tuning
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

# Initialize GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=inner_cv, scoring='balanced_accuracy', n_jobs=-1)

# random_search = RandomizedSearchCV(model, param_distributions=param_grid, cv=inner_cv, scoring='balanced_accuracy', n_iter=100, random_state=7)

# Perform nested cross-validation
nested_scores = cross_val_score(grid_search, X_resampled, y_resampled, cv=outer_cv, scoring='balanced_accuracy')

# Mean and Confidence Interval of the nested cross-validation scores
print(f"Nested balanced accuracy: {np.mean(nested_scores):.4f} ({np.percentile(nested_scores, 2.5):.4f}-{np.percentile(nested_scores, 97.5):.4f})")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=7)

# Apply GridSearchCV on training data
grid_search.fit(X_train, y_train)

# Extract the best hyperparameters from the nested cross-validation
best_params = grid_search.best_params_

print("Best Parameters from Nested CV:", best_params)

# Initialize the model with the best parameters
best_model = RandomForestClassifier(**best_params)

# Train the model on the training set
best_model.fit(X_train, y_train)

dump(best_model, '/AMIVA_F.joblib')  # here we store the model for future use.

# Get feature importances
feature_importances = best_model.feature_importances_

# Print feature importances
print("Feature Importances:")
for feature, importance in zip(X.columns, feature_importances):
    print(f"{feature}: {importance}")

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Initialize lists to store evaluation metric scores from each bootstrap sample
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
roc_auc_scores = []
f2_scores = []
mcc_scores = []

# Lists to store false positive rates, true positive rates, and AUC-ROC scores for plotting
all_fpr = []
all_tpr = []
all_auc = []

# Number of bootstrap samples
n_bootstrap_samples = 1000 

for _ in range(n_bootstrap_samples):
    # Create a bootstrap sample, randomly chosen, with replacement.
    bootstrap_indices = np.random.choice(len(y_test), len(y_test), replace=True)
    X_bootstrap = X_test.iloc[bootstrap_indices]
    y_bootstrap = y_test.iloc[bootstrap_indices]
    
    # Make predictions on the bootstrap sample
    y_pred_bootstrap = best_model.predict(X_bootstrap)
        
    # Calculate evaluation metrics for the bootstrap sample
    accuracy_scores.append(accuracy_score(y_bootstrap, y_pred_bootstrap))
    precision_scores.append(precision_score(y_bootstrap, y_pred_bootstrap))
    recall_scores.append(recall_score(y_bootstrap, y_pred_bootstrap))
    f1_scores.append(f1_score(y_bootstrap, y_pred_bootstrap))
    roc_auc_scores.append(roc_auc_score(y_bootstrap, y_pred_bootstrap))
    f2_scores.append(fbeta_score(y_bootstrap, y_pred_bootstrap, beta=2))
    mcc_scores.append(matthews_corrcoef(y_bootstrap, y_pred_bootstrap))

    # Calculate ROC curve for the bootstrap sample
    fpr, tpr, _ = roc_curve(y_bootstrap, y_pred_bootstrap)
    roc_auc = auc(fpr, tpr)

    # Store false positive rates, true positive rates, and AUC-ROC scores for later plotting
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_auc.append(roc_auc)

def calculate_confidence_interval(metric_scores):
    lower_bound = 2.5
    upper_bound = 97.5
    
    lower = np.percentile(metric_scores, lower_bound)
    upper = np.percentile(metric_scores, upper_bound)
    return lower, upper

# Calculate mean and confidence intervals for AUC-ROC
mean_auc = np.mean(all_auc)
lower_auc, upper_auc = calculate_confidence_interval(all_auc)

mean_f2 = np.mean(f2_scores)
lower_f2, upper_f2 = calculate_confidence_interval(f2_scores)

mean_mcc = np.mean(mcc_scores)
lower_mcc, upper_mcc = calculate_confidence_interval(mcc_scores)

print(f"F2 Score: {mean_f2:.4f} ({lower_f2:.4f}-{upper_f2:.4f})")
print(f"Matthews Correlation Coefficient (MCC): {mean_mcc:.4f} ({lower_mcc:.4f}-{upper_mcc:.4f})")


# Print results
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nTest Set Metrics:")
print(f"Accuracy: {np.mean(accuracy_scores):.4f} ({calculate_confidence_interval(accuracy_scores)})")
print(f"Precision: {np.mean(precision_scores):.4f} ({calculate_confidence_interval(precision_scores)})")
print(f"Recall: {np.mean(recall_scores):.4f} ({calculate_confidence_interval(recall_scores)})")
print(f"F1 Score: {np.mean(f1_scores):.4f} ({calculate_confidence_interval(f1_scores)})")
print(f"AUC-ROC: {np.mean(roc_auc_scores):.4f} ({calculate_confidence_interval(roc_auc_scores)}")

# 95 % Conf Int.
lower_bound = 2.5
upper_bound = 97.5

# Plotting ROC curve
plt.figure(figsize=(8, 8))
for i in range(n_bootstrap_samples):
    plt.plot(all_fpr[i], all_tpr[i], color='grey', alpha=0.1)

plt.plot([0, 1], [0, 1], linestyle='--', color='blue', label='Random')

# grab mean fpr and tpr for avg balanced ROC curve
mean_fpr = np.mean(all_fpr, axis=0)
mean_tpr = np.mean(all_tpr, axis=0)


plt.plot(mean_fpr, mean_tpr, color='red', label=f'Mean AUC-ROC = {mean_auc:.4f} ({lower_auc:.4f}-{upper_auc:.4f})')

#indicate 95% conf.intv. as requested.
plt.fill_between(mean_fpr, np.percentile(all_tpr, lower_bound, axis=0), np.percentile(all_tpr, upper_bound, axis=0), color='red', alpha=0.2)

plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()
