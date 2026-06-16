import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import os

# Load the raw dataset with utf-7 encoding
try:
    df = pd.read_csv('raw_dataset.csv', encoding='utf-7')
    print("Successfully loaded the full raw_dataset.csv with utf-7 encoding.")
    print(f"Full dataset shape: {df.shape}")
except FileNotFoundError:
    print("Error: raw_dataset.csv not found. Please ensure the file is in the current working directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the full data: {e}")
    exit()

# --- Initial Data Exploration and Visualization (Before Cleaning) ---
# Create visualizations directory if it doesn't exist
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

print("\n--- Initial Data Exploration and Visualization ---")
# Pair plots removed as per user request.

# Visualize koi_disposition distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='koi_disposition', palette='viridis')
plt.title('Distribution of KOI Disposition')
plt.xlabel('KOI Disposition')
plt.ylabel('Count')
plt.savefig('visualizations/koi_disposition_distribution.png')
plt.close()
print("Saved 'koi_disposition_distribution.png'")

# Visualize distributions of all numerical features (histograms) before scaling
print("\n--- Visualizing Distributions of Numerical Features (Before Scaling) ---")
# numerical_features_before_selection = df.select_dtypes(include=np.number).columns.tolist()
# # Exclude the encoded target variable from these plots if it's present
# if 'koi_disposition_encoded' in numerical_features_before_selection:
#     numerical_features_before_selection.remove('koi_disposition_encoded')

# if numerical_features_before_selection:
#     num_plots = len(numerical_features_before_selection)
#     num_cols = 4 # Adjust as needed for better layout
#     num_rows = (num_plots + num_cols - 1) // num_cols
    
#     plt.figure(figsize=(num_cols * 5, num_rows * 4))
#     for i, col in enumerate(numerical_features_before_selection):
#         plt.subplot(num_rows, num_cols, i + 1)
#         sns.histplot(df[col].dropna(), kde=True)
#         plt.title(f'Distribution of {col}')
#         plt.xlabel(col)
#         plt.ylabel('Count')
#     plt.suptitle('Histograms of All Numerical Features (Before Scaling)', y=1.02)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.98])
#     plt.savefig('visualizations/all_numerical_feature_histograms_before_scaling.png')
#     plt.close()
#     print("Saved 'all_numerical_feature_histograms_before_scaling.png'")
# else:
#     print("No numerical features found for histograms before scaling.")

# --- Data Cleaning and Preprocessing ---

print("\n--- Data Cleaning and Preprocessing ---")

print("\nColumn Names (after loading with utf-7 encoding):")
print(df.columns.tolist())

# Identify and remove irrelevant columns
irrelevant_columns = ['kepid', 'kepoi_name', 'kepler_name', 'koi_tce_delivname']
df = df.drop(columns=irrelevant_columns, errors='ignore')
print(f"\nRemoved irrelevant columns. New shape: {df.shape}")

# Convert relevant 'object' type columns to numeric
# Identify columns that should be numeric but are 'object' type
# These are typically the error columns and 'koi_depth' which might have non-numeric values
object_cols_to_convert = [
    'koi_period_err1', 'koi_period_err2', 'koi_impact_err2',
    'koi_duration_err2', 'koi_depth', 'koi_depth_err1', 'koi_depth_err2',
    'koi_prad_err1', 'koi_prad_err2', 'koi_insol_err2',
    'koi_steff_err2', 'koi_slogg_err2', 'koi_srad_err1', 'koi_srad_err2'
]

print("\nChecking object columns for non-numeric values before conversion:")
for col in object_cols_to_convert:
    if col in df.columns and df[col].dtype == 'object':
        non_numeric_values = df[col][pd.to_numeric(df[col], errors='coerce').isna()].unique()
        if len(non_numeric_values) > 0:
            print(f"Column '{col}' contains non-numeric values: {non_numeric_values[:5]} (showing first 5 unique)")
        else:
            print(f"Column '{col}' contains only numeric values or NaN (as object type).")

for col in object_cols_to_convert:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"Converted '{col}' to numeric, coercing errors.")

# Check for NaNs introduced by coercion before dropping rows
print("\nNaN counts per column after numeric conversion (before dropping rows):")
nan_counts = df.isnull().sum()
print(nan_counts[nan_counts > 0])

# Drop columns with a very high percentage of NaN values (e.g., > 90%)
# This is a pragmatic step to avoid losing the entire dataset if some columns are almost entirely empty.
# This is done before row-wise dropping to preserve as much data as possible for other features.
initial_cols = df.shape[1]
cols_to_drop_high_nan = []
for col in nan_counts[nan_counts > 0].index:
    if nan_counts[col] / df.shape[0] > 0.90: # More than 90% NaN values
        cols_to_drop_high_nan.append(col)

if cols_to_drop_high_nan:
    df.drop(columns=cols_to_drop_high_nan, inplace=True)
    print(f"\nDropped columns with >90% NaN values: {cols_to_drop_high_nan}")
    print(f"Dataset shape after dropping high-NaN columns: {df.shape}")
else:
    print("\nNo columns with >90% NaN values to drop.")

# Handle remaining missing values by removing rows
initial_rows = df.shape[0]
df.dropna(inplace=True)
rows_after_na = df.shape[0]
print(f"\nRemoved {initial_rows - rows_after_na} rows with remaining missing values.")
print(f"Dataset shape after dropping all NA rows: {df.shape}")

# Target Feature Encoding
le = LabelEncoder()
df['koi_disposition_encoded'] = le.fit_transform(df['koi_disposition'])
print(f"\nEncoded 'koi_disposition' to numerical labels: {le.classes_} -> {le.transform(le.classes_)}")
print(df[['koi_disposition', 'koi_disposition_encoded']].head())

# Separate features (X) and target (y)
X = df.drop(columns=['koi_disposition', 'koi_disposition_encoded'])
y = df['koi_disposition_encoded']
print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# --- Data Splitting ---
# Split data into training (70%), validation (15%), and test (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"\nTraining set shape: {X_train.shape}, {y_train.shape}")
print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")

# --- Feature Selection (mRMR - using Pearson Correlation for Redundancy) ---
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr

def mrmr_feature_selection(X_data, y_data, n_features_to_select):
    """
    Performs mRMR feature selection using a greedy approach.
    Maximizes relevance to target (Mutual Information), minimizes redundancy among selected features (Pearson Correlation).
    """
    # Calculate mutual information between each feature and the target
    # Ensure y_data is 1D array for mutual_info_classif
    mi_features_target = mutual_info_classif(X_data, y_data.values.ravel(), random_state=42)
    
    selected_features = []
    candidate_features = list(X_data.columns)
    
    print(f"\nStarting mRMR feature selection to select {n_features_to_select} features...")
    for i in range(n_features_to_select):
        best_feature = None
        best_mrmr_score = -np.inf
        
        for feature in candidate_features:
            # Relevance term: Mutual Information between feature and target
            relevance = mi_features_target[X_data.columns.get_loc(feature)]
            
            # Redundancy term: Average absolute Pearson correlation between candidate and selected features
            redundancy = 0.0
            if selected_features:
                redundancy_sum = 0.0
                for s_feature in selected_features:
                    # Calculate absolute Pearson correlation
                    # Handle cases where correlation might be NaN (e.g., constant features)
                    if X_data[feature].std() == 0 or X_data[s_feature].std() == 0:
                        corr_val = 0.0 # Assume no correlation if one feature is constant
                    else:
                        corr_val, _ = pearsonr(X_data[feature], X_data[s_feature])
                        corr_val = abs(corr_val)
                    redundancy_sum += corr_val
                redundancy = redundancy_sum / len(selected_features)
            
            # mRMR score (difference criterion: Relevance - Redundancy)
            mrmr_score = relevance - redundancy
            
            if mrmr_score > best_mrmr_score:
                best_mrmr_score = mrmr_score
                best_feature = feature
        
        if best_feature is not None:
            selected_features.append(best_feature)
            candidate_features.remove(best_feature)
            print(f"Selected feature {i+1}: {best_feature} (mRMR score: {best_mrmr_score:.4f})")
        else:
            print(f"Could not select {n_features_to_select} features. Only {len(selected_features)} features selected.")
            break
            
    return selected_features

# Apply mRMR feature selection on the training data
n_features_to_select = 15 # You can adjust this number
selected_features_list = mrmr_feature_selection(X_train, y_train, n_features_to_select)

# Subset the datasets to include only the selected features
X_train = X_train[selected_features_list]
X_val = X_val[selected_features_list]
X_test = X_test[selected_features_list]

print(f"\nFeatures selected using mRMR: {selected_features_list}")
print(f"New Training set shape after mRMR: {X_train.shape}")
print(f"New Validation set shape after mRMR: {X_val.shape}")
print(f"New Test set shape after mRMR: {X_test.shape}")

# --- Normalization/Scaling ---
# Identify numerical columns for scaling (these are now the selected features)
numerical_cols = X_train.select_dtypes(include=np.number).columns

if not numerical_cols.empty:
    scaler = StandardScaler()
    # Fit scaler only on the training data
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    # Transform validation and test data using the scaler fitted on training data
    X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    print("\nNumerical features scaled using StandardScaler (fitted on training data).")
else:
    print("\nNo numerical columns to scale after mRMR selection.")

# --- Final Visualization (Post-Scaling) ---
# Create visualizations directory if it doesn't exist
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Visualize distributions of numerical features after scaling (histograms)
if not numerical_cols.empty:
    X_train[numerical_cols].hist(bins=30, figsize=(20, 15))
    plt.suptitle('Histograms of Numerical Features (After Scaling - Training Data)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig('visualizations/numerical_feature_histograms_after_scaling_train.png')
    plt.close()
    print("Saved 'numerical_feature_histograms_after_scaling_train.png'")
else:
    print("No numerical columns to generate histograms after scaling.")

# Visualize distributions of numerical features before scaling (box plots)
print("\n--- Visualizing Box Plots of Numerical Features (Before Scaling) ---")
numerical_features_for_boxplots = df.select_dtypes(include=np.number).columns.tolist()
if 'koi_disposition_encoded' in numerical_features_for_boxplots:
    numerical_features_for_boxplots.remove('koi_disposition_encoded')

# Exclude koi_insol_err1 and koi_model_snr from the main boxplots
features_for_general_boxplots = [col for col in numerical_features_for_boxplots if col not in ['koi_insol_err1', 'koi_model_snr']]

if features_for_general_boxplots:
    num_plots = len(features_for_general_boxplots)
    num_cols = 4 # Adjust as needed for better layout
    num_rows = (num_plots + num_cols - 1) // num_cols
    
    plt.figure(figsize=(num_cols * 5, num_rows * 4))
    for i, col in enumerate(features_for_general_boxplots):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.boxplot(y=df[col].dropna())
        plt.title(f'Box Plot of {col}')
        plt.ylabel(col)
    plt.suptitle('Box Plots of Numerical Features (Before Scaling - Excl. koi_insol_err1, koi_model_snr)', y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig('visualizations/numerical_feature_boxplots_before_scaling_excluded.png')
    plt.close()
    print("Saved 'numerical_feature_boxplots_before_scaling_excluded.png'")
else:
    print("No numerical features found for general box plots before scaling.")

# Separate boxplot for koi_model_snr
if 'koi_model_snr' in numerical_features_for_boxplots:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df['koi_model_snr'].dropna())
    plt.title('Box Plot of koi_model_snr (Before Scaling)')
    plt.ylabel('koi_model_snr')
    plt.tight_layout()
    plt.savefig('visualizations/koi_model_snr_boxplot_before_scaling.png')
    plt.close()
    print("Saved 'koi_model_snr_boxplot_before_scaling.png'")
else:
    print("koi_model_snr not found for separate box plot.")


# Visualize distributions of numerical features after scaling (box plots)
if not numerical_cols.empty:
    plt.figure(figsize=(20, 15))
    X_train[numerical_cols].boxplot(rot=90)
    plt.title('Box Plots of Numerical Features (After Scaling - Training Data)')
    # Set y-axis limits for scaled data, typically -5 to 5 or -10 to 10 for StandardScaler
    plt.ylim(-5, 5) # Adjust limits to make the box plots more discernible
    plt.tight_layout()
    plt.savefig('visualizations/numerical_feature_boxplots_after_scaling_train.png')
    plt.close()
    print("Saved 'numerical_feature_boxplots_after_scaling_train.png'")
else:
    print("No numerical columns to generate box plots after scaling.")

# Note: Correlation matrix and scatter plots are generally not re-generated after scaling
# as scaling only changes the magnitude, not the relationships or correlations.
# The initial visualizations are sufficient for understanding feature relationships.

# --- Algorithm Exploration & Training and Optimization ---
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score

# Define classifiers and their hyperparameter grids
# Base Classifiers
classifiers = {
    "KNeighborsClassifier": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        }
    },
    "GaussianNB": {
        "model": GaussianNB(),
        "params": {} # GaussianNB typically has no hyperparameters to tune in this context
    },
    "LinearDiscriminantAnalysis": {
        "model": LinearDiscriminantAnalysis(),
        "params": {
            "solver": ["svd", "lsqr", "eigen"]
        }
    }
}

# Advanced Classifiers
classifiers.update({
    "SVC_Linear": {
        "model": SVC(random_state=42, probability=True), # Added probability=True
        "params": {
            "kernel": ["linear"],
            "C": [0.1, 1, 10],
            "class_weight": [None, "balanced"]
        }
    },
    "SVC_RBF": {
        "model": SVC(random_state=42, probability=True), # Added probability=True
        "params": {
            "kernel": ["rbf"],
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
            "class_weight": [None, "balanced"]
        }
    },
    "DecisionTreeClassifier": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": [None, "balanced"]
        }
    },
    "RandomForestClassifier": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "class_weight": [None, "balanced"]
        }
    }
})

best_models = {}
best_scores = {}

print("\n--- Starting Model Exploration and Optimization ---")

for name, config in classifiers.items():
    print(f"\nOptimizing {name}...")
    grid_search = GridSearchCV(config["model"], config["params"], cv=3, scoring='balanced_accuracy', n_jobs=-1, verbose=1)
    
    # Fit GridSearchCV on training data (X_train, y_train)
    grid_search.fit(X_train, y_train)
    
    best_models[name] = grid_search.best_estimator_
    
    # Evaluate on the training set to check for overfitting
    y_train_pred = best_models[name].predict(X_train)
    train_balanced_accuracy = balanced_accuracy_score(y_train, y_train_pred)

    # Evaluate on the validation set
    y_val_pred = best_models[name].predict(X_val)
    val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)
    best_scores[name] = val_balanced_accuracy
    
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Training Balanced Accuracy for {name}: {train_balanced_accuracy:.4f}")
    print(f"Validation Balanced Accuracy for {name}: {val_balanced_accuracy:.4f}")
    
    # Calculate AUC-ROC for validation set
    # For multi-class, use predict_proba if available, otherwise decision_function (for SVC)
    if hasattr(best_models[name], "predict_proba"):
        y_val_proba = best_models[name].predict_proba(X_val)
        val_roc_auc = roc_auc_score(y_val, y_val_proba, multi_class='ovr', average='weighted')
    elif hasattr(best_models[name], "decision_function"):
        y_val_scores = best_models[name].decision_function(X_val)
        val_roc_auc = roc_auc_score(y_val, y_val_scores, multi_class='ovr', average='weighted')
    else:
        val_roc_auc = "N/A (proba/decision_function not available)"
    
    print(f"Validation AUC-ROC for {name}: {val_roc_auc:.4f}")
    print(f"Validation Classification Report for {name}:\n{classification_report(y_val, y_val_pred, target_names=le.classes_)}")
    print(f"Validation Confusion Matrix for {name}:\n{confusion_matrix(y_val, y_val_pred)}")

# Select the best base classifier and best advanced classifier based on validation balanced accuracy
base_classifiers_names = ["KNeighborsClassifier", "GaussianNB", "LinearDiscriminantAnalysis"]
advanced_classifiers_names = ["SVC_Linear", "SVC_RBF", "DecisionTreeClassifier", "RandomForestClassifier"]

best_base_classifier_name = max(base_classifiers_names, key=lambda k: best_scores.get(k, -1))
best_advanced_classifier_name = max(advanced_classifiers_names, key=lambda k: best_scores.get(k, -1))

print(f"\n--- Best Classifiers Selected ---")
print(f"Best Base Classifier: {best_base_classifier_name} with Validation Balanced Accuracy: {best_scores[best_base_classifier_name]:.4f}")
print(f"Best Advanced Classifier: {best_advanced_classifier_name} with Validation Balanced Accuracy: {best_scores[best_advanced_classifier_name]:.4f}")

# --- Final Machine Learning Evaluation ---
print("\n--- Starting Final Evaluation on Test Set ---")

# Evaluate Best Base Classifier
best_base_model = best_models[best_base_classifier_name]
y_train_pred_base = best_base_model.predict(X_train)
train_balanced_accuracy_base = balanced_accuracy_score(y_train, y_train_pred_base)
y_test_pred_base = best_base_model.predict(X_test)
test_balanced_accuracy_base = balanced_accuracy_score(y_test, y_test_pred_base)

print(f"\nFinal Evaluation for Best Base Classifier ({best_base_classifier_name}):")
print(f"Training Balanced Accuracy: {train_balanced_accuracy_base:.4f}")
print(f"Test Balanced Accuracy: {test_balanced_accuracy_base:.4f}")

# Calculate AUC-ROC for test set (Best Base Classifier)
if hasattr(best_base_model, "predict_proba"):
    y_test_proba_base = best_base_model.predict_proba(X_test)
    test_roc_auc_base = roc_auc_score(y_test, y_test_proba_base, multi_class='ovr', average='weighted')
elif hasattr(best_base_model, "decision_function"):
    y_test_scores_base = best_base_model.decision_function(X_test)
    test_roc_auc_base = roc_auc_score(y_test, y_test_scores_base, multi_class='ovr', average='weighted')
else:
    test_roc_auc_base = "N/A (proba/decision_function not available)"

print(f"Test AUC-ROC: {test_roc_auc_base:.4f}")
print(f"Test Classification Report:\n{classification_report(y_test, y_test_pred_base, target_names=le.classes_)}")
print(f"Test Confusion Matrix:\n{confusion_matrix(y_test, y_test_pred_base)}")

# Evaluate Best Advanced Classifier
best_advanced_model = best_models[best_advanced_classifier_name]
y_train_pred_advanced = best_advanced_model.predict(X_train)
train_balanced_accuracy_advanced = balanced_accuracy_score(y_train, y_train_pred_advanced)
y_test_pred_advanced = best_advanced_model.predict(X_test)
test_balanced_accuracy_advanced = balanced_accuracy_score(y_test, y_test_pred_advanced)

print(f"\nFinal Evaluation for Best Advanced Classifier ({best_advanced_classifier_name}):")
print(f"Training Balanced Accuracy: {train_balanced_accuracy_advanced:.4f}")
print(f"Test Balanced Accuracy: {test_balanced_accuracy_advanced:.4f}")

# Calculate AUC-ROC for test set (Best Advanced Classifier)
if hasattr(best_advanced_model, "predict_proba"):
    y_test_proba_advanced = best_advanced_model.predict_proba(X_test)
    test_roc_auc_advanced = roc_auc_score(y_test, y_test_proba_advanced, multi_class='ovr', average='weighted')
elif hasattr(best_advanced_model, "decision_function"):
    y_test_scores_advanced = best_advanced_model.decision_function(X_test)
    test_roc_auc_advanced = roc_auc_score(y_test, y_test_scores_advanced, multi_class='ovr', average='weighted')
else:
    test_roc_auc_advanced = "N/A (proba/decision_function not available)"

print(f"Test AUC-ROC: {test_roc_auc_advanced:.4f}")
print(f"Test Classification Report:\n{classification_report(y_test, y_test_pred_advanced, target_names=le.classes_)}\n")
print(f"Test Confusion Matrix:\n{confusion_matrix(y_test, y_test_pred_advanced)}")

# Save final results to a text file
with open('ml_evaluation_results.txt', 'w') as f:
    f.write("--- Machine Learning Evaluation Results ---\n\n")
    f.write(f"Dataset Shape after Cleaning: {df.shape}\n")
    f.write(f"Training set shape: {X_train.shape}, {y_train.shape}\n")
    f.write(f"Validation set shape: {X_val.shape}, {y_val.shape}\n")
    f.write(f"Test set shape: {X_test.shape}, {y_test.shape}\n\n")

    f.write("--- Best Classifiers Selected (based on Validation Balanced Accuracy) ---\n")
    f.write(f"Best Base Classifier: {best_base_classifier_name}\n")
    f.write(f"Validation Balanced Accuracy: {best_scores[best_base_classifier_name]:.4f}\n\n")
    f.write(f"Best Advanced Classifier: {best_advanced_classifier_name}\n")
    f.write(f"Validation Balanced Accuracy: {best_scores[best_advanced_classifier_name]:.4f}\n\n")

    f.write(f"\n--- Final Evaluation for Best Base Classifier ({best_base_classifier_name}) ---\n")
    f.write(f"Training Balanced Accuracy: {train_balanced_accuracy_base:.4f}\n")
    f.write(f"Test Balanced Accuracy: {test_balanced_accuracy_base:.4f}\n")
    f.write(f"Test AUC-ROC: {test_roc_auc_base:.4f}\n")
    f.write(f"Test Classification Report:\n{classification_report(y_test, y_test_pred_base, target_names=le.classes_)}\n")
    f.write(f"Test Confusion Matrix:\n{confusion_matrix(y_test, y_test_pred_base)}\n\n")

    f.write(f"\n--- Final Evaluation for Best Advanced Classifier ({best_advanced_classifier_name}) ---\n")
    f.write(f"Training Balanced Accuracy: {train_balanced_accuracy_advanced:.4f}\n")
    f.write(f"Test Balanced Accuracy: {test_balanced_accuracy_advanced:.4f}\n")
    f.write(f"Test AUC-ROC: {test_roc_auc_advanced:.4f}\n")
    f.write(f"Test Classification Report:\n{classification_report(y_test, y_test_pred_advanced, target_names=le.classes_)}\n")
    f.write(f"Test Confusion Matrix:\n{confusion_matrix(y_test, y_test_pred_advanced)}\n")

print("\nML evaluation results saved to 'ml_evaluation_results.txt'")
