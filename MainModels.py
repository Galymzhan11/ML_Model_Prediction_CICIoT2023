import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('C:/Users/Galym Beketai/Downloads/0.001percent_2classes.csv')

# Handling missing values
data = data.dropna()

# Encoding categorical variables
label_encoder = LabelEncoder()
data['protocol_type'] = label_encoder.fit_transform(data['protocol_type'])

# Encoding boolean flags
boolean_columns = ['fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
                   'ece_flag_number', 'cwr_flag_number', 'http', 'https', 'dns', 'telnet', 'smtp', 'ssh', 'irc',
                   'tcp', 'udp', 'dhcp', 'arp', 'icmp', 'ipv', 'llc', 'benign']

for col in boolean_columns:
    data[col] = data[col].astype(int)

# Separating features and target variable
X = data.drop('benign', axis=1)
y = data['benign']

# Balance the dataset using undersampling
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)

# Introduce noise into the training labels
noise_factor = 0.05
num_noisy_labels = int(len(y_res) * noise_factor)
noisy_indices = np.random.choice(len(y_res), num_noisy_labels, replace=False)
y_res.iloc[noisy_indices] = 1 - y_res.iloc[noisy_indices]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_res, test_size=0.2, random_state=42)

# Define models with smoothing for Naive Bayes
models = {
    "Naive Bayes": GaussianNB(var_smoothing=1e-9),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False)
}

# Hyperparameter grids for tuning
param_grids = {
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    },
    "Decision Tree": {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    "AdaBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    },
    "KNN": {
        'n_neighbors': [3, 5, 7]
    },
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10]
    },
    "XGBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9]
    }
}

# Evaluate models with cross-validation and hyperparameter tuning
best_models = {}
results = []
for name, model in models.items():
    print(f"Training {name}...")
    if name in param_grids:
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = model
        best_model.fit(X_train, y_train)

    best_models[name] = best_model  # Save the best model

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC-ROC': auc_score
    })

    # Plot ROC curve
    if hasattr(best_model, "predict_proba"):
        y_pred_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_pred_prob = best_model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# Plot the ROC curves
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Create a DataFrame to store the results
results_df = pd.DataFrame(results)
print(results_df)


# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# Plot confusion matrix for each model
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, f'Confusion Matrix - {name}')

# Plot barplots for each metric
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']

# Generate a unique color for each model
palette = sns.color_palette("husl", len(models))

for i, metric in enumerate(metrics):
    ax = sns.barplot(x='Model', y=metric, data=results_df, ax=axes[i // 2, i % 2], palette=palette)
    axes[i // 2, i % 2].set_title(f'Model {metric}')
    axes[i // 2, i % 2].set_xticklabels(axes[i // 2, i % 2].get_xticklabels(), rotation=45)

    # Add data labels
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points')

plt.tight_layout()
plt.show()

# Evaluate models with cross-validation and hyperparameter tuning
best_models = {}
results = []
train_test_accuracy = []  # To store train and test accuracy

for name, model in models.items():
    print(f"Training {name}...")
    if name in param_grids:
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = model
        best_model.fit(X_train, y_train)

    best_models[name] = best_model  # Save the best model

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Calculate train and test accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    train_test_accuracy.append({
        'Model': name,
        'Train Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy
    })

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    auc_score = roc_auc_score(y_test, y_test_pred)

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC-ROC': auc_score
    })

    # Plot ROC curve
    if hasattr(best_model, "predict_proba"):
        y_pred_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_pred_prob = best_model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# Plot the ROC curves
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Create a DataFrame to store the results
results_df = pd.DataFrame(results)
train_test_accuracy_df = pd.DataFrame(train_test_accuracy)

print(results_df)
print(train_test_accuracy_df)




# Plot train and test accuracy
fig, ax = plt.subplots(figsize=(12, 6))
train_test_accuracy_df_melted = pd.melt(train_test_accuracy_df, id_vars='Model', var_name='Dataset', value_name='Accuracy')
sns.barplot(x='Model', y='Accuracy', hue='Dataset', data=train_test_accuracy_df_melted, palette='viridis', ax=ax)

# Add data labels
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points')

ax.set_title('Train and Test Accuracy for Each Model')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the layout to make space for the legend
plt.show()