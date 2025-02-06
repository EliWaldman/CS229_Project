import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Load your dataset (replace 'your_dataset.csv' with your actual dataset file)
df = pd.read_pickle('next_wave_admission_prediction.pkl')

# these r cols of interest from the pdf (found by hand, all are present in our next_wave_admission_prediction.pkl)
cols_of_interest = [
    "rabyear", "ragender", "raracem", "rahispan", "rcenreg", "rurbrur", "raedyrs", "raedegrm", 
    "raeduc", "rameduc", "rafeduc", "rmstat", "rmcurln", "rarelig", "ravetrn", "rabplace", 
    "rshlt", "rhltc3", "rhosp", "rnrshom", "rdoctor", "rhomcar", "rdrugs", "routpt", 
    "rdentst", "rspcfac", "roopmd", "roopmdo", "rdepres", "reffort", "rsleepr", "rwhappy", "rflone", 
    "rfsad", "rfsad", "rgoing", "rhibp", "rdiab", "rcancr", "rlung", "rheart", "rstrok", "rpsych", 
    "rarthr", "rsleep", "rmemry", "rmemrye2", "ralzhee2", "rdemene2", "rbmi", "rweight", "rdrink", 
    "rcholst", "rflusht", "rbreast", "rmammog", "rprost", "rsmokev", "rnsscre", 
    "rslfmem", "rpstmem", "rser7", "rbwc20", "rdy", "rmo", "ryr", "rwalkra", 
    "rdressa", "rbatha", "reata", "rtoilta", "rbeda", "hatotb", "hahous", "hadebt", "hachck", "rhigov", 
    "rmrprem", "rhiothp", "rlifein", "rpnhm5y"
    ]

# these r the cols that have greater than 50% of their values filled in (ie not nan)
columns_to_convert = [
    "ragender", "raedegrm", "rfsad", "rabyear", "raeduc", "rabplace", "ravetrn", "rahispan", 
    "raracem", "raedyrs", "rarelig", "rameduc", "rafeduc", "hatotb", "hahous", "hadebt", 
    "hachck", "rdrink", "rshlt", "rcenreg", "rstrok", "rheart", "rlung", "rcancr", "rmstat", 
    "rdiab", "rpsych", "rhibp", "rarthr", "rwalkra", "rdressa", "reata", "rbatha", "rbeda", 
    "roopmd", "rdrugs"
]


# cols from cols_of_interest that have more than 50% values filled in
for col in columns_to_convert:
    # print(f"Processing column: {col}")

    # Extract numbers before '.' and convert to numeric, ignoring NaNs for now
    df[col] = df[col].astype(str).str.extract(r'(\d+)')[0].astype(float)

    # CLEANING METHOD 1: replace all nan values with the mean of that column (test accuracy = 0.8482)
    # Replace NaNs with the mean of the column
    # mean_value = df[col].mean()
    # df[col].fillna(mean_value, inplace=True)

    # CLEANING METHOD 2: drop all rows that have nan (test accuracy = 0.7420)
    df = df.dropna(subset=[col])  # 

    #print(f"Sample cleaned data for {col}:")
    #print(df[col].sample(20).reset_index(drop=True))
    #print("\n")

print(df.shape)
print("------shape------")



# Define the target variable (assuming 'target' is the name of the target column)
X = df[columns_to_convert] 
y = df['will_admit_next']  # Replace with your actual target column name 

# Split the dataset into train, validation, and test sets (70% train, 15% val, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize the logistic regression model
log_reg_model = LogisticRegression()

# Fit the model to the training data
log_reg_model.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = log_reg_model.predict(X_val)

# Evaluate the model on the validation set
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy:.4f}')
print('Classification Report (Validation):')
print(classification_report(y_val, y_val_pred))

# Predict on the test set
y_test_pred = log_reg_model.predict(X_test)

# Evaluate the model on the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy:.4f}')
print('Classification Report (Test):')
print(classification_report(y_test, y_test_pred))

# Print the feature importance based on logistic regression coefficients
coef = log_reg_model.coef_[0]
feature_importance = pd.DataFrame({
    'Feature': columns_to_convert,
    'Coefficient': coef,
    'Absolute Coefficient': abs(coef)
})

# Sort by the absolute coefficient value to identify the most influential features
feature_importance_sorted = feature_importance.sort_values(by='Absolute Coefficient', ascending=False)

print('\nFeature Importance (Sorted by Absolute Coefficient):')
print(feature_importance_sorted[['Feature', 'Coefficient']])


# -----------------------------------
# PLOT

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["No Admit", "Admit"], yticklabels=["No Admit", "Admit"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()