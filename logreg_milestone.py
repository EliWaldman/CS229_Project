import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Load your dataset (replace 'your_dataset.csv' with your actual dataset file)
df = pd.read_pickle('next_wave_admission_prediction.pkl')

# List of columns to convert to numeric values
columns_to_convert = [
    'sagey_b', 'seduc', 'swork', 'sfsize', 'sfinr', 
    'shltc5', 'sshlt', 'shosp', 'sdeprex', 'slifein'
]

# bigger feature set, not using for now
"""
social_determinants = [
    # Income & Pensions
    "riearn", "ripena", "ripen", "riann", "rissdi", "risret", "riunwc", "rigxfr", "risdi", 
    "riunem", "riwcmp", "rissi", "rifsdi", "rifunem", "rifwcmp", "rifssi", "rifearn", 
    "rifpena", "rifpen", "rifann", "rifssdi", "rifsret", "rifunwc", "rifgxfr",

    # Public Assistance & Benefits
    "rassrecv", "rssdi", "rssiapp", "rssiden", "rssiapl", "rssirec", "rssistp", "rssappy", 
    "rssaply", "rssrecy", "rssstpy", "rssappm", "rssaplm", "rssrecm", "rssstpm",

    # Retirement & Work History
    "rwork", "rwork62", "rwork65", "rworklm", "rrplnyr", "rrplnya", "rretemp", "rsamemp", 
    "rsamejob", "rwork70", "rwork70a", "rwork70f", "rwork70af",

    # Poverty & Household Financial Struggles
    "hpovthr", "hpovfam", "hpovhhi", "hinpov", "hinpovr", "hinpovd", "hinpovrd",

    # Healthcare Utilization & Costs
    "rdoctor", "rhomcar", "rhsptim", "rnrstim", "rhspnit", "rnrsnit", "rdoctim", "rnhmliv", 
    "rhlthlm", "rshltc", "rshltcf", "rhiltc", "rltcprm",

    # Insurance Coverage
    "rcovr", "rcovs", "rhigov", "rgovmr", "rgovmd", "rgovva", "rgovot", "rhiothp", 
    "rhesrc1", "rhesrc2", "rhecov1", "rhecov2", "rhecov3",

    # Health Conditions & Chronic Illnesses
    "rhibps", "rdiabs", "rcancrs", "rlungs", "rhearts", "rstroks", "rpsychs", "rarthrs", 
    "rcondsf", "rcondsp", "rconds",

    # Cognitive & Mental Health
    "rdepres", "reffort", "rsleepr", "rwhappy", "rflone", "renlife", "rfsad", "rcesd", 
    "rmemryq", "rmemry", "rmemrye", "rmemrye2", "rnotics", "rprchmem", "rpnhm5y",

    # Preventive Care & Health Screenings
    "rflusht", "rcholst", "rbreast", "rmammog", "rpapsm", "rprost", "rdrinkd", "rdrinkn",

    # Living Arrangements & Family Support
    "rdadliv", "rmomliv", "rlivpar", "rdadage", "rmomage", "rlivsib", "rlivbro", "rlivsis", 
    "raevbrn", "raevbrnf",

    # Loneliness & Social Networks
    "rflone", "rlblonely3", "rlblonely11",

    # Perceived Support & Well-being
    "rlbsatwlf", "rlbsatfam", "rlbsathlth", "rlbsatlife", "rlbsathome",

    # Housing & Residential Stability
    "hhhresp", "hcpl", "hanyfin", "hanyfam", "haoahdhh", "hohrshh", "hnhmliv", "hahous", 
    "hamort", "hahmln", "hafhous", "hatoth", "hatotn", "hatotf", "hatotb", "hatotw", 
    "hanethb", "hafira", "hafmort", "hafhmln"
]
"""

for col in columns_to_convert:
    #print(f"Processing column: {col}")

    # Extract numbers before '.' and convert to numeric, ignoring NaNs for now
    df[col] = df[col].astype(str).str.extract(r'(\d+)')[0].astype(float)

    # Replace NaNs with the mean of the column
    mean_value = df[col].mean()
    df[col].fillna(mean_value, inplace=True)

    #print(f"Sample cleaned data for {col}:")
    #print(df[col].sample(20).reset_index(drop=True))
    #print("\n")

print(df.shape)
print("shape^^^^^^")

# ------------
"""
# clean the data, drop any rows that have NaN within our subset of features
def clean_column(col):
    return col.dropna().apply(lambda x: int(str(x).split('.')[0]) if isinstance(x, str) and '.' in x else x).astype(float)

# Apply cleaning only to specified columns
df_cleaned = df.copy()
df_cleaned[columns_to_convert] = df_cleaned[columns_to_convert].apply(clean_column)

# Drop rows where **any** of the `columns_to_convert` contain NaNs
df_cleaned = df_cleaned.dropna(subset=columns_to_convert)
"""
# ------------



"""
# Function to extract numeric values before any non-numeric characters
def extract_numeric(value):
    # Ensure we handle cases like '12352.' or '1.male' by extracting the numeric part before any non-numeric character
    if isinstance(value, str):
        return ''.join(filter(str.isdigit, value.split('.')[0]))
    return str(value).split('.')[0]

# Convert selected columns to numeric by applying the extraction function
for col in columns_to_convert:
    df[col] = df[col].apply(extract_numeric).astype(float)

# THIS IS COMMENTED OUT BC IT REMOVES ALL BUT 6 ROWS
# Handle any missing values (NaN) that result from coercion or missing data
# df = df.dropna(subset=columns_to_convert)
"""



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