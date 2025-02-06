import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Load dataset 
df = pd.read_pickle('next_wave_admission_prediction.pkl')

# all of these cols r in the file above
cols_to_check = [ 
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

# all of these cols are more than 50% filled in
columns_more_than_50_full = [
    "ragender", "raedegrm", "rfsad", "rabyear", "raeduc", "rabplace", "ravetrn", "rahispan", 
    "raracem", "raedyrs", "rarelig", "rameduc", "rafeduc", "hatotb", "hahous", "hadebt", 
    "hachck", "rdrink", "rshlt", "rcenreg", "rstrok", "rheart", "rlung", "rcancr", "rmstat", 
    "rdiab", "rpsych", "rhibp", "rarthr", "rwalkra", "rdressa", "reata", "rbatha", "rbeda", 
    "roopmd", "rdrugs"
]


col_stats = []

for col in cols_to_check:
    total = len(df[col])
    
    # Count NaNs and empty strings/spaces as missing values
    missing = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
    filled = total - missing
    
    filled_percentage = (filled / total) * 100
    missing_percentage = (missing / total) * 100
    
    col_stats.append((col, filled, missing, filled_percentage, missing_percentage))

# Sort by the most filled percentage in descending order
col_stats_sorted = sorted(col_stats, key=lambda x: x[3], reverse=True)

# Print the sorted stats
for col, filled, missing, filled_percentage, missing_percentage in col_stats_sorted:
    print(f"Column: {col}")
    print(f"  - Filled: {filled} ({filled_percentage:.2f}%)")
    print(f"  - Missing: {missing} ({missing_percentage:.2f}%)\n")