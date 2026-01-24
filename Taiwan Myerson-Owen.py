# ============================================================
# TAIWAN CREDIT CARD DATASET – 9-METHOD UNIFIED EXPLAINABILITY
# SHAP, BANZHAF, MYERSON, OWEN (3), MYERSON–OWEN (3)
# WITH FULL STATISTICAL ANALYSIS AND PLOTS
# ============================================================

# -----------------------------
# 0. INSTALL & IMPORT PACKAGES
# -----------------------------
!pip install xgboost shap imbalanced-learn lightgbm scikit-posthocs xlrd openpyxl -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import shap
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine
from scipy import stats
from scipy.stats import spearmanr, wilcoxon, friedmanchisquare, levene, shapiro
import scikit_posthocs as sp

plt.style.use('default')
sns.set_palette("husl")
np.random.seed(42)

# ============================================================
# 1. LOAD AND PREPROCESS TAIWAN CREDIT CARD DATASET
# ============================================================

print("\n" + "="*80)
print("LOADING AND PREPROCESSING TAIWAN CREDIT CARD DEFAULT DATASET")
print("="*80)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

try:
    df = pd.read_excel(url, header=1, engine='openpyxl')
    print("✓ Dataset loaded successfully with openpyxl")
except:
    try:
        df = pd.read_excel(url, header=1, engine='xlrd')
        print("✓ Dataset loaded successfully with xlrd")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        import urllib.request
        urllib.request.urlretrieve(url, 'taiwan_credit_card.xls')
        df = pd.read_excel('taiwan_credit_card.xls', header=1, engine='openpyxl')
        print("✓ Dataset downloaded and loaded successfully")

df = df.rename(columns={'default payment next month': 'target'})
df = df.drop('ID', axis=1)

new_col_names = {
    'LIMIT_BAL': 'credit_limit',
    'SEX': 'gender',
    'EDUCATION': 'education',
    'MARRIAGE': 'marriage',
    'AGE': 'age',
    'PAY_0': 'repayment_status_sep',
    'PAY_2': 'repayment_status_aug',
    'PAY_3': 'repayment_status_jul',
    'PAY_4': 'repayment_status_jun',
    'PAY_5': 'repayment_status_may',
    'PAY_6': 'repayment_status_apr',
    'BILL_AMT1': 'bill_amount_sep',
    'BILL_AMT2': 'bill_amount_aug',
    'BILL_AMT3': 'bill_amount_jul',
    'BILL_AMT4': 'bill_amount_jun',
    'BILL_AMT5': 'bill_amount_may',
    'BILL_AMT6': 'bill_amount_apr',
    'PAY_AMT1': 'payment_amount_sep',
    'PAY_AMT2': 'payment_amount_aug',
    'PAY_AMT3': 'payment_amount_jul',
    'PAY_AMT4': 'payment_amount_jun',
    'PAY_AMT5': 'payment_amount_may',
    'PAY_AMT6': 'payment_amount_apr'
}
df = df.rename(columns=new_col_names)

X = df.drop('target', axis=1)
y = df['target']

print(f"Dataset shape: {df.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")
print(f"Default rate: {(y.mean() * 100):.2f}%")
print(f"Imbalance ratio: {sum(y==0)/sum(y==1):.2f}:1")

categorical_features = ['gender', 'education', 'marriage'] + \
                      [col for col in X.columns if 'repayment_status' in col]
numeric_features = [col for col in X.columns if col not in categorical_features]

for col in categorical_features:
    if 'repayment_status' in col:
        X[col] = X[col].astype(str)

print(f"\nCategorical columns ({len(categorical_features)}): {categorical_features}")
print(f"Numeric columns ({len(numeric_features)}): {numeric_features[:10]}...")

cat_categories = [sorted(X[col].dropna().unique().tolist()) for col in categorical_features]
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore',
                          sparse_output=False, categories=cat_categories), categorical_features),
    ('num', StandardScaler(), numeric_features)
])

X_processed = preprocessor.fit_transform(X)
d = X_processed.shape[1]
print(f"\nNumber of processed features: {d}")

fnames = preprocessor.get_feature_names_out()

# ============================================================
# 2. MODELS & SAMPLING STRATEGIES
# ============================================================

models = {
    'RF': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'XGB': xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42,
                             eval_metric='logloss', n_jobs=-1),
    'LGB': lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42,
                              verbose=-1, n_jobs=-1)
}

resamplers = {
    'None': None,
    'SMOTE': SMOTE(random_state=42),
    'SMOTETomek': SMOTETomek(random_state=42),
    'CostSensitive': 'cost'
}

classes = np.unique(y)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, weights)}
print(f"\nClass weights: {class_weight_dict}")

# ============================================================
# 3. EXPLANATION METHODS (9 METHODS)
# ============================================================

def get_shap_reliable(pipe, X_test):
    """
    SHAP values using TreeExplainer.
    If it fails, fallback to permutation importance.
    """
    clf = pipe.named_steps['clf']
    X_proc = pipe.named_steps['prep'].transform(X_test)
    try:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_proc)
        if isinstance(sv, list):
            return sv[1]
        elif sv.ndim == 3:
            return sv[:, :, 1]
        else:
            return sv
    except Exception as e:
        print(f"SHAP failed: {e}. Using permutation importance fallback...")
        from sklearn.inspection import permutation_importance
        res = permutation_importance(clf, X_proc, pipe.predict(X_proc),
                                     n_repeats=3, random_state=42)
        return np.tile(res.importances_mean, (X_proc.shape[0], 1))

def compute_banzhaf(pipe, X_test, n_samples=5, max_instances=5):
    """
    Banzhaf value approximation.
    For each feature f and instance x:
        φ_f ≈ E[ f(x with f) - f(x without f) ] over random coalitions.
    """
    clf = pipe.named_steps['clf']
    X_proc = pipe.named_steps['prep'].transform(X_test)
    n_feat = X_proc.shape[1]
    n_inst = min(max_instances, X_proc.shape[0])
    mat = np.zeros((n_inst, n_feat))
    for i in range(n_inst):
        x = X_proc[i:i+1]
        for f in range(n_feat):
            contrib = []
            for _ in range(n_samples):
                coal = np.random.binomial(1, 0.5, n_feat)
                x_with = x.copy()
                x_with[0, f] = x[0, f] * coal[f]
                p1 = clf.predict_proba(x_with)[0, 1]
                x_without = x.copy()
                x_without[0, f] = 0
                p0 = clf.predict_proba(x_without)[0, 1]
                contrib.append(p1 - p0)
            mat[i, f] = np.mean(contrib)
    return mat

def compute_myerson(pipe, X_test, G, alpha=0.5):
    """
    Myerson value:
        φ_i^M = (1 - α) * φ_i^SHAP + α * mean_{j in N(i)} φ_j^SHAP
    where N(i) are neighbors in the feature graph G.
    """
    shap_vals = get_shap_reliable(pipe, X_test)
    shap_mean = shap_vals.mean(axis=0)
    d_local = len(shap_mean)
    phi = np.zeros(d_local)
    for i in range(d_local):
        neighbors = list(G.neighbors(i))
        if len(neighbors) == 0:
            phi[i] = shap_mean[i]
        else:
            neighbor_mean = np.mean([shap_mean[j] for j in neighbors])
            phi[i] = (1 - alpha) * shap_mean[i] + alpha * neighbor_mean
    return np.tile(phi, (min(50, len(X_test)), 1)), shap_vals, shap_mean

def build_domain_groups_taiwan(fnames):
    """
    Domain-based Owen groups:
        - Demographic
        - Financial
        - Repayment
    """
    groups = {
        "Domain_Demographic": [],
        "Domain_Financial": [],
        "Domain_Repayment": []
    }
    for i, name in enumerate(fnames):
        lname = name.lower()
        if any(k in lname for k in ["gender", "education", "marriage", "age"]):
            groups["Domain_Demographic"].append(i)
        if any(k in lname for k in ["credit_limit", "bill_amount", "payment_amount"]):
            groups["Domain_Financial"].append(i)
        if "repayment_status" in lname:
            groups["Domain_Repayment"].append(i)
    return {g: v for g, v in groups.items() if len(v) > 0}

def build_data_groups(X_proc, n_groups=6):
    """
    Data-driven Owen groups via correlation clustering.
    Distance = 1 - |corr|.
    """
    try:
        X_proc = np.nan_to_num(X_proc, nan=0.0)
        with np.errstate(invalid='ignore'):
            corr = np.corrcoef(X_proc.T)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)
        dist = 1 - np.abs(corr)
        dist = np.clip(dist, 0.0, 2.0)

        if X_proc.shape[1] < n_groups:
            n_groups = max(2, X_proc.shape[1] // 2)
        clustering = AgglomerativeClustering(
            n_clusters=n_groups,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(dist)
        groups = {}
        for idx, lab in enumerate(labels):
            groups.setdefault(f"Data_Cluster_{lab}", []).append(idx)
        return groups, np.abs(corr)
    except Exception as e:
        print(f"Warning: Data-driven grouping failed: {e}")
        n_features = X_proc.shape[1]
        return {"Data_Default": list(range(n_features))}, np.eye(n_features)

def build_model_groups(shap_vals, n_groups=6):
    """
    Model-driven Owen groups via SHAP correlation.
    """
    try:
        shap_vals = np.nan_to_num(shap_vals, nan=0.0)
        if shap_vals.ndim == 1:
            shap_vals = shap_vals.reshape(1, -1)
        elif shap_vals.ndim == 3:
            shap_vals = shap_vals.mean(axis=2) if shap_vals.shape[2] > 1 else shap_vals.squeeze()

        if shap_vals.shape[0] < 2:
            n_features = shap_vals.shape[1] if shap_vals.shape[0] == 1 else shap_vals.shape[0]
            return {"Model_Default": list(range(n_features))}, np.eye(n_features)

        with np.errstate(invalid='ignore'):
            corr = np.corrcoef(shap_vals)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)
        dist = 1 - np.abs(corr)
        dist = np.clip(dist, 0.0, 2.0)

        n_features = corr.shape[0]
        if n_features < n_groups:
            n_groups = max(2, n_features // 2)
        clustering = AgglomerativeClustering(
            n_clusters=n_groups,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(dist)
        groups = {}
        for idx, lab in enumerate(labels):
            groups.setdefault(f"Model_Group_{lab}", []).append(idx)
        return groups, np.abs(corr)
    except Exception as e:
        print(f"Warning: Model-driven grouping failed: {e}")
        n_features = shap_vals.shape[1] if shap_vals.ndim > 1 else shap_vals.shape[0]
        return {"Model_Default": list(range(n_features))}, np.eye(n_features)

def compute_owen_from_vector(phi, groups):
    """
    Owen value redistribution from a base vector φ (SHAP or Myerson):
        For each group G:
            total = Σ |φ_j|, j in G
            per = total / |G|
            each feature in G gets ±per with sign of φ_j.
    """
    d_local = len(phi)
    out = np.zeros(d_local)
    for _, feats in groups.items():
        if len(feats) == 0:
            continue
        valid_feats = [f for f in feats if f < d_local]
        if not valid_feats:
            continue
        total = np.sum(np.abs(phi[valid_feats]))
        per = total / len(valid_feats)
        for f in valid_feats:
            sign = np.sign(phi[f]) if phi[f] != 0 else 1
            out[f] = sign * per
    return out

def compute_Q(groups, dep):
    """
    Group quality Q:
        Q = mean(within-group dependency) / mean(across-group dependency)
    Values > 1 indicate stronger within-group coherence.
    """
    try:
        d_local = dep.shape[0]
        group_id = np.full(d_local, -1)
        for gid, (_, feats) in enumerate(groups.items()):
            for f in feats:
                if f < d_local:
                    group_id[f] = gid
        within, across = [], []
        for i in range(d_local):
            for j in range(i+1, d_local):
                if group_id[i] == group_id[j] and group_id[i] != -1:
                    within.append(dep[i, j])
                elif group_id[i] != -1 and group_id[j] != -1:
                    across.append(dep[i, j])
        if len(within) == 0 or len(across) == 0:
            return 1.0
        within_mean = np.mean(within)
        across_mean = np.mean(across)
        if across_mean == 0:
            return 1.0 if within_mean == 0 else 10.0
        return float(within_mean / across_mean)
    except:
        return 1.0

# ============================================================
# 4. INTERPRETABILITY METRICS
# ============================================================

def stability_cv(expl_list):
    """
    Coefficient of variation (CV) across folds:
        CV = mean( std(|φ|) / mean(|φ|) )
    Lower CV → higher stability.
    """
    if len(expl_list) < 2:
        return 1.0
    try:
        arr = np.stack([np.abs(e) for e in expl_list])
        mean = arr.mean(axis=0) + 1e-8
        std = arr.std(axis=0)
        cv_per_feature = std / mean
        return float(np.mean(cv_per_feature))
    except:
        return 1.0

def kuncheva_index(expl_list, k=5):
    """
    Kuncheva index for top-k feature set stability:
        KI = (|A∩B| - k^2/d) / (k - k^2/d)
    """
    if len(expl_list) < 2:
        return 0.0
    sets = []
    for exp in expl_list:
        try:
            imp = np.abs(exp).mean(axis=0).ravel()
            topk = min(k, len(imp))
            topk_indices = set(np.argsort(imp)[-topk:].tolist())
            sets.append(topk_indices)
        except:
            sets.append(set())
    if len(sets) < 2:
        return 0.0
    ki_values = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            intersection = len(sets[i] & sets[j])
            if k == d:
                ki = 0.0
            else:
                ki = (intersection - (k**2)/d) / (k - (k**2)/d)
            ki = max(min(ki, 1.0), -1.0)
            ki_values.append(ki)
    return float(np.mean(ki_values)) if ki_values else 0.0

def cosine_directional_consistency(expl_list):
    """
    Cosine similarity between mean |φ| vectors across folds.
    Higher → more consistent direction of importance.
    """
    if len(expl_list) < 2:
        return 0.0
    mean_vecs = [np.mean(np.abs(e), axis=0).ravel() for e in expl_list]
    K = len(mean_vecs)
    cos_sims = []
    for p in range(K):
        for s in range(p + 1, K):
            v_p = mean_vecs[p]
            v_s = mean_vecs[s]
            cos = 1 - cosine(v_p, v_s)
            cos_sims.append(cos)
    if not cos_sims:
        return 0.0
    return np.mean(cos_sims)

def interpretability_score(cv, kuncheva, cosine_score, beta=1/3):
    """
    Overall interpretability:
        I = β(1 - CV) + β * KI + β * Cosine
    """
    return beta * (1 - cv) + beta * kuncheva + beta * cosine_score

def normalize(s):
    """Normalize to [0,1]."""
    return (s - s.min()) / (s.max() - s.min() + 1e-8)

def tradeoff_metric(auc_series, I_series, alpha=0.5):
    """
    Trade-off metric:
        T(α) = α * norm(AUC) + (1 - α) * norm(I)
    """
    return alpha * normalize(auc_series) + (1 - alpha) * normalize(I_series)

# ============================================================
# 5. UNIFIED EXPERIMENT LOOP – 9 METHODS
# ============================================================

print("\n" + "="*80)
print("STARTING UNIFIED EXPERIMENT - 9 EXPLANATION METHODS")
print("="*80)

# Build feature graph for Myerson
print("\nBuilding feature graph for Myerson value...")
X_proc_full = preprocessor.transform(X)
corr_full = np.corrcoef(X_proc_full.T)
G_myerson = nx.Graph()
d_graph = corr_full.shape[0]
G_myerson.add_nodes_from(range(d_graph))
for i in range(d_graph):
    for j in range(i+1, d_graph):
        if abs(corr_full[i, j]) >= 0.25:
            G_myerson.add_edge(i, j, weight=corr_full[i, j])
print(f"Graph built with {G_myerson.number_of_nodes()} nodes and {G_myerson.number_of_edges()} edges")

cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
all_records = []

for mname, model in models.items():
    for sname, sampler in resamplers.items():
        print(f"\n{'='*60}")
        print(f"Processing: {mname} + {sname}")
        print(f"{'='*60}")

        aucs = []
        shap_runs, banzhaf_runs, myerson_runs = [], [], []
        owen_dom_runs, owen_data_runs, owen_model_runs = [], [], []
        owen_dom_my_runs, owen_data_my_runs, owen_model_my_runs = [], [], []

        Q_dom_list, Q_data_list, Q_model_list = [], [], []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            steps = [('prep', preprocessor)]
            if sampler and sampler != 'cost':
                steps.append(('samp', sampler))
            steps.append(('clf', model))
            pipe = ImbPipeline(steps)

            if sname == 'CostSensitive':
                if mname == 'RF':
                    pipe.named_steps['clf'].set_params(class_weight=class_weight_dict)
                elif mname == 'XGB':
                    ratio = class_weight_dict[1] / class_weight_dict[0] if 0 in class_weight_dict else 1
                    pipe.named_steps['clf'].set_params(scale_pos_weight=ratio)
                elif mname == 'LGB':
                    pipe.named_steps['clf'].set_params(class_weight=class_weight_dict)

            pipe.fit(X_tr, y_tr)

            # AUC per fold
            y_proba = pipe.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, y_proba)
            aucs.append(auc)
            print(f" Fold {fold} | {mname}+{sname:15} → AUC: {auc:.4f}")

            X_sample = X_te.sample(n=min(50, len(X_te)), random_state=42)

            # 1. SHAP
            try:
                shap_vals = get_shap_reliable(pipe, X_sample)
                shap_runs.append(shap_vals)
            except Exception as e:
                print(f"  SHAP failed: {e}")
                n_features = preprocessor.transform(X_sample[:1]).shape[1]
                shap_vals = np.zeros((len(X_sample), n_features))
                shap_runs.append(shap_vals)

            # 2. Banzhaf
            try:
                banzhaf_vals = compute_banzhaf(pipe, X_sample)
                banzhaf_runs.append(banzhaf_vals)
            except Exception as e:
                print(f"  Banzhaf failed: {e}")
                n_features = preprocessor.transform(X_sample[:1]).shape[1]
                banzhaf_vals = np.zeros((min(5, len(X_sample)), n_features))
                banzhaf_runs.append(banzhaf_vals)

            # 3. Myerson (and SHAP mean)
            try:
                myerson_vals, shap_vals_full, shap_mean = compute_myerson(pipe, X_sample, G_myerson)
                myerson_runs.append(myerson_vals)
            except Exception as e:
                print(f"  Myerson failed: {e}")
                n_features = preprocessor.transform(X_sample[:1]).shape[1]
                myerson_vals = np.zeros((min(50, len(X_sample)), n_features))
                myerson_runs.append(myerson_vals)
                shap_vals_full = shap_vals
                shap_mean = shap_vals.mean(axis=0)

            # Domain groups
            dom_groups = build_domain_groups_taiwan(fnames)

            # Data groups
            X_tr_proc = preprocessor.transform(X_tr)
            data_groups, dep_data = build_data_groups(X_tr_proc)

            # Model groups (using background SHAP)
            bg_sample = X_tr.sample(n=min(100, len(X_tr)), random_state=42)
            shap_bg = get_shap_reliable(pipe, bg_sample)
            model_groups, dep_mod = build_model_groups(shap_bg)

            # Correlation for Q (domain)
            corr_tr = np.abs(np.corrcoef(X_tr_proc.T))
            corr_tr = np.nan_to_num(corr_tr, nan=0.0)

            # 4–6. Owen from SHAP
            phi_shap = shap_vals_full.mean(axis=0)

            owen_dom_vec = compute_owen_from_vector(phi_shap, dom_groups)
            owen_dom = np.tile(owen_dom_vec, (len(X_sample), 1))
            owen_dom_runs.append(owen_dom)
            Q_dom_list.append(compute_Q(dom_groups, corr_tr))

            owen_data_vec = compute_owen_from_vector(phi_shap, data_groups)
            owen_data = np.tile(owen_data_vec, (len(X_sample), 1))
            owen_data_runs.append(owen_data)
            Q_data_list.append(compute_Q(data_groups, dep_data))

            owen_model_vec = compute_owen_from_vector(phi_shap, model_groups)
            owen_model = np.tile(owen_model_vec, (len(X_sample), 1))
            owen_model_runs.append(owen_model)
            Q_model_list.append(compute_Q(model_groups, dep_mod))

            # 7–9. Owen from Myerson (Myerson–Owen hybrids)
            phi_my = myerson_vals.mean(axis=0)

            owen_dom_my_vec = compute_owen_from_vector(phi_my, dom_groups)
            owen_dom_my = np.tile(owen_dom_my_vec, (len(X_sample), 1))
            owen_dom_my_runs.append(owen_dom_my)

            owen_data_my_vec = compute_owen_from_vector(phi_my, data_groups)
            owen_data_my = np.tile(owen_data_my_vec, (len(X_sample), 1))
            owen_data_my_runs.append(owen_data_my)

            owen_model_my_vec = compute_owen_from_vector(phi_my, model_groups)
            owen_model_my = np.tile(owen_model_my_vec, (len(X_sample), 1))
            owen_model_my_runs.append(owen_model_my)

        auc_mean = np.mean(aucs)

        explanation_methods = [
            ('SHAP', shap_runs),
            ('Banzhaf', banzhaf_runs),
            ('Myerson', myerson_runs),
            ('Owen-Domain', owen_dom_runs),
            ('Owen-Data', owen_data_runs),
            ('Owen-Model', owen_model_runs),
            ('Owen-Domain-Myerson', owen_dom_my_runs),
            ('Owen-Data-Myerson', owen_data_my_runs),
            ('Owen-Model-Myerson', owen_model_my_runs)
        ]

        Q_values = {
            'Owen-Domain': np.mean(Q_dom_list) if Q_dom_list else 1.0,
            'Owen-Data': np.mean(Q_data_list) if Q_data_list else 1.0,
            'Owen-Model': np.mean(Q_model_list) if Q_model_list else 1.0,
            'Owen-Domain-Myerson': np.mean(Q_dom_list) if Q_dom_list else 1.0,
            'Owen-Data-Myerson': np.mean(Q_data_list) if Q_data_list else 1.0,
            'Owen-Model-Myerson': np.mean(Q_model_list) if Q_model_list else 1.0
        }

        for method_name, runs in explanation_methods:
            cv_val = stability_cv(runs)
            kun_val = kuncheva_index(runs, k=5)
            cos_val = cosine_directional_consistency(runs)
            I = interpretability_score(cv_val, kun_val, cos_val)

            record = {
                'Dataset': 'Taiwan',
                'Model': mname,
                'Sampler': sname,
                'Method': method_name,
                'AUC': auc_mean,
                'CV': cv_val,
                'Stability': 1 - cv_val,
                'Kuncheva': kun_val,
                'Cosine': cos_val,
                'I': I,
                'Q': Q_values.get(method_name, np.nan)
            }
            all_records.append(record)

# ============================================================
# 6. PROCESS AND SAVE RESULTS
# ============================================================

metrics = pd.DataFrame(all_records)
metrics['T(α=0.5)'] = tradeoff_metric(metrics['AUC'], metrics['I'])

print("\n" + "="*80)
print("COMPLETE RESULTS - ALL 9 EXPLANATION METHODS (TAIWAN)")
print("="*80)
print(metrics.round(4).to_string(index=False))

metrics.to_csv('taiwan_unified_results_9methods.csv', index=False)
print("\n✓ Results saved to: taiwan_unified_results_9methods.csv")

# ============================================================
# 7. PLOTS – MODEL ROC CURVES
# ============================================================

print("\n" + "="*80)
print("PLOTTING MODEL ROC CURVES")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

plt.figure(figsize=(8, 7))
for mname, model in models.items():
    pipe = ImbPipeline([
        ('prep', preprocessor),
        ('clf', model)
    ])
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{mname} (AUC={auc_val:.3f})", linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Model ROC Curves – Taiwan Credit")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 8. VISUALIZATIONS – 9 METHODS
# ============================================================

method_order = [
    'SHAP', 'Banzhaf', 'Myerson',
    'Owen-Domain', 'Owen-Data', 'Owen-Model',
    'Owen-Domain-Myerson', 'Owen-Data-Myerson', 'Owen-Model-Myerson'
]

method_colors = {
    'SHAP': '#4C78A8',
    'Banzhaf': '#F58518',
    'Myerson': '#E45756',
    'Owen-Domain': '#D62728',           # RED as requested
    'Owen-Data': '#54A24B',
    'Owen-Model': '#EECA3B',
    'Owen-Domain-Myerson': '#8E6C8A',   # distinct from Owen-Domain
    'Owen-Data-Myerson': '#FF9DA6',
    'Owen-Model-Myerson': '#9C755F'
}

metrics_g = metrics.copy()

# ------------------------------------------------------------
# Pareto front: AUC vs I (scatter, legends outside)
# ------------------------------------------------------------
plt.figure(figsize=(9, 7))
for method in method_order:
    if method in metrics_g['Method'].unique():
        sub = metrics_g[metrics_g['Method'] == method]
        plt.scatter(sub['AUC'], sub['I'],
                    s=120, label=method,
                    color=method_colors[method],
                    alpha=0.8, edgecolors='black')

plt.xlabel("AUC-ROC")
plt.ylabel("Interpretability Score I")
plt.title("Pareto Front – AUC vs Interpretability (9 Methods)")
x_min, x_max = metrics_g['AUC'].min(), metrics_g['AUC'].max()
y_min, y_max = metrics_g['I'].min(), metrics_g['I'].max()
plt.xlim(x_min - 0.01, x_max + 0.01)
plt.ylim(y_min - 0.02, y_max + 0.02)
plt.grid(alpha=0.3)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Trade-off T(α=0.5) comparison
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
T_means = metrics_g.groupby("Method")["T(α=0.5)"].mean().reindex(method_order)
bars = plt.bar(T_means.index, T_means.values,
               color=[method_colors[m] for m in T_means.index],
               edgecolor='black')
plt.bar_label(bars, fmt="%.3f", padding=3)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Mean Trade-off T(α=0.5)")
plt.title("Trade-off Comparison Across 9 Methods – Taiwan")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Interpretability I comparison
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
I_means = metrics_g.groupby("Method")["I"].mean().reindex(method_order)
bars = plt.bar(I_means.index, I_means.values,
               color=[method_colors[m] for m in I_means.index],
               edgecolor='black')
plt.bar_label(bars, fmt="%.3f", padding=3)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Mean Interpretability I")
plt.title("Interpretability Comparison Across 9 Methods – Taiwan")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Model × Sampler performance (AUC)
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.barplot(
    data=metrics_g,
    x="Model",
    y="AUC",
    hue="Sampler",
    ci=None,
    edgecolor='black'
)
plt.title("AUC by Model × Sampler (All Methods Pooled) – Taiwan")
plt.grid(axis='y', alpha=0.3)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Group quality Q for Owen & Myerson–Owen
# ------------------------------------------------------------
owen_methods = [
    'Owen-Domain', 'Owen-Data', 'Owen-Model',
    'Owen-Domain-Myerson', 'Owen-Data-Myerson', 'Owen-Model-Myerson'
]
owen_data = metrics_g[metrics_g["Method"].isin(owen_methods)]

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=owen_data,
    x="Method",
    y="Q",
    order=owen_methods,
    palette=[method_colors[m] for m in owen_methods]
)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Group Quality Q")
plt.title("Group Quality (Q) – Owen & Myerson–Owen (Taiwan)")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Q vs I scatter for Owen-only
# ------------------------------------------------------------
plt.figure(figsize=(9, 7))
for method in owen_methods:
    sub = owen_data[owen_data["Method"] == method]
    plt.scatter(sub["Q"], sub["I"],
                s=120, label=method,
                color=method_colors[method],
                alpha=0.8, edgecolors='black')
plt.xlabel("Group Quality Q")
plt.ylabel("Interpretability I")
plt.title("Q vs I – Owen & Myerson–Owen Methods (Taiwan)")
plt.grid(alpha=0.3)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

# ============================================================
# 9. STATISTICAL ANALYSIS
# ============================================================

print("\n" + "="*80)
print("STATISTICAL ANALYSIS – 9 METHODS (TAIWAN)")
print("="*80)

# ------------------------------------------------------------
# Spearman correlation between AUC and I
# ------------------------------------------------------------
rho, p_spear = spearmanr(metrics_g["AUC"], metrics_g["I"])
print(f"\nSpearman correlation (AUC vs I): ρ={rho:.4f}, p={p_spear:.6f}")

# ------------------------------------------------------------
# Levene’s test for homogeneity of variances on T(α=0.5)
# ------------------------------------------------------------
print("\nLevene’s Test on T(α=0.5) across methods:")
groups_T = [metrics_g[metrics_g["Method"] == m]["T(α=0.5)"].values
            for m in method_order if m in metrics_g["Method"].unique()]
stat_lev, p_lev = levene(*groups_T)
print(f"Levene’s statistic={stat_lev:.4f}, p={p_lev:.6f}")

# ------------------------------------------------------------
# Shapiro–Wilk normality test per method on T(α=0.5)
# ------------------------------------------------------------
print("\nShapiro–Wilk Normality Test on T(α=0.5) per method:")
for m in method_order:
    vals = metrics_g[metrics_g["Method"] == m]["T(α=0.5)"].values
    if len(vals) >= 3:
        stat_sh, p_sh = shapiro(vals)
        print(f"{m}: W={stat_sh:.4f}, p={p_sh:.6f}")
    else:
        print(f"{m}: Not enough samples for Shapiro–Wilk")

# ------------------------------------------------------------
# Cliff’s Delta for all pairwise method comparisons (T(α=0.5))
# ------------------------------------------------------------
def cliffs_delta(x, y):
    """
    Cliff’s delta:
        δ = ( (#x>y - #x<y) ) / (n1 * n2)
    Magnitude thresholds:
        |δ| < 0.147 → negligible
        < 0.33 → small
        < 0.474 → medium
        ≥ 0.474 → large
    """
    x = np.array(x)
    y = np.array(y)
    n1, n2 = len(x), len(y)
    greater = sum(xi > yj for xi in x for yj in y)
    less = sum(xi < yj for xi in x for yj in y)
    delta = (greater - less) / (n1 * n2)
    ad = abs(delta)
    if ad < 0.147:
        mag = "negligible"
    elif ad < 0.33:
        mag = "small"
    elif ad < 0.474:
        mag = "medium"
    else:
        mag = "large"
    return delta, mag

print("\nCliff’s Delta for all pairwise method comparisons (T(α=0.5)):")
for i in range(len(method_order)):
    for j in range(i+1, len(method_order)):
        m1, m2 = method_order[i], method_order[j]
        d1 = metrics_g[metrics_g["Method"] == m1]["T(α=0.5)"].values
        d2 = metrics_g[metrics_g["Method"] == m2]["T(α=0.5)"].values
        if len(d1) > 1 and len(d2) > 1:
            delta, mag = cliffs_delta(d1, d2)
            print(f"{m1} vs {m2}: δ={delta:.4f} ({mag})")

# ------------------------------------------------------------
# Wilcoxon signed-rank tests for all pairwise method comparisons
# (T(α=0.5)) – only when paired lengths match
# ------------------------------------------------------------
print("\nWilcoxon Signed-Rank Tests (T(α=0.5)) – pairwise methods:")
for i in range(len(method_order)):
    for j in range(i+1, len(method_order)):
        m1, m2 = method_order[i], method_order[j]
        d1 = metrics_g[metrics_g["Method"] == m1]["T(α=0.5)"].values
        d2 = metrics_g[metrics_g["Method"] == m2]["T(α=0.5)"].values
        if len(d1) == len(d2) and len(d1) > 1:
            try:
                stat_w, p_w = wilcoxon(d1, d2)
                sig = "SIGNIFICANT" if p_w < 0.05 else "ns"
                print(f"{m1} vs {m2}: p={p_w:.6f} ({sig})")
            except Exception:
                print(f"{m1} vs {m2}: Wilcoxon could not be computed")
        else:
            # Not strictly paired; skip
            pass

# ------------------------------------------------------------
# Friedman test + Nemenyi post-hoc on T(α=0.5)
# ------------------------------------------------------------
print("\nFriedman Test Across Methods (T(α=0.5)):")

pivot = metrics_g.pivot_table(
    values="T(α=0.5)",
    index=["Model", "Sampler"],
    columns="Method",
    aggfunc="mean"
)

complete_methods = pivot.dropna(axis=1).columns.tolist()
complete_methods = [m for m in method_order if m in complete_methods]

if len(complete_methods) >= 3:
    data = pivot[complete_methods].values
    stat_f, p_f = friedmanchisquare(*[data[:, i] for i in range(data.shape[1])])
    print(f"Methods: {complete_methods}")
    print(f"Friedman χ²={stat_f:.4f}, p={p_f:.6f}")
    if p_f < 0.05:
        print("→ Significant differences detected (Friedman)")
        nemenyi = sp.posthoc_nemenyi_friedman(data)
        nemenyi.index = complete_methods
        nemenyi.columns = complete_methods
        print("\nNemenyi Post-hoc Test (p-values):")
        print(nemenyi.round(4))
    else:
        print("→ No significant differences (Friedman)")
else:
    print("Not enough complete methods for Friedman test.")

print("\n" + "="*80)
print("UNIFIED 9-METHOD EXPERIMENT & ANALYSIS COMPLETE – TAIWAN")
print("="*80)
