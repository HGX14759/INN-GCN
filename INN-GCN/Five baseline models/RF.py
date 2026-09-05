import os
import numpy as np
import random
import pandas as pd
import joblib
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, hp, STATUS_OK

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

BASE_SEED = 42
CV_FOLD_SEED = BASE_SEED
HP_OPT_SEED = BASE_SEED

EXCEL_PATH = r"D:\YWGJ\gjsj001.xlsx"
MAX_EVALS = 100
N_FOLDS = 5
USE_DATA_AUGMENTATION = True

VIS_SAVE_DIR = "./paper_visualizations"
MODEL_SAVE_DIR = r"D:\project\contrast model\cocrystal_prediction_RF"
os.makedirs(VIS_SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

ECFP_RADIUS = 2
ECFP_NBITS = 1024

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

np.random.seed(BASE_SEED)
random.seed(BASE_SEED)

def smiles_to_ecfp4(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(ECFP_NBITS, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=ECFP_RADIUS, nBits=ECFP_NBITS)
    return np.array(fp, dtype=np.float32)

def calculate_complementary_features(api_desc, ccf_desc, api_hba_hbd_homo_lumo, ccf_hba_hbd_homo_lumo):
    api_hba, api_hbd, api_homo, api_lumo = api_hba_hbd_homo_lumo
    ccf_hba, ccf_hbd, ccf_homo, ccf_lumo = ccf_hba_hbd_homo_lumo
    api_rbn, api_s, api_s_l, api_s_m, api_m_l, api_fr_no, api_fr_aromaticAtom, \
        api_xlogp3, api_tpsa, api_acd_logp, api_mv, api_polarizability, api_dipole = api_desc
    ccf_rbn, ccf_s, ccf_s_l, ccf_s_m, ccf_m_l, ccf_fr_no, ccf_fr_aromaticAtom, \
        ccf_xlogp3, ccf_tpsa, ccf_acd_logp, ccf_mv, ccf_polarizability, ccf_dipole = ccf_desc

    pair_feat = [
        api_hba * ccf_hbd + api_hbd * ccf_hba,
        api_homo - ccf_lumo,
        ccf_homo - api_lumo,
        api_polarizability * ccf_polarizability,
        abs(api_rbn - ccf_rbn), abs(api_s - ccf_s), abs(api_s_l - ccf_s_l),
        abs(api_s_m - ccf_s_m), abs(api_m_l - ccf_m_l), abs(api_fr_no - ccf_fr_no),
        abs(api_fr_aromaticAtom - ccf_fr_aromaticAtom), abs(api_xlogp3 - ccf_xlogp3),
        abs(api_tpsa - ccf_tpsa), abs(api_acd_logp - ccf_acd_logp),
        abs(api_mv - ccf_mv), abs(api_dipole - ccf_dipole)
    ]
    return np.array(pair_feat, dtype=np.float32)

class CocrystalDataset:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.df = pd.read_excel(excel_path, sheet_name='Sheet1')
        self.X, self.y, self.row_indices = self._load_original()

    def _load_original(self):
        X_list = []
        y_list = []
        rows = []
        for idx, row in self.df.iterrows():
            y = int(row['Target'])
            smi1 = row['SMILES1']
            smi2 = row['SMILES2']

            api_fp = smiles_to_ecfp4(smi1)
            ccf_fp = smiles_to_ecfp4(smi2)

            api_17 = [row[c] for c in [
                'API_RBN', 'API_S', 'API_S_L', 'API_S_M', 'API_M_L', 'API_Fr_NO', 'API_Fr_aromaticAtom',
                'API_XLogP3', 'API_Topological Polar Surface Area', 'API_ACD/LogP', 'API_MV',
                'API_Polarizability', 'API_Dipole Moment', 'API_HBA', 'API_HBD', 'API_homo', 'API_lumo'
            ]]
            ccf_17 = [row[c] for c in [
                'CCF_RBN', 'CCF_S', 'CCF_S_L', 'CCF_S_M', 'CCF_M_L', 'CCF_Fr_NO', 'CCF_Fr_aromaticAtom',
                'CCF_XLogP3', 'CCF_Topological Polar Surface Area', 'CCF_ACD/LogP', 'CCF_MV',
                'CCF_Polarizability', 'CCF_Dipole Moment', 'CCF_HBA', 'CCF_HBD', 'CCF_homo', 'CCF_lumo'
            ]]
            api13, api4 = api_17[:13], api_17[13:]
            ccf13, ccf4 = ccf_17[:13], ccf_17[13:]
            pair_feat = calculate_complementary_features(api13, ccf13, api4, ccf4)

            full_feat = np.concatenate([api_fp, ccf_fp, pair_feat])
            X_list.append(full_feat)
            y_list.append(y)
            rows.append(idx)
        return np.array(X_list), np.array(y_list), np.array(rows)

    def augment_train_set(self, train_indices):
        X_aug = []
        y_aug = []
        for idx in train_indices:
            row = self.df.iloc[idx]
            y = int(row['Target'])
            smi1 = row['SMILES1']
            smi2 = row['SMILES2']

            api_fp = smiles_to_ecfp4(smi1)
            ccf_fp = smiles_to_ecfp4(smi2)
            api_17 = [row[c] for c in [
                'API_RBN', 'API_S', 'API_S_L', 'API_S_M', 'API_M_L', 'API_Fr_NO', 'API_Fr_aromaticAtom',
                'API_XLogP3', 'API_Topological Polar Surface Area', 'API_ACD/LogP', 'API_MV',
                'API_Polarizability', 'API_Dipole Moment', 'API_HBA', 'API_HBD', 'API_homo', 'API_lumo'
            ]]
            ccf_17 = [row[c] for c in [
                'CCF_RBN', 'CCF_S', 'CCF_S_L', 'CCF_S_M', 'CCF_M_L', 'CCF_Fr_NO', 'CCF_Fr_aromaticAtom',
                'CCF_XLogP3', 'CCF_Topological Polar Surface Area', 'CCF_ACD/LogP', 'CCF_MV',
                'CCF_Polarizability', 'CCF_Dipole Moment', 'CCF_HBA', 'CCF_HBD', 'CCF_homo', 'CCF_lumo'
            ]]
            api13, api4 = api_17[:13], api_17[13:]
            ccf13, ccf4 = ccf_17[:13], ccf_17[13:]
            pair = calculate_complementary_features(api13, ccf13, api4, ccf4)
            feat_original = np.concatenate([api_fp, ccf_fp, pair])
            X_aug.append(feat_original)
            y_aug.append(y)

            if USE_DATA_AUGMENTATION:
                pair_aug = calculate_complementary_features(ccf13, api13, ccf4, api4)
                feat_aug = np.concatenate([ccf_fp, api_fp, pair_aug])
                X_aug.append(feat_aug)
                y_aug.append(y)

        return np.array(X_aug), np.array(y_aug)

    def get_val_set(self, val_indices):
        X_val = []
        y_val = []
        for idx in val_indices:
            row = self.df.iloc[idx]
            y = int(row['Target'])
            smi1 = row['SMILES1']
            smi2 = row['SMILES2']

            api_fp = smiles_to_ecfp4(smi1)
            ccf_fp = smiles_to_ecfp4(smi2)

            api_17 = [row[c] for c in [
                'API_RBN', 'API_S', 'API_S_L', 'API_S_M', 'API_M_L', 'API_Fr_NO', 'API_Fr_aromaticAtom',
                'API_XLogP3', 'API_Topological Polar Surface Area', 'API_ACD/LogP', 'API_MV',
                'API_Polarizability', 'API_Dipole Moment', 'API_HBA', 'API_HBD', 'API_homo', 'API_lumo'
            ]]
            ccf_17 = [row[c] for c in [
                'CCF_RBN', 'CCF_S', 'CCF_S_L', 'CCF_S_M', 'CCF_M_L', 'CCF_Fr_NO', 'CCF_Fr_aromaticAtom',
                'CCF_XLogP3', 'CCF_Topological Polar Surface Area', 'CCF_ACD/LogP', 'CCF_MV',
                'CCF_Polarizability', 'CCF_Dipole Moment', 'CCF_HBA', 'CCF_HBD', 'CCF_homo', 'CCF_lumo'
            ]]
            api13, api4 = api_17[:13], api_17[13:]
            ccf13, ccf4 = ccf_17[:13], ccf_17[13:]
            pair = calculate_complementary_features(api13, ccf13, api4, ccf4)

            full_feat = np.concatenate([api_fp, ccf_fp, pair])
            X_val.append(full_feat)
            y_val.append(y)
        return np.array(X_val), np.array(y_val)

    def get_cv_folds(self):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_FOLD_SEED)
        folds = []
        for train_idx, val_idx in skf.split(self.X, self.y):
            X_tr, y_tr = self.augment_train_set(self.row_indices[train_idx])
            X_va, y_va = self.get_val_set(self.row_indices[val_idx])

            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_va = sc.transform(X_va)

            folds.append({
                'x_tr': X_tr, 'y_tr': y_tr,
                'x_va': X_va, 'y_va': y_va,
                'sc': sc
            })
        return folds

def build_model(params):
    return RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
        random_state=BASE_SEED,
        n_jobs=-1,
        class_weight='balanced'
    )

def get_metrics(y_true, y_pred_proba):
    y_pred = (y_pred_proba > 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_pred_proba)
    }

class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        preds = [m.predict_proba(X)[:, 1] for m in self.models]
        return np.mean(preds, axis=0)

def plot_roc(y, prob):
    fpr, tpr, _ = roc_curve(y, prob)
    plt.figure(figsize=(10,7))
    plt.plot(fpr, tpr, label=f'AUC={roc_auc_score(y, prob):.3f}')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{VIS_SAVE_DIR}/roc.png')
    plt.close()

def plot_fold_box(metrics_list):
    df = pd.DataFrame(metrics_list)
    plt.figure(figsize=(14,8))
    df[['accuracy','precision','recall','f1','auc']].boxplot()
    plt.title('5-Fold Validation Metrics Boxplot')
    plt.tight_layout()
    plt.savefig(f'{VIS_SAVE_DIR}/fold_box.png')
    plt.close()

def plot_train_val_bar(train_metrics, val_metrics):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    train_scores = [train_metrics[m] for m in metrics]
    val_scores = [val_metrics[m] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(12,7))
    plt.bar(x-width/2, train_scores, width, label='Train', color='#457b9d')
    plt.bar(x+width/2, val_scores, width, label='Validation', color='#e63946')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Train vs Validation')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{VIS_SAVE_DIR}/train_val_bar.png')
    plt.close()

def main():
    print("Loading data...")
    dataset = CocrystalDataset(EXCEL_PATH)
    folds = dataset.get_cv_folds()

    space = {
        'n_estimators': hp.randint('n_estimators', 100, 500),
        'max_depth': hp.randint('max_depth', 5, 30),
        'min_samples_split': hp.randint('min_samples_split', 2, 10),
        'min_samples_leaf': hp.randint('min_samples_leaf', 1, 5),
        'max_features': hp.choice('max_features', ['sqrt', 'log2'])
    }

    def objective(args):
        f1_list = []
        for f in folds:
            m = RandomForestClassifier(
                n_estimators=args['n_estimators'],max_depth=args['max_depth'],
                min_samples_split=args['min_samples_split'],min_samples_leaf=args['min_samples_leaf'],
                max_features=args['max_features'],random_state=BASE_SEED,n_jobs=-1
            )
            m.fit(f['x_tr'], f['y_tr'])
            pred = m.predict_proba(f['x_va'])[:,1]
            f1 = f1_score(f['y_va'], (pred>0.5).astype(int), zero_division=0)
            f1_list.append(f1)
        return {'loss': -np.mean(f1_list), 'status': STATUS_OK}

    best = fmin(objective, space, algo=tpe.suggest, max_evals=MAX_EVALS, rstate=np.random.default_rng(HP_OPT_SEED))
    best_params = {
        'n_estimators': best['n_estimators'],
        'max_depth': best['max_depth'],
        'min_samples_split': best['min_samples_split'],
        'min_samples_leaf': best['min_samples_leaf'],
        'max_features': ['sqrt','log2'][best['max_features']]
    }
    print("Best Params:", best_params)

    models = []
    train_metrics_list = []
    val_metrics_list = []
    oof_proba = []
    oof_true = []

    print("\nTraining final model...")
    for f in folds:
        m = build_model(best_params)
        m.fit(f['x_tr'], f['y_tr'])
        models.append(m)

        tr_pred = m.predict_proba(f['x_tr'])[:,1]
        tr_met = get_metrics(f['y_tr'], tr_pred)
        train_metrics_list.append(tr_met)

        va_pred = m.predict_proba(f['x_va'])[:,1]
        va_met = get_metrics(f['y_va'], va_pred)
        val_metrics_list.append(va_met)

        oof_proba.extend(va_pred)
        oof_true.extend(f['y_va'])

        print(f"Fold | Train F1={tr_met['f1']:.3f} | Val F1={va_met['f1']:.3f} | Val AUC={va_met['auc']:.3f}")

    avg_train = {k: np.mean([m[k] for m in train_metrics_list]) for k in ['accuracy','precision','recall','f1','auc']}
    avg_val = {k: np.mean([m[k] for m in val_metrics_list]) for k in ['accuracy','precision','recall','f1','auc']}

    ens = EnsembleModel(models)
    joblib.dump(ens, f'{MODEL_SAVE_DIR}/ensemble_rf_final.pkl')

    oof_proba = np.array(oof_proba)
    oof_true = np.array(oof_true)
    final_met = get_metrics(oof_true, oof_proba)

    print("\n=== Ensemble OOF Performance ===")
    print(f"accuracy  : {final_met['accuracy']:.4f}")
    print(f"precision : {final_met['precision']:.4f}")
    print(f"recall    : {final_met['recall']:.4f}")
    print(f"f1        : {final_met['f1']:.4f}")
    print(f"auc       : {final_met['auc']:.4f}")

    plot_roc(oof_true, oof_proba)
    plot_fold_box(val_metrics_list)
    plot_train_val_bar(avg_train, avg_val)

    print("\nDone! ✅ 已加入 ECFP4 指纹 ✅ 验证集原始不增强")

if __name__ == "__main__":
    main()