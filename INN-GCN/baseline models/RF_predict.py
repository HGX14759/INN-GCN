import os
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
from sklearn.preprocessing import StandardScaler

class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        preds = [m.predict_proba(X)[:, 1] for m in self.models]
        return np.mean(preds, axis=0)

BASE_SEED = 42
BEST_PARAMS = {
    'n_estimators': 291,
    'max_depth': 27,
    'min_samples_split': 4,
    'min_samples_leaf': 1,
    'max_features': 'sqrt'
}
EXCEL_PATH = r"D:\project\predict\sjyc-csj.xlsx"
MODEL_SAVE_DIR = r"D:\project\contrast model\cocrystal_prediction_RF"
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "ensemble_rf_final.pkl")
SCALER_LIST_PATH = os.path.join(MODEL_SAVE_DIR, "scaler_list.pkl")
RESULT_SAVE_PATH = r"D:\project\predict\RF_prediction_result.xlsx"
ECFP_RADIUS = 2
ECFP_NBITS = 1024

np.random.seed(BASE_SEED)

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

def generate_features_from_df(df):
    X_list = []
    for idx, row in df.iterrows():
        api_fp = smiles_to_ecfp4(row['SMILES1'])
        ccf_fp = smiles_to_ecfp4(row['SMILES2'])

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
    return np.array(X_list)

def main():
    print("="*60)
    print("共晶形成预测模型 - 全新数据集预测")
    print("="*60)
    print(f"固定最优超参数: {BEST_PARAMS}")
    print(f"输入数据集路径: {EXCEL_PATH}")
    print(f"集成模型路径: {ENSEMBLE_MODEL_PATH}")
    print(f"结果保存路径: {RESULT_SAVE_PATH}")
    print("="*60)

    print("\n[1/6] 加载输入数据集...")
    if not os.path.exists(EXCEL_PATH):
        print(f"❌ 错误：数据集文件不存在于 {EXCEL_PATH}，请检查路径是否正确")
        return
    df = pd.read_excel(EXCEL_PATH, sheet_name='Sheet1')
    print(f"✅ 数据集加载完成，共 {len(df)} 条样本")

    print("\n[2/6] 生成特征（ECFP4指纹+配对互补特征）...")
    X = generate_features_from_df(df)
    print(f"✅ 特征生成完成，特征维度: {X.shape}")

    print("\n[3/6] 加载训练好的集成模型...")
    if not os.path.exists(ENSEMBLE_MODEL_PATH):
        print(f"❌ 错误：集成模型文件不存在 {ENSEMBLE_MODEL_PATH}")
        print("请先运行训练代码生成模型文件")
        return
    ensemble_model = joblib.load(ENSEMBLE_MODEL_PATH)
    print(f"✅ 集成模型加载完成，基模型数量：{len(ensemble_model.models)}")

    print("\n[4/6] 标准化特征...")
    if os.path.exists(SCALER_LIST_PATH):
        scaler_list = joblib.load(SCALER_LIST_PATH)
        X_scaled_list = [scaler.transform(X) for scaler in scaler_list]
    else:
        print("⚠️ 未找到fold标准化器，临时拟合标准化器")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_list = [X_scaled] * len(ensemble_model.models)
    print("✅ 特征标准化完成")

    print("\n[5/6] 执行预测...")
    pred_proba_list = []
    for model, X_scaled in zip(ensemble_model.models, X_scaled_list):
        pred_proba = model.predict_proba(X_scaled)[:, 1]
        pred_proba_list.append(pred_proba)
    y_pred_proba = np.mean(pred_proba_list, axis=0)
    y_pred = (y_pred_proba > 0.5).astype(int)
    print("✅ 预测完成")

    print("\n[6/6] 整理并保存预测结果...")
    result_df = df.copy()
    result_df['预测标签'] = y_pred
    result_df['预测概率(共晶形成)'] = y_pred_proba
    result_df['预测结果说明'] = result_df['预测标签'].map({1: '共晶形成', 0: '无共晶形成'})
    result_df.to_excel(RESULT_SAVE_PATH, index=False)
    print(f"✅ 预测结果已保存至: {RESULT_SAVE_PATH}")

    print("\n" + "="*60)
    print("📊 预测结果统计")
    print("="*60)
    print(f"总样本数: {len(result_df)}")
    print(f"预测共晶形成: {sum(y_pred)} 条")
    print(f"预测无共晶形成: {len(y_pred) - sum(y_pred)} 条")
    print(f"共晶形成占比: {sum(y_pred)/len(y_pred):.2%}")
    print(f"平均预测概率: {np.mean(y_pred_proba):.4f}")
    print("="*60)
    print("🎉 预测全部完成！")

if __name__ == "__main__":
    main()