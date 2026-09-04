import os
import numpy as np
import random
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

BASE_SEED = 42
ECFP_RADIUS = 2
ECFP_NBITS = 1024

BEST_PARAMS = {
    'C': 0.002110897991492277,
    'kernel': 'linear',
    'gamma': 0.7192901084136052,
    'random_state': BASE_SEED,
    'probability': True,
    'class_weight': 'balanced'
}

PREDICT_EXCEL_PATH = r"D:\project\predict\sjyc-csj.xlsx"
TRAIN_EXCEL_PATH = r"D:\YWGJ\gjsj001.xlsx"
RESULT_SAVE_PATH = r"D:\project\predict\SVM_prediction_result.xlsx"
MODEL_SAVE_PATH = r"D:\project\predict\svm_model.pkl"
SCALER_SAVE_PATH = r"D:\project\predict\standard_scaler.pkl"

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


def generate_features_from_df(df):
    """
    从DataFrame生成与训练代码完全一致的特征向量
    输入：包含SMILES1、SMILES2及所有描述符列的DataFrame
    输出：特征矩阵X，形状为(n_samples, 2064)，与训练时维度完全一致
    """
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
    print("=" * 60)
    print("共晶形成预测脚本（固定最优超参数版）")
    print("=" * 60)

    print(f"\n1. 正在读取预测数据集：{PREDICT_EXCEL_PATH}")
    pred_df = pd.read_excel(PREDICT_EXCEL_PATH, sheet_name='Sheet1')
    print(f"✅ 预测数据集加载完成，共 {len(pred_df)} 条样本")

    print("\n2. 正在生成预测数据集特征...")
    X_pred = generate_features_from_df(pred_df)
    print(f"✅ 特征生成完成，特征形状：{X_pred.shape}（与训练时维度完全一致）")

    print("\n3. 正在准备模型与标准化器...")
    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(SCALER_SAVE_PATH):
        print("✅ 检测到已保存的模型和标准化器，直接加载")
        model = joblib.load(MODEL_SAVE_PATH)
        scaler = joblib.load(SCALER_SAVE_PATH)
    else:
        print("⚠️ 未检测到已保存的模型，开始用原训练集训练（首次运行必须确保训练集路径正确）")
        if not os.path.exists(TRAIN_EXCEL_PATH):
            print(f"❌ 错误：原训练集路径 {TRAIN_EXCEL_PATH} 不存在，请修改TRAIN_EXCEL_PATH为您的实际训练集路径")
            return

        train_df = pd.read_excel(TRAIN_EXCEL_PATH, sheet_name='Sheet1')
        if 'Target' not in train_df.columns:
            print("❌ 错误：训练集必须包含Target列（标签列）")
            return

        print("正在生成训练集特征...")
        X_train = generate_features_from_df(train_df)
        y_train = train_df['Target'].values
        print(f"✅ 训练集特征生成完成，形状：{X_train.shape}，标签形状：{y_train.shape}")

        print("正在拟合标准化器...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        print("正在训练SVM模型（固定最优超参数）...")
        model = SVC(**BEST_PARAMS)
        model.fit(X_train_scaled, y_train)
        print("✅ 模型训练完成")

        joblib.dump(model, MODEL_SAVE_PATH)
        joblib.dump(scaler, SCALER_SAVE_PATH)
        print(f"✅ 模型已保存至：{MODEL_SAVE_PATH}")
        print(f"✅ 标准化器已保存至：{SCALER_SAVE_PATH}")

    print("\n4. 正在标准化预测数据...")
    X_pred_scaled = scaler.transform(X_pred)
    print("✅ 预测数据标准化完成")

    print("\n5. 正在执行预测...")
    y_pred_proba = model.predict_proba(X_pred_scaled)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    print("✅ 预测完成")

    print("\n6. 正在整理预测结果...")
    result_df = pred_df.copy()
    result_df['预测类别'] = y_pred
    result_df['形成共晶的概率'] = y_pred_proba.round(4)
    result_df['预测结果说明'] = result_df['预测类别'].map({0: '预测不会形成共晶', 1: '预测会形成共晶'})

    print(f"\n7. 正在保存预测结果至：{RESULT_SAVE_PATH}")
    result_df.to_excel(RESULT_SAVE_PATH, index=False)
    print("✅ 结果保存完成")

    print("\n" + "=" * 60)
    print("🎉 全流程完成！预测结果统计")
    print("=" * 60)
    print(f"总样本数：{len(result_df)}")
    print(f"预测会形成共晶的样本数：{sum(y_pred)}")
    print(f"预测不会形成共晶的样本数：{len(y_pred) - sum(y_pred)}")
    print(f"共晶预测占比：{sum(y_pred) / len(y_pred):.2%}")
    print("\n📁 预测结果已保存到Excel文件，包含：")
    print("  - 原数据集的所有列")
    print("  - 预测类别（0/1）")
    print("  - 形成共晶的概率（0-1之间）")
    print("  - 预测结果说明（中文）")


if __name__ == "__main__":
    main()