import os
import sys
import numpy as np
import random
import tensorflow as tf
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import pandas as pd
import copy
import gc
import joblib
import json

BASE_SEED = 42
BEST_HYPERPARAMS = {
    'dense_layer_size': 64,
    'dropout_rate': 0.26739594784902165,
    'l2_regularization': 0.005598037107874646,
    'learning_rate': 0.001068620235026263,
    'label_smoothing': 0.01285737266190985
}
PREDICT_EXCEL_PATH = r"D:\project\predict\sjyc-csj.xlsx"
MODEL_SAVE_DIR = r"D:\project\contrast model\cocrystal_prediction_MLP"
RESULT_SAVE_PATH = r"D:\project\predict\MLP_result.xlsx"

BATCH_SIZE = 32
MAX_ATOM_NUM = 50
ECFP_RADIUS = 2
ECFP_N_BITS = 1024
EXTRA_FEAT_SIZE = 16
DESC_FEAT_SIZE = 13

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
RDLogger.DisableLog('rdApp.*')
np.random.seed(BASE_SEED)
random.seed(BASE_SEED)
tf.random.set_seed(BASE_SEED)

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.greater(y_pred, self.threshold), tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        self.true_positives.assign_add(tf.reduce_sum(y_true * y_pred))
        self.false_positives.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.false_negatives.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        return f1

    def reset_state(self):
        self.true_positives.assign(0.)
        self.false_positives.assign(0.)
        self.false_negatives.assign(0.)

    def get_config(self):
        config = super().get_config()
        config.update({'threshold': self.threshold})
        return config

def focal_dice_loss_with_label_smoothing(alpha=0.51, gamma=1.5, smooth=1e-6, label_smoothing=0.1):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, smooth, 1.0 - smooth)
        y_true_smoothed = y_true * (1.0 - label_smoothing) + (1.0 - y_true) * label_smoothing
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        focal_weight = alpha_t * tf.pow(1 - pt, gamma)
        focal_loss = -focal_weight * tf.math.log(pt)
        intersection = tf.reduce_sum(y_true_smoothed * y_pred, axis=-1)
        union = tf.reduce_sum(y_true_smoothed, axis=-1) + tf.reduce_sum(y_pred, axis=-1)
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice
        focal_loss = tf.reduce_mean(focal_loss, axis=-1) if len(focal_loss.shape) > 1 else focal_loss
        total_loss = 0.5 * focal_loss + 0.5 * dice_loss
        return tf.reduce_mean(total_loss)
    loss.__name__ = "focal_dice_loss_with_label_smoothing"
    return loss

def clean_memory(verbose=True):
    tf.keras.backend.clear_session()
    collected_base = gc.collect()
    collected_gen2 = gc.collect(2)
    if verbose:
        print(f"内存回收：{collected_base + collected_gen2} 对象")

def smiles_to_ecfp(smiles, radius=ECFP_RADIUS, n_bits=ECFP_N_BITS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)

def calculate_complementary_features(api_desc, ccf_desc, api_hba_hbd_homo_lumo, ccf_hba_hbd_homo_lumo):
    api_hba, api_hbd, api_homo, api_lumo = api_hba_hbd_homo_lumo
    ccf_hba, ccf_hbd, ccf_homo, ccf_lumo = ccf_hba_hbd_homo_lumo
    api_rbn, api_s, api_s_l, api_s_m, api_m_l, api_fr_no, api_fr_aromaticAtom, api_xlogp3, api_tpsa, api_acd_logp, api_mv, api_polarizability, api_dipole = api_desc
    ccf_rbn, ccf_s, ccf_s_l, ccf_s_m, ccf_m_l, ccf_fr_no, ccf_fr_aromaticAtom, ccf_xlogp3, ccf_tpsa, ccf_acd_logp, ccf_mv, ccf_polarizability, ccf_dipole = ccf_desc
    pair_feat = [
        api_hba * ccf_hbd + api_hbd * ccf_hba,
        api_homo - ccf_lumo,
        ccf_homo - api_lumo,
        api_polarizability * ccf_polarizability,
        abs(api_rbn - ccf_rbn),
        abs(api_s - ccf_s),
        abs(api_s_l - ccf_s_l),
        abs(api_s_m - ccf_s_m),
        abs(api_m_l - ccf_m_l),
        abs(api_fr_no - ccf_fr_no),
        abs(api_fr_aromaticAtom - ccf_fr_aromaticAtom),
        abs(api_xlogp3 - ccf_xlogp3),
        abs(api_tpsa - ccf_tpsa),
        abs(api_acd_logp - ccf_acd_logp),
        abs(api_mv - ccf_mv),
        abs(api_dipole - ccf_dipole)
    ]
    return np.array(pair_feat, dtype=np.float32)

class PredictDataset:
    def __init__(self, excel_path, scaler_dict):
        self.scalers = scaler_dict
        self.raw_data = []
        self.graph_feats = []
        self.extra_feats_raw = None
        self.extra_feats_scaled = None
        self._load_data(excel_path)
        self._scale_all()

    def _load_data(self, excel_path):
        df = pd.read_excel(excel_path, sheet_name="Sheet1")
        print(f"读取预测表格，总行数：{len(df)}")
        api_17_cols = [
            'API_RBN', 'API_S', 'API_S_L', 'API_S_M', 'API_M_L',
            'API_Fr_NO', 'API_Fr_aromaticAtom', 'API_XLogP3', 'API_Topological Polar Surface Area',
            'API_ACD/LogP', 'API_MV', 'API_Polarizability', 'API_Dipole Moment',
            'API_HBA', 'API_HBD', 'API_homo', 'API_lumo'
        ]
        ccf_17_cols = [
            'CCF_RBN', 'CCF_S', 'CCF_S_L', 'CCF_S_M', 'CCF_M_L',
            'CCF_Fr_NO', 'CCF_Fr_aromaticAtom', 'CCF_XLogP3', 'CCF_Topological Polar Surface Area',
            'CCF_ACD/LogP', 'CCF_MV', 'CCF_Polarizability', 'CCF_Dipole Moment',
            'CCF_HBA', 'CCF_HBD', 'CCF_homo', 'CCF_lumo'
        ]
        for idx, row in df.iterrows():
            smi1 = str(row["SMILES1"])
            smi2 = str(row["SMILES2"])
            mol1 = Chem.MolFromSmiles(smi1)
            mol2 = Chem.MolFromSmiles(smi2)
            if mol1 is None or mol2 is None:
                print(f"跳过无效SMILES 行{idx}")
                continue
            n1, n2 = mol1.GetNumAtoms(), mol2.GetNumAtoms()
            if n1 <3 or n2 <3 or n1>MAX_ATOM_NUM or n2>MAX_ATOM_NUM:
                print(f"原子数异常 行{idx}，跳过")
                continue
            api_desc_all = [float(row[c]) if pd.notna(row[c]) else 0.0 for c in api_17_cols]
            ccf_desc_all = [float(row[c]) if pd.notna(row[c]) else 0.0 for c in ccf_17_cols]
            api_desc = api_desc_all[:13]
            api_hhhl = api_desc_all[13:]
            ccf_desc = ccf_desc_all[:13]
            ccf_hhhl = ccf_desc_all[13:]
            api_fp = smiles_to_ecfp(smi1)
            ccf_fp = smiles_to_ecfp(smi2)
            pair_feat = calculate_complementary_features(api_desc, ccf_desc, api_hhhl, ccf_hhhl)
            feat_dict = {
                "api_global_desc": np.array(api_desc, dtype=np.float32),
                "ccf_global_desc": np.array(ccf_desc, dtype=np.float32),
                "api_ecfp": api_fp,
                "ccf_ecfp": ccf_fp,
                "pair_raw": pair_feat,
                "origin_row": row.to_dict()
            }
            self.raw_data.append(feat_dict)
        print(f"有效预测样本数量：{len(self.raw_data)}")

    def _scale_all(self):
        extra_raw = []
        for item in self.raw_data:
            item["api_scaled"] = self.scalers["api"].transform(item["api_global_desc"].reshape(1,-1)).flatten()
            item["ccf_scaled"] = self.scalers["ccf"].transform(item["ccf_global_desc"].reshape(1,-1)).flatten()
            item["api_fp_scaled"] = self.scalers["api_ecfp"].transform(item["api_ecfp"].reshape(1,-1)).flatten()
            item["ccf_fp_scaled"] = self.scalers["ccf_ecfp"].transform(item["ccf_ecfp"].reshape(1,-1)).flatten()
            extra_raw.append(item["pair_raw"])
        self.extra_feats_raw = np.array(extra_raw)
        self.extra_feats_scaled = self.scalers["pair"].transform(self.extra_feats_raw)

    def get_all_inputs(self):
        api_in = []
        ccf_in = []
        api_fp_in = []
        ccf_fp_in = []
        extra_in = []
        for i, item in enumerate(self.raw_data):
            api_in.append(item["api_scaled"])
            ccf_in.append(item["ccf_scaled"])
            api_fp_in.append(item["api_fp_scaled"])
            ccf_fp_in.append(item["ccf_fp_scaled"])
            extra_in.append(self.extra_feats_scaled[i])
        return [
            np.array(api_in),
            np.array(ccf_in),
            np.array(api_fp_in),
            np.array(ccf_fp_in),
            np.array(extra_in)
        ]

class EnsembleModel:
    def __init__(self, models=None, scalers=None, weights=None):
        self.models = models or []
        self.scalers = scalers or []
        self.weights = weights

    def predict(self, input_x):
        pred_list = []
        for m in self.models:
            p = m.predict(input_x, verbose=0)
            pred_list.append(p)
        pred_stack = np.concatenate(pred_list, axis=-1)
        weighted_pred = np.zeros_like(pred_stack[:,0])
        for idx, w in enumerate(self.weights):
            weighted_pred += pred_stack[:, idx] * w
        return weighted_pred

    @classmethod
    def load_ensemble(cls, load_dir):
        models = []
        i = 0
        while os.path.exists(os.path.join(load_dir, f"model_{i}.keras")):
            m_path = os.path.join(load_dir, f"model_{i}.keras")
            model = tf.keras.models.load_model(
                m_path,
                custom_objects={
                    "F1Score": F1Score,
                    "focal_dice_loss_with_label_smoothing": focal_dice_loss_with_label_smoothing()
                }
            )
            models.append(model)
            i += 1
        scaler_path = os.path.join(load_dir, "scalers.pkl")
        all_scalers = joblib.load(scaler_path)
        cfg_path = os.path.join(load_dir, "config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        weight_arr = np.array(cfg["weights"]) if cfg["weights"] is not None else np.ones(len(models))/len(models)
        return cls(models=models, scalers=all_scalers, weights=weight_arr)

def main():
    print("===== 加载已训练完成的集成模型 =====")
    ens = EnsembleModel.load_ensemble(MODEL_SAVE_DIR)
    use_scaler = ens.scalers[0]
    print("模型加载完成，开始读取预测数据集")

    pred_ds = PredictDataset(PREDICT_EXCEL_PATH, use_scaler)
    inputs = pred_ds.get_all_inputs()

    print("开始预测......")
    pred_proba = ens.predict(inputs)
    pred_label = (pred_proba > 0.5).astype(int)

    output_rows = []
    for i, item in enumerate(pred_ds.raw_data):
        row_dict = item["origin_row"]
        row_dict["预测共晶概率"] = round(float(pred_proba[i]), 4)
        row_dict["预测标签(1=共晶)"] = int(pred_label[i])
        row_dict["预测结论"] = "共晶" if pred_label[i]==1 else "非共晶"
        output_rows.append(row_dict)
    out_df = pd.DataFrame(output_rows)
    out_df.to_excel(RESULT_SAVE_PATH, index=False)
    print(f"\n预测结果已保存至：{RESULT_SAVE_PATH}")

    total = len(pred_proba)
    pos = np.sum(pred_label)
    neg = total - pos
    print(f"\n===== 预测汇总 =====")
    print(f"总样本：{total}")
    print(f"预测共晶：{pos} 个，占比 {pos/total:.2%}")
    print(f"预测非共晶：{neg} 个")
    clean_memory()

if __name__ == "__main__":
    main()