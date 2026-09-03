import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import Chem, RDLogger
import gc
import warnings
import copy
import json
import joblib

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

BASE_SEED = 42
np.random.seed(BASE_SEED)
tf.random.set_seed(BASE_SEED)

MAX_ATOM_NUM = 50
EXTRA_FEAT_SIZE = 16
DESC_FEAT_SIZE = 13

MODEL_SAVE_DIR = r"D:\project\contrast model\cocrystal_prediction_INN-GCNtp1"
INPUT_EXCEL_PATH = r"D:\project\predict\sjyc-csj.xlsx"
OUTPUT_EXCEL_PATH = r"D:\project\predict\IG-csj-sjyc.xlsx"
HYPERPARAMS_LOAD_PATH = os.path.join(MODEL_SAVE_DIR, "best_hyperparams.json")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
else:
    tf.config.set_soft_device_placement(True)
    tf.keras.mixed_precision.set_global_policy('float32')

def clean_memory(verbose=True):
    tf.keras.backend.clear_session()
    if tf.__version__ >= '2.0':
        tf.compat.v1.reset_default_graph()

    try:
        tf.config.experimental.reset_memory_stats('CPU:0')
        if tf.config.list_physical_devices('GPU'):
            tf.config.experimental.reset_memory_stats('GPU:0')
    except Exception as e:
        if verbose:
            print(f"⚠️ 设备缓存清理失败：{e}")

    collected_base = gc.collect()
    collected_gen2 = gc.collect(2)
    total_collected = collected_base + collected_gen2

    if verbose:
        print(f"✅ 内存清理完成：回收 {total_collected} 个对象")
    return total_collected


def atom_to_features(atom):
    atom_types = ['Cl', 'N', 'P', 'Br', 'Si', 'S', 'I', 'F', 'C', 'O', 'H']
    atom_symbol = atom.GetSymbol()
    atom_type_feat = [1.0 if t == atom_symbol else 0.0 for t in atom_types]

    hybrid_types = [Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.S,
                    Chem.rdchem.HybridizationType.SP]
    hybrid = atom.GetHybridization()
    hybrid_feat = [1.0 if h == hybrid else 0.0 for h in hybrid_types]

    chiral_tag = atom.GetChiralTag()
    chirality_feat = [
        1.0 if chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW else 0.0,
        1.0 if chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW else 0.0,
    ]

    binary_feat = [
        1.0 if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED else 0.0,
        1.0 if atom.IsInRingSize(3) or atom.IsInRingSize(4) else 0.0,
        1.0 if atom.IsInRing() else 0.0,
        1.0 if atom.GetIsAromatic() else 0.0,
        1.0 if atom.GetAtomicNum() in [7, 8, 15, 16] else 0.0,
        1.0 if atom.GetAtomicNum() in [7, 8] else 0.0,
        1.0 if atom.GetFormalCharge() != 0 else 0.0,
    ]

    max_valence = 6
    max_degree = 6
    max_h = 4
    max_vdw = 2.0
    max_atomic_num = 53

    from rdkit.Chem import rdchem
    pt = rdchem.GetPeriodicTable()

    atomic_num = atom.GetAtomicNum()
    element_symbol = pt.GetElementSymbol(atomic_num)
    vdw_rad = pt.GetRvdw(atomic_num) if element_symbol else 1.0

    numeric_feat = [
        atom.GetExplicitValence() / max_valence,
        atom.GetImplicitValence() / max_valence,
        atom.GetFormalCharge() / 1.0,
        atom.GetDegree() / max_degree,
        atom.GetTotalNumHs() / max_h,
        vdw_rad / max_vdw,
        atom.GetAtomicNum() / max_atomic_num
    ]

    return np.array(atom_type_feat + hybrid_feat + chirality_feat + binary_feat + numeric_feat, dtype=np.float32)


def standardize_atom_feats(atom_feats):
    """原子特征标准化（与训练一致）"""
    if atom_feats.shape[0] == 0:
        return atom_feats
    mean = np.mean(atom_feats, axis=0, keepdims=True)
    std = np.std(atom_feats, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    atom_feats_std = (atom_feats - mean) / std
    return atom_feats_std


def mol_to_adj_list(mol):
    """分子转邻接列表（与训练一致）"""
    adj_list = []
    for atom in mol.GetAtoms():
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        adj_list.append(neighbors)
    return adj_list


def adj_list_to_matrix(adj_list):
    """邻接列表转邻接矩阵（与训练一致）"""
    n = len(adj_list)
    adj_mat = np.zeros((n, n), dtype=np.float32)
    for i, neighbors in enumerate(adj_list):
        for j in neighbors:
            adj_mat[i, j] = 1.0
    return adj_mat


def adj_list_to_matrix_with_edge_feats(mol):
    """提取邻接矩阵+边特征（与训练一致）"""
    n = mol.GetNumAtoms()
    adj_mat = np.zeros((n, n), dtype=np.float32)
    edge_feats = np.zeros((n, n, 4), dtype=np.float32)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj_mat[i, j] = 1.0
        adj_mat[j, i] = 1.0

        bond_type = bond.GetBondType()
        bond_type_feat = [
            1.0 if bond_type == Chem.rdchem.BondType.SINGLE else 0.0,
            1.0 if bond_type == Chem.rdchem.BondType.DOUBLE else 0.0,
            1.0 if bond_type == Chem.rdchem.BondType.TRIPLE else 0.0,
            1.0 if bond_type == Chem.rdchem.BondType.AROMATIC else 0.0
        ]
        edge_feats[i, j] = bond_type_feat
        edge_feats[j, i] = bond_type_feat

    return adj_mat, edge_feats


def calculate_complementary_features(api_desc, ccf_desc, api_hba_hbd_homo_lumo, ccf_hba_hbd_homo_lumo):
    """计算16维配对交互特征（与训练一致）"""
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


def pad_sequences(sequences, padding='post'):
    """原子特征填充（与训练一致）"""
    max_len = MAX_ATOM_NUM
    feat_dim = sequences[0].shape[1] if sequences else 44
    padded = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        seq_trunc = seq[:max_len] if len(seq) > max_len else seq
        if pad_len > 0:
            pad = np.zeros((pad_len, feat_dim), dtype=np.float32)
            if padding == 'post':
                padded_seq = np.concatenate([seq_trunc, pad], axis=0)
            else:
                padded_seq = np.concatenate([pad, seq_trunc], axis=0)
        else:
            padded_seq = seq_trunc
        padded.append(padded_seq)
    return np.array(padded)


def pad_adj_matrices(adj_mats):
    """邻接矩阵填充（与训练一致）"""
    max_size = MAX_ATOM_NUM
    padded = []
    for mat in adj_mats:
        pad_mat = np.zeros((max_size, max_size), dtype=np.float32)
        mat_trunc = mat[:max_size, :max_size]
        pad_mat[:mat_trunc.shape[0], :mat_trunc.shape[1]] = mat_trunc
        padded.append(pad_mat)
    return np.array(padded)


def pad_edge_feats(edge_feats_list):
    """边特征填充（与训练一致）"""
    max_size = MAX_ATOM_NUM
    n_edge_feats = 4 if not edge_feats_list else edge_feats_list[0].shape[-1]
    padded = []
    for ef in edge_feats_list:
        pad_ef = np.zeros((max_size, max_size, n_edge_feats), dtype=np.float32)
        ef_trunc = ef[:max_size, :max_size]
        pad_ef[:ef_trunc.shape[0], :ef_trunc.shape[1]] = ef_trunc
        padded.append(pad_ef)
    return np.array(padded)


def preload_all_data(graph_feats, extra_feats):
    """预加载数据（与训练一致，无标签）"""
    preloaded = []
    for i in range(len(graph_feats)):
        gf = graph_feats[i]
        api_adj = adj_list_to_matrix(gf['api_adj_list'])
        ccf_adj = adj_list_to_matrix(gf['ccf_adj_list'])

        if 'api_edge_feats' in gf and 'ccf_edge_feats' in gf:
            api_edge_feats = gf['api_edge_feats']
            ccf_edge_feats = gf['ccf_edge_feats']
        elif 'api_smiles' in gf and 'ccf_smiles' in gf:
            api_mol = Chem.MolFromSmiles(gf['api_smiles'])
            ccf_mol = Chem.MolFromSmiles(gf['ccf_smiles'])
            _, api_edge_feats = adj_list_to_matrix_with_edge_feats(api_mol)
            _, ccf_edge_feats = adj_list_to_matrix_with_edge_feats(ccf_mol)
        else:
            api_size = api_adj.shape[0]
            ccf_size = ccf_adj.shape[0]
            api_edge_feats = np.zeros((api_size, api_size, 4), dtype=np.float32)
            ccf_edge_feats = np.zeros((ccf_size, ccf_size, 4), dtype=np.float32)

        api_global_desc_scaled = gf.get('api_global_desc_scaled', gf['api_global_desc'])
        ccf_global_desc_scaled = gf.get('ccf_global_desc_scaled', gf['ccf_global_desc'])

        api_atom_feats = gf['api_atom_feats']
        api_atom_num = api_atom_feats.shape[0]
        api_global_broadcast = np.tile(api_global_desc_scaled, (api_atom_num, 1))
        api_atom_feats_new = np.concatenate([api_atom_feats, api_global_broadcast], axis=1)

        ccf_atom_feats = gf['ccf_atom_feats']
        ccf_atom_num = ccf_atom_feats.shape[0]
        ccf_global_broadcast = np.tile(ccf_global_desc_scaled, (ccf_atom_num, 1))
        ccf_atom_feats_new = np.concatenate([ccf_atom_feats, ccf_global_broadcast], axis=1)

        preloaded.append({
            'api_atom': api_atom_feats_new,
            'api_adj': api_adj,
            'api_edge_feats': api_edge_feats,
            'api_global_desc': api_global_desc_scaled,
            'ccf_atom': ccf_atom_feats_new,
            'ccf_adj': ccf_adj,
            'ccf_edge_feats': ccf_edge_feats,
            'ccf_global_desc': ccf_global_desc_scaled,
            'extra': extra_feats[i]
        })
    return preloaded


def prediction_data_generator(preloaded_data, batch_size):
    """修复版：预测数据生成器（无标签，无死循环）"""
    data_len = len(preloaded_data)
    indices = np.arange(data_len)

    for start in range(0, data_len, batch_size):
        end = min(start + batch_size, data_len)
        batch_idx = indices[start:end]
        batch_data = [preloaded_data[i] for i in batch_idx]

        api_atom = [d['api_atom'] for d in batch_data]
        api_adj = [d['api_adj'] for d in batch_data]
        api_edge_feats = [d['api_edge_feats'] for d in batch_data]
        ccf_atom = [d['ccf_atom'] for d in batch_data]
        ccf_adj = [d['ccf_adj'] for d in batch_data]
        ccf_edge_feats = [d['ccf_edge_feats'] for d in batch_data]
        extra = [d['extra'] for d in batch_data]

        api_atom_pad = pad_sequences(api_atom, padding='post')
        api_adj_pad = pad_adj_matrices(api_adj)
        api_edge_pad = pad_edge_feats(api_edge_feats)
        ccf_atom_pad = pad_sequences(ccf_atom, padding='post')
        ccf_adj_pad = pad_adj_matrices(ccf_adj)
        ccf_edge_pad = pad_edge_feats(ccf_edge_feats)

        yield [
            api_atom_pad, api_adj_pad, api_edge_pad,
            ccf_atom_pad, ccf_adj_pad, ccf_edge_pad,
            np.array(extra)
        ]


class F1Score(tf.keras.metrics.Metric):
    """自定义F1指标（与训练一致）"""

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

class EnsembleModel:
    """集成模型加载类（简化版，仅用于预测）"""

    def __init__(self, models=None, scalers=None, weights=None):
        self.models = models if models is not None else []
        self.scalers = scalers if scalers is not None else []
        self.weights = weights

    def predict(self, x, batch_size=32, verbose=0):
        """加权集成预测"""
        if len(self.models) == 0:
            raise ValueError("集成模型为空！")

        if self.weights is None:
            self.weights = np.ones(len(self.models)) / len(self.models)

        all_preds = []
        for model in self.models:
            pred = model.predict(x, batch_size=batch_size, verbose=verbose)
            all_preds.append(pred)

        all_preds = np.array(all_preds)
        weights_expanded = np.expand_dims(np.expand_dims(self.weights, axis=-1), axis=-1)
        weighted_preds = np.sum(all_preds * weights_expanded, axis=0)

        return weighted_preds

    @classmethod
    def load_ensemble(cls, save_dir):
        """加载训练好的集成模型"""
        config_path = os.path.join(save_dir, "ensemble_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在：{config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        sub_models_dir = config["sub_models_dir"]
        models = []
        custom_objects = {
            "INN_GraphConvLayer": INN_GraphConvLayer,
            "INN_HybridGraphLayer": INN_HybridGraphLayer,
            "INN_GraphAttentionPoolLayer": INN_GraphAttentionPoolLayer,
            "INN_MolCrossAttentionLayer": INN_MolCrossAttentionLayer,
            "INN_MolPairMemoryLayer": INN_MolPairMemoryLayer,
            "F1Score": F1Score,
            "focal_dice_loss_with_label_smoothing": focal_dice_loss_with_label_smoothing()
        }

        for idx in range(config["n_models"]):
            model_path = os.path.join(sub_models_dir, f"fold_{idx + 1}_model.keras")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在：{model_path}")

            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            models.append(model)

        scalers_dir = config["scalers_dir"]
        scalers = []
        for idx in range(config["n_models"]):
            api_scaler = joblib.load(os.path.join(scalers_dir, f"fold_{idx + 1}_api_desc_scaler.pkl"))
            ccf_scaler = joblib.load(os.path.join(scalers_dir, f"fold_{idx + 1}_ccf_desc_scaler.pkl"))
            pair_scaler = joblib.load(os.path.join(scalers_dir, f"fold_{idx + 1}_pair_feat_scaler.pkl"))

            scalers.append({
                'api_desc_scaler': api_scaler,
                'ccf_desc_scaler': ccf_scaler,
                'pair_feat_scaler': pair_scaler
            })

        weights = np.array(config["weights"]) if config["weights"] is not None else None

        return cls(models=models, scalers=scalers, weights=weights)

def preprocess_unknown_data(excel_path):
    """预处理未知数据集（完全复用训练流程）"""
    print(f"\n加载未知数据集：{excel_path}")
    df = pd.read_excel(excel_path, sheet_name='Sheet1')
    processed_data = []

    api_17_desc_cols = [
        'API_RBN', 'API_S', 'API_S_L', 'API_S_M', 'API_M_L',
        'API_Fr_NO', 'API_Fr_aromaticAtom', 'API_XLogP3', 'API_Topological Polar Surface Area',
        'API_ACD/LogP', 'API_MV', 'API_Polarizability', 'API_Dipole Moment',
        'API_HBA', 'API_HBD', 'API_homo', 'API_lumo'
    ]

    ccf_17_desc_cols = [
        'CCF_RBN', 'CCF_S', 'CCF_S_L', 'CCF_S_M', 'CCF_M_L',
        'CCF_Fr_NO', 'CCF_Fr_aromaticAtom', 'CCF_XLogP3', 'CCF_Topological Polar Surface Area',
        'CCF_ACD/LogP', 'CCF_MV', 'CCF_Polarizability', 'CCF_Dipole Moment',
        'CCF_HBA', 'CCF_HBD', 'CCF_homo', 'CCF_lumo'
    ]

    for index, row in df.iterrows():
        api_smiles = row.get('SMILES1', '')
        ccf_smiles = row.get('SMILES2', '')

        if pd.isna(api_smiles) or pd.isna(ccf_smiles):
            print(f"警告：行{index} SMILES为空，跳过")
            continue

        api_mol = Chem.MolFromSmiles(api_smiles)
        ccf_mol = Chem.MolFromSmiles(ccf_smiles)
        if api_mol is None or ccf_mol is None:
            print(f"警告：行{index} SMILES解析失败，跳过")
            continue

        api_atom_num = api_mol.GetNumAtoms()
        ccf_atom_num = ccf_mol.GetNumAtoms()
        if (api_atom_num < 3 or ccf_atom_num < 3 or
                api_atom_num > MAX_ATOM_NUM or ccf_atom_num > MAX_ATOM_NUM):
            print(f"警告：行{index} 原子数异常，跳过")
            continue

        api_atom_feats = np.array([atom_to_features(atom) for atom in api_mol.GetAtoms()], dtype=np.float32)
        api_atom_feats = standardize_atom_feats(api_atom_feats)

        ccf_atom_feats = np.array([atom_to_features(atom) for atom in ccf_mol.GetAtoms()], dtype=np.float32)
        ccf_atom_feats = standardize_atom_feats(ccf_atom_feats)

        api_adj_list = mol_to_adj_list(api_mol)
        ccf_adj_list = mol_to_adj_list(ccf_mol)
        api_adj_mat, api_edge_feats = adj_list_to_matrix_with_edge_feats(api_mol)
        ccf_adj_mat, ccf_edge_feats = adj_list_to_matrix_with_edge_feats(ccf_mol)

        api_17_desc = []
        for col in api_17_desc_cols:
            val = row.get(col, 0.0)
            api_17_desc.append(float(val) if pd.notna(val) else 0.0)

        ccf_17_desc = []
        for col in ccf_17_desc_cols:
            val = row.get(col, 0.0)
            ccf_17_desc.append(float(val) if pd.notna(val) else 0.0)

        api_desc = api_17_desc[:13]
        api_hba_hbd_homo_lumo = api_17_desc[13:]
        ccf_desc = ccf_17_desc[:13]
        ccf_hba_hbd_homo_lumo = ccf_17_desc[13:]

        pair_desc = calculate_complementary_features(
            api_desc,
            ccf_desc,
            api_hba_hbd_homo_lumo,
            ccf_hba_hbd_homo_lumo
        )
        extra_feat = pair_desc

        graph_feat = {
            'api_atom_feats': api_atom_feats,
            'api_adj_list': api_adj_list,
            'api_edge_feats': api_edge_feats,
            'api_deg_list': [len(neighbors) for neighbors in api_adj_list],
            'ccf_atom_feats': ccf_atom_feats,
            'ccf_adj_list': ccf_adj_list,
            'ccf_edge_feats': ccf_edge_feats,
            'ccf_deg_list': [len(neighbors) for neighbors in ccf_adj_list],
            'api_smiles': api_smiles,
            'ccf_smiles': ccf_smiles,
            'api_17_desc': np.array(api_17_desc, dtype=np.float32),
            'ccf_17_desc': np.array(ccf_17_desc, dtype=np.float32),
            'api_global_desc': np.array(api_desc, dtype=np.float32),
            'ccf_global_desc': np.array(ccf_desc, dtype=np.float32),
            'api_hba_hbd_homo_lumo': np.array(api_hba_hbd_homo_lumo, dtype=np.float32),
            'ccf_hba_hbd_homo_lumo': np.array(ccf_hba_hbd_homo_lumo, dtype=np.float32)
        }

        processed_data.append({
            'index': index,
            'graph_feat': graph_feat,
            'extra_feat': extra_feat,
            'api_smiles': api_smiles,
            'ccf_smiles': ccf_smiles
        })

        del api_mol, ccf_mol

    print(f"预处理完成，有效样本数：{len(processed_data)}")
    return processed_data, df

def load_best_hyperparams(load_path):
    """加载训练保存的最优超参数"""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"超参数文件不存在：{load_path}，请先运行训练代码保存超参！")
    with open(load_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    print(f"✅ 成功加载训练超参数：{params}")
    return params

TRAINED_HYPERPARAMS = load_best_hyperparams(HYPERPARAMS_LOAD_PATH)
GCN1_SIZE = TRAINED_HYPERPARAMS['gcn_layer1_size']
GCN2_SIZE = TRAINED_HYPERPARAMS['gcn_layer2_size']
DENSE_SIZE = TRAINED_HYPERPARAMS['dense_layer_size']
DROPOUT_RATE = TRAINED_HYPERPARAMS['dropout_rate']
L2_REG = TRAINED_HYPERPARAMS['l2_regularization']
LEARNING_RATE = TRAINED_HYPERPARAMS['learning_rate']
NUM_HEADS = TRAINED_HYPERPARAMS['num_heads']
LABEL_SMOOTHING = TRAINED_HYPERPARAMS['label_smoothing']

class INN_GraphConvLayer(tf.keras.layers.Layer):
    """INN图卷积层（使用优化超参）"""
    def __init__(self, units, activation='relu', dropout_rate=DROPOUT_RATE, gate_type='adaptive', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.dropout_rate = dropout_rate
        self.gate_type = gate_type

        self.dense = tf.keras.layers.Dense(units, kernel_regularizer=tf.keras.regularizers.l2(L2_REG))
        self.edge_dense = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(L2_REG))
        self.residual_dense = tf.keras.layers.Dense(units, kernel_regularizer=tf.keras.regularizers.l2(L2_REG))

        if self.gate_type == 'adaptive':
            self.gate = tf.keras.Sequential([
                tf.keras.layers.Dense(units * 2, activation='relu'),
                tf.keras.layers.LayerNormalization(epsilon=1e-6),
                tf.keras.layers.Dense(units, activation='sigmoid')
            ])
        elif self.gate_type == 'hybrid':
            self.static_gate = self.add_weight(
                shape=(1, 1, units), initializer=tf.keras.initializers.Constant(0.5), trainable=True, name='static_gate'
            )
            self.dynamic_gate = tf.keras.layers.Dense(units, activation='sigmoid')
        elif self.gate_type == 'hard':
            self.gate = tf.keras.layers.Dense(units)
            self.gate_threshold = self.add_weight(
                shape=(1, 1, units), initializer=tf.keras.initializers.Constant(0.0), trainable=True, name='gate_threshold'
            )

        self.dropout = tf.keras.layers.SpatialDropout1D(rate=dropout_rate)

    def build(self, input_shape):
        atom_feat_shape = input_shape[0]
        edge_feat_shape = input_shape[2]
        if edge_feat_shape[-1] is None:
            edge_feat_shape = (None, None, 4)
        self.edge_dense.build(edge_feat_shape)
        self.dense.build(atom_feat_shape)
        self.residual_dense.build(atom_feat_shape)

        if self.gate_type == 'adaptive':
            self.gate.build(atom_feat_shape)
        elif self.gate_type == 'hybrid':
            self.dynamic_gate.build(atom_feat_shape)
        elif self.gate_type == 'hard':
            self.gate.build(atom_feat_shape)
        super().build(input_shape)

    def call(self, inputs, training=False):
        atom_features, adj_matrix, edge_features = inputs
        edge_weights = tf.squeeze(self.edge_dense(edge_features), axis=-1)
        weighted_adj = adj_matrix * edge_weights
        degree = tf.reduce_sum(weighted_adj, axis=-1, keepdims=True)
        degree_inv_sqrt = tf.math.rsqrt(tf.maximum(degree, 1e-8))
        adj_norm = weighted_adj * degree_inv_sqrt * tf.transpose(degree_inv_sqrt, [0, 2, 1])
        agg_features = tf.matmul(adj_norm, atom_features)
        gcn_out = self.dense(agg_features)
        gcn_out = self.dropout(gcn_out, training=training)

        if self.gate_type == 'adaptive':
            gate_value = self.gate(atom_features)
        elif self.gate_type == 'hybrid':
            dynamic_gate = self.dynamic_gate(atom_features)
            gate_value = tf.nn.sigmoid(self.static_gate + dynamic_gate)
        elif self.gate_type == 'hard':
            gate_logits = self.gate(atom_features)
            gate_value = tf.cast(gate_logits > self.gate_threshold, tf.float32)

        gated_gcn_out = gcn_out * gate_value
        residual = self.residual_dense(atom_features)
        out = gated_gcn_out + residual
        if self.activation is not None:
            out = self.activation(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units, 'activation': tf.keras.activations.serialize(self.activation),
            'dropout_rate': self.dropout_rate, 'gate_type': self.gate_type
        })
        return config

class INN_HybridGraphLayer(tf.keras.layers.Layer):
    """INN混合图层（使用优化超参）"""
    def __init__(self, units, num_heads=NUM_HEADS, activation='relu', dropout_rate=DROPOUT_RATE,
                 gate_type='adaptive', global_gate=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.activation = tf.keras.activations.get(activation)
        self.dropout_rate = dropout_rate
        self.gate_type = gate_type
        self.global_gate = global_gate

        self.local_gcn = INN_GraphConvLayer(units, activation=activation, dropout_rate=dropout_rate, gate_type=gate_type)
        self.attn_dim_adapt = tf.keras.layers.Dense(units)
        self.global_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=units//num_heads, dropout=0.1)
        self.attn_dense = tf.keras.layers.Dense(units)
        self.attn_gate = tf.keras.layers.Dense(units, activation='sigmoid')
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.fuse_weight = None

    def build(self, input_shape):
        atom_feat_shape = input_shape[0]
        edge_feat_shape = input_shape[2]
        if edge_feat_shape[-1] is None:
            edge_feat_shape = (None, None, 4)
        self.local_gcn.build(input_shape[:3])
        self.attn_dim_adapt.build(atom_feat_shape)
        self.global_attn.build([(None, None, self.units)] * 3)
        self.attn_dense.build((None, None, self.units))
        self.attn_gate.build(atom_feat_shape)
        self.layernorm.build((None, None, self.units))
        self.fuse_weight = self.add_weight(
            shape=(1,1,self.units), initializer=tf.keras.initializers.Constant(0.5), trainable=True, name='fuse_weight'
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        atom_features, adj_matrix, edge_features = inputs
        local_out = self.local_gcn([atom_features, adj_matrix, edge_features], training=training)
        atom_feat_adapt = self.attn_dim_adapt(atom_features)
        global_out = self.global_attn(query=atom_feat_adapt, key=atom_feat_adapt, value=atom_feat_adapt, training=training)
        global_out = self.attn_dense(global_out)
        gate_value = self.attn_gate(atom_features)
        global_out = global_out * gate_value
        fuse_weight_sigmoid = tf.nn.sigmoid(self.fuse_weight)
        out = fuse_weight_sigmoid * local_out + (1 - fuse_weight_sigmoid) * global_out
        out = self.layernorm(out)
        if self.activation is not None:
            out = self.activation(out)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.units)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units, 'num_heads': self.num_heads,
            'activation': tf.keras.activations.serialize(self.activation),
            'dropout_rate': self.dropout_rate, 'gate_type': self.gate_type, 'global_gate': self.global_gate
        })
        return config

class INN_GraphAttentionPoolLayer(tf.keras.layers.Layer):
    def __init__(self, units=GCN2_SIZE, num_heads=NUM_HEADS, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=units//num_heads, dropout=0.1)
        self.dense = tf.keras.layers.Dense(units)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        atom_feats = inputs
        attn_out = self.attention(query=atom_feats, key=atom_feats, value=atom_feats, training=training)
        out = self.layernorm(atom_feats + attn_out)
        attn_weights = tf.nn.softmax(self.dense(out), axis=1)
        pooled = tf.reduce_sum(out * attn_weights, axis=1)
        return pooled

class INN_MolCrossAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units=DENSE_SIZE, num_heads=NUM_HEADS, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.cross_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=units, dropout=0.1)
        self.dense1 = tf.keras.layers.Dense(units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, api_feat, ccf_feat, training=False):
        api_feat_exp = tf.expand_dims(api_feat, axis=1)
        ccf_feat_exp = tf.expand_dims(ccf_feat, axis=1)
        api2ccf = self.cross_attention(query=ccf_feat_exp, key=api_feat_exp, value=api_feat_exp, training=training)
        ccf2api = self.cross_attention(query=api_feat_exp, key=ccf_feat_exp, value=ccf_feat_exp, training=training)
        api_fused = self.layernorm(api_feat_exp + ccf2api)
        ccf_fused = self.layernorm(ccf_feat_exp + api2ccf)
        fused = tf.concat([tf.squeeze(api_fused, 1), tf.squeeze(ccf_fused, 1)], axis=1)
        fused = self.dense2(self.dense1(fused))
        return fused

class INN_MolPairMemoryLayer(tf.keras.layers.Layer):
    """INN双分子协同记忆层（与训练一致）"""
    def __init__(self, units=256, memory_scale=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.memory_scale = memory_scale

    def build(self, input_shape):
        input_units = input_shape[-1]
        self.units = input_units
        self.pair_interact_dense = tf.keras.layers.Dense(self.units, activation='relu', kernel_initializer='he_normal')
        self.enhance_gate = tf.keras.layers.Dense(self.units, activation='sigmoid', kernel_initializer='glorot_uniform')
        self.initial_memory = tf.zeros((1, self.units))
        super().build(input_shape)

    def call(self, api_feat, ccf_feat, training=False):
        input_units = api_feat.shape[-1]
        if input_units != self.units:
            api_feat = tf.keras.layers.Dense(self.units)(api_feat)
            ccf_feat = tf.keras.layers.Dense(self.units)(ccf_feat)
        pair_interact = tf.multiply(api_feat, ccf_feat)
        pair_interact = self.pair_interact_dense(pair_interact)
        pair_interact_mean = tf.reduce_mean(pair_interact, axis=0, keepdims=True)
        if training:
            current_memory = (1 - self.memory_scale) * self.initial_memory + self.memory_scale * pair_interact_mean
        else:
            current_memory = self.initial_memory
        api_gate = self.enhance_gate(api_feat)
        ccf_gate = self.enhance_gate(ccf_feat)
        api_feat_enhanced = api_feat + (api_gate * current_memory)
        ccf_feat_enhanced = ccf_feat + (ccf_gate * current_memory)
        return api_feat_enhanced, ccf_feat_enhanced

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units), (input_shape[1], self.units)

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'memory_scale': self.memory_scale})
        return config

def focal_dice_loss_with_label_smoothing(alpha=0.5, gamma=1.5, smooth=1e-6, label_smoothing=LABEL_SMOOTHING):
    """损失函数（使用超参）"""
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

def main_prediction():
    """主预测流程"""
    clean_memory()
    print("\n=== 加载训练好的集成模型 ===")
    try:
        ensemble_model = EnsembleModel.load_ensemble(MODEL_SAVE_DIR)
        print(f"✅ 模型加载成功，包含 {len(ensemble_model.models)} 个子模型")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return

    print("\n=== 预处理未知数据集 ===")
    processed_data, original_df = preprocess_unknown_data(INPUT_EXCEL_PATH)
    if len(processed_data) == 0:
        print("❌ 无有效数据，预测终止")
        return

    raw_graph_feats = [sample['graph_feat'] for sample in processed_data]
    raw_extra_feats = np.array([sample['extra_feat'] for sample in processed_data], dtype=np.float32)
    sample_count = len(processed_data)
    batch_size = 32

    all_model_preds = []
    print("\n=== 开始逐模型独立预测（使用对应Scaler）===")

    for model_idx, (model, scaler_dict) in enumerate(zip(ensemble_model.models, ensemble_model.scalers)):
        print(f"\n📌 正在预测 第{model_idx + 1}折 模型（使用专属Scaler）")
        scaler_api = scaler_dict['api_desc_scaler']
        scaler_ccf = scaler_dict['ccf_desc_scaler']
        scaler_pair = scaler_dict['pair_feat_scaler']

        graph_feats = copy.deepcopy(raw_graph_feats)
        for gf in graph_feats:
            gf['api_global_desc_scaled'] = scaler_api.transform(gf['api_global_desc'].reshape(1, -1)).flatten()
            gf['ccf_global_desc_scaled'] = scaler_ccf.transform(gf['ccf_global_desc'].reshape(1, -1)).flatten()
        extra_feats_scaled = scaler_pair.transform(raw_extra_feats)

        preloaded_data = preload_all_data(graph_feats, extra_feats_scaled)
        pred_generator = prediction_data_generator(preloaded_data, batch_size)
        steps = len(preloaded_data) // batch_size + 1

        model_preds = []
        for _ in range(steps):
            try:
                X = next(pred_generator)
                preds = model.predict(X, batch_size=batch_size, verbose=0)
                model_preds.extend(preds.flatten())
            except StopIteration:
                break
        model_preds = np.array(model_preds[:sample_count])
        all_model_preds.append(model_preds)
        del graph_feats, preloaded_data, pred_generator
        clean_memory(verbose=False)

    all_model_preds = np.array(all_model_preds)
    weights = ensemble_model.weights
    if weights is None:
        weights = np.ones(len(ensemble_model.models)) / len(ensemble_model.models)
    final_preds = np.average(all_model_preds, axis=0, weights=weights)

    print("\n=== 生成预测结果 ===")
    original_df = original_df.iloc[[sample['index'] for sample in processed_data]].reset_index(drop=True)
    original_df['Prediction_Probability'] = final_preds
    original_df['Prediction_Label'] = (final_preds > 0.5).astype(int)
    original_df.to_excel(OUTPUT_EXCEL_PATH, index=False)
    print(f"✅ 预测结果已保存到：{OUTPUT_EXCEL_PATH}")

    print("\n=== 预测结果统计 ===")
    print(f"总样本数：{len(original_df)}")
    print(f"预测为正样本（1）的数量：{original_df['Prediction_Label'].sum()}")
    print(f"预测为负样本（0）的数量：{len(original_df) - original_df['Prediction_Label'].sum()}")
    print(f"预测概率均值：{original_df['Prediction_Probability'].mean():.4f}")
    print(f"预测概率标准差：{original_df['Prediction_Probability'].std():.4f}")
    clean_memory()

if __name__ == "__main__":
    print("=" * 80)
    print("         INN-GCN 共晶预测模型 - 未知数据集预测脚本")
    print("=" * 80)

    if not os.path.exists(MODEL_SAVE_DIR):
        print(f"❌ 模型目录不存在：{MODEL_SAVE_DIR}")
        sys.exit(1)
    if not os.path.exists(INPUT_EXCEL_PATH):
        print(f"❌ 输入文件不存在：{INPUT_EXCEL_PATH}")
        sys.exit(1)

    main_prediction()
    print("\n🎉 预测完成！")