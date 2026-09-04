import os
import sys
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import deepchem as dc
from rdkit import Chem, RDLogger
from deepchem.feat import ConvMolFeaturizer
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             precision_recall_curve, roc_curve, roc_auc_score)
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import multiprocessing
import joblib
import json

BASE_SEED = 42

FIXED_BEST_PARAMS = {
    'gcn_layer1_size': 256,
    'gcn_layer2_size': 128,
    'dense_layer_size': 256,
    'dropout_rate': 0.2148,
    'l2_regularization': 0.003912,
    'learning_rate': 0.000222,
    'label_smoothing': 0.042
}

EXCEL_PATH = r"D:\project\predict\sjyc-csj.xlsx"
MODEL_SAVE_DIR = r"D:\project\contrast model\cocrystal_prediction_GCN"
PREDICTION_RESULT_SAVE_PATH = r"D:\project\predict\prediction_result.xlsx"
VIS_SAVE_DIR = "./prediction_visualizations"
os.makedirs(VIS_SAVE_DIR, exist_ok=True)

BATCH_SIZE = 32
MAX_ATOM_NUM = 50
EXTRA_FEAT_SIZE = 16
DESC_FEAT_SIZE = 13

RDLogger.DisableLog('rdApp.*')
np.random.seed(BASE_SEED)
random.seed(BASE_SEED)
tf.random.set_seed(BASE_SEED)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.size'] = 10

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

class StandardGCNLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation='relu', dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.dropout_rate = dropout_rate
        self.dense = tf.keras.layers.Dense(self.units)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        atom_feat_shape = input_shape[0]
        input_dim = atom_feat_shape[-1]
        self.dense = tf.keras.layers.Dense(self.units, input_dim=input_dim)
        self.dense.build(atom_feat_shape)
        super().build(input_shape)

    def call(self, inputs, training=False):
        atom_features, adj_matrix, edge_features = inputs
        degree = tf.reduce_sum(adj_matrix, axis=-1, keepdims=True)
        degree_inv_sqrt = tf.math.rsqrt(tf.maximum(degree, 1e-8))
        adj_norm = adj_matrix * degree_inv_sqrt * tf.transpose(degree_inv_sqrt, [0, 2, 1])
        agg_features = tf.matmul(adj_norm, atom_features)
        gcn_out = self.dense(agg_features)
        gcn_out = self.dropout(gcn_out, training=training)
        if self.activation is not None:
            gcn_out = self.activation(gcn_out)
        gcn_out = self.layernorm(gcn_out)
        return gcn_out

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'dropout_rate': self.dropout_rate
        })
        return config

class StandardGCNStackLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_heads=4, activation='relu', dropout_rate=0.2,
                 gate_type='adaptive', global_gate=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.activation = tf.keras.activations.get(activation)
        self.dropout_rate = dropout_rate
        self.gate_type = gate_type
        self.global_gate = global_gate
        self.gcn1 = None
        self.gcn2 = None

    def build(self, input_shape):
        atom_feat_shape = input_shape[0]
        input_dim = atom_feat_shape[-1]
        self.gcn1 = StandardGCNLayer(
            units=self.units,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        )
        self.gcn1.build(input_shape)
        gcn1_output_shape = (atom_feat_shape[0], atom_feat_shape[1], self.units)
        gcn2_input_shape = [gcn1_output_shape, input_shape[1], input_shape[2]]
        self.gcn2 = StandardGCNLayer(
            units=self.units,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        )
        self.gcn2.build(gcn2_input_shape)
        super().build(input_shape)

    def call(self, inputs, training=False):
        atom_features, adj_matrix, edge_features = inputs
        out = self.gcn1([atom_features, adj_matrix, edge_features], training=training)
        out = self.gcn2([out, adj_matrix, edge_features], training=training)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.units)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'num_heads': self.num_heads,
            'activation': tf.keras.activations.serialize(self.activation),
            'dropout_rate': self.dropout_rate,
            'gate_type': self.gate_type,
        })
        return config

class StandardGraphPoolLayer(tf.keras.layers.Layer):
    def __init__(self, units=64, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dense = tf.keras.layers.Dense(units, activation='relu')
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        atom_feats = inputs
        pooled = self.global_avg_pool(atom_feats)
        pooled = self.dense(pooled)
        pooled = self.layernorm(pooled)
        return pooled

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'num_heads': self.num_heads
        })
        return config

class StandardMolCrossLayer(tf.keras.layers.Layer):
    def __init__(self, units=128, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dense1 = tf.keras.layers.Dense(units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, api_feat, ccf_feat, training=False):
        fused = tf.concat([api_feat, ccf_feat], axis=1)
        fused = self.dense1(fused)
        fused = self.layernorm(fused)
        fused = self.dense2(fused)
        return fused

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'num_heads': self.num_heads
        })
        return config

def focal_dice_loss_with_label_smoothing(alpha=0.5, gamma=1.5, smooth=1e-6, label_smoothing=0.042):
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
        1.0 if chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW else 0.0
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
    element_symbol = pt.GetElementSymbol(atom.GetAtomicNum())
    vdw_rad = pt.GetRvdw(atom.GetAtomicNum()) if element_symbol else 1.0

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
    if atom_feats.shape[0] == 0:
        return atom_feats
    mean = np.mean(atom_feats, axis=0, keepdims=True)
    std = np.std(atom_feats, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    atom_feats_std = (atom_feats - mean) / std
    return atom_feats_std

def adj_list_to_matrix(adj_list):
    n = len(adj_list)
    adj_mat = np.zeros((n, n), dtype=np.float32)
    for i, neighbors in enumerate(adj_list):
        for j in neighbors:
            adj_mat[i, j] = 1.0
    return adj_mat

def adj_list_to_matrix_with_edge_feats(mol):
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

def pad_sequences(sequences, padding='post'):
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
    max_size = MAX_ATOM_NUM
    padded = []
    for mat in adj_mats:
        pad_mat = np.zeros((max_size, max_size), dtype=np.float32)
        mat_trunc = mat[:max_size, :max_size]
        pad_mat[:mat_trunc.shape[0], :mat_trunc.shape[1]] = mat_trunc
        padded.append(pad_mat)
    return np.array(padded)

def pad_edge_feats(edge_feats_list):
    max_size = MAX_ATOM_NUM
    n_edge_feats = 4 if not edge_feats_list else edge_feats_list[0].shape[-1]
    padded = []
    for ef in edge_feats_list:
        pad_ef = np.zeros((max_size, max_size, n_edge_feats), dtype=np.float32)
        ef_trunc = ef[:max_size, :max_size]
        pad_ef[:ef_trunc.shape[0], :ef_trunc.shape[1]] = ef_trunc
        padded.append(pad_ef)
    return np.array(padded)

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

def preload_all_data(graph_feats, extra_feats):
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

def predict_data_generator(preloaded_data, batch_size):
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

class EnsembleModel:
    def __init__(self, models=None, scalers=None, weights=None):
        self.models = models if models is not None else []
        self.scalers = scalers if scalers is not None else []
        self.weights = weights

    def predict(self, x, batch_size=BATCH_SIZE, verbose=0):
        if len(self.models) == 0:
            raise ValueError("集成模型为空，请先添加子模型！")
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

    def predict_proba(self, x, batch_size=BATCH_SIZE, verbose=0):
        return self.predict(x, batch_size, verbose)

    @classmethod
    def load_ensemble(cls, save_dir):
        config_path = os.path.join(save_dir, "ensemble_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"集成配置文件不存在：{config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        sub_models_dir = config["sub_models_dir"]
        models = []
        for idx in range(config["n_models"]):
            model_path = os.path.join(sub_models_dir, f"fold_{idx + 1}_model.keras")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"子模型文件不存在：{model_path}")

            custom_objects = {
                "StandardGCNLayer": StandardGCNLayer,
                "StandardGCNStackLayer": StandardGCNStackLayer,
                "StandardGraphPoolLayer": StandardGraphPoolLayer,
                "StandardMolCrossLayer": StandardMolCrossLayer,
                "F1Score": F1Score,
                "focal_dice_loss_with_label_smoothing": focal_dice_loss_with_label_smoothing(alpha=0.5, gamma=1.5, label_smoothing=FIXED_BEST_PARAMS['label_smoothing'])
            }
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
        ensemble_model = cls(models=models, scalers=scalers, weights=weights)
        print(f"✅ 加权集成模型加载完成：{config['n_models']} 个子模型 + {len(scalers)} 组scaler")
        if weights is not None:
            print(f"   - 加载的模型权重：{weights}")
        return ensemble_model

class PredictDataset:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.raw_data = self._load_raw_data()
        self.all_graph_feats = None
        self.all_extra_feats = None
        self._preprocess_all_data()

    def _load_raw_data(self):
        print(f"\nLoading raw data from {self.excel_path}...")
        df = pd.read_excel(self.excel_path, sheet_name='Sheet1')
        raw_data = []

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
            api_smiles = row['SMILES1']
            ccf_smiles = row['SMILES2']

            api_mol = Chem.MolFromSmiles(api_smiles)
            ccf_mol = Chem.MolFromSmiles(ccf_smiles)
            if api_mol is None or ccf_mol is None:
                print(f"Warning: Parse failed, skip row {index}")
                continue

            api_atom_num = api_mol.GetNumAtoms()
            ccf_atom_num = ccf_mol.GetNumAtoms()
            if (api_atom_num < 3 or ccf_atom_num < 3 or
                    api_atom_num > MAX_ATOM_NUM or ccf_atom_num > MAX_ATOM_NUM):
                print(f"Warning: Abnormal atom count, skip row {index}")
                continue

            api_atom_feats = np.array([atom_to_features(atom) for atom in api_mol.GetAtoms()], dtype=np.float32)
            api_atom_feats = standardize_atom_feats(api_atom_feats)

            ccf_atom_feats = np.array([atom_to_features(atom) for atom in ccf_mol.GetAtoms()], dtype=np.float32)
            ccf_atom_feats = standardize_atom_feats(ccf_atom_feats)

            def mol_to_adj_list(mol):
                adj_list = []
                for atom in mol.GetAtoms():
                    neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
                    adj_list.append(neighbors)
                return adj_list

            api_adj_list = mol_to_adj_list(api_mol)
            ccf_adj_list = mol_to_adj_list(ccf_mol)

            api_adj_mat, api_edge_feats = adj_list_to_matrix_with_edge_feats(api_mol)
            ccf_adj_mat, ccf_edge_feats = adj_list_to_matrix_with_edge_feats(ccf_mol)

            api_17_desc = []
            for col in api_17_desc_cols:
                val = row[col]
                api_17_desc.append(float(val) if pd.notna(val) else 0.0)

            ccf_17_desc = []
            for col in ccf_17_desc_cols:
                val = row[col]
                ccf_17_desc.append(float(val) if pd.notna(val) else 0.0)

            api_desc = api_17_desc[:13]
            api_hba_hbd_homo_lumo = api_17_desc[13:]
            ccf_desc = ccf_17_desc[:13]
            ccf_hba_hbd_homo_lumo = ccf_17_desc[13:]

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

            pair_desc = calculate_complementary_features(
                api_desc,
                ccf_desc,
                api_hba_hbd_homo_lumo,
                ccf_hba_hbd_homo_lumo
            )
            extra_feat = pair_desc

            assert len(pair_desc) == EXTRA_FEAT_SIZE, f"Pair desc feat size error: {len(pair_desc)} (expected {EXTRA_FEAT_SIZE})"

            raw_data.append({
                'index': index,
                'graph_feat': graph_feat,
                'extra_feat': extra_feat,
                'original_row': row
            })

            del api_mol, ccf_mol
            del api_desc, ccf_desc, pair_desc

        print(f"Loaded {len(raw_data)} valid raw samples.")
        return raw_data

    def _preprocess_all_data(self):
        self.all_graph_feats = [sample['graph_feat'] for sample in self.raw_data]
        self.all_extra_feats = np.array([sample['extra_feat'] for sample in self.raw_data], dtype=np.float32)
        print(f"\nData preprocess completed:")
        print(f"Total samples: {len(self.all_graph_feats)}")

    def get_original_data(self):
        return [sample['original_row'] for sample in self.raw_data]

def plot_prediction_scatter(y_pred_proba, threshold=0.5, save_name='prediction_scatter.png'):
    save_path = os.path.join(VIS_SAVE_DIR, save_name)
    plt.figure(figsize=(12, 8))

    pos_idx = y_pred_proba >= threshold
    neg_idx = y_pred_proba < threshold

    plt.scatter(
        np.where(pos_idx)[0], y_pred_proba[pos_idx],
        color='#E74C3C', alpha=0.7, s=60, edgecolor='black', linewidth=0.5,
        label=f'Positive Prediction (n={np.sum(pos_idx)})'
    )
    plt.scatter(
        np.where(neg_idx)[0], y_pred_proba[neg_idx],
        color='#3498DB', alpha=0.7, s=60, edgecolor='black', linewidth=0.5,
        label=f'Negative Prediction (n={np.sum(neg_idx)})'
    )

    plt.axhline(
        y=threshold, color='#F39C12', linestyle='--', linewidth=2.5,
        label=f'Classification Threshold = {threshold}'
    )

    plt.xlabel('Sample Index', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Probability of Cocrystal Formation', fontsize=12, fontweight='bold')
    plt.title('Model Prediction Probability Distribution', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.ylim(-0.05, 1.05)
    plt.xlim(-5, len(y_pred_proba) + 5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 预测概率散点图已保存到：{save_path}")

def main():
    print(f"\n{'=' * 80}")
    print(f"【共晶形成预测模型 - 全新数据集预测】")
    print(f"{'=' * 80}")
    print(f"使用固定最优超参数：")
    for param_name, param_value in FIXED_BEST_PARAMS.items():
        if isinstance(param_value, float):
            if param_name in ['learning_rate', 'l2_regularization']:
                print(f"  {param_name}: {param_value:.6f}")
            elif param_name == 'dropout_rate':
                print(f"  {param_name}: {param_value:.4f}")
            else:
                print(f"  {param_name}: {param_value:.3f}")
        else:
            print(f"  {param_name}: {param_value}")

    print(f"\n{'=' * 80}")
    print(f"【步骤1】加载训练好的集成模型")
    print(f"{'=' * 80}")
    try:
        ensemble_model = EnsembleModel.load_ensemble(MODEL_SAVE_DIR)
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        print(f"请检查MODEL_SAVE_DIR路径是否正确，确保模型文件已保存到该路径")
        return

    print(f"\n{'=' * 80}")
    print(f"【步骤2】加载并预处理数据集")
    print(f"{'=' * 80}")
    try:
        dataset = PredictDataset(EXCEL_PATH)
    except Exception as e:
        print(f"❌ 数据集加载失败：{e}")
        return

    print(f"\n{'=' * 80}")
    print(f"【步骤3】特征标准化（使用训练集scaler）")
    print(f"{'=' * 80}")
    api_scaler = ensemble_model.scalers[0]['api_desc_scaler']
    ccf_scaler = ensemble_model.scalers[0]['ccf_desc_scaler']
    pair_scaler = ensemble_model.scalers[0]['pair_feat_scaler']

    for i, gf in enumerate(dataset.all_graph_feats):
        gf['api_global_desc_scaled'] = api_scaler.transform(gf['api_global_desc'].reshape(1, -1)).flatten()
        gf['ccf_global_desc_scaled'] = ccf_scaler.transform(gf['ccf_global_desc'].reshape(1, -1)).flatten()

    all_extra_feats_scaled = pair_scaler.transform(dataset.all_extra_feats)

    print(f"\n{'=' * 80}")
    print(f"【步骤4】预加载预测数据")
    print(f"{'=' * 80}")
    preloaded_data = preload_all_data(dataset.all_graph_feats, all_extra_feats_scaled)

    print(f"\n{'=' * 80}")
    print(f"【步骤5】执行模型预测")
    print(f"{'=' * 80}")
    test_gen = predict_data_generator(preloaded_data, batch_size=BATCH_SIZE)
    steps = len(preloaded_data) // BATCH_SIZE + 1

    y_pred_proba = []
    for _ in range(steps):
        try:
            X = next(test_gen)
            pred = ensemble_model.predict(X, verbose=0)
            y_pred_proba.extend(pred.flatten())
        except StopIteration:
            break

    y_pred_proba = np.array(y_pred_proba[:len(preloaded_data)])
    y_pred = (y_pred_proba > 0.5).astype(int)

    print(f"✅ 预测完成！共预测 {len(y_pred_proba)} 个样本")
    print(f"   预测为正样本（可形成共晶）的数量：{np.sum(y_pred)}")
    print(f"   预测为负样本（不可形成共晶）的数量：{len(y_pred) - np.sum(y_pred)}")

    print(f"\n{'=' * 80}")
    print(f"【步骤6】生成预测结果文件")
    print(f"{'=' * 80}")
    original_data = dataset.get_original_data()
    result_df = pd.DataFrame(original_data)

    result_df['predicted_probability'] = y_pred_proba
    result_df['predicted_label'] = y_pred
    result_df['predicted_label_desc'] = result_df['predicted_label'].map({
        0: '不可形成共晶',
        1: '可形成共晶'
    })

    result_df.to_excel(PREDICTION_RESULT_SAVE_PATH, index=False)
    print(f"✅ 预测结果已保存到：{PREDICTION_RESULT_SAVE_PATH}")

    print(f"\n{'=' * 80}")
    print(f"【步骤7】生成预测可视化图表")
    print(f"{'=' * 80}")
    plot_prediction_scatter(y_pred_proba, threshold=0.5)

    print(f"\n{'=' * 80}")
    print(f"【预测结果预览（前10个样本）】")
    print(f"{'=' * 80}")
    preview_df = result_df[['API', 'CCF', 'predicted_probability', 'predicted_label_desc']].head(10)
    print(preview_df.to_string(index=False))

    print(f"\n{'=' * 80}")
    print(f"【预测流程全部完成！】")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()