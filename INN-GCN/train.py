import os
import sys
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import deepchem as dc
from rdkit import Chem, RDLogger
from deepchem.feat import ConvMolFeaturizer
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             precision_recall_curve, roc_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import copy
import gc
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer, Dense
import multiprocessing
import json
import pickle
matplotlib.use('Agg')

MAX_CPU = multiprocessing.cpu_count()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], False)
else:
    tf.config.set_soft_device_placement(True)
tf.config.threading.set_intra_op_parallelism_threads(MAX_CPU)
tf.config.threading.set_inter_op_parallelism_threads(MAX_CPU)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CPU_ALLOCATOR'] = 'cpu_malloc_async'
if gpus:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
else:
    tf.keras.mixed_precision.set_global_policy('float32')

BASE_SEED = 42
CV_FOLD_SEED = BASE_SEED
HP_OPT_SEED = BASE_SEED

EXCEL_PATH = r"D:\YWGJ\gjsj001.xlsx"
BATCH_SIZE = 32
MAX_EVALS = 5
TRAIN_EPOCHS = 60
EARLY_STOP_PATIENCE = 8
N_FOLDS = 5
MODEL_SAVE_DIR = r"D:\project\contrast model\cocrystal_prediction_INN-GCNtp1"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
HYPERPARAMS_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "best_hyperparams.json")

EXTRA_FEAT_SIZE = 16
DESC_FEAT_SIZE = 17
MAX_ATOM_NUM = 50
AUG_SWAP = True
AUG_NOISE = True
AUG_MASK = True

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

print(f"Python Version: {sys.version.split()[0]}")
print(f"RDKit Version: {Chem.rdbase.rdkitVersion}")
print(f"TensorFlow Version: {tf.__version__}")
print(f"DeepChem Version: {dc.__version__}")

def clean_memory(verbose=True):
    tf.keras.backend.clear_session()
    if tf.__version__ >= '2.0':
        tf.compat.v1.reset_default_graph()
    try:
        tf.config.experimental.reset_memory_stats('CPU:0')
        if tf.config.list_physical_devices('GPU'):
            tf.config.experimental.reset_memory_stats('GPU:0')
    except Exception:
        pass
    collected_base = gc.collect()
    collected_gen2 = gc.collect(2)
    total_collected = collected_base + collected_gen2
    np.random.seed(BASE_SEED)
    try:
        import psutil
        process = psutil.Process(os.getpid())
        process.memory_full_info()
    except:
        pass
    if verbose:
        print(f"✅ 内存清理完成：回收 {total_collected} 个对象")
    return total_collected

def pad_adj_matrices(adj_mats):
    max_size = MAX_ATOM_NUM
    padded = []
    for mat in adj_mats:
        pad_mat = np.zeros((max_size, max_size), dtype=np.float32)
        mat_trunc = mat[:max_size, :max_size]
        pad_mat[:mat_trunc.shape[0], :mat_trunc.shape[1]] = mat_trunc
        padded.append(pad_mat)
    return np.array(padded)

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

def preload_all_data(graph_feats, extra_feats, labels):
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
            'extra': extra_feats[i],
            'label': float(labels[i])
        })
    return preloaded

def val_test_data_generator(preloaded_data, batch_size):
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
        labels = [d['label'] for d in batch_data]

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
        ], np.array(labels)

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

class INN_GraphConvLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation='relu', dropout_rate=0.2, gate_type='adaptive', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.dropout_rate = dropout_rate
        self.gate_type = gate_type
        self.dense = tf.keras.layers.Dense(units)
        self.edge_dense = tf.keras.layers.Dense(1)
        self.residual_dense = tf.keras.layers.Dense(units)
        if self.gate_type == 'adaptive':
            self.gate = tf.keras.Sequential([
                tf.keras.layers.Dense(units * 2, activation='relu'),
                tf.keras.layers.LayerNormalization(epsilon=1e-6),
                tf.keras.layers.Dense(units, activation='sigmoid')
            ])
        elif self.gate_type == 'hybrid':
            self.static_gate = self.add_weight(
                shape=(1, 1, units),
                initializer=tf.keras.initializers.Constant(0.5),
                trainable=True,
                name='static_gate'
            )
            self.dynamic_gate = tf.keras.layers.Dense(units, activation='sigmoid')
        elif self.gate_type == 'hard':
            self.gate = tf.keras.layers.Dense(units)
            self.gate_threshold = self.add_weight(
                shape=(1, 1, units),
                initializer=tf.keras.initializers.Constant(0.0),
                trainable=True,
                name='gate_threshold'
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
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'dropout_rate': self.dropout_rate,
            'gate_type': self.gate_type
        })
        return config

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
    return (atom_feats - mean) / std

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

def create_batched_data(graph_feats, extra_feats, labels, batch_size, shuffle=True):
    precomputed = []
    for i, gf in enumerate(graph_feats):
        api_adj = adj_list_to_matrix(gf['api_adj_list'])
        ccf_adj = adj_list_to_matrix(gf['ccf_adj_list'])
        if 'api_edge_feats' in gf:
            api_edge_feats = gf['api_edge_feats']
        else:
            api_size = api_adj.shape[0]
            api_edge_feats = np.zeros((api_size, api_size, 4), dtype=np.float32)
        if 'ccf_edge_feats' in gf:
            ccf_edge_feats = gf['ccf_edge_feats']
        else:
            ccf_size = ccf_adj.shape[0]
            ccf_edge_feats = np.zeros((ccf_size, ccf_size, 4), dtype=np.float32)
        precomputed.append({
            'api_global_desc': gf['api_global_desc_scaled'],
            'ccf_global_desc': gf['ccf_global_desc_scaled'],
            'api_atom': gf['api_atom_feats'],
            'api_adj': api_adj,
            'api_edge_feats': api_edge_feats,
            'ccf_atom': gf['ccf_atom_feats'],
            'ccf_adj': ccf_adj,
            'ccf_edge_feats': ccf_edge_feats,
            'extra': extra_feats[i],
            'label': float(labels[i])
        })
    indices = np.arange(len(precomputed))
    if shuffle:
        np.random.shuffle(indices)
    batches_x, batches_y = [], []
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        batch_idx = indices[start:end]
        batch_data = [precomputed[i] for i in batch_idx]
        api_max_len = MAX_ATOM_NUM
        ccf_max_len = MAX_ATOM_NUM
        batch_api_atom = np.zeros((len(batch_data), api_max_len, 31), dtype=np.float32)
        batch_api_adj = np.zeros((len(batch_data), api_max_len, api_max_len), dtype=np.float32)
        batch_api_edge = np.zeros((len(batch_data), api_max_len, api_max_len, 4), dtype=np.float32)
        batch_ccf_atom = np.zeros((len(batch_data), ccf_max_len, 31), dtype=np.float32)
        batch_ccf_adj = np.zeros((len(batch_data), ccf_max_len, ccf_max_len), dtype=np.float32)
        batch_ccf_edge = np.zeros((len(batch_data), ccf_max_len, ccf_max_len, 4), dtype=np.float32)
        batch_extra = np.zeros((len(batch_data), EXTRA_FEAT_SIZE), dtype=np.float32)
        batch_labels = np.zeros(len(batch_data), dtype=np.float32)
        for i, d in enumerate(batch_data):
            api_len = d['api_atom'].shape[0]
            batch_api_atom[i, :api_len] = d['api_atom']
            batch_api_adj[i, :api_len, :api_len] = d['api_adj']
            batch_api_edge[i, :api_len, :api_len] = d['api_edge_feats']
            ccf_len = d['ccf_atom'].shape[0]
            batch_ccf_atom[i, :ccf_len] = d['ccf_atom']
            batch_ccf_adj[i, :ccf_len, :ccf_len] = d['ccf_adj']
            batch_ccf_edge[i, :ccf_len, :ccf_len] = d['ccf_edge_feats']
            batch_extra[i] = d['extra']
            batch_labels[i] = d['label']
        batches_x.append([
            batch_api_atom, batch_api_adj, batch_api_edge,
            batch_ccf_atom, batch_ccf_adj, batch_ccf_edge,
            batch_extra
        ])
        batches_y.append(batch_labels)
    return batches_x, batches_y

def focal_dice_loss_with_label_smoothing(alpha=0.5, gamma=1.5, smooth=1e-6, label_smoothing=0.1):
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

class CrossValCocrystalDataset:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.raw_data = self._load_raw_data()
        self.all_graph_feats = None
        self.all_extra_feats = None
        self.all_labels = None
        self.train_val_graph_feats = None
        self.train_val_extra_feats_raw = None
        self.train_val_labels_raw = None
        self.fold_original_indices = []
        self._preprocess_all_data()

    def _load_raw_data(self):
        print(f"\nLoading raw data from {self.excel_path}...")
        df = pd.read_excel(self.excel_path, sheet_name='Sheet1')
        raw_data = []
        conv_featurizer = ConvMolFeaturizer()
        for index, row in df.iterrows():
            api_smiles = row['SMILES1']
            ccf_smiles = row['SMILES2']
            target = int(row['Target'])
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
                'api_global_desc': np.array(api_17_desc, dtype=np.float32),
                'ccf_global_desc': np.array(ccf_17_desc, dtype=np.float32),
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
            assert len(pair_desc) == EXTRA_FEAT_SIZE, f"Pair desc feat size error: {len(pair_desc)}"
            raw_data.append({
                'index': index,
                'graph_feat': graph_feat,
                'extra_feat': extra_feat,
                'target': target,
                'api_smiles': api_smiles,
                'ccf_smiles': ccf_smiles,
                'api_17_desc': np.array(api_17_desc, dtype=np.float32),
                'ccf_17_desc': np.array(ccf_17_desc, dtype=np.float32),
                'api_atom_num': api_atom_num,
                'ccf_atom_num': ccf_atom_num
            })
            del api_mol, ccf_mol
            del api_desc, ccf_desc, pair_desc
        print(f"Loaded {len(raw_data)} valid raw samples (GCN graph features).")
        return raw_data

    def get_data_statistics(self):
        targets = [x['target'] for x in self.raw_data]
        api_atoms = [x['api_atom_num'] for x in self.raw_data]
        ccf_atoms = [x['ccf_atom_num'] for x in self.raw_data]
        return {
            'target_distribution': pd.Series(targets).value_counts(),
            'api_atom_dist': np.array(api_atoms),
            'ccf_atom_dist': np.array(ccf_atoms),
            'all_labels': self.all_labels
        }

    def _preprocess_all_data(self):
        self.all_graph_feats = [sample['graph_feat'] for sample in self.raw_data]
        self.all_extra_feats = np.array([sample['extra_feat'] for sample in self.raw_data], dtype=np.float32)
        self.all_labels = np.array([sample['target'] for sample in self.raw_data], dtype=np.int32)
        self.train_val_graph_feats = self.all_graph_feats
        self.train_val_extra_feats_raw = self.all_extra_feats.copy()
        self.train_val_labels_raw = self.all_labels.copy()
        print(f"\nData preprocess completed (no test set):")
        print(f"Total train+val samples: {len(self.train_val_graph_feats)}")
        print(f"Positive rate: {np.mean(self.train_val_labels_raw):.4f}")

    def _augment_data(self, graph_feats, extra_feats, labels, fold_idx=0):
        print(f"\n[Fold {fold_idx}] 开始数据增强，原始样本数：{len(graph_feats)}")
        if len(graph_feats) == 0:
            return graph_feats, extra_feats, labels
        aug_seed = BASE_SEED + fold_idx
        np.random.seed(aug_seed)
        random.seed(aug_seed)
        aug_graph_feats = graph_feats.copy()
        aug_extra_feats = extra_feats.copy()
        aug_labels = labels.copy()

        if AUG_SWAP:
            swap_ratio = 1.0
            n_swap = int(len(graph_feats) * swap_ratio)
            swap_indices = np.random.choice(len(graph_feats), n_swap, replace=False)
            for idx in swap_indices:
                original_graph = graph_feats[idx]
                swapped_api_smiles = original_graph['ccf_smiles']
                swapped_ccf_smiles = original_graph['api_smiles']
                swapped_api_17_desc = original_graph['ccf_17_desc']
                swapped_ccf_17_desc = original_graph['api_17_desc']
                swapped_api_desc = swapped_api_17_desc[:13]
                swapped_api_hba_hbd = swapped_api_17_desc[13:]
                swapped_ccf_desc = swapped_ccf_17_desc[:13]
                swapped_ccf_hba_hbd = swapped_ccf_17_desc[13:]
                aug_graph = {
                    'api_atom_feats': original_graph['ccf_atom_feats'],
                    'api_adj_list': original_graph['ccf_adj_list'],
                    'api_edge_feats': original_graph['ccf_edge_feats'],
                    'api_deg_list': original_graph['ccf_deg_list'],
                    'ccf_atom_feats': original_graph['api_atom_feats'],
                    'ccf_adj_list': original_graph['api_adj_list'],
                    'ccf_edge_feats': original_graph['api_edge_feats'],
                    'ccf_deg_list': original_graph['api_deg_list'],
                    'api_smiles': swapped_api_smiles,
                    'ccf_smiles': swapped_ccf_smiles,
                    'api_17_desc': swapped_api_17_desc,
                    'ccf_17_desc': swapped_ccf_17_desc,
                    'api_global_desc': swapped_api_desc,
                    'ccf_global_desc': swapped_ccf_desc,
                    'api_hba_hbd_homo_lumo': swapped_api_hba_hbd,
                    'ccf_hba_hbd_homo_lumo': swapped_ccf_hba_hbd,
                    'api_global_desc_scaled': original_graph.get('ccf_global_desc_scaled', swapped_api_desc),
                    'ccf_global_desc_scaled': original_graph.get('api_global_desc_scaled', swapped_ccf_desc)
                }
                swapped_pair_desc = calculate_complementary_features(
                    swapped_api_desc,
                    swapped_ccf_desc,
                    swapped_api_hba_hbd,
                    swapped_ccf_hba_hbd
                )
                aug_graph_feats.append(aug_graph)
                aug_extra_feats = np.vstack([aug_extra_feats, swapped_pair_desc])
                aug_labels = np.append(aug_labels, labels[idx])
            print(f"✅ 分子交换增强完成：新增 {n_swap} 个样本")

        if AUG_NOISE:
            noise_ratio = 0.2
            n_noise = int(len(graph_feats) * noise_ratio)
            noise_indices = np.random.choice(len(graph_feats), n_noise, replace=False)
            noise_std = 0.01
            for idx in noise_indices:
                noisy_extra = extra_feats[idx] + np.random.normal(0, noise_std, extra_feats[idx].shape)
                noisy_extra = np.clip(noisy_extra, 0, 1)
                aug_graph_feats.append(graph_feats[idx])
                aug_extra_feats = np.vstack([aug_extra_feats, noisy_extra])
                aug_labels = np.append(aug_labels, labels[idx])
            print(f"✅ 特征噪声增强完成：新增 {n_noise} 个样本")

        if AUG_MASK:
            mask_ratio = 0.2
            n_mask = int(len(graph_feats) * mask_ratio)
            mask_indices = np.random.choice(len(graph_feats), n_mask, replace=False)
            mask_prob = 0.1
            for idx in mask_indices:
                original_graph = graph_feats[idx].copy()
                api_atom_feats = original_graph['api_atom_feats']
                mask = np.random.choice([0, 1], size=api_atom_feats.shape, p=[mask_prob, 1 - mask_prob])
                original_graph['api_atom_feats'] = api_atom_feats * mask
                ccf_atom_feats = original_graph['ccf_atom_feats']
                mask = np.random.choice([0, 1], size=ccf_atom_feats.shape, p=[mask_prob, 1 - mask_prob])
                original_graph['ccf_atom_feats'] = ccf_atom_feats * mask
                aug_graph_feats.append(original_graph)
                aug_extra_feats = np.vstack([aug_extra_feats, extra_feats[idx]])
                aug_labels = np.append(aug_labels, labels[idx])
            print(f"✅ 原子掩码增强完成：新增 {n_mask} 个样本")

        shuffle_indices = np.arange(len(aug_graph_feats))
        np.random.shuffle(shuffle_indices)
        aug_graph_feats = [aug_graph_feats[i] for i in shuffle_indices]
        aug_extra_feats = aug_extra_feats[shuffle_indices]
        aug_labels = aug_labels[shuffle_indices]
        print(f"📊 数据增强总览：{len(graph_feats)} → {len(aug_graph_feats)} 样本")
        np.random.seed(BASE_SEED)
        random.seed(BASE_SEED)
        return aug_graph_feats, aug_extra_feats, aug_labels

    def get_cv_folds(self):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_FOLD_SEED)
        folds = []
        self.fold_original_indices = []
        for fold_idx, (train_idx, val_idx) in enumerate(
                skf.split(self.train_val_extra_feats_raw, self.train_val_labels_raw)):
            train_graph_feats = [copy.deepcopy(self.train_val_graph_feats[i]) for i in train_idx]
            val_graph_feats = [copy.deepcopy(self.train_val_graph_feats[i]) for i in val_idx]
            train_extra_feats_raw = self.train_val_extra_feats_raw[train_idx].copy()
            val_extra_feats_raw = self.train_val_extra_feats_raw[val_idx].copy()
            train_labels_raw = self.train_val_labels_raw[train_idx].copy()
            val_labels_raw = self.train_val_labels_raw[val_idx].copy()

            scaler = StandardScaler()
            scaler.fit(train_extra_feats_raw)

            train_api_global_desc = np.array([gf['api_global_desc'] for gf in train_graph_feats])
            train_ccf_global_desc = np.array([gf['ccf_global_desc'] for gf in train_graph_feats])
            train_pair_feat = train_extra_feats_raw

            api_desc_scaler = StandardScaler()
            ccf_desc_scaler = StandardScaler()
            pair_feat_scaler = StandardScaler()
            api_desc_scaler.fit(train_api_global_desc)
            ccf_desc_scaler.fit(train_ccf_global_desc)
            pair_feat_scaler.fit(train_pair_feat)

            for i, gf in enumerate(train_graph_feats):
                gf['api_global_desc_scaled'] = api_desc_scaler.transform(gf['api_global_desc'].reshape(1, -1)).flatten()
                gf['ccf_global_desc_scaled'] = ccf_desc_scaler.transform(gf['ccf_global_desc'].reshape(1, -1)).flatten()
            for i, gf in enumerate(val_graph_feats):
                gf['api_global_desc_scaled'] = api_desc_scaler.transform(gf['api_global_desc'].reshape(1, -1)).flatten()
                gf['ccf_global_desc_scaled'] = ccf_desc_scaler.transform(gf['ccf_global_desc'].reshape(1, -1)).flatten()

            train_extra_feats_scaled = pair_feat_scaler.transform(train_extra_feats_raw)
            val_extra_feats_scaled = pair_feat_scaler.transform(val_extra_feats_raw)

            folds.append({
                'fold_idx': fold_idx,
                'train_graph_feats': train_graph_feats,
                'train_extra_feats': train_extra_feats_scaled,
                'train_labels': train_labels_raw,
                'val_graph_feats': val_graph_feats,
                'val_extra_feats': val_extra_feats_scaled,
                'val_labels': val_labels_raw,
                'scaler': scaler,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'val_original_indices': val_idx,
                'api_desc_scaler': api_desc_scaler,
                'ccf_desc_scaler': ccf_desc_scaler,
                'pair_feat_scaler': pair_feat_scaler
            })
            self.fold_original_indices.append({
                'train_idx': train_idx,
                'val_idx': val_idx
            })
            print(f"Fold {fold_idx}: Train/Val object identity check")
            print(f"Train graph feat 0 id: {id(train_graph_feats[0])}")
            print(f"Val graph feat 0 id: {id(val_graph_feats[0])}")
            assert id(train_graph_feats[0]) != id(val_graph_feats[0]), "数据未隔离！"
        return folds

def calculate_metrics(model, graph_feats, extra_feats, labels, batch_size=BATCH_SIZE):
    preloaded = preload_all_data(graph_feats, extra_feats, labels)
    test_gen = val_test_data_generator(preloaded, batch_size)
    steps = len(preloaded) // batch_size + 1
    y_pred_proba = []
    y_true = []
    for _ in range(steps):
        try:
            X, y = next(test_gen)
            pred = model.predict(X, verbose=0)
            y_pred_proba.extend(pred.flatten())
            y_true.extend(y.flatten())
        except StopIteration:
            break
    y_pred_proba = np.array(y_pred_proba[:len(labels)])
    y_true = np.array(y_true[:len(labels)]).astype(int)

    print(f"\n=== Probability Distribution Stats ===")
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    if np.sum(pos_mask) > 0:
        print(f"Positive Mean: {np.mean(y_pred_proba[pos_mask]):.4f}, Median: {np.median(y_pred_proba[pos_mask]):.4f}")
    if np.sum(neg_mask) > 0:
        print(f"Negative Mean: {np.mean(y_pred_proba[neg_mask]):.4f}, Median: {np.median(y_pred_proba[neg_mask]):.4f}")
    if np.sum(pos_mask) > 0 and np.sum(neg_mask) > 0:
        print(f"Positive/Negative Gap: {np.mean(y_pred_proba[pos_mask]) - np.mean(y_pred_proba[neg_mask]):.4f}")

    threshold = 0.5
    y_pred = (y_pred_proba > threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'threshold': 0.5
    }

def one_cycle_lr(epoch, lr, max_lr=4e-4, epochs=TRAIN_EPOCHS, pct_start=None):
    if pct_start is None:
        pct_start = 0.3
    if epoch < epochs * pct_start:
        return lr + (max_lr - lr) * (epoch / (epochs * pct_start))
    else:
        decay_ratio = (epoch - epochs * pct_start) / (epochs * (1 - pct_start))
        min_lr = max_lr * 0.1
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + tf.math.cos(np.pi * decay_ratio))

def train_with_loss_recording(
        model,
        train_preloaded,
        val_preloaded,
        batch_size=BATCH_SIZE,
        epochs=TRAIN_EPOCHS,
        patience=EARLY_STOP_PATIENCE
):
    def train_gen():
        while True:
            indices = np.arange(len(train_preloaded))
            np.random.shuffle(indices)
            for start in range(0, len(train_preloaded), batch_size):
                end = min(start + batch_size, len(train_preloaded))
                batch_idx = indices[start:end]
                batch_data = [train_preloaded[i] for i in batch_idx]
                api_atom = [d['api_atom'] for d in batch_data]
                api_adj = [d['api_adj'] for d in batch_data]
                api_edge_feats = [d['api_edge_feats'] for d in batch_data]
                ccf_atom = [d['ccf_atom'] for d in batch_data]
                ccf_adj = [d['ccf_adj'] for d in batch_data]
                ccf_edge_feats = [d['ccf_edge_feats'] for d in batch_data]
                extra = [d['extra'] for d in batch_data]
                labels = [d['label'] for d in batch_data]
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
                ], np.array(labels)

    def val_gen():
        while True:
            indices = np.arange(len(val_preloaded))
            for start in range(0, len(val_preloaded), batch_size):
                end = min(start + batch_size, len(val_preloaded))
                batch_idx = indices[start:end]
                batch_data = [val_preloaded[i] for i in batch_idx]
                api_atom = [d['api_atom'] for d in batch_data]
                api_adj = [d['api_adj'] for d in batch_data]
                api_edge_feats = [d['api_edge_feats'] for d in batch_data]
                ccf_atom = [d['ccf_atom'] for d in batch_data]
                ccf_adj = [d['ccf_adj'] for d in batch_data]
                ccf_edge_feats = [d['ccf_edge_feats'] for d in batch_data]
                extra = [d['extra'] for d in batch_data]
                labels = [d['label'] for d in batch_data]
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
                ], np.array(labels)

    steps_per_epoch = len(train_preloaded) // batch_size + 1
    val_steps = len(val_preloaded) // batch_size + 1

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch, lr: one_cycle_lr(epoch, lr, max_lr=model.optimizer.learning_rate.numpy()),
        verbose=1
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_f1_score',
        min_delta=0.0001,
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
        verbose=1,
        mode='max',
        baseline=None,
    )

    history = model.fit(
        train_gen(),
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen(),
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=[early_stopping, lr_scheduler],
        verbose=2,
        workers=0,
        use_multiprocessing=False,
        max_queue_size=100,
        shuffle=False
    )
    return history.history['loss'], history.history['val_loss'], early_stopping.best_epoch + 1

class INN_HybridGraphLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_heads=4, activation='relu', dropout_rate=0.2,
                 gate_type='adaptive', global_gate=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.activation = tf.keras.activations.get(activation)
        self.dropout_rate = dropout_rate
        self.gate_type = gate_type
        self.global_gate = global_gate
        self.local_gcn = INN_GraphConvLayer(
            units, activation=activation,
            dropout_rate=dropout_rate, gate_type=gate_type
        )
        self.attn_dim_adapt = tf.keras.layers.Dense(units)
        self.global_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=units // num_heads,
            dropout=0.1
        )
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
            shape=(1, 1, self.units),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
            name='fuse_weight'
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        atom_features, adj_matrix, edge_features = inputs
        local_out = self.local_gcn([atom_features, adj_matrix, edge_features], training=training)
        atom_feat_adapt = self.attn_dim_adapt(atom_features)
        global_out = self.global_attn(
            query=atom_feat_adapt,
            key=atom_feat_adapt,
            value=atom_feat_adapt,
            training=training
        )
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
            'units': self.units,
            'num_heads': self.num_heads,
            'activation': tf.keras.activations.serialize(self.activation),
            'dropout_rate': self.dropout_rate,
            'gate_type': self.gate_type,
            'global_gate': self.global_gate
        })
        return config

class INN_GraphAttentionPoolLayer(tf.keras.layers.Layer):
    def __init__(self, units=64, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=units // num_heads,
            dropout=0.1
        )
        self.dense = tf.keras.layers.Dense(units)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        atom_feats = inputs
        attn_out = self.attention(
            query=atom_feats,
            key=atom_feats,
            value=atom_feats,
            training=training
        )
        out = self.layernorm(atom_feats + attn_out)
        attn_weights = tf.nn.softmax(self.dense(out), axis=1)
        pooled = tf.reduce_sum(out * attn_weights, axis=1)
        return pooled

class INN_MolCrossAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units=128, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.cross_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=units,
            dropout=0.1
        )
        self.dense1 = tf.keras.layers.Dense(units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, api_feat, ccf_feat, training=False):
        api_feat_exp = tf.expand_dims(api_feat, axis=1)
        ccf_feat_exp = tf.expand_dims(ccf_feat, axis=1)
        api2ccf = self.cross_attention(
            query=ccf_feat_exp,
            key=api_feat_exp,
            value=api_feat_exp,
            training=training
        )
        ccf2api = self.cross_attention(
            query=api_feat_exp,
            key=ccf_feat_exp,
            value=ccf_feat_exp,
            training=training
        )
        api_fused = self.layernorm(api_feat_exp + ccf2api)
        ccf_fused = self.layernorm(ccf_feat_exp + api2ccf)
        fused = tf.concat([tf.squeeze(api_fused, 1), tf.squeeze(ccf_fused, 1)], axis=1)
        fused = self.dense2(self.dense1(fused))
        return fused

class INN_MolPairMemoryLayer(tf.keras.layers.Layer):
    def __init__(self, units=256, memory_scale=0.1, **kwargs):
        super(INN_MolPairMemoryLayer, self).__init__(**kwargs)
        self.units = units
        self.memory_scale = memory_scale

    def build(self, input_shape):
        input_units = input_shape[-1]
        self.units = input_units
        self.pair_interact_dense = tf.keras.layers.Dense(
            self.units,
            activation='relu',
            kernel_initializer='he_normal',
            name='pair_interact_dense'
        )
        self.enhance_gate = tf.keras.layers.Dense(
            self.units,
            activation='sigmoid',
            kernel_initializer='glorot_uniform',
            name='enhance_gate'
        )
        self.initial_memory = tf.zeros((1, self.units))
        super(INN_MolPairMemoryLayer, self).build(input_shape)

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
        config = super(INN_MolPairMemoryLayer, self).get_config()
        config.update({
            'units': self.units,
            'memory_scale': self.memory_scale
        })
        return config

def build_model(hp_params):
    api_atom_input = tf.keras.Input(shape=(None, 48), name='api_atom_input')
    api_adj_input = tf.keras.Input(shape=(None, None), name='api_adj_input')
    api_edge_input = tf.keras.Input(shape=(None, None, 4), name='api_edge_input')
    ccf_atom_input = tf.keras.Input(shape=(None, 48), name='ccf_atom_input')
    ccf_adj_input = tf.keras.Input(shape=(None, None), name='ccf_adj_input')
    ccf_edge_input = tf.keras.Input(shape=(None, None, 4), name='ccf_edge_input')
    extra_input = tf.keras.Input(shape=(EXTRA_FEAT_SIZE,), name='extra_input')

    api_inn_hybrid1 = INN_HybridGraphLayer(
        hp_params['gcn_layer1_size'],
        num_heads=hp_params.get('num_heads', 4),
        activation='relu',
        dropout_rate=hp_params.get('dropout_rate', 0.2),
        gate_type='adaptive',
        global_gate=True,
        name='api_inn_hybrid1'
    )
    api_inn_hybrid2 = INN_HybridGraphLayer(
        hp_params['gcn_layer2_size'],
        num_heads=hp_params.get('num_heads', 4),
        activation='relu',
        dropout_rate=hp_params.get('dropout_rate', 0.2),
        gate_type='adaptive',
        global_gate=True,
        name='api_inn_hybrid2'
    )
    ccf_inn_hybrid1 = INN_HybridGraphLayer(
        hp_params['gcn_layer1_size'],
        num_heads=hp_params.get('num_heads', 4),
        activation='relu',
        dropout_rate=hp_params.get('dropout_rate', 0.2),
        gate_type='adaptive',
        global_gate=True,
        name='ccf_inn_hybrid1'
    )
    ccf_inn_hybrid2 = INN_HybridGraphLayer(
        hp_params['gcn_layer2_size'],
        num_heads=hp_params.get('num_heads', 4),
        activation='relu',
        dropout_rate=hp_params.get('dropout_rate', 0.2),
        gate_type='adaptive',
        global_gate=True,
        name='ccf_inn_hybrid2'
    )

    api_inn1 = api_inn_hybrid1([api_atom_input, api_adj_input, api_edge_input])
    api_inn2 = api_inn_hybrid2([api_inn1, api_adj_input, api_edge_input])
    api_inn_pool = INN_GraphAttentionPoolLayer(
        units=hp_params['gcn_layer2_size'],
        num_heads=hp_params.get('num_heads', 4),
        name='api_inn_pool'
    )(api_inn2)

    ccf_inn1 = ccf_inn_hybrid1([ccf_atom_input, ccf_adj_input, ccf_edge_input])
    ccf_inn2 = ccf_inn_hybrid2([ccf_inn1, ccf_adj_input, ccf_edge_input])
    ccf_inn_pool = INN_GraphAttentionPoolLayer(
        units=hp_params['gcn_layer2_size'],
        num_heads=hp_params.get('num_heads', 4),
        name='ccf_inn_pool'
    )(ccf_inn2)

    inn_memory_layer = INN_MolPairMemoryLayer(
        units=hp_params['dense_layer_size'],
        memory_scale=0.1
    )
    api_feat_enhanced, ccf_feat_enhanced = inn_memory_layer(api_inn_pool, ccf_inn_pool)

    inn_cross_attn = INN_MolCrossAttentionLayer(
        units=hp_params['dense_layer_size'],
        num_heads=hp_params.get('num_heads', 4)
    )(api_feat_enhanced, ccf_feat_enhanced)

    combined = tf.keras.layers.Concatenate(name='combined_features')(
        [inn_cross_attn, extra_input]
    )

    dropout = tf.keras.layers.Dropout(hp_params['dropout_rate'])(combined)
    dense1 = tf.keras.layers.Dense(
        hp_params['dense_layer_size'],
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(hp_params['l2_regularization'])
    )(dropout)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense1)

    model = tf.keras.Model(
        inputs=[
            api_atom_input, api_adj_input, api_edge_input,
            ccf_atom_input, ccf_adj_input, ccf_edge_input,
            extra_input
        ],
        outputs=output
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=hp_params['learning_rate'],
            weight_decay=1e-4
        ),
        loss=focal_dice_loss_with_label_smoothing(
            alpha=0.5,
            gamma=1.5,
            label_smoothing=hp_params['label_smoothing']
        ),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            F1Score(name='f1_score')
        ]
    )
    return model

class EnsembleModel:
    def __init__(self, models=None, scalers=None, weights=None):
        self.models = models if models is not None else []
        self.scalers = scalers if scalers is not None else []
        self.weights = weights

    def _calculate_weights(self, fold_val_metrics, weight_metric='f1'):
        if not fold_val_metrics or len(fold_val_metrics) != len(self.models):
            raise ValueError("指标数量必须与模型数量匹配！")
        metric_values = [fold[weight_metric] for fold in fold_val_metrics]
        metric_values = np.array(metric_values)
        if np.sum(metric_values) == 0:
            weights = np.ones_like(metric_values) / len(metric_values)
        else:
            weights = metric_values / np.sum(metric_values)
        self.weights = weights
        return weights

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

    def save_ensemble(self, save_dir):
        import shutil
        sub_models_dir = os.path.join(save_dir, "sub_models")
        if os.path.exists(sub_models_dir):
            shutil.rmtree(sub_models_dir)
        os.makedirs(sub_models_dir, exist_ok=True)
        scalers_dir = os.path.join(save_dir, "scalers")
        if os.path.exists(scalers_dir):
            shutil.rmtree(scalers_dir)
        os.makedirs(scalers_dir, exist_ok=True)

        for idx, model in enumerate(self.models):
            model_path = os.path.join(sub_models_dir, f"fold_{idx + 1}_model.keras")
            model.save(model_path)

        import joblib
        for idx, fold_scalers in enumerate(self.scalers):
            joblib.dump(fold_scalers['api_desc_scaler'],
                        os.path.join(scalers_dir, f"fold_{idx + 1}_api_desc_scaler.pkl"))
            joblib.dump(fold_scalers['ccf_desc_scaler'],
                        os.path.join(scalers_dir, f"fold_{idx + 1}_ccf_desc_scaler.pkl"))
            joblib.dump(fold_scalers['pair_feat_scaler'],
                        os.path.join(scalers_dir, f"fold_{idx + 1}_pair_feat_scaler.pkl"))

        config = {
            "n_models": len(self.models),
            "sub_models_dir": sub_models_dir,
            "scalers_dir": scalers_dir,
            "ensemble_type": "weighted_average_probability",
            "base_seed": BASE_SEED,
            "weights": self.weights.tolist() if self.weights is not None else None
        }
        config_path = os.path.join(save_dir, "ensemble_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)

        print(f"✅ 加权集成模型已保存到：{save_dir}")
        print(f"   - 子模型数量：{len(self.models)}")
        print(f"   - 子模型路径：{sub_models_dir}")
        print(f"   - Scaler路径：{scalers_dir}")
        print(f"   - 配置文件：{config_path}")
        if self.weights is not None:
            print(f"   - 模型权重：{self.weights}")

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
                "INN_GraphConvLayer": INN_GraphConvLayer,
                "INN_HybridGraphLayer": INN_HybridGraphLayer,
                "INN_GraphAttentionPoolLayer": INN_GraphAttentionPoolLayer,
                "INN_MolCrossAttentionLayer": INN_MolCrossAttentionLayer,
                "INN_MolPairMemoryLayer": INN_MolPairMemoryLayer,
                "F1Score": F1Score,
                "focal_dice_loss_with_label_smoothing": focal_dice_loss_with_label_smoothing(alpha=0.5, gamma=1.5)
            }
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            models.append(model)
        scalers_dir = config["scalers_dir"]
        scalers = []
        import joblib
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

def compute_descriptor_importance(ensemble_model, data_loader, fold_idx=0):
    """
    计算描述符重要性（Grad-CAM风格），只返回重要性数组和名称列表，不绘图
    """
    fold = data_loader.get_cv_folds()[fold_idx]
    val_graph_feats = fold['val_graph_feats']
    val_extra_feats = fold['val_extra_feats']
    val_labels = fold['val_labels']
    model = ensemble_model.models[fold_idx]

    all_desc_names = [
        'HBA_HBD', 'API_homo-CCF_lumo', 'CCF_homo-API_lumo', 'RBN',
        'S', 'S/L', 'S/M', 'M/L', 'Fr_NO',
        'Fr_aromaticAtom', 'XLogP3', 'TPSA', 'ACD/LogP',
        'MV', 'Polarizability', 'Dipole_Moment'
    ]

    @tf.function
    def get_gradients(inputs, label):
        with tf.GradientTape() as tape:
            tape.watch(inputs[-1])
            pred = model(inputs, training=False)
            loss = tf.keras.losses.binary_crossentropy(label, tf.squeeze(pred))
        grads = tape.gradient(loss, inputs[-1])
        return grads, pred

    total_grads = np.zeros(EXTRA_FEAT_SIZE)
    total_preds = []
    preloaded = preload_all_data(val_graph_feats, val_extra_feats, val_labels)
    test_gen = val_test_data_generator(preloaded, batch_size=BATCH_SIZE)
    steps = len(preloaded) // BATCH_SIZE + 1

    for _ in range(steps):
        try:
            X, y = next(test_gen)
            grads, preds = get_gradients(X, y)
            total_grads += np.sum(np.abs(grads), axis=0)
            total_preds.extend(preds.numpy().flatten())
        except StopIteration:
            break

    total_grads = total_grads / np.sum(total_grads)
    desc_importance = total_grads[:16]
    return desc_importance, all_desc_names

def save_best_hyperparams(params, save_path):
    serializable_params = {}
    for k, v in params.items():
        if isinstance(v, np.generic):
            serializable_params[k] = v.item()
        else:
            serializable_params[k] = v
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_params, f, indent=4, ensure_ascii=False)
    print(f"✅ 最优超参数已保存到：{save_path}")

def main():
    fixed_best_params = None

    print(f"\n{'=' * 80}")
    print(f"【阶段1】执行超参数优化")
    print(f"{'=' * 80}")

    clean_memory(verbose=True)

    data_loader_init = CrossValCocrystalDataset(EXCEL_PATH)
    cv_folds_init = data_loader_init.get_cv_folds()
    data_stats = data_loader_init.get_data_statistics()

    space = {
        'gcn_layer1_size': hp.choice('gcn1', [64, 128, 256]),
        'gcn_layer2_size': hp.choice('gcn2', [128, 256, 512]),
        'dense_layer_size': hp.choice('dense', [64, 128, 256]),
        'dropout_rate': hp.uniform('dropout', 0.2, 0.4),
        'l2_regularization': hp.loguniform('l2', np.log(1e-3), np.log(1e-2)),
        'learning_rate': hp.loguniform('lr', np.log(5e-5), np.log(2e-3)),
        'num_heads': hp.choice('heads', [2, 4]),
        'label_smoothing': hp.uniform('label_smoothing', 0.0, 0.2)
    }

    def black_box_function(args_dict):
        print(f"\nTesting hyperparameters: {args_dict}")
        cv_metrics = {'roc_auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for fold in cv_folds_init[:N_FOLDS]:
            fold_idx = fold['fold_idx']
            print(f"\n=== Training fold {fold_idx + 1}/{N_FOLDS} (Hyperopt) ===")
            train_graph_feats_aug, train_extra_feats_aug, train_labels_aug = data_loader_init._augment_data(
                fold['train_graph_feats'],
                fold['train_extra_feats'],
                fold['train_labels'],
                fold_idx=fold_idx
            )
            train_preloaded = preload_all_data(
                train_graph_feats_aug,
                train_extra_feats_aug,
                train_labels_aug
            )
            val_preloaded = preload_all_data(
                fold['val_graph_feats'],
                fold['val_extra_feats'],
                fold['val_labels']
            )
            model = build_model(args_dict)
            _, _, _ = train_with_loss_recording(
                model, train_preloaded, val_preloaded,
                batch_size=BATCH_SIZE,
                epochs=TRAIN_EPOCHS,
                patience=EARLY_STOP_PATIENCE
            )
            del train_preloaded, val_preloaded
            del train_graph_feats_aug, train_extra_feats_aug, train_labels_aug

            val_metrics = calculate_metrics(
                model,
                fold['val_graph_feats'],
                fold['val_extra_feats'],
                fold['val_labels'],
                batch_size=BATCH_SIZE
            )
            cv_metrics['roc_auc'].append(val_metrics['auc'])
            cv_metrics['accuracy'].append(val_metrics['accuracy'])
            cv_metrics['precision'].append(val_metrics['precision'])
            cv_metrics['recall'].append(val_metrics['recall'])
            cv_metrics['f1'].append(val_metrics['f1'])

            print(f"Fold {fold_idx + 1} - AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")
            del model, val_metrics
            tf.keras.backend.clear_session()
            gc.collect(2)
            clean_memory(verbose=False)

        mean_metrics = {k: np.mean(v) for k, v in cv_metrics.items()}
        print(f"\nCV Average - AUC: {mean_metrics['roc_auc']:.4f}, F1: {mean_metrics['f1']:.4f}")
        clean_memory(verbose=False)
        return {'loss': -mean_metrics['f1'], 'status': STATUS_OK, 'eval_info': mean_metrics}

    print("\nStarting hyperparameter optimization (only once)...")
    trials = Trials()
    best = fmin(
        fn=black_box_function,
        space=space,
        algo=tpe.suggest,
        max_evals=MAX_EVALS,
        trials=trials,
        rstate=np.random.default_rng(HP_OPT_SEED)
    )

    fixed_best_params = {
        'gcn_layer1_size': [64, 128, 256][best['gcn1']],
        'gcn_layer2_size': [128, 256, 512][best['gcn2']],
        'dense_layer_size': [64, 128, 256][best['dense']],
        'dropout_rate': best['dropout'],
        'l2_regularization': best['l2'],
        'learning_rate': best['lr'],
        'num_heads': [2, 4][best['heads']],
        'label_smoothing': best['label_smoothing']
    }
    print("\nOptimization completed. Fixed best hyperparameters for all repeats:")
    save_best_hyperparams(fixed_best_params, HYPERPARAMS_SAVE_PATH)
    for param_name, param_value in fixed_best_params.items():
        if isinstance(param_value, float):
            if param_name in ['learning_rate', 'l2_regularization']:
                print(f"{param_name.replace('_', ' ').title()}: {param_value:.6f}")
            elif param_name == 'dropout_rate':
                print(f"{param_name.replace('_', ' ').title()}: {param_value:.4f}")
            else:
                print(f"{param_name.replace('_', ' ').title()}: {param_value:.3f}")
        else:
            print(f"{param_name.replace('_', ' ').title()}: {param_value}")

    del data_loader_init, cv_folds_init
    clean_memory()

    print(f"\n{'=' * 80}")
    print(f"【阶段2】训练交叉验证模型并保存")
    print(f"{'=' * 80}")

    data_loader = CrossValCocrystalDataset(EXCEL_PATH)
    cv_folds = data_loader.get_cv_folds()

    final_models = []
    fold_val_metrics = []
    fold_train_metrics = []

    for fold in cv_folds:
        fold_idx = fold['fold_idx']
        print(f"\n=== Training fold {fold_idx + 1}/{N_FOLDS} ===")

        train_graph_feats_aug, train_extra_feats_aug, train_labels_aug = data_loader._augment_data(
            fold['train_graph_feats'],
            fold['train_extra_feats'],
            fold['train_labels'],
            fold_idx=fold_idx
        )

        train_preloaded = preload_all_data(
            train_graph_feats_aug,
            train_extra_feats_aug,
            train_labels_aug
        )
        val_preloaded = preload_all_data(
            fold['val_graph_feats'],
            fold['val_extra_feats'],
            fold['val_labels']
        )

        model = build_model(fixed_best_params)
        _, _, _ = train_with_loss_recording(
            model, train_preloaded, val_preloaded,
            batch_size=BATCH_SIZE,
            epochs=TRAIN_EPOCHS,
            patience=EARLY_STOP_PATIENCE
        )

        del train_preloaded

        val_metrics = calculate_metrics(
            model,
            fold['val_graph_feats'],
            fold['val_extra_feats'],
            fold['val_labels'],
            batch_size=BATCH_SIZE,
        )
        fold_val_metrics.append(val_metrics)

        train_metrics = calculate_metrics(
            model,
            fold['train_graph_feats'],
            fold['train_extra_feats'],
            fold['train_labels'],
            batch_size=BATCH_SIZE
        )
        fold_train_metrics.append(train_metrics)

        final_models.append(model)

        print(f"Fold {fold_idx + 1} 训练集性能：")
        print(f"  准确率: {fold_train_metrics[-1]['accuracy']:.4f}, F1: {fold_train_metrics[-1]['f1']:.4f}")
        print(f"Fold {fold_idx + 1} 验证集性能：")
        print(f"  准确率: {fold_val_metrics[-1]['accuracy']:.4f}, F1: {fold_val_metrics[-1]['f1']:.4f}")

        del model, val_metrics, train_metrics, val_preloaded
        tf.keras.backend.clear_session()
        gc.collect(2)
        clean_memory(verbose=False)

    print(f"\n{'=' * 80}")
    print(f"【阶段3】构建并保存加权平均概率集成模型")
    print(f"{'=' * 80}")

    fold_scalers = []
    for fold in cv_folds:
        fold_scalers.append({
            'api_desc_scaler': fold['api_desc_scaler'],
            'ccf_desc_scaler': fold['ccf_desc_scaler'],
            'pair_feat_scaler': fold['pair_feat_scaler']
        })

    ensemble_model = EnsembleModel(models=final_models, scalers=fold_scalers)
    print("\n计算各折模型的权重（基于验证集F1）：")
    weights = ensemble_model._calculate_weights(fold_val_metrics, weight_metric='f1')
    print(f"各折模型权重：{weights}")
    print(f"权重和：{np.sum(weights):.4f}")
    ensemble_model.save_ensemble(MODEL_SAVE_DIR)

    all_oof_preds = []
    all_oof_true = []
    for fold_idx, fold in enumerate(cv_folds):
        print(f"\nProcessing OOF prediction for fold {fold_idx + 1}/{N_FOLDS}...")
        fold_model = final_models[fold_idx]
        val_graph_feats = fold['val_graph_feats']
        val_extra_feats = fold['val_extra_feats']
        val_labels = fold['val_labels']
        val_preloaded = preload_all_data(val_graph_feats, val_extra_feats, val_labels)
        val_gen = val_test_data_generator(val_preloaded, batch_size=BATCH_SIZE)
        val_steps = len(val_preloaded) // BATCH_SIZE + 1
        fold_preds = []
        fold_true = []
        for _ in range(val_steps):
            try:
                X, y = next(val_gen)
                pred = fold_model.predict(X, verbose=0)
                fold_preds.extend(pred.flatten())
                fold_true.extend(y.flatten())
            except StopIteration:
                break
        fold_preds = np.array(fold_preds[:len(val_labels)])
        fold_true = np.array(fold_true[:len(val_labels)]).astype(int)
        all_oof_preds.extend(fold_preds)
        all_oof_true.extend(fold_true)
        del fold_model, val_preloaded, val_gen, fold_preds, fold_true
        clean_memory(verbose=False)

    all_oof_preds = np.array(all_oof_preds)
    all_oof_true = np.array(all_oof_true).astype(int)

    ensemble_auc = roc_auc_score(all_oof_true, all_oof_preds) if len(np.unique(all_oof_true)) > 1 else 0.5
    ensemble_pred = (all_oof_preds > 0.5).astype(int)
    ensemble_acc = accuracy_score(all_oof_true, ensemble_pred)
    ensemble_precision = precision_score(all_oof_true, ensemble_pred, pos_label=1, average='binary', zero_division=0)
    ensemble_recall = recall_score(all_oof_true, ensemble_pred, pos_label=1, average='binary', zero_division=0)
    ensemble_f1 = f1_score(all_oof_true, ensemble_pred, pos_label=1, average='binary', zero_division=0)

    print(f"\n=== 修复后集成模型验证集性能（Out-of-Fold） ===")
    print(f"  准确率: {ensemble_acc:.4f}, 精确率(Precision): {ensemble_precision:.4f}")
    print(f"  召回率(Recall): {ensemble_recall:.4f}, F1: {ensemble_f1:.4f}, AUC: {ensemble_auc:.4f}")

    print("\n=== Computing Descriptor Importance ===")
    desc_importance, desc_names = compute_descriptor_importance(ensemble_model, data_loader, fold_idx=0)
    print("\n=== Top 5 Most Important Descriptors ===")
    top5_idx = np.argsort(desc_importance)[-5:][::-1]
    for i in top5_idx:
        print(f"{desc_names[i]}: {desc_importance[i]:.4f}")

    result_dir = "./train_results"
    os.makedirs(result_dir, exist_ok=True)

    np.save(os.path.join(result_dir, "all_oof_preds.npy"), all_oof_preds)
    np.save(os.path.join(result_dir, "all_oof_true.npy"), all_oof_true)

    with open(os.path.join(result_dir, "fold_train_metrics.pkl"), "wb") as f:
        pickle.dump(fold_train_metrics, f)
    with open(os.path.join(result_dir, "fold_val_metrics.pkl"), "wb") as f:
        pickle.dump(fold_val_metrics, f)

    losses = [-trial['result']['loss'] for trial in trials.trials]
    np.save(os.path.join(result_dir, "hyperopt_losses.npy"), np.array(losses))

    with open(os.path.join(result_dir, "data_stats.pkl"), "wb") as f:
        pickle.dump(data_stats, f)

    np.save(os.path.join(result_dir, "desc_importance.npy"), desc_importance)
    with open(os.path.join(result_dir, "desc_names.pkl"), "wb") as f:
        pickle.dump(desc_names, f)

    np.save(os.path.join(result_dir, "extra_feats.npy"), data_loader.all_extra_feats)

    with open(os.path.join(result_dir, "fixed_best_params.pkl"), "wb") as f:
        pickle.dump(fixed_best_params, f)

    print(f"\n所有训练结果和中间数据已保存到：{result_dir}")
    print("请运行 visualize.py 生成图表。")

    del data_loader, cv_folds, final_models
    del fold_train_metrics, fold_val_metrics
    tf.keras.backend.clear_session()
    gc.collect(2)
    clean_memory(verbose=True)

if __name__ == "__main__":
    main()