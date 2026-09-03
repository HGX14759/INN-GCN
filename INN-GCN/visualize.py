import os
import sys
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, Dense
import deepchem as dc
from rdkit import Chem, RDLogger
from hyperopt import hp
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score
import copy
from rdkit.Chem import Draw
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from rdkit.Chem.Draw import rdMolDraw2D
import json
import joblib
import multiprocessing
import pickle

BASE_SEED = 42
EXCEL_PATH = r"D:\YWGJ\gjsj001.xlsx"
MODEL_SAVE_DIR = r"D:\project\contrast model\cocrystal_prediction_INN-GCNtp1"
VIS_SAVE_DIR = "./paper_visualizations"
os.makedirs(VIS_SAVE_DIR, exist_ok=True)

BATCH_SIZE = 32
N_FOLDS = 5
MAX_ATOM_NUM = 50
EXTRA_FEAT_SIZE = 16
DESC_FEAT_SIZE = 17

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
            self.static_gate = self.add_weight(shape=(1, 1, units), initializer=tf.keras.initializers.Constant(0.5),
                                               trainable=True, name='static_gate')
            self.dynamic_gate = tf.keras.layers.Dense(units, activation='sigmoid')
        elif self.gate_type == 'hard':
            self.gate = tf.keras.layers.Dense(units)
            self.gate_threshold = self.add_weight(shape=(1, 1, units), initializer=tf.keras.initializers.Constant(0.0),
                                                  trainable=True, name='gate_threshold')
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
        self.local_gcn = INN_GraphConvLayer(units, activation=activation, dropout_rate=dropout_rate, gate_type=gate_type)
        self.attn_dim_adapt = tf.keras.layers.Dense(units)
        self.global_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=units // num_heads, dropout=0.1)
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
        self.fuse_weight = self.add_weight(shape=(1, 1, self.units), initializer=tf.keras.initializers.Constant(0.5),
                                           trainable=True, name='fuse_weight')
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
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=units // num_heads, dropout=0.1)
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
    def __init__(self, units=128, num_heads=4, **kwargs):
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

class INN_MolPairMemoryLayer(Layer):
    def __init__(self, units=256, memory_scale=0.1, **kwargs):
        super(INN_MolPairMemoryLayer, self).__init__(**kwargs)
        self.units = units
        self.memory_scale = memory_scale

    def build(self, input_shape):
        input_units = input_shape[-1]
        self.units = input_units
        self.pair_interact_dense = Dense(self.units, activation='relu', kernel_initializer='he_normal',
                                         name='pair_interact_dense')
        self.enhance_gate = Dense(self.units, activation='sigmoid', kernel_initializer='glorot_uniform',
                                  name='enhance_gate')
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
        config.update({'units': self.units, 'memory_scale': self.memory_scale})
        return config

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

def atom_to_features(atom):
    atom_types = ['Cl', 'N', 'P', 'Br', 'Si', 'S', 'I', 'F', 'C', 'O', 'H']
    atom_symbol = atom.GetSymbol()
    atom_type_feat = [1.0 if t == atom_symbol else 0.0 for t in atom_types]
    hybrid_types = [Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP]
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
    max_valence = 6; max_degree = 6; max_h = 4; max_vdw = 2.0; max_atomic_num = 53
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

def pad_sequences(sequences, padding='post'):
    max_len = MAX_ATOM_NUM
    feat_dim = sequences[0].shape[1] if sequences else 48
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
            'label': float(labels[i]),
            'api_smiles': gf['api_smiles'],
            'ccf_smiles': gf['ccf_smiles']
        })
    return preloaded

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
        self._preprocess_all_data()

    def _load_raw_data(self):
        print(f"\nLoading raw data from {self.excel_path}...")
        df = pd.read_excel(self.excel_path, sheet_name='Sheet1')
        raw_data = []
        for index, row in df.iterrows():
            api_smiles = row['SMILES1']
            ccf_smiles = row['SMILES2']
            target = int(row['Target'])
            api_mol = Chem.MolFromSmiles(api_smiles)
            ccf_mol = Chem.MolFromSmiles(ccf_smiles)
            if api_mol is None or ccf_mol is None:
                continue
            api_atom_num = api_mol.GetNumAtoms()
            ccf_atom_num = ccf_mol.GetNumAtoms()
            if (api_atom_num < 3 or ccf_atom_num < 3 or api_atom_num > MAX_ATOM_NUM or ccf_atom_num > MAX_ATOM_NUM):
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
            api_17_desc_cols = ['API_RBN','API_S','API_S_L','API_S_M','API_M_L','API_Fr_NO','API_Fr_aromaticAtom',
                                'API_XLogP3','API_Topological Polar Surface Area','API_ACD/LogP','API_MV',
                                'API_Polarizability','API_Dipole Moment','API_HBA','API_HBD','API_homo','API_lumo']
            ccf_17_desc_cols = ['CCF_RBN','CCF_S','CCF_S_L','CCF_S_M','CCF_M_L','CCF_Fr_NO','CCF_Fr_aromaticAtom',
                                'CCF_XLogP3','CCF_Topological Polar Surface Area','CCF_ACD/LogP','CCF_MV',
                                'CCF_Polarizability','CCF_Dipole Moment','CCF_HBA','CCF_HBD','CCF_homo','CCF_lumo']
            api_17_desc = [float(row[col]) if pd.notna(row[col]) else 0.0 for col in api_17_desc_cols]
            ccf_17_desc = [float(row[col]) if pd.notna(row[col]) else 0.0 for col in ccf_17_desc_cols]
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
            pair_desc = calculate_complementary_features(api_desc, ccf_desc, api_hba_hbd_homo_lumo, ccf_hba_hbd_homo_lumo)
            extra_feat = pair_desc
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
        print(f"Loaded {len(raw_data)} valid raw samples.")
        return raw_data

    def _preprocess_all_data(self):
        self.all_graph_feats = [sample['graph_feat'] for sample in self.raw_data]
        self.all_extra_feats = np.array([sample['extra_feat'] for sample in self.raw_data], dtype=np.float32)
        self.all_labels = np.array([sample['target'] for sample in self.raw_data], dtype=np.int32)
        self.train_val_graph_feats = self.all_graph_feats
        self.train_val_extra_feats_raw = self.all_extra_feats.copy()
        self.train_val_labels_raw = self.all_labels.copy()
        print(f"Total samples: {len(self.train_val_graph_feats)}")

    def get_cv_folds(self):
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=BASE_SEED)
        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self.train_val_extra_feats_raw, self.train_val_labels_raw)):
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
                'api_desc_scaler': api_desc_scaler,
                'ccf_desc_scaler': ccf_desc_scaler,
                'pair_feat_scaler': pair_feat_scaler
            })
        return folds

class EnsembleModel:
    def __init__(self, models=None, scalers=None, weights=None):
        self.models = models if models is not None else []
        self.scalers = scalers if scalers is not None else []
        self.weights = weights

    @classmethod
    def load_ensemble(cls, save_dir):
        config_path = os.path.join(save_dir, "ensemble_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, 'r') as f:
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
            "focal_dice_loss_with_label_smoothing": focal_dice_loss_with_label_smoothing(alpha=0.5, gamma=1.5)
        }
        for idx in range(config["n_models"]):
            model_path = os.path.join(sub_models_dir, f"fold_{idx+1}_model.keras")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            models.append(model)
        scalers_dir = config["scalers_dir"]
        scalers = []
        for idx in range(config["n_models"]):
            api_scaler = joblib.load(os.path.join(scalers_dir, f"fold_{idx+1}_api_desc_scaler.pkl"))
            ccf_scaler = joblib.load(os.path.join(scalers_dir, f"fold_{idx+1}_ccf_desc_scaler.pkl"))
            pair_scaler = joblib.load(os.path.join(scalers_dir, f"fold_{idx+1}_pair_feat_scaler.pkl"))
            scalers.append({'api_desc_scaler': api_scaler, 'ccf_desc_scaler': ccf_scaler, 'pair_feat_scaler': pair_scaler})
        weights = np.array(config["weights"]) if config["weights"] is not None else None
        ensemble = cls(models=models, scalers=scalers, weights=weights)
        print(f"Loaded {len(models)} sub-models.")
        return ensemble

def get_sample_input_from_preloaded(sample_preloaded):
    api_atom_pad = pad_sequences([sample_preloaded['api_atom']], padding='post')
    api_adj_pad = pad_adj_matrices([sample_preloaded['api_adj']])
    api_edge_pad = pad_edge_feats([sample_preloaded['api_edge_feats']])
    ccf_atom_pad = pad_sequences([sample_preloaded['ccf_atom']], padding='post')
    ccf_adj_pad = pad_adj_matrices([sample_preloaded['ccf_adj']])
    ccf_edge_pad = pad_edge_feats([sample_preloaded['ccf_edge_feats']])
    extra_in = np.expand_dims(sample_preloaded['extra'], axis=0)
    return [api_atom_pad, api_adj_pad, api_edge_pad,
            ccf_atom_pad, ccf_adj_pad, ccf_edge_pad, extra_in]

def shap_only_extra_features(ensemble_model, preloaded_data, n_background=80, n_sample_explain=40,
                             save_dir="./paper_visualizations/shap"):
    os.makedirs(save_dir, exist_ok=True)
    model = ensemble_model.models[0]

    bg_idx = np.random.choice(np.arange(len(preloaded_data)), size=n_background, replace=False)
    bg_extra = np.array([preloaded_data[i]['extra'] for i in bg_idx])

    zero_api_atom = tf.zeros((1, MAX_ATOM_NUM, 48), dtype=tf.float32)
    zero_api_adj  = tf.zeros((1, MAX_ATOM_NUM, MAX_ATOM_NUM), dtype=tf.float32)
    zero_api_edge = tf.zeros((1, MAX_ATOM_NUM, MAX_ATOM_NUM, 4), dtype=tf.float32)
    zero_ccf_atom = tf.zeros((1, MAX_ATOM_NUM, 48), dtype=tf.float32)
    zero_ccf_adj  = tf.zeros((1, MAX_ATOM_NUM, MAX_ATOM_NUM), dtype=tf.float32)
    zero_ccf_edge = tf.zeros((1, MAX_ATOM_NUM, MAX_ATOM_NUM, 4), dtype=tf.float32)

    def predict_extra_only(extra_np):
        """
        extra_np: (N, 16)  批量化预测，但内部再分小批次以防止 OOM
        返回: (N,) 预测概率
        """
        internal_batch_size = 64
        num_samples = extra_np.shape[0]
        all_preds = []
        for start in range(0, num_samples, internal_batch_size):
            end = min(start + internal_batch_size, num_samples)
            batch_extra = extra_np[start:end]
            bsz = batch_extra.shape[0]
            api_atom_batch = tf.tile(zero_api_atom, [bsz, 1, 1])
            api_adj_batch = tf.tile(zero_api_adj, [bsz, 1, 1])
            api_edge_batch = tf.tile(zero_api_edge, [bsz, 1, 1, 1])
            ccf_atom_batch = tf.tile(zero_ccf_atom, [bsz, 1, 1])
            ccf_adj_batch = tf.tile(zero_ccf_adj, [bsz, 1, 1])
            ccf_edge_batch = tf.tile(zero_ccf_edge, [bsz, 1, 1, 1])
            x = [api_atom_batch, api_adj_batch, api_edge_batch,
                 ccf_atom_batch, ccf_adj_batch, ccf_edge_batch,
                 tf.constant(batch_extra, dtype=tf.float32)]
            pred = model(x, training=False)
            all_preds.append(pred.numpy().flatten())
        return np.concatenate(all_preds)

    explainer = shap.KernelExplainer(predict_extra_only, bg_extra)
    labels = np.array([d['label'] for d in preloaded_data])
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    n_per_class = n_sample_explain // 2
    if n_per_class == 0:
        n_per_class = 1

    selected_pos = np.random.choice(pos_idx, size=min(n_per_class, len(pos_idx)), replace=False)
    selected_neg = np.random.choice(neg_idx, size=min(n_per_class, len(neg_idx)), replace=False)

    test_idx = np.concatenate([selected_pos, selected_neg])
    np.random.shuffle(test_idx)
    test_extra = np.array([preloaded_data[i]['extra'] for i in test_idx])
    shap_values = explainer.shap_values(test_extra, nsamples=200)

    print(f"SHAP values - min: {shap_values.min():.4f}, max: {shap_values.max():.4f}, mean: {shap_values.mean():.4f}")

    desc_names_16 = ['HBA_HBD','API_homo‑CCF_lumo','CCF_homo‑API_lumo','RBN_diff',
                     'S_diff','S/L_diff','S/M_diff','M/L_diff','Fr_NO_diff',
                     'Fr_aromaticAtom_diff','XLogP3_diff','TPSA_diff','ACD/LogP_diff',
                     'MV_diff','Polarizability_diff','Dipole_Moment_diff']
    records = []
    for i, s_idx in enumerate(test_idx):
        samp = preloaded_data[s_idx]
        x_in = get_sample_input_from_preloaded(samp)
        prob = float(model.predict(x_in, verbose=0)[0,0])
        row = {"sample_index": int(s_idx), "true_label": samp['label'], "pred_prob": prob}
        for feat_i in range(EXTRA_FEAT_SIZE):
            row[f"feat_{feat_i}"] = float(shap_values[i, feat_i])
        records.append(row)
    extra_df = pd.DataFrame(records)
    extra_df.to_csv(os.path.join(save_dir, "shap_extra_features.csv"), index=False, encoding="utf-8-sig")

    plt.figure(figsize=(14, 9))
    shap.summary_plot(shap_values, test_extra, feature_names=desc_names_16,
                      show=False, max_display=20, cmap="coolwarm", color_bar_label="Feature value")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_summary_dotplot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 8))
    shap.summary_plot(shap_values, test_extra, feature_names=desc_names_16,
                      plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_bar_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    atom_records = []
    for s_idx in test_idx:
        sample = preloaded_data[s_idx]
        x_in = get_sample_input_from_preloaded(sample)
        api_atom_tf = tf.constant(x_in[0])
        api_adj_tf = tf.constant(x_in[1])
        api_edge_tf = tf.constant(x_in[2])
        ccf_atom_tf = tf.constant(x_in[3])
        ccf_adj_tf = tf.constant(x_in[4])
        ccf_edge_tf = tf.constant(x_in[5])
        extra_tf = tf.constant(x_in[6])
        with tf.GradientTape() as tape:
            tape.watch([api_atom_tf, ccf_atom_tf])
            pred = model([api_atom_tf, api_adj_tf, api_edge_tf,
                          ccf_atom_tf, ccf_adj_tf, ccf_edge_tf, extra_tf], training=False)
        grad_api, grad_ccf = tape.gradient(pred, [api_atom_tf, ccf_atom_tf])
        n_api_real = sample['api_atom'].shape[0]
        n_ccf_real = sample['ccf_atom'].shape[0]
        api_sum_grad = np.sum(np.abs(grad_api.numpy()[0,:n_api_real]), axis=1)
        ccf_sum_grad = np.sum(np.abs(grad_ccf.numpy()[0,:n_ccf_real]), axis=1)
        for atom_i, val in enumerate(api_sum_grad):
            atom_records.append({"sample_idx": int(s_idx), "mol_type":"API","atom_id":atom_i,"grad_attribution":float(val)})
        for atom_i, val in enumerate(ccf_sum_grad):
            atom_records.append({"sample_idx": int(s_idx), "mol_type":"CCF","atom_id":atom_i,"grad_attribution":float(val)})
    atom_df = pd.DataFrame(atom_records)
    atom_df.to_csv(os.path.join(save_dir, "grad_atom_attribution.csv"), index=False, encoding="utf-8-sig")

    print(f"✅ SHAP + 梯度原子归因完成，输出：{save_dir}")
    return explainer, shap_values, extra_df, atom_df

def gnn_explainer_inngcn_single_sample(model, sample_preloaded, epochs_explainer=200, lr=0.005,
                                       save_dir="./paper_visualizations/gnnexplainer", suffix=""):
    os.makedirs(save_dir, exist_ok=True)
    sample_input = get_sample_input_from_preloaded(sample_preloaded)
    api_atom_ori, api_adj_ori, api_edge_ori, ccf_atom_ori, ccf_adj_ori, ccf_edge_ori, extra_ori = sample_input
    n_api = sample_preloaded['api_atom'].shape[0]
    n_ccf = sample_preloaded['ccf_atom'].shape[0]

    api_atom_mask_var = tf.Variable(tf.random.normal((1, MAX_ATOM_NUM,1), mean=0.0, stddev=0.01), trainable=True)
    ccf_atom_mask_var = tf.Variable(tf.random.normal((1, MAX_ATOM_NUM,1), mean=0.0, stddev=0.01), trainable=True)
    api_edge_mask_var = tf.Variable(tf.random.normal((1, MAX_ATOM_NUM, MAX_ATOM_NUM,1), mean=0.0, stddev=0.01), trainable=True)
    ccf_edge_mask_var = tf.Variable(tf.random.normal((1, MAX_ATOM_NUM, MAX_ATOM_NUM,1), mean=0.0, stddev=0.01), trainable=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    orig_pred = model(sample_input, training=False)

    for ep in range(epochs_explainer):
        with tf.GradientTape() as tape:
            api_atom_mask = tf.sigmoid(api_atom_mask_var)
            ccf_atom_mask = tf.sigmoid(ccf_atom_mask_var)
            api_edge_mask = tf.sigmoid(api_edge_mask_var)
            ccf_edge_mask = tf.sigmoid(ccf_edge_mask_var)
            api_atom_masked = api_atom_ori * api_atom_mask
            api_edge_masked = api_edge_ori * api_edge_mask
            ccf_atom_masked = ccf_atom_ori * ccf_atom_mask
            ccf_edge_masked = ccf_edge_ori * ccf_edge_mask
            x_masked = [api_atom_masked, api_adj_ori, api_edge_masked,
                        ccf_atom_masked, ccf_adj_ori, ccf_edge_masked, extra_ori]
            pred_masked = model(x_masked, training=False)
            pred_loss = tf.keras.losses.binary_crossentropy(orig_pred, pred_masked)
            ent_api_atom = - tf.reduce_sum(api_atom_mask * tf.math.log(api_atom_mask + 1e-8))
            ent_ccf_atom = - tf.reduce_sum(ccf_atom_mask * tf.math.log(ccf_atom_mask + 1e-8))
            ent_api_edge = - tf.reduce_sum(api_edge_mask * tf.math.log(api_edge_mask + 1e-8))
            ent_ccf_edge = - tf.reduce_sum(ccf_edge_mask * tf.math.log(ccf_edge_mask + 1e-8))
            loss = pred_loss + 0.03*(ent_api_atom+ent_ccf_atom+ent_api_edge+ent_ccf_edge)
        grads = tape.gradient(loss, [api_atom_mask_var,ccf_atom_mask_var,api_edge_mask_var,ccf_edge_mask_var])
        optimizer.apply_gradients(zip(grads,[api_atom_mask_var,ccf_atom_mask_var,api_edge_mask_var,ccf_edge_mask_var]))
        if ep % 30 == 0:
            print(f"[GNNExplainer epoch {ep}] loss={float(loss):.4f}, orig_pred={float(orig_pred):.4f}, masked_pred={float(pred_masked):.4f}")

    api_atom_mask_np = tf.sigmoid(api_atom_mask_var).numpy()[0,:,0]
    ccf_atom_mask_np = tf.sigmoid(ccf_atom_mask_var).numpy()[0,:,0]
    api_edge_mask_np = tf.sigmoid(api_edge_mask_var).numpy()[0,:,:,0]
    ccf_edge_mask_np = tf.sigmoid(ccf_edge_mask_var).numpy()[0,:,:,0]

    api_mol = Chem.MolFromSmiles(sample_preloaded['api_smiles'])
    ccf_mol = Chem.MolFromSmiles(sample_preloaded['ccf_smiles'])

    def draw_mol_highlight(mol, atom_weights, real_n, out_png):
        d2d = rdMolDraw2D.MolDraw2DCairo(800, 600)
        aw = atom_weights[:real_n]
        w_min = np.min(aw)
        w_max = np.max(aw)
        if abs(w_max - w_min) < 1e-6:
            norm_weights = np.ones_like(aw)
        else:
            norm_weights = (aw - w_min) / (w_max - w_min)
        hit_atoms = list(range(real_n))
        atom_colors = {}
        for i in hit_atoms:
            v = float(norm_weights[i])
            if v < 0.5:
                r = 2*v; g = 2*v; b = 1.0
            else:
                r = 1.0; g = 2*(1-v); b = 2*(1-v)
            atom_colors[i] = (r, g, b)
        d2d.DrawMolecule(mol, highlightAtoms=hit_atoms, highlightAtomColors=atom_colors)
        d2d.FinishDrawing()
        with open(out_png, "wb") as f:
            f.write(d2d.GetDrawingText())

    def _get_atom_colors(weights, real_n):
        """从权重生成原子颜色字典（归一化后映射到蓝-红渐变）"""
        aw = weights[:real_n]
        if len(aw) == 0:
            return {}
        w_min, w_max = np.min(aw), np.max(aw)
        if abs(w_max - w_min) < 1e-6:
            norm_weights = np.ones_like(aw)
        else:
            norm_weights = (aw - w_min) / (w_max - w_min)
        colors = {}
        for i in range(real_n):
            v = float(norm_weights[i])
            if v < 0.5:
                r, g, b = 2 * v, 2 * v, 1.0
            else:
                r, g, b = 1.0, 2 * (1 - v), 2 * (1 - v)
            colors[i] = (r, g, b)
        return colors

    api_colors = _get_atom_colors(api_atom_mask_np, n_api)
    ccf_colors = _get_atom_colors(ccf_atom_mask_np, n_ccf)

    api_highlight_atoms = list(range(n_api))
    ccf_highlight_atoms = list(range(n_ccf))

    combined_img = Draw.MolsToGridImage(
        [api_mol, ccf_mol],
        molsPerRow=2,
        subImgSize=(500, 400),
        highlightAtomLists=[api_highlight_atoms, ccf_highlight_atoms],
        highlightAtomColors=[api_colors, ccf_colors],
        legends=['API', 'CCF']
    )

    out_combined = os.path.join(save_dir, f"combined_mol_explain_{suffix}.png")
    combined_img.save(out_combined)
    print(f"✅ GNNExplainer完成，合并分子图保存：{out_combined}")
    return {
        "api_atom_mask": api_atom_mask_np,
        "ccf_atom_mask": ccf_atom_mask_np,
        "api_edge_mask": api_edge_mask_np,
        "ccf_edge_mask": ccf_edge_mask_np,
        "original_pred": float(orig_pred),
        "masked_pred": float(pred_masked)
    }

def run_explanation_pipeline(ensemble_model, val_preloaded_all):
    print("\n" + "="*70)
    print("📌 开始执行模型解释：SHAP(extra features)+梯度原子归因 + 自定义GNNExplainer")
    print("="*70)

    explainer, shap_values, extra_df, atom_df = shap_only_extra_features(
        ensemble_model, val_preloaded_all, n_background=100, n_sample_explain=60
    )

    labels = np.array([d['label'] for d in val_preloaded_all])
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    n_per = 4
    selected_pos = np.random.choice(pos_idx, size=min(n_per, len(pos_idx)), replace=False)
    selected_neg = np.random.choice(neg_idx, size=min(n_per, len(neg_idx)), replace=False)

    explain_sample_idx = np.concatenate([selected_pos, selected_neg])
    np.random.shuffle(explain_sample_idx)
    for s_idx in explain_sample_idx:
        sample = val_preloaded_all[s_idx]
        _ = gnn_explainer_inngcn_single_sample(
            ensemble_model.models[0], sample, epochs_explainer=200, suffix=f"sample_{s_idx}"
        )
    print("\n✅全部解释任务完成！图片&csv输出在 ./paper_visualizations/shap 和 ./paper_visualizations/gnnexplainer")
    return

RESULT_DIR = "./train_results"

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.size'] = 10

def plot_data_distribution(stats, save_name='data_distribution.png'):
    save_path = os.path.join(VIS_SAVE_DIR, save_name)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax1 = axes[0]
    stats['target_distribution'].plot(kind='bar', ax=ax1, color=['#3498DB', '#E74C3C'], edgecolor='black', linewidth=1)
    ax1.set_title('Sample Label Distribution', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Class Label', fontsize=11)
    ax1.set_ylabel('Number of Samples', fontsize=11)
    ax1.tick_params(axis='x', rotation=0)
    for i, v in enumerate(stats['target_distribution']):
        ax1.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=10)

    ax2 = axes[1]
    ax2.hist(stats['api_atom_dist'], bins=15, color='#9B59B6', edgecolor='black', alpha=0.7)
    ax2.set_title('API Molecule Atom Count Distribution', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Number of Atoms', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.axvline(np.mean(stats['api_atom_dist']), color='red', linestyle='--',
                label=f'Mean: {np.mean(stats["api_atom_dist"]):.1f}')
    ax2.legend(fontsize=9)

    ax3 = axes[2]
    ax3.hist(stats['ccf_atom_dist'], bins=15, color='#F39C12', edgecolor='black', alpha=0.7)
    ax3.set_title('CCF Molecule Atom Count Distribution', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Number of Atoms', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.axvline(np.mean(stats['ccf_atom_dist']), color='red', linestyle='--',
                label=f'Mean: {np.mean(stats["ccf_atom_dist"]):.1f}')
    ax3.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_fold_metrics_boxplot(fold_metrics_df, save_name='fold_metrics_boxplot.png'):
    save_path = os.path.join(VIS_SAVE_DIR, save_name)
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    plt.figure(figsize=(14, 8))
    boxplot = plt.boxplot([fold_metrics_df[metric] for metric in metrics],
                          labels=labels, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2),
                          whiskerprops=dict(color='black'),
                          capprops=dict(color='black'))
    colors = ['#2ECC71', '#3498DB', '#E74C3C', '#F39C12', '#9B59B6']
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, metric in enumerate(metrics):
        mean_val = fold_metrics_df[metric].mean()
        plt.scatter(i + 1, mean_val, color='black', s=100, marker='*', label='Mean' if i == 0 else "", zorder=5)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title('Cross-Validation Metrics Distribution Across Folds', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.legend(fontsize=11)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_hyperopt_progress(losses, save_name='hyperopt_progress.png'):
    save_path = os.path.join(VIS_SAVE_DIR, save_name)
    iterations = list(range(1, len(losses) + 1))
    plt.figure(figsize=(12, 7))
    plt.plot(iterations, losses, color='#9B59B6', linewidth=2, marker='o', markersize=6)
    best_idx = np.argmax(losses)
    plt.scatter(iterations[best_idx], losses[best_idx],
                color='#E74C3C', s=200, zorder=5, edgecolor='black',
                label=f'Best Iteration = {iterations[best_idx]}\nBest F1 = {losses[best_idx]:.4f}')
    plt.xlabel('Hyperopt Iteration', fontsize=12)
    plt.ylabel('Cross-Validation F1 Score', fontsize=12)
    plt.title('Hyperparameter Optimization Progress (TPE Algorithm)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(0, len(iterations) + 1)
    plt.ylim(np.min(losses) - 0.02, np.max(losses) + 0.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pr_curve(all_metrics, best_metrics, best_threshold, save_name='pr_curve.png'):
    save_path = os.path.join(VIS_SAVE_DIR, save_name)
    plt.figure(figsize=(10, 7))
    plt.plot(all_metrics['recall'], all_metrics['precision'],
             label='PR Curve', color='#2E86AB', linewidth=2)
    plt.scatter(best_metrics['recall'], best_metrics['precision'],
                color='#A23B72', s=150, zorder=5, edgecolor='black',
                label=f'F1 = {best_metrics["f1"]:.4f} (Threshold=0.5)')
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
    plt.title('Precision-Recall Curve (Test Set - Mean Val Threshold)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, save_name='roc_curve.png'):
    save_path = os.path.join(VIS_SAVE_DIR, save_name)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    best_threshold_idx = np.argmax(tpr - fpr)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='#F18F01', linewidth=2,
             label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='#C73E1D', linewidth=1.5, linestyle='--', label='Random Guess')
    plt.scatter(fpr[best_threshold_idx], tpr[best_threshold_idx],
                color='#000000', s=150, zorder=5, edgecolor='white',
                label=f'Threshold = 0.5')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve (Test Set)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_train_val_metrics_comparison(train_metrics_list, val_metrics_list,
                                      save_name='train_val_metrics_comparison.png'):
    save_path = os.path.join(VIS_SAVE_DIR, save_name)
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    train_vals = {m: [fold[m] for fold in train_metrics_list] for m in metrics}
    val_vals = {m: [fold[m] for fold in val_metrics_list] for m in metrics}
    train_means = [np.mean(train_vals[m]) for m in metrics]
    val_means = [np.mean(val_vals[m]) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width / 2, train_means, width,
                    label='Training Set', color='#3498DB', alpha=0.8, edgecolor='black')
    rects2 = ax.bar(x + width / 2, val_means, width,
                    label='Validation Set', color='#E74C3C', alpha=0.8, edgecolor='black')
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Training vs Validation Metrics (Cross-Validation Average)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim(0, 1.1)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    autolabel(rects1)
    autolabel(rects2)
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_train_val_trend(fold_train_metrics, fold_val_metrics, save_name='train_val_trend.png'):
    save_path = os.path.join(VIS_SAVE_DIR, save_name)
    folds = list(range(1, len(fold_train_metrics) + 1))
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        train_vals = [fold[metric] for fold in fold_train_metrics]
        val_vals = [fold[metric] for fold in fold_val_metrics]
        ax.plot(folds, train_vals, marker='o', linewidth=2, markersize=6,
                color='#3498DB', label='Training', alpha=0.8)
        ax.plot(folds, val_vals, marker='s', linewidth=2, markersize=6,
                color='#E74C3C', label='Validation', alpha=0.8)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Fold Number', fontsize=10)
        ax.set_ylabel('Metric Value', fontsize=10)
        ax.set_xticks(folds)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.1)
    axes[-1].axis('off')
    fig.suptitle('Training vs Validation Metrics Trend Across Folds', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_descriptor_importance(desc_importance, desc_names, save_name='descriptor_importance.png'):
    """绘制描述符重要性柱状图（从高到低排序）"""
    save_path = os.path.join(VIS_SAVE_DIR, save_name)
    sorted_indices = np.argsort(desc_importance)[::-1]
    sorted_importance = desc_importance[sorted_indices]
    sorted_names = [desc_names[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(sorted_names))
    bars = ax.bar(x, sorted_importance, color='#9B59B6', alpha=0.8, edgecolor='black')
    ax.set_ylabel('Normalized Importance (Gradient Weight)', fontsize=12)
    ax.set_title('Descriptor Importance (API + CCF) - Gradient-Based (Sorted Descending)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    # 标注前三
    for i in range(min(3, len(sorted_importance))):
        ax.annotate(f'{sorted_importance[i]:.3f}',
                    xy=(i, sorted_importance[i]),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color='red')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_descriptor_correlation_heatmap(extra_feats, save_name='descriptor_correlation.png'):
    save_path = os.path.join(VIS_SAVE_DIR, save_name)
    desc_names = [
        'HBA_HBD', 'API_homo-CCF_lumo', 'CCF_homo-API_lumo', 'RBN_diff',
        'S_diff', 'S/L_diff', 'S/M_diff', 'M/L_diff', 'Fr_NO_diff',
        'Fr_aromaticAtom_diff', 'XLogP3_diff', 'TPSA_diff', 'ACD/LogP_diff',
        'MV_diff', 'Polarizability_diff', 'Dipole_Moment_diff'
    ]
    corr_matrix = np.corrcoef(extra_feats.T)
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        corr_matrix,
        xticklabels=desc_names,
        yticklabels=desc_names,
        cmap='coolwarm',
        center=0,
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        annot_kws={'size': 8}
    )
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title('Descriptor Correlation Heatmap (16D Pair Features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_ensemble_prediction_scatter(y_true, y_pred_proba, threshold, save_name='ensemble_prediction_scatter.png'):
    save_path = os.path.join(VIS_SAVE_DIR, save_name)
    plt.figure(figsize=(12, 8))
    pos_idx = y_true == 1
    neg_idx = y_true == 0
    plt.scatter(
        np.where(pos_idx)[0], y_pred_proba[pos_idx],
        color='#E74C3C', alpha=0.7, s=60, edgecolor='black', linewidth=0.5,
        label=f'Positive Samples (n={np.sum(pos_idx)})'
    )
    plt.scatter(
        np.where(neg_idx)[0], y_pred_proba[neg_idx],
        color='#3498DB', alpha=0.7, s=60, edgecolor='black', linewidth=0.5,
        label=f'Negative Samples (n={np.sum(neg_idx)})'
    )
    plt.axhline(y=threshold, color='#F39C12', linestyle='--', linewidth=2.5,
                label=f'Classification Threshold = {threshold}')
    plt.xlabel('Sample Index (Out-of-Fold)', fontsize=12, fontweight='bold')
    plt.ylabel('Ensemble Model Predicted Probability', fontsize=12, fontweight='bold')
    plt.title('Ensemble Model Prediction Probabilities (Out-of-Fold Validation)',
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.ylim(-0.05, 1.05)
    plt.xlim(-50, len(y_true) + 50)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Loading training results from", RESULT_DIR)
    all_oof_preds = np.load(os.path.join(RESULT_DIR, "all_oof_preds.npy"))
    all_oof_true = np.load(os.path.join(RESULT_DIR, "all_oof_true.npy"))
    with open(os.path.join(RESULT_DIR, "fold_train_metrics.pkl"), "rb") as f:
        fold_train_metrics = pickle.load(f)
    with open(os.path.join(RESULT_DIR, "fold_val_metrics.pkl"), "rb") as f:
        fold_val_metrics = pickle.load(f)
    hyperopt_losses = np.load(os.path.join(RESULT_DIR, "hyperopt_losses.npy"))
    with open(os.path.join(RESULT_DIR, "data_stats.pkl"), "rb") as f:
        data_stats = pickle.load(f)
    desc_importance = np.load(os.path.join(RESULT_DIR, "desc_importance.npy"))
    with open(os.path.join(RESULT_DIR, "desc_names.pkl"), "rb") as f:
        desc_names = pickle.load(f)
    extra_feats = np.load(os.path.join(RESULT_DIR, "extra_feats.npy"))
    with open(os.path.join(RESULT_DIR, "fixed_best_params.pkl"), "rb") as f:
        fixed_best_params = pickle.load(f)

    print("Data loaded successfully.")

    print("\n=== Generating Data Distribution ===")
    plot_data_distribution(data_stats)

    print("\n=== Generating Fold Metrics Boxplot ===")
    df_val = pd.DataFrame(fold_val_metrics)
    plot_fold_metrics_boxplot(df_val)

    print("\n=== Generating Hyperopt Progress ===")
    plot_hyperopt_progress(hyperopt_losses)

    print("\n=== Generating PR and ROC Curves ===")
    precision, recall, _ = precision_recall_curve(all_oof_true, all_oof_preds)
    all_metrics_pr = {'precision': precision, 'recall': recall}
    current_threshold = 0.5
    ensemble_pred = (all_oof_preds > current_threshold).astype(int)
    current_f1 = f1_score(all_oof_true, ensemble_pred, pos_label=1, average='binary', zero_division=0)
    best_metrics_pr = {
        'precision': precision_score(all_oof_true, ensemble_pred, pos_label=1, average='binary', zero_division=0),
        'recall': recall_score(all_oof_true, ensemble_pred, pos_label=1, average='binary', zero_division=0),
        'f1': current_f1
    }
    plot_pr_curve(all_metrics_pr, best_metrics_pr, best_threshold=current_threshold)
    plot_roc_curve(all_oof_true, all_oof_preds)

    print("\n=== Generating Train/Val Comparison ===")
    plot_train_val_metrics_comparison(fold_train_metrics, fold_val_metrics)
    plot_train_val_trend(fold_train_metrics, fold_val_metrics)

    print("\n=== Generating Descriptor Importance ===")
    plot_descriptor_importance(desc_importance, desc_names)

    print("\n=== Generating Descriptor Correlation Heatmap ===")
    plot_descriptor_correlation_heatmap(extra_feats)

    print("\n=== Generating Ensemble Prediction Scatter ===")
    plot_ensemble_prediction_scatter(all_oof_true, all_oof_preds, threshold=0.5)

    print("\nAll visualizations saved to:", VIS_SAVE_DIR)

    print("\n" + "="*70)
    print("Now running model explanation (SHAP + GNNExplainer)...")
    print("="*70)
    try:
        print("Loading ensemble model...")
        ensemble_model = EnsembleModel.load_ensemble(MODEL_SAVE_DIR)
        print("Model loaded successfully.")

        print("Preparing data for explanation...")
        data_loader = CrossValCocrystalDataset(EXCEL_PATH)
        cv_folds = data_loader.get_cv_folds()
        all_val_preloaded = []
        for fold in cv_folds:
            vp = preload_all_data(fold['val_graph_feats'], fold['val_extra_feats'], fold['val_labels'])
            all_val_preloaded.extend(vp)
        print(f"Total {len(all_val_preloaded)} validation samples for explanation.")

        run_explanation_pipeline(ensemble_model, all_val_preloaded)

    except Exception as e:
        print(f"⚠️ Explanation pipeline failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== All tasks completed. ===")

if __name__ == "__main__":
    main()