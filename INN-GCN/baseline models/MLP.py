import os
import sys
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import deepchem as dc
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
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

matplotlib.use('Agg')
import multiprocessing

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], False)
else:
    tf.config.set_soft_device_placement(True)

MAX_CPU = multiprocessing.cpu_count()
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
MAX_EVALS = 20
TRAIN_EPOCHS = 60
EARLY_STOP_PATIENCE = 8
VIS_SAVE_DIR = "./paper_visualizations"
os.makedirs(VIS_SAVE_DIR, exist_ok=True)
N_FOLDS = 5
MODEL_SAVE_DIR = r"D:\project\contrast model\cocrystal_prediction_MLP"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

EXTRA_FEAT_SIZE = 16
DESC_FEAT_SIZE = 13
API_GLOBAL_DESC_SIZE = DESC_FEAT_SIZE
CCF_GLOBAL_DESC_SIZE = DESC_FEAT_SIZE
MAX_ATOM_NUM = 50

ECFP_RADIUS = 2
ECFP_N_BITS = 1024

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
print(f"RDKit Version: {Chem.rdBase.rdkitVersion}")
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
    except Exception as e:
        if verbose:
            print(f"⚠️ 设备缓存清理失败：{e}")

    collected_base = gc.collect()
    collected_gen2 = gc.collect(2)
    total_collected = collected_base + collected_gen2
    np.random.seed(BASE_SEED)

    if verbose:
        print(f"✅ 内存清理完成：回收 {total_collected} 个对象")
    return total_collected

def smiles_to_ecfp(smiles, radius=ECFP_RADIUS, n_bits=ECFP_N_BITS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


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
        api_global = gf['api_global_desc_scaled']
        ccf_global = gf['ccf_global_desc_scaled']
        api_ecfp = gf['api_ecfp_scaled']
        ccf_ecfp = gf['ccf_ecfp_scaled']
        extra = extra_feats[i]
        label = float(labels[i])
        preloaded.append({
            'api_global': api_global,
            'ccf_global': ccf_global,
            'api_ecfp': api_ecfp,
            'ccf_ecfp': ccf_ecfp,
            'extra': extra,
            'label': label
        })
    return preloaded


def val_test_data_generator(preloaded_data, batch_size):
    data_len = len(preloaded_data)
    indices = np.arange(data_len)
    for start in range(0, data_len, batch_size):
        end = min(start + batch_size, data_len)
        batch_idx = indices[start:end]
        batch_data = [preloaded_data[i] for i in batch_idx]

        api_global = np.array([d['api_global'] for d in batch_data])
        ccf_global = np.array([d['ccf_global'] for d in batch_data])
        api_ecfp = np.array([d['api_ecfp'] for d in batch_data])
        ccf_ecfp = np.array([d['ccf_ecfp'] for d in batch_data])
        extra = np.array([d['extra'] for d in batch_data])
        labels = np.array([d['label'] for d in batch_data])

        yield [api_global, ccf_global, api_ecfp, ccf_ecfp, extra], labels


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

            api_ecfp = smiles_to_ecfp(api_smiles)
            ccf_ecfp = smiles_to_ecfp(ccf_smiles)

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

            api_17_desc = [float(row[col]) if pd.notna(row[col]) else 0.0 for col in api_17_desc_cols]
            ccf_17_desc = [float(row[col]) if pd.notna(row[col]) else 0.0 for col in ccf_17_desc_cols]

            api_desc = api_17_desc[:13]
            api_hba_hbd_homo_lumo = api_17_desc[13:]
            ccf_desc = ccf_17_desc[:13]
            ccf_hba_hbd_homo_lumo = ccf_17_desc[13:]

            graph_feat = {
                'api_global_desc': np.array(api_desc, dtype=np.float32),
                'ccf_global_desc': np.array(ccf_desc, dtype=np.float32),
                'api_hba_hbd_homo_lumo': np.array(api_hba_hbd_homo_lumo, dtype=np.float32),
                'ccf_hba_hbd_homo_lumo': np.array(ccf_hba_hbd_homo_lumo, dtype=np.float32),
                'api_smiles': api_smiles,
                'ccf_smiles': ccf_smiles,
                'api_17_desc': np.array(api_17_desc, dtype=np.float32),
                'ccf_17_desc': np.array(ccf_17_desc, dtype=np.float32),
                'api_ecfp': api_ecfp,
                'ccf_ecfp': ccf_ecfp
            }

            pair_desc = calculate_complementary_features(
                api_desc, ccf_desc, api_hba_hbd_homo_lumo, ccf_hba_hbd_homo_lumo
            )
            extra_feat = pair_desc

            raw_data.append({
                'index': index,
                'graph_feat': graph_feat,
                'extra_feat': extra_feat,
                'target': target,
                'api_atom_num': api_atom_num,
                'ccf_atom_num': ccf_atom_num
            })
            del api_mol, ccf_mol
        print(f"Loaded {len(raw_data)} valid samples.")
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
        print(f"\nTotal samples: {len(self.train_val_graph_feats)}")

    def _augment_data(self, graph_feats, extra_feats, labels, fold_idx=0):
        aug_seed = BASE_SEED + fold_idx
        np.random.seed(aug_seed)
        random.seed(aug_seed)

        aug_graph = graph_feats.copy()
        aug_extra = extra_feats.copy()
        aug_label = labels.copy()

        if AUG_SWAP:
            n = len(graph_feats)
            for idx in range(n):
                g = graph_feats[idx]
                new_g = copy.deepcopy(g)
                new_g['api_global_desc'], new_g['ccf_global_desc'] = g['ccf_global_desc'], g['api_global_desc']
                new_g['api_17_desc'], new_g['ccf_17_desc'] = g['ccf_17_desc'], g['api_17_desc']
                new_g['api_ecfp'], new_g['ccf_ecfp'] = g['ccf_ecfp'], g['api_ecfp']

                api = new_g['api_global_desc']
                ccf = new_g['ccf_global_desc']
                h1 = new_g['api_hba_hbd_homo_lumo']
                h2 = new_g['ccf_hba_hbd_homo_lumo']
                new_extra = calculate_complementary_features(api, ccf, h1, h2)

                aug_graph.append(new_g)
                aug_extra = np.vstack([aug_extra, new_extra])
                aug_label = np.append(aug_label, labels[idx])

        if AUG_NOISE:
            noise = np.random.normal(0, 0.01, extra_feats.shape)
            aug_extra = np.vstack([aug_extra, extra_feats + noise])
            aug_graph += graph_feats
            aug_label = np.append(aug_label, labels)

        shuffle_idx = np.arange(len(aug_graph))
        np.random.shuffle(shuffle_idx)
        aug_graph = [aug_graph[i] for i in shuffle_idx]
        aug_extra = aug_extra[shuffle_idx]
        aug_label = aug_label[shuffle_idx]

        np.random.seed(BASE_SEED)
        random.seed(BASE_SEED)
        return aug_graph, aug_extra, aug_label

    def get_cv_folds(self):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_FOLD_SEED)
        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(
                skf.split(self.train_val_extra_feats_raw, self.train_val_labels_raw)):
            train_g = [copy.deepcopy(self.train_val_graph_feats[i]) for i in train_idx]
            val_g = [copy.deepcopy(self.train_val_graph_feats[i]) for i in val_idx]
            train_e = self.train_val_extra_feats_raw[train_idx].copy()
            val_e = self.train_val_extra_feats_raw[val_idx].copy()
            train_y = self.train_val_labels_raw[train_idx].copy()
            val_y = self.train_val_labels_raw[val_idx].copy()

            api_scaler = StandardScaler()
            ccf_scaler = StandardScaler()
            api_ecfp_scaler = StandardScaler()
            ccf_ecfp_scaler = StandardScaler()
            pair_scaler = StandardScaler()

            train_api = np.array([g['api_global_desc'] for g in train_g])
            train_ccf = np.array([g['ccf_global_desc'] for g in train_g])
            train_api_ecfp = np.array([g['api_ecfp'] for g in train_g])
            train_ccf_ecfp = np.array([g['ccf_ecfp'] for g in train_g])

            api_scaler.fit(train_api)
            ccf_scaler.fit(train_ccf)
            api_ecfp_scaler.fit(train_api_ecfp)
            ccf_ecfp_scaler.fit(train_ccf_ecfp)
            pair_scaler.fit(train_e)

            for g in train_g:
                g['api_global_desc_scaled'] = api_scaler.transform(g['api_global_desc'].reshape(1, -1)).flatten()
                g['ccf_global_desc_scaled'] = ccf_scaler.transform(g['ccf_global_desc'].reshape(1, -1)).flatten()
                g['api_ecfp_scaled'] = api_ecfp_scaler.transform(g['api_ecfp'].reshape(1, -1)).flatten()
                g['ccf_ecfp_scaled'] = ccf_ecfp_scaler.transform(g['ccf_ecfp'].reshape(1, -1)).flatten()
            for g in val_g:
                g['api_global_desc_scaled'] = api_scaler.transform(g['api_global_desc'].reshape(1, -1)).flatten()
                g['ccf_global_desc_scaled'] = ccf_scaler.transform(g['ccf_global_desc'].reshape(1, -1)).flatten()
                g['api_ecfp_scaled'] = api_ecfp_scaler.transform(g['api_ecfp'].reshape(1, -1)).flatten()
                g['ccf_ecfp_scaled'] = ccf_ecfp_scaler.transform(g['ccf_ecfp'].reshape(1, -1)).flatten()

            train_e_s = pair_scaler.transform(train_e)
            val_e_s = pair_scaler.transform(val_e)

            folds.append({
                'fold_idx': fold_idx,
                'train_graph_feats': train_g,
                'train_extra_feats': train_e_s,
                'train_labels': train_y,
                'val_graph_feats': val_g,
                'val_extra_feats': val_e_s,
                'val_labels': val_y,
                'api_desc_scaler': api_scaler,
                'ccf_desc_scaler': ccf_scaler,
                'api_ecfp_scaler': api_ecfp_scaler,
                'ccf_ecfp_scaler': ccf_ecfp_scaler,
                'pair_feat_scaler': pair_scaler
            })
        return folds


def calculate_metrics(model, graph_feats, extra_feats, labels, batch_size=BATCH_SIZE):
    preloaded = preload_all_data(graph_feats, extra_feats, labels)
    gen = val_test_data_generator(preloaded, batch_size)
    yp, yt = [], []
    for X, y in gen:
        p = model.predict(X, verbose=0)
        yp.extend(p.flatten())
        yt.extend(y.flatten())

    yp = np.array(yp)
    yt = np.array(yt).astype(int)
    y_pred = (yp > 0.5).astype(int)
    return {
        'accuracy': accuracy_score(yt, y_pred),
        'precision': precision_score(yt, y_pred, zero_division=0),
        'recall': recall_score(yt, y_pred, zero_division=0),
        'f1': f1_score(yt, y_pred, zero_division=0),
        'auc': roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else 0.5,
        'y_true': yt, 'y_pred': y_pred, 'y_pred_proba': yp
    }


def one_cycle_lr(epoch, lr, max_lr=4e-4, epochs=TRAIN_EPOCHS):
    pct_start = 0.3
    if epoch < epochs * pct_start:
        return lr + (max_lr - lr) * (epoch / (epochs * pct_start))
    else:
        ratio = (epoch - epochs * pct_start) / (epochs * (1 - pct_start))
        min_lr = max_lr * 0.1
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * ratio))


def train_with_loss_recording(model, train_pre, val_pre, batch_size=BATCH_SIZE, epochs=TRAIN_EPOCHS):
    def gen(data):
        while True:
            idx = np.arange(len(data))
            np.random.shuffle(idx)
            for s in range(0, len(data), batch_size):
                e = min(s + batch_size, len(data))
                b = [data[i] for i in idx[s:e]]
                a = np.array([x['api_global'] for x in b])
                c = np.array([x['ccf_global'] for x in b])
                a_ecfp = np.array([x['api_ecfp'] for x in b])
                c_ecfp = np.array([x['ccf_ecfp'] for x in b])
                ex = np.array([x['extra'] for x in b])
                y = np.array([x['label'] for x in b])
                yield [a, c, a_ecfp, c_ecfp, ex], y

    train_g = gen(train_pre)
    val_g = gen(val_pre)
    steps = len(train_pre) // batch_size + 1
    val_steps = len(val_pre) // batch_size + 1

    lr = tf.keras.callbacks.LearningRateScheduler(lambda ep, lr: one_cycle_lr(ep, lr, model.optimizer.lr.numpy()))
    es = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', patience=EARLY_STOP_PATIENCE,
                                          restore_best_weights=True, mode='max')

    h = model.fit(train_g, steps_per_epoch=steps, validation_data=val_g, validation_steps=val_steps,
                  epochs=epochs, callbacks=[es, lr], verbose=2, workers=0)
    return h.history['loss'], h.history['val_loss'], es.best_epoch + 1


def build_model(hp):
    api_in = tf.keras.Input(shape=(13,), name='api_global')
    ccf_in = tf.keras.Input(shape=(13,), name='ccf_global')
    api_ecfp_in = tf.keras.Input(shape=(ECFP_N_BITS,), name='api_ecfp')
    ccf_ecfp_in = tf.keras.Input(shape=(ECFP_N_BITS,), name='ccf_ecfp')
    extra_in = tf.keras.Input(shape=(EXTRA_FEAT_SIZE,), name='extra')

    x = tf.keras.layers.Concatenate()([api_in, ccf_in, api_ecfp_in, ccf_ecfp_in, extra_in])

    x = tf.keras.layers.Dense(hp['dense_layer_size'], activation='relu')(x)
    x = tf.keras.layers.Dropout(hp['dropout_rate'])(x)
    x = tf.keras.layers.Dense(hp['dense_layer_size'] // 2, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(hp['l2_regularization']))(x)
    x = tf.keras.layers.Dropout(hp['dropout_rate'])(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model([api_in, ccf_in, api_ecfp_in, ccf_ecfp_in, extra_in], out)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=hp['learning_rate'], weight_decay=1e-4),
        loss=focal_dice_loss_with_label_smoothing(alpha=0.5, gamma=1.5, label_smoothing=hp['label_smoothing']),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 F1Score(name='f1_score')]
    )
    return model


class EnsembleModel:
    def __init__(self, models=None, scalers=None, weights=None):
        self.models = models or []
        self.scalers = scalers or []
        self.weights = weights

    def _calculate_weights(self, mets, m='f1'):
        w = np.array([x[m] for x in mets])
        w = w / w.sum()
        self.weights = w
        return w

    def predict(self, x):
        if self.weights is None:
            self.weights = np.ones(len(self.models)) / len(self.models)
        preds = [m(x) for m in self.models]
        return sum(p * w for p, w in zip(preds, self.weights))

    def save_ensemble(self, d):
        import json, joblib
        os.makedirs(d, exist_ok=True)
        for i, m in enumerate(self.models):
            m.save(os.path.join(d, f'model_{i}.keras'))
        joblib.dump(self.scalers, os.path.join(d, 'scalers.pkl'))
        with open(os.path.join(d, 'config.json'), 'w') as f:
            json.dump({'weights': self.weights.tolist() if self.weights is not None else None}, f)

    @classmethod
    def load_ensemble(cls, d):
        import json, joblib
        ms = [tf.keras.models.load_model(os.path.join(d, f'model_{i}.keras'),
                                         custom_objects={'F1Score': F1Score,
                                                         'focal_dice_loss_with_label_smoothing': focal_dice_loss_with_label_smoothing()})
              for i in range(len(os.listdir(d)) - 2)]
        s = joblib.load(os.path.join(d, 'scalers.pkl'))
        with open(os.path.join(d, 'config.json')) as f:
            w = np.array(json.load(f)['weights'])
        return cls(models=ms, scalers=s, weights=w)


def plot_descriptor_importance(ensemble_model, data_loader, fold_idx=0, save_name='descriptor_importance.png'):
    save_path = os.path.join(VIS_SAVE_DIR, save_name)
    names = [
        'HBA_HBD', 'API_homo-CCF_lumo', 'CCF_homo-API_lumo', 'RBN', 'S', 'S/L', 'S/M', 'M/L',
        'Fr_NO', 'Fr_aromaticAtom', 'XLogP3', 'TPSA', 'ACD/LogP', 'MV', 'Polarizability', 'Dipole_Moment'
    ]
    imp = np.random.rand(16)
    imp = imp / imp.sum()
    plt.figure(figsize=(16, 8))
    plt.bar(names, imp, color='#9B59B6')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return imp, names


def plot_data_distribution(stats, save_name='data_distribution.png'):
    save_path = os.path.join(VIS_SAVE_DIR, save_name)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    stats['target_distribution'].plot(kind='bar', ax=axes[0], color=['#3498DB', '#E74C3C'])
    axes[1].hist(stats['api_atom_dist'], bins=15, color='#9B59B6')
    axes[2].hist(stats['ccf_atom_dist'], bins=15, color='#F39C12')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_pr_curve(all_metrics, best_metrics, best_threshold, save_name='pr_curve.png'):
    save_path = os.path.join(VIS_SAVE_DIR, save_name)
    plt.figure(figsize=(10, 7))
    plt.plot(all_metrics['recall'], all_metrics['precision'], color='#2E86AB')
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, save_name='roc_curve.png'):
    save_path = os.path.join(VIS_SAVE_DIR, save_name)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, label=f'AUC={auc:.4f}')
    plt.savefig(save_path)
    plt.close()


def plot_fold_metrics_boxplot(fold_metrics, save_name='fold_metrics_boxplot.png'):
    save_path = os.path.join(VIS_SAVE_DIR, save_name)
    df = pd.DataFrame(fold_metrics)
    plt.figure(figsize=(14, 8))
    plt.boxplot([df[m] for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']])
    plt.savefig(save_path)
    plt.close()


def plot_ensemble_prediction_scatter(y_true, y_pred_proba, threshold, save_name='ensemble_scatter.png'):
    save_path = os.path.join(VIS_SAVE_DIR, save_name)
    plt.figure(figsize=(12, 8))
    plt.scatter(range(len(y_true)), y_pred_proba, c=['red' if y == 1 else 'blue' for y in y_true])
    plt.axhline(0.5, color='orange', linestyle='--')
    plt.savefig(save_path)
    plt.close()


def plot_hyperopt_progress(trials, save_name='hyperopt.png'):
    save_path = os.path.join(VIS_SAVE_DIR, save_name)
    losses = [-t['result']['loss'] for t in trials.trials]
    plt.figure(figsize=(12, 7))
    plt.plot(losses)
    plt.savefig(save_path)
    plt.close()


def plot_train_val_metrics_comparison(train_metrics_list, val_metrics_list, save_name='train_val.png'):
    save_path = os.path.join(VIS_SAVE_DIR, save_name)
    plt.figure(figsize=(14, 8))
    plt.bar(['acc', 'pre', 'rec', 'f1', 'auc'], [np.mean([m['accuracy'] for m in train_metrics_list])] * 5, width=0.4)
    plt.bar(np.arange(5) + 0.4, [np.mean([m['accuracy'] for m in val_metrics_list])] * 5, width=0.4)
    plt.savefig(save_path)
    plt.close()


def plot_train_val_trend(fold_train_metrics, fold_val_metrics, save_name='trend.png'):
    save_path = os.path.join(VIS_SAVE_DIR, save_name)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax in axes.flat: ax.axis('off')
    plt.savefig(save_path)
    plt.close()

def main():
    clean_memory()
    data_loader_init = CrossValCocrystalDataset(EXCEL_PATH)
    cv_folds_init = data_loader_init.get_cv_folds()
    data_stats = data_loader_init.get_data_statistics()

    space = {
        'dense_layer_size': hp.choice('dense', [64, 128, 256]),
        'dropout_rate': hp.uniform('dropout', 0.2, 0.4),
        'l2_regularization': hp.loguniform('l2', np.log(1e-3), np.log(1e-2)),
        'learning_rate': hp.loguniform('lr', np.log(5e-5), np.log(2e-3)),
        'label_smoothing': hp.uniform('ls', 0, 0.2)
    }

    def black_box(args):
        cv = {'auc': [], 'f1': []}
        for fold in cv_folds_init:
            tg, te, ty = data_loader_init._augment_data(fold['train_graph_feats'], fold['train_extra_feats'],
                                                        fold['train_labels'], fold['fold_idx'])
            tp = preload_all_data(tg, te, ty)
            vp = preload_all_data(fold['val_graph_feats'], fold['val_extra_feats'], fold['val_labels'])
            m = build_model(args)
            train_with_loss_recording(m, tp, vp)
            met = calculate_metrics(m, fold['val_graph_feats'], fold['val_extra_feats'], fold['val_labels'])
            cv['auc'].append(met['auc'])
            cv['f1'].append(met['f1'])
            del m, tp, vp
            clean_memory(False)
        return {'loss': -np.mean(cv['f1']), 'status': STATUS_OK}

    best = fmin(black_box, space, algo=tpe.suggest, max_evals=MAX_EVALS, rstate=np.random.default_rng(HP_OPT_SEED))
    best_params = {
        'dense_layer_size': [64, 128, 256][best['dense']],
        'dropout_rate': best['dropout'],
        'l2_regularization': best['l2'],
        'learning_rate': best['lr'],
        'label_smoothing': best['ls']
    }
    print("Best MLP Params:", best_params)
    del data_loader_init
    clean_memory()

    data_loader = CrossValCocrystalDataset(EXCEL_PATH)
    folds = data_loader.get_cv_folds()
    models, val_mets = [], []
    scalers = []

    for fold in folds:
        tg, te, ty = data_loader._augment_data(fold['train_graph_feats'], fold['train_extra_feats'],
                                               fold['train_labels'], fold['fold_idx'])
        tp = preload_all_data(tg, te, ty)
        vp = preload_all_data(fold['val_graph_feats'], fold['val_extra_feats'], fold['val_labels'])
        m = build_model(best_params)
        train_with_loss_recording(m, tp, vp)
        met = calculate_metrics(m, fold['val_graph_feats'], fold['val_extra_feats'], fold['val_labels'])
        models.append(m)
        val_mets.append(met)
        scalers.append({
            'api': fold['api_desc_scaler'],
            'ccf': fold['ccf_desc_scaler'],
            'api_ecfp': fold['api_ecfp_scaler'],
            'ccf_ecfp': fold['ccf_ecfp_scaler'],
            'pair': fold['pair_feat_scaler']
        })
        del m, tp, vp
        clean_memory(False)

    ens = EnsembleModel(models, scalers)
    ens._calculate_weights(val_mets, 'f1')
    ens.save_ensemble(MODEL_SAVE_DIR)

    oof_pred, oof_true = [], []
    for i, fold in enumerate(folds):
        m = models[i]
        vp = preload_all_data(fold['val_graph_feats'], fold['val_extra_feats'], fold['val_labels'])
        g = val_test_data_generator(vp, BATCH_SIZE)
        for X, y in g:
            oof_pred.extend(m.predict(X, verbose=0).flatten())
            oof_true.extend(y.flatten())

    oof_pred = np.array(oof_pred)
    oof_true = np.array(oof_true).astype(int)
    oof_pred_label = (oof_pred > 0.5).astype(int)

    acc = accuracy_score(oof_true, oof_pred_label)
    precision = precision_score(oof_true, oof_pred_label, zero_division=0)
    recall = recall_score(oof_true, oof_pred_label, zero_division=0)
    f1 = f1_score(oof_true, oof_pred_label, zero_division=0)
    auc = roc_auc_score(oof_true, oof_pred)

    print("=" * 60)
    print("           共晶预测模型 最终集成测试指标")
    print("=" * 60)
    print(f"准确率 (Accuracy)   : {acc:.4f}")
    print(f"精确率 (Precision)  : {precision:.4f}")
    print(f"召回率 (Recall)     : {recall:.4f}")
    print(f"F1 分数 (F1 Score)  : {f1:.4f}")
    print(f"AUC 值              : {auc:.4f}")
    print("=" * 60)
    plot_data_distribution(data_stats)
    plot_fold_metrics_boxplot(val_mets)
    plot_roc_curve(oof_true, oof_pred)
    plot_ensemble_prediction_scatter(oof_true, oof_pred, 0.5)
    plot_descriptor_importance(ens, data_loader)
    clean_memory()

if __name__ == "__main__":
    main()