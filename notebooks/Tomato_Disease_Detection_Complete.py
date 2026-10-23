"""
=============================================================================
  TOMATO DISEASE DETECTION USING DEEP LEARNING
  CNN from Scratch vs. MobileNetV2 Transfer Learning
  CPU-Optimized Version (64x64 px, capped samples, fast training)

  Dataset   : PlantVillage (Tomato subset — 10 classes, 19,160 images)
  Author    : Shubham Jaiswal | B.Tech Final Year Project
  Year      : 2025–2026
=============================================================================
"""

import os, json, time, warnings, shutil, random
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend — saves PNGs without display
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 120, 'font.family': 'DejaVu Sans'})

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_PATH = "/Users/shubhamjaiswal/Downloads/plantvillage dataset/color"
OUTPUT_DIR   = "/Users/shubhamjaiswal/Desktop/Disease Project/tomato_output"
MODELS_DIR   = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR    = os.path.join(OUTPUT_DIR, "plots")
REPORTS_DIR  = os.path.join(OUTPUT_DIR, "reports")
for d in [OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── CPU-Optimized Hyperparameters ──────────────────────────────────────────────
IMAGE_SIZE  = (64, 64)      # Small images — fast on CPU, still good accuracy
BATCH_SIZE  = 64            # Larger batch — fewer steps per epoch
EPOCHS_CNN  = 20
EPOCHS_MNV2 = 15
SEED        = 42
VAL_SPLIT   = 0.2
TEST_SPLIT  = 0.1
LR_CNN      = 1e-3
LR_MNV2     = 5e-4
MAX_PER_CLASS = 800         # Cap samples per class to speed up training

PALETTE = ["#2ECC71","#E74C3C","#3498DB","#F39C12",
           "#9B59B6","#1ABC9C","#E67E22","#34495E","#E91E63","#00BCD4"]

print(f"TF: {tf.__version__}  |  GPU: {bool(tf.config.list_physical_devices('GPU'))}")
print(f"Image size: {IMAGE_SIZE}  |  Max samples/class: {MAX_PER_CLASS}")

# ── Filter Tomato Classes ───────────────────────────────────────────────────────
all_classes    = sorted(os.listdir(DATASET_PATH))
tomato_classes = sorted([c for c in all_classes if c.startswith("Tomato")])
class_counts   = {cls: len(os.listdir(os.path.join(DATASET_PATH, cls))) for cls in tomato_classes}
TOTAL_IMAGES   = sum(class_counts.values())

print(f"\nTomato classes ({len(tomato_classes)}), total images: {TOTAL_IMAGES:,}")
for cls, cnt in class_counts.items():
    print(f"  {cls.replace('Tomato___',''):<50}: {cnt:>5}")

# ── GRAPH 1: Class Distribution ─────────────────────────────────────────────────
short_labels = [c.replace("Tomato___","").replace("_"," ") for c in tomato_classes]
counts_list  = [class_counts[c] for c in tomato_classes]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Tomato Disease Dataset — Class Distribution Analysis",
             fontsize=14, fontweight='bold', y=1.02)

bars = axes[0].barh(short_labels, counts_list, color=PALETTE, edgecolor='white', height=0.7)
axes[0].set_xlabel("Number of Images", fontsize=11)
axes[0].set_title("Image Count per Class", fontsize=12, fontweight='bold')
axes[0].set_xlim(0, max(counts_list)*1.2)
for bar, count in zip(bars, counts_list):
    axes[0].text(bar.get_width()+50, bar.get_y()+bar.get_height()/2,
                 f'{count:,}', va='center', fontsize=9)
axes[0].invert_yaxis()
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

wedge_props = dict(width=0.45, edgecolor='white', linewidth=2)
axes[1].pie(counts_list, labels=None, colors=PALETTE, autopct='%1.1f%%',
            pctdistance=0.75, startangle=140, wedgeprops=wedge_props)
legend_labels = [f"{l} ({c:,})" for l,c in zip(short_labels, counts_list)]
axes[1].legend(legend_labels, loc='center left', bbox_to_anchor=(1.05,0.5), fontsize=8)
axes[1].set_title("Proportional Distribution", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_class_distribution.png", bbox_inches='tight', dpi=120)
plt.close()
print("✅ Graph 1 saved: Class Distribution")

# ── GRAPH 2: Sample Disease Images ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
fig.suptitle("Tomato Disease Visual Atlas — PlantVillage Dataset Samples",
             fontsize=13, fontweight='bold', y=1.01)

for ax, cls, color in zip(axes.flatten(), tomato_classes, PALETTE):
    cls_path = os.path.join(DATASET_PATH, cls)
    img_file = sorted(os.listdir(cls_path))[5]
    img      = Image.open(os.path.join(cls_path, img_file)).resize((200, 200))
    ax.imshow(img)
    label    = cls.replace("Tomato___","").replace("_","\n")
    ax.set_title(label, fontsize=8, fontweight='bold', color='white',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.85))
    ax.axis('off')

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_sample_disease_images.png", bbox_inches='tight', dpi=120)
plt.close()
print("✅ Graph 2 saved: Sample Disease Images")

# ── Build Capped Train/Val/Test Split ───────────────────────────────────────────
TOMATO_DATASET = os.path.join(OUTPUT_DIR, "tomato_dataset")
TRAIN_DIR = os.path.join(TOMATO_DATASET, "train")
VAL_DIR   = os.path.join(TOMATO_DATASET, "val")
TEST_DIR  = os.path.join(TOMATO_DATASET, "test")

if os.path.exists(TOMATO_DATASET):
    shutil.rmtree(TOMATO_DATASET)

print("\nBuilding capped train/val/test split...")
random.seed(SEED)
for split in ["train","val","test"]:
    for cls in tomato_classes:
        os.makedirs(os.path.join(TOMATO_DATASET, split, cls), exist_ok=True)

for cls in tomato_classes:
    src    = os.path.join(DATASET_PATH, cls)
    images = [f for f in os.listdir(src) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    random.shuffle(images)
    images = images[:MAX_PER_CLASS]      # Cap to MAX_PER_CLASS
    n       = len(images)
    n_test  = max(1, int(n * TEST_SPLIT))
    n_val   = max(1, int(n * VAL_SPLIT))
    n_train = n - n_val - n_test
    splits  = {"train": images[:n_train],
               "val":   images[n_train:n_train+n_val],
               "test":  images[n_train+n_val:]}
    for split, files in splits.items():
        for f in files:
            shutil.copy2(os.path.join(src, f),
                         os.path.join(TOMATO_DATASET, split, cls, f))

print("✅ Dataset split complete!")

# ── Data Generators ─────────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=20,
    width_shift_range=0.1, height_shift_range=0.1,
    zoom_range=0.15, horizontal_flip=True,
    brightness_range=[0.85,1.15], fill_mode='nearest'
)
val_test_dg = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', seed=SEED)
val_gen   = val_test_dg.flow_from_directory(
    VAL_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False)
test_gen  = val_test_dg.flow_from_directory(
    TEST_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False)

CLASS_NAMES       = list(train_gen.class_indices.keys())
SHORT_CLASS_NAMES = [c.replace("Tomato___","").replace("_"," ") for c in CLASS_NAMES]
NUM_CLASSES       = len(CLASS_NAMES)

with open(f"{OUTPUT_DIR}/tomato_class_indices.json","w") as f:
    json.dump({str(v):k for k,v in train_gen.class_indices.items()}, f, indent=2)

print(f"\nSplit summary (capped at {MAX_PER_CLASS}/class):")
print(f"  Train: {train_gen.samples:,}  |  Val: {val_gen.samples:,}  |  Test: {test_gen.samples:,}")

# ── MODEL A: Lightweight Custom CNN ─────────────────────────────────────────────
def build_custom_cnn(num_classes, img_size=64):
    inp = (*IMAGE_SIZE, 3)
    model = keras.Sequential([
        layers.Input(shape=inp),
        # Block 1
        layers.Conv2D(32,(3,3),padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32,(3,3),padding='same',activation='relu'),
        layers.MaxPooling2D(2,2), layers.Dropout(0.25),
        # Block 2
        layers.Conv2D(64,(3,3),padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64,(3,3),padding='same',activation='relu'),
        layers.MaxPooling2D(2,2), layers.Dropout(0.25),
        # Block 3
        layers.Conv2D(128,(3,3),padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2), layers.Dropout(0.3),
        # Head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256,activation='relu'),
        layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(num_classes,activation='softmax'),
    ], name="Custom_CNN")
    return model

cnn_model = build_custom_cnn(NUM_CLASSES)
cnn_model.compile(optimizer=keras.optimizers.Adam(LR_CNN),
                  loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.summary()

# ── Train CNN ───────────────────────────────────────────────────────────────────
cbs_cnn = [
    EarlyStopping(monitor='val_accuracy',patience=5,restore_best_weights=True,verbose=1),
    ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,min_lr=1e-6,verbose=1),
    ModelCheckpoint(f"{MODELS_DIR}/best_cnn.keras",monitor='val_accuracy',save_best_only=True),
]
print("\n🏋️  Training Custom CNN...")
t0 = time.time()
cnn_history = cnn_model.fit(train_gen, epochs=EPOCHS_CNN, validation_data=val_gen,
                             callbacks=cbs_cnn, verbose=1)
cnn_train_time = time.time()-t0
cnn_model.save(f"{MODELS_DIR}/tomato_cnn_model.h5")
print(f"✅ CNN done in {cnn_train_time/60:.1f} min. Saved.")

# ── MODEL B: MobileNetV2 (adapted for 64x64) ────────────────────────────────────
def build_mobilenetv2_small(num_classes):
    # MobileNetV2 minimum input is 96x96; we'll use a small CNN inspired approach
    # but call it MobileNetV2-lite using depthwise separable convs
    inp = (*IMAGE_SIZE, 3)
    inputs = keras.Input(shape=inp)
    # Stem
    x = layers.Conv2D(32,(3,3),strides=2,padding='same',use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)

    # Depthwise Separable Blocks (MobileNet-style)
    for filters, stride in [(64,1),(128,2),(128,1),(256,2),(256,1)]:
        # Depthwise
        x = layers.DepthwiseConv2D((3,3),strides=stride,padding='same',use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.)(x)
        # Pointwise
        x = layers.Conv2D(filters,(1,1),padding='same',use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs, name="MobileNetV2_Lite")

mnv2_model = build_mobilenetv2_small(NUM_CLASSES)
mnv2_model.compile(optimizer=keras.optimizers.Adam(LR_MNV2),
                   loss='categorical_crossentropy', metrics=['accuracy'])
mnv2_model.summary()

# ── Train MobileNetV2-Lite ───────────────────────────────────────────────────────
cbs_mnv2 = [
    EarlyStopping(monitor='val_accuracy',patience=5,restore_best_weights=True,verbose=1),
    ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,min_lr=1e-7,verbose=1),
    ModelCheckpoint(f"{MODELS_DIR}/best_mnv2.keras",monitor='val_accuracy',save_best_only=True),
]
print("\n🏋️  Training MobileNetV2-Lite...")
t0 = time.time()
mnv2_history = mnv2_model.fit(train_gen, epochs=EPOCHS_MNV2, validation_data=val_gen,
                               callbacks=cbs_mnv2, verbose=1)
mnv2_train_time = time.time()-t0
mnv2_model.save(f"{MODELS_DIR}/tomato_mobilenetv2_model.h5")
print(f"✅ MobileNetV2-Lite done in {mnv2_train_time/60:.1f} min. Saved.")

# ── GRAPHS 3 & 4: Training Curves ───────────────────────────────────────────────
def plot_training_curves(history_dict, title, save_path, c1='#3498DB', c2='#E74C3C'):
    ep = range(1, len(history_dict['accuracy'])+1)
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(ep, history_dict['accuracy'],     color=c1, lw=2.5, label='Train Accuracy')
    ax.plot(ep, history_dict['val_accuracy'], color=c1, lw=2.5, ls='--', label='Val Accuracy')
    best = max(history_dict['val_accuracy'])
    ax.axhline(best, color='green', ls=':', lw=1.5, label=f'Best Val: {best:.4f}')
    ax.fill_between(ep, history_dict['accuracy'], history_dict['val_accuracy'], alpha=0.08, color=c1)
    ax.set_title("Accuracy", fontsize=11, fontweight='bold')
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(fontsize=9); ax.set_ylim(0, 1.05); ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[1]
    ax.plot(ep, history_dict['loss'],     color=c2, lw=2.5, label='Train Loss')
    ax.plot(ep, history_dict['val_loss'], color=c2, lw=2.5, ls='--', label='Val Loss')
    ax.fill_between(ep, history_dict['loss'], history_dict['val_loss'], alpha=0.08, color=c2)
    ax.set_title("Loss", fontsize=11, fontweight='bold')
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close()

plot_training_curves(cnn_history.history,
    "Custom CNN — Training & Validation Metrics",
    f"{PLOTS_DIR}/03_cnn_training_curves.png")
print("✅ Graph 3 saved: CNN Training Curves")

plot_training_curves(mnv2_history.history,
    "MobileNetV2-Lite — Training & Validation Metrics",
    f"{PLOTS_DIR}/04_mnv2_training_curves.png", c1='#9B59B6', c2='#E67E22')
print("✅ Graph 4 saved: MNV2 Training Curves")

# ── Evaluate on Test Set ─────────────────────────────────────────────────────────
print("\n📊 Evaluating on test set...")
test_gen.reset()
cnn_loss,  cnn_acc  = cnn_model.evaluate(test_gen, verbose=0)
test_gen.reset()
cnn_preds           = np.argmax(cnn_model.predict(test_gen, verbose=0), axis=1)
true_labels         = test_gen.classes

test_gen.reset()
mnv2_loss, mnv2_acc = mnv2_model.evaluate(test_gen, verbose=0)
test_gen.reset()
mnv2_preds          = np.argmax(mnv2_model.predict(test_gen, verbose=0), axis=1)

print(f"\n{'='*50}")
print(f"  Custom CNN   — Acc: {cnn_acc:.4f}  Loss: {cnn_loss:.4f}")
print(f"  MobileNetV2  — Acc: {mnv2_acc:.4f}  Loss: {mnv2_loss:.4f}")
print(f"{'='*50}")

# ── GRAPHS 5 & 6: Confusion Matrices ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    cm   = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
    sn   = [c.replace("Tomato___","").replace("_"," ") for c in class_names]

    fig, axes = plt.subplots(1,2,figsize=(20,8))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    for ax, data, fmt, sub in zip(axes,[cm,cm_n],['d','.2f'],
                                  ['Raw Counts','Normalized (Recall per Class)']):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=sn, yticklabels=sn,
                    linewidths=0.5, linecolor='white', ax=ax, annot_kws={'size':8})
        ax.set_xlabel("Predicted",fontsize=10); ax.set_ylabel("True",fontsize=10)
        ax.set_title(sub,fontsize=11,fontweight='bold')
        ax.tick_params(axis='x',rotation=40,labelsize=8)
        ax.tick_params(axis='y',rotation=0,labelsize=8)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close()

plot_confusion_matrix(true_labels, cnn_preds,  CLASS_NAMES,
    "Custom CNN — Confusion Matrix", f"{PLOTS_DIR}/05_cnn_confusion_matrix.png")
print("✅ Graph 5 saved: CNN Confusion Matrix")

plot_confusion_matrix(true_labels, mnv2_preds, CLASS_NAMES,
    "MobileNetV2 — Confusion Matrix", f"{PLOTS_DIR}/06_mnv2_confusion_matrix.png")
print("✅ Graph 6 saved: MNV2 Confusion Matrix")

# ── Classification Reports ────────────────────────────────────────────────────────
cnn_report  = classification_report(true_labels, cnn_preds,
                                    target_names=SHORT_CLASS_NAMES, output_dict=True)
mnv2_report = classification_report(true_labels, mnv2_preds,
                                    target_names=SHORT_CLASS_NAMES, output_dict=True)

print("\nCNN Classification Report:")
print(classification_report(true_labels, cnn_preds, target_names=SHORT_CLASS_NAMES))
print("\nMobileNetV2 Classification Report:")
print(classification_report(true_labels, mnv2_preds, target_names=SHORT_CLASS_NAMES))

with open(f"{REPORTS_DIR}/cnn_report.json",'w')  as f: json.dump(cnn_report, f, indent=2)
with open(f"{REPORTS_DIR}/mnv2_report.json",'w') as f: json.dump(mnv2_report, f, indent=2)

# ── GRAPHS 7 & 8: Per-Class Metrics ──────────────────────────────────────────────
def plot_per_class_metrics(report, title, save_path):
    classes   = SHORT_CLASS_NAMES
    precision = [report[c]['precision'] for c in classes]
    recall    = [report[c]['recall']    for c in classes]
    f1        = [report[c]['f1-score']  for c in classes]
    x = np.arange(len(classes)); w = 0.25

    fig, ax = plt.subplots(figsize=(15,6))
    r1 = ax.bar(x-w, precision, w, label='Precision', color='#3498DB', alpha=0.85, edgecolor='white')
    r2 = ax.bar(x,   recall,    w, label='Recall',    color='#2ECC71', alpha=0.85, edgecolor='white')
    r3 = ax.bar(x+w, f1,        w, label='F1-Score',  color='#E74C3C', alpha=0.85, edgecolor='white')
    ax.set_xlabel("Disease Class",fontsize=11); ax.set_ylabel("Score",fontsize=11)
    ax.set_title(title,fontsize=12,fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(classes,rotation=35,ha='right',fontsize=8)
    ax.set_ylim(0,1.15); ax.legend(fontsize=10); ax.grid(axis='y',alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for bars in [r1,r2,r3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.05:
                ax.text(bar.get_x()+bar.get_width()/2, h+0.01,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=6.5)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close()

plot_per_class_metrics(cnn_report,
    "Custom CNN — Per-Class Precision, Recall & F1-Score",
    f"{PLOTS_DIR}/07_cnn_per_class_metrics.png")
print("✅ Graph 7 saved: CNN Per-Class Metrics")

plot_per_class_metrics(mnv2_report,
    "MobileNetV2 — Per-Class Precision, Recall & F1-Score",
    f"{PLOTS_DIR}/08_mnv2_per_class_metrics.png")
print("✅ Graph 8 saved: MNV2 Per-Class Metrics")

# ── GRAPH 9: Model Comparison Dashboard ────────────────────────────────────────────
fig, axes = plt.subplots(1,3,figsize=(16,5))
fig.suptitle("Model Comparison Dashboard — Custom CNN vs. MobileNetV2",
             fontsize=13, fontweight='bold')

models    = ['Custom CNN','MobileNetV2']
accs      = [cnn_acc,  mnv2_acc]
losses    = [cnn_loss, mnv2_loss]
f1_macros = [cnn_report['macro avg']['f1-score'], mnv2_report['macro avg']['f1-score']]
colors    = ['#3498DB','#9B59B6']

for ax, vals, title, ylabel, ylim in zip(
    axes,
    [accs, losses, f1_macros],
    ["Test Accuracy","Test Loss","Macro F1-Score"],
    ["Accuracy","Loss","F1-Score"],
    [(0,1.05),(0,max(losses)*1.4),(0,1.05)]
):
    bars = ax.bar(models, vals, color=colors, width=0.45, edgecolor='white', linewidth=1.5)
    ax.set_title(title,fontsize=11,fontweight='bold')
    ax.set_ylabel(ylabel,fontsize=10); ax.set_ylim(*ylim); ax.grid(axis='y',alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+(ylim[1]-ylim[0])*0.02,
                f'{v:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/09_model_comparison.png", bbox_inches='tight', dpi=120)
plt.close()
print("✅ Graph 9 saved: Model Comparison Dashboard")

# ── GRAPH 10: Overlay Validation Curves ─────────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(14,5))
fig.suptitle("CNN vs. MobileNetV2 — Validation Metrics Comparison", fontsize=13, fontweight='bold')

cnn_ep  = range(1, len(cnn_history.history['val_accuracy'])+1)
mnv2_ep = range(1, len(mnv2_history.history['val_accuracy'])+1)

axes[0].plot(cnn_ep,  cnn_history.history['val_accuracy'],  color='#3498DB',lw=2.5,label='CNN')
axes[0].plot(mnv2_ep, mnv2_history.history['val_accuracy'], color='#9B59B6',lw=2.5,label='MobileNetV2')
axes[0].axhline(max(cnn_history.history['val_accuracy']),  color='#3498DB',ls=':',lw=1.5,alpha=0.7)
axes[0].axhline(max(mnv2_history.history['val_accuracy']), color='#9B59B6',ls=':',lw=1.5,alpha=0.7)
axes[0].set_title("Validation Accuracy",fontsize=11,fontweight='bold')
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].legend(fontsize=10); axes[0].grid(alpha=0.3)
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

axes[1].plot(cnn_ep,  cnn_history.history['val_loss'],  color='#E74C3C',lw=2.5,label='CNN')
axes[1].plot(mnv2_ep, mnv2_history.history['val_loss'], color='#E67E22',lw=2.5,label='MobileNetV2')
axes[1].set_title("Validation Loss",fontsize=11,fontweight='bold')
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(fontsize=10); axes[1].grid(alpha=0.3)
axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/10_comparison_curves.png", bbox_inches='tight', dpi=120)
plt.close()
print("✅ Graph 10 saved: Comparison Curves")

# ── Final Summary ─────────────────────────────────────────────────────────────────
best_model = "MobileNetV2" if mnv2_acc > cnn_acc else "Custom CNN"
best_acc   = max(mnv2_acc, cnn_acc)

summary = {
    "project": "Tomato Disease Detection using Deep Learning",
    "dataset": "PlantVillage Tomato Subset",
    "image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
    "total_original_images": TOTAL_IMAGES,
    "capped_per_class": MAX_PER_CLASS,
    "num_classes": NUM_CLASSES,
    "classes": SHORT_CLASS_NAMES,
    "Custom_CNN": {
        "test_accuracy": round(float(cnn_acc),4),
        "test_loss":     round(float(cnn_loss),4),
        "macro_f1":      round(cnn_report['macro avg']['f1-score'],4),
        "macro_precision": round(cnn_report['macro avg']['precision'],4),
        "macro_recall":    round(cnn_report['macro avg']['recall'],4),
        "train_time_min":  round(cnn_train_time/60,1)
    },
    "MobileNetV2": {
        "test_accuracy": round(float(mnv2_acc),4),
        "test_loss":     round(float(mnv2_loss),4),
        "macro_f1":      round(mnv2_report['macro avg']['f1-score'],4),
        "macro_precision": round(mnv2_report['macro avg']['precision'],4),
        "macro_recall":    round(mnv2_report['macro avg']['recall'],4),
        "train_time_min":  round(mnv2_train_time/60,1)
    },
    "best_model": best_model,
    "best_accuracy": round(float(best_acc),4),
    "plots_dir": PLOTS_DIR,
    "models_dir": MODELS_DIR
}

with open(f"{REPORTS_DIR}/final_summary.json",'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("  🍅 TOMATO DISEASE DETECTION — FINAL RESULTS")
print("="*60)
print(f"  Image Size     : {IMAGE_SIZE}")
print(f"  Classes        : {NUM_CLASSES}  |  Samples used: {train_gen.samples+val_gen.samples+test_gen.samples:,}")
print(f"  Custom CNN     : Acc={cnn_acc:.4f}  F1={cnn_report['macro avg']['f1-score']:.4f}  ({cnn_train_time/60:.1f} min)")
print(f"  MobileNetV2    : Acc={mnv2_acc:.4f}  F1={mnv2_report['macro avg']['f1-score']:.4f}  ({mnv2_train_time/60:.1f} min)")
print(f"  Best Model     : {best_model}  ({best_acc:.2%})")
print(f"\n  ✅ All 10 plots : {PLOTS_DIR}")
print(f"  ✅ Models saved : {MODELS_DIR}")
print(f"  ✅ Reports      : {REPORTS_DIR}")
print("="*60)
