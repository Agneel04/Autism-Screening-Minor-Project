import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from collections import Counter

# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32

TEST_DIR   = r"C:\Users\KIIT0001\OneDrive\Desktop\Autism_Minor\test"
MODEL_PATH = r"C:\Users\KIIT0001\OneDrive\Desktop\Autism_Minor\model.h5"

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
model = load_model(r"C:\Users\KIIT0001\OneDrive\Desktop\Autism_Minor\model.h5")
print("Model loaded successfully\n")

# ─────────────────────────────────────────────────────────────
# TEST DATA GENERATOR
# ─────────────────────────────────────────────────────────────
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH_SIZE,
    class_mode  = "binary",
    shuffle     = False        # critical — must stay False
)

class_names = list(test_data.class_indices.keys())
print(f"Classes found : {class_names}")
print(f"Test images   : {test_data.samples}\n")

# ─────────────────────────────────────────────────────────────
# BASIC EVALUATION
# ─────────────────────────────────────────────────────────────
loss, accuracy = model.evaluate(test_data, verbose=1)

print("\n" + "=" * 50)
print("  BASIC TEST RESULTS")
print("=" * 50)
print(f"  Test Accuracy : {accuracy * 100:.2f}%")
print(f"  Test Loss     : {loss:.4f}")
print("=" * 50 + "\n")

# ─────────────────────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────────────────────
print("Running predictions on test set...")
y_pred_prob = model.predict(test_data, verbose=1).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)
y_true      = test_data.classes

# ─────────────────────────────────────────────────────────────
# CLASSIFICATION REPORT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(y_true, y_pred, target_names=class_names))

# ─────────────────────────────────────────────────────────────
# SEVERITY BREAKDOWN
# ─────────────────────────────────────────────────────────────
def classify_severity(prob):
    if prob >= 0.85:   return "High"
    elif prob >= 0.65: return "Medium"
    elif prob >= 0.50: return "Low"
    else:              return "Non-Autistic"

severity_labels = [classify_severity(p) for p in y_pred_prob]
counts          = Counter(severity_labels)

print("=" * 50)
print("  SEVERITY BREAKDOWN ON TEST SET")
print("=" * 50)
for level in ["High", "Medium", "Low", "Non-Autistic"]:
    bar   = "█" * counts.get(level, 0)
    print(f"  {level:15s}: {counts.get(level, 0):4d} images  {bar}")
print("=" * 50 + "\n")

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Autism Screening Model — Test Evaluation", fontsize=15)

# ── Plot 1: Confusion Matrix ──
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(
    cm,
    annot        = True,
    fmt          = "d",
    cmap         = "Blues",
    xticklabels  = class_names,
    yticklabels  = class_names,
    ax           = axes[0, 0],
    linewidths   = 0.5
)
axes[0, 0].set_title("Confusion Matrix", fontsize=13)
axes[0, 0].set_ylabel("Actual Label")
axes[0, 0].set_xlabel("Predicted Label")

# Annotate TP TN FP FN
tn, fp, fn, tp = cm.ravel()
axes[0, 0].text(
    0.5, -0.18,
    f"TP={tp}   TN={tn}   FP={fp}   FN={fn}",
    ha="center", transform=axes[0, 0].transAxes,
    fontsize=10, color="dimgray"
)

# ── Plot 2: Per-class Metrics Bar Chart ──
report  = classification_report(
    y_true, y_pred,
    target_names = class_names,
    output_dict  = True
)
metrics = ["precision", "recall", "f1-score"]
x       = np.arange(len(metrics))
width   = 0.3
colors  = ["#4472C4", "#ED7D31"]

for i, cls in enumerate(class_names):
    values = [report[cls][m] for m in metrics]
    bars   = axes[0, 1].bar(x + i * width, values, width,
                             label=cls, color=colors[i], alpha=0.85)
    for bar, val in zip(bars, values):
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=9
        )

axes[0, 1].set_xticks(x + width / 2)
axes[0, 1].set_xticklabels(metrics)
axes[0, 1].set_ylim(0, 1.15)
axes[0, 1].set_title("Per-class Precision / Recall / F1", fontsize=13)
axes[0, 1].set_ylabel("Score")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis="y")

# ── Plot 3: ROC Curve ──
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
roc_auc              = auc(fpr, tpr)

axes[1, 0].plot(fpr, tpr, color="steelblue", linewidth=2,
                label=f"ROC Curve (AUC = {roc_auc:.3f})")
axes[1, 0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
axes[1, 0].fill_between(fpr, tpr, alpha=0.1, color="steelblue")
axes[1, 0].set_xlabel("False Positive Rate")
axes[1, 0].set_ylabel("True Positive Rate")
axes[1, 0].set_title("ROC Curve", fontsize=13)
axes[1, 0].legend(loc="lower right")
axes[1, 0].grid(True, alpha=0.3)

# ── Plot 4: Severity Distribution Pie Chart ──
severity_order  = ["High", "Medium", "Low", "Non-Autistic"]
severity_counts = [counts.get(s, 0) for s in severity_order]
severity_colors = ["#E57373", "#FFB74D", "#FFF176", "#81C784"]

wedges, texts, autotexts = axes[1, 1].pie(
    severity_counts,
    labels     = severity_order,
    colors     = severity_colors,
    autopct    = lambda p: f"{p:.1f}%\n({int(round(p * sum(severity_counts) / 100))})",
    startangle = 140,
    wedgeprops = dict(edgecolor="white", linewidth=1.5)
)
for at in autotexts:
    at.set_fontsize(9)

axes[1, 1].set_title("Severity Distribution on Test Set", fontsize=13)

plt.tight_layout()
plt.savefig("evaluation_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Evaluation plots saved as evaluation_results.png")

# ─────────────────────────────────────────────────────────────
# FALSE NEGATIVE ANALYSIS
# (Most critical for medical screening)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  FALSE NEGATIVE ANALYSIS")
print("  (Autistic children predicted as Non-Autistic)")
print("=" * 50)
false_negatives = np.sum((y_true == 1) & (y_pred == 0))
total_autistic  = np.sum(y_true == 1)
fn_rate         = false_negatives / total_autistic * 100 if total_autistic > 0 else 0

print(f"  Total Autistic in test set : {total_autistic}")
print(f"  Missed (False Negatives)   : {false_negatives}")
print(f"  Miss Rate                  : {fn_rate:.2f}%")
print(f"  Recall (Autistic class)    : {report[class_names[1]]['recall']:.4f}")
print()
if fn_rate > 15:
    print("  Warning: Miss rate is high.")
    print("  Consider lowering the decision threshold below 0.5")
    print("  to catch more autistic cases at the cost of more false positives.")
else:
    print("  Miss rate is within acceptable range for a screening tool.")
print("=" * 50)
