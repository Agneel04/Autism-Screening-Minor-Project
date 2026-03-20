import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────
IMG_SIZE        = 224
BATCH_SIZE      = 32
EPOCHS          = 20
FINE_TUNE_EPOCHS= 10

TRAIN_DIR  = r"C:\Users\KIIT0001\OneDrive\Desktop\Autism_Minor\train"
VALID_DIR  = r"C:\Users\KIIT0001\OneDrive\Desktop\Autism_Minor\valid"
MODEL_PATH = r"C:\Users\KIIT0001\OneDrive\Desktop\Autism_Minor\model.h5"

# ─────────────────────────────────────────────────────────────
# DATA GENERATORS
# ─────────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale           = 1./255,
    rotation_range    = 20,
    zoom_range        = 0.2,
    horizontal_flip   = True,
    width_shift_range = 0.1,
    height_shift_range= 0.1
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH_SIZE,
    class_mode  = "binary"
)

valid_data = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH_SIZE,
    class_mode  = "binary"
)

print(f"\nClass indices : {train_data.class_indices}")
print(f"Training images : {train_data.samples}")
print(f"Validation images: {valid_data.samples}\n")

# ─────────────────────────────────────────────────────────────
# HANDLE CLASS IMBALANCE
# ─────────────────────────────────────────────────────────────
class_counts = np.bincount(train_data.classes)
class_weights = {
    0: len(train_data.classes) / (2 * class_counts[0]),
    1: len(train_data.classes) / (2 * class_counts[1])
}
print(f"Class counts  : {dict(enumerate(class_counts))}")
print(f"Class weights : {class_weights}\n")

# ─────────────────────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────────────────────
base_model = MobileNetV2(
    weights      = "imagenet",
    include_top  = False,
    input_shape  = (IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

x      = base_model.output
x      = GlobalAveragePooling2D()(x)
x      = Dense(128, activation="relu")(x)
x      = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model  = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer = "adam",
    loss      = "binary_crossentropy",
    metrics   = ["accuracy"]
)

model.summary()

# ─────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor            = "val_loss",
        patience           = 5,
        restore_best_weights = True,
        verbose            = 1
    ),
    ModelCheckpoint(
        "best_model.h5",
        save_best_only = True,
        monitor        = "val_accuracy",
        verbose        = 1
    ),
    ReduceLROnPlateau(
        monitor  = "val_loss",
        factor   = 0.3,
        patience = 3,
        verbose  = 1
    )
]

# ─────────────────────────────────────────────────────────────
# PHASE 1 — TRAIN TOP LAYERS ONLY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  PHASE 1: Training top layers (base frozen)")
print("=" * 55 + "\n")

history1 = model.fit(
    train_data,
    validation_data = valid_data,
    epochs          = EPOCHS,
    class_weight    = class_weights,
    callbacks       = callbacks
)

# ─────────────────────────────────────────────────────────────
# PHASE 2 — FINE-TUNE LAST 30 LAYERS OF BASE MODEL
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  PHASE 2: Fine-tuning last 30 layers")
print("=" * 55 + "\n")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with a much lower learning rate for fine-tuning
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss      = "binary_crossentropy",
    metrics   = ["accuracy"]
)

history2 = model.fit(
    train_data,
    validation_data = valid_data,
    epochs          = FINE_TUNE_EPOCHS,
    class_weight    = class_weights,
    callbacks       = callbacks
)

# ─────────────────────────────────────────────────────────────
# SAVE FINAL MODEL
# ─────────────────────────────────────────────────────────────
model.save(MODEL_PATH)
print(f"\nModel saved as {MODEL_PATH}")

# ─────────────────────────────────────────────────────────────
# PLOT TRAINING CURVES
# ─────────────────────────────────────────────────────────────
def plot_history(h1, h2):
    acc     = h1.history["accuracy"]     + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss    = h1.history["loss"]         + h2.history["loss"]
    val_loss= h1.history["val_loss"]     + h2.history["val_loss"]

    phase2_start = len(h1.history["accuracy"])
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=14)

    # Accuracy plot
    ax1.plot(epochs_range, acc,     label="Train Accuracy",      color="steelblue")
    ax1.plot(epochs_range, val_acc, label="Validation Accuracy",  color="orange")
    ax1.axvline(phase2_start, color="gray", linestyle="--",
                linewidth=1.5, label="Fine-tune start")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(epochs_range, loss,     label="Train Loss",      color="steelblue")
    ax2.plot(epochs_range, val_loss, label="Validation Loss",  color="orange")
    ax2.axvline(phase2_start, color="gray", linestyle="--",
                linewidth=1.5, label="Fine-tune start")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("Training curves saved as training_curves.png")

plot_history(history1, history2)

# ─────────────────────────────────────────────────────────────
# FINAL EVALUATION ON VALIDATION SET
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  FINAL VALIDATION RESULTS")
print("=" * 55)
val_loss, val_acc = model.evaluate(valid_data, verbose=0)
print(f"  Validation Accuracy : {val_acc  * 100:.2f}%")
print(f"  Validation Loss     : {val_loss:.4f}")
print("=" * 55)
