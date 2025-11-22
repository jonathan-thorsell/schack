import os
import cv2
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# --- Configuration ---
IMAGE_SIZE = (64, 64)
TRAIN_FOLDER = r"training\data\training"
TEST_FOLDER = r"training\data\testing"
CLASSES = ["black", "white"]  # keep all 3 folders
EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 32
MODEL_FILENAME = r"training\models\color_classifier.keras"

# --- Callbacks ---
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=PATIENCE,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# --- Data Loading ---
def load_data(data_dir, augment=False):
    X, y = [], []

    for label_name in CLASSES:
        label_dir = os.path.abspath(os.path.join(data_dir, label_name))
        print(f"🔍 Scanning folder: {label_dir}")

        if not os.path.exists(label_dir):
            print(f"⚠️ Folder missing: {label_dir}")
            continue

        for filename in os.listdir(label_dir):
            if not (filename.lower().endswith(".jpg") or filename.lower().endswith(".png")):
                continue

            img_path = os.path.join(label_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Failed to load image: {img_path}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMAGE_SIZE)

            if augment:
                img = augment_image(img)

            img = img.astype("float32") / 255.0  # normalize once
            X.append(img)

            # map empty=0, black/white=1
            label = 0 if label_name == "black" else 1
            y.append(label)

    print(f"✅ Loaded {len(X)} samples from {data_dir}.")
    return np.array(X), np.array(y)

def augment_image(img):
    # Augment in 0-255 range
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)

    if np.random.rand() > 0.3:
        brightness = np.random.uniform(0.9, 1.1)
        img = np.clip(img * brightness, 0, 255)

    return img

# --- Model Definition ---
def build_model():
    base = MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False  # freeze base first

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# --- Evaluation ---
def evaluate_model(model, *_):
    print("🔍 Loading test data...")
    X_test, y_test = load_data(TEST_FOLDER, augment=False)
    if len(X_test) == 0:
        print("⚠️ No test samples found.")
        return

    print(f"🧪 Testing samples: {len(X_test)}")
    print("🔍 Evaluating model on test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"📉 Test Loss: {loss:.4f}, 🎯 Test Accuracy: {accuracy:.4f}")

    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob > 0.5).astype(int)
    print("\n📊 Classification report (test set):")
    print(classification_report(y_test, y_pred, target_names=["black", "white"]))

# --- Main ---
def main():
    # Load and split data
    X, y = load_data(TRAIN_FOLDER, augment=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"📊 Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Compute class weights for imbalance
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    print("⚖️ Class weights:", class_weights)

    # Build and train model
    print("🧠 Building model...")
    model = build_model()

    print("🚀 Training model (frozen base)...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weights
    )

    # --- Save after fine-tuning ---
    model.save(MODEL_FILENAME)
    print(f"✅ Model saved as {MODEL_FILENAME}")

    # Evaluate final model
    evaluate_model(model, X_val, y_val)

if __name__ == "__main__":
    main()