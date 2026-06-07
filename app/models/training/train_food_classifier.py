"""Entrenamiento de la CNN de clasificación de alimentos (CU9 — MobileNetV2).

Transfer learning sobre MobileNetV2 (pesos ImageNet) con una cabeza densa para
las clases del dataset. Entrena en dos fases: (1) base congelada, (2) fine-tuning
de las últimas capas. Guarda dos artefactos en ``app/models/artifacts/``:

* ``food_classifier.h5``       — el modelo Keras entrenado
* ``food_class_names.json``    — nombres de clase en el orden de salida del modelo

Está preparado para **Food-101** (estructura ``images/<clase>/*.jpg``). La ruta
por defecto apunta al dataset local del proyecto.

Ejemplos::

    # Entrenamiento completo (101 clases, 1000 img/clase) — requiere GPU o mucho tiempo
    python -m app.models.training.train_food_classifier

    # Subconjunto rápido y factible en CPU (todas las clases, 150 img/clase)
    python -m app.models.training.train_food_classifier --max-per-class 150 \
        --epochs-frozen 4 --epochs-finetune 2

REQUISITOS: TensorFlow (>=2.20 para Python 3.13). El microservicio funciona sin
este script (CU9/plate responde 503 hasta que exista el artefacto).
"""

import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "app" / "models" / "artifacts"
ARTIFACT_PATH = ARTIFACTS_DIR / "food_classifier.h5"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "food_class_names.json"

# Ruta por defecto al dataset Food-101 local (carpeta que contiene <clase>/*.jpg)
DEFAULT_DATA_DIR = r"D:\FoodNetProject\FoodNet\Food Datasets\food-101\images"

IMG_SIZE = (224, 224)
VALID_EXT = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _collect_files(
    data_dir: Path,
    max_per_class: int | None,
    limit_classes: int | None,
    seed: int,
) -> tuple[list[str], list[int], list[str]]:
    """Recolecta rutas de imagen + etiquetas, con caps opcionales por clase."""
    rng = random.Random(seed)
    class_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if limit_classes:
        class_dirs = class_dirs[:limit_classes]

    class_names = [d.name for d in class_dirs]
    filepaths: list[str] = []
    labels: list[int] = []

    for idx, cdir in enumerate(class_dirs):
        files = [str(f) for f in cdir.iterdir() if f.suffix.lower() in VALID_EXT]
        rng.shuffle(files)
        if max_per_class:
            files = files[:max_per_class]
        filepaths.extend(files)
        labels.extend([idx] * len(files))

    return filepaths, labels, class_names


def _build_datasets(
    filepaths: list[str],
    labels: list[int],
    num_classes: int,
    batch_size: int,
    val_split: float,
    seed: int,
):
    import tensorflow as tf

    # Mezcla y split train/val
    paths = list(filepaths)
    labs = list(labels)
    combined = list(zip(paths, labs))
    random.Random(seed).shuffle(combined)
    paths, labs = zip(*combined)
    n_val = int(len(paths) * val_split)

    def make_ds(p, y, training: bool):
        ds = tf.data.Dataset.from_tensor_slices((list(p), list(y)))

        def _load(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, IMG_SIZE)
            img = img / 255.0
            return img, tf.one_hot(label, num_classes)

        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.map(
                lambda x, y: (tf.image.random_flip_left_right(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            ds = ds.shuffle(1000, seed=seed)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = make_ds(paths[:n_val], labs[:n_val], training=False)
    train_ds = make_ds(paths[n_val:], labs[n_val:], training=True)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def build_model(num_classes: int):
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model

    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # Fase 1: entrenar solo la cabeza

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=outputs), base_model


def train(
    data_dir: str = DEFAULT_DATA_DIR,
    epochs_frozen: int = 10,
    epochs_finetune: int = 5,
    batch_size: int = 32,
    max_per_class: int | None = None,
    limit_classes: int | None = None,
    val_split: float = 0.2,
    seed: int = 42,
) -> None:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__} | GPUs detectadas: {[g.name for g in gpus] or 'ninguna (CPU)'}")

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado en {data_path}. "
            "Esperado: carpeta con subdirectorios <clase>/*.jpg (estructura Food-101)."
        )

    print("Recolectando archivos...")
    filepaths, labels, class_names = _collect_files(
        data_path, max_per_class, limit_classes, seed
    )
    num_classes = len(class_names)
    print(f"  {len(filepaths)} imágenes en {num_classes} clases")

    train_ds, val_ds = _build_datasets(
        filepaths, labels, num_classes, batch_size, val_split, seed
    )

    model, base_model = build_model(num_classes)

    # Fase 1: cabeza nueva con base congelada
    print(f"\n=== FASE 1: base congelada ({epochs_frozen} épocas) ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=epochs_frozen)

    # Fase 2: fine-tuning de las últimas 30 capas de la base
    if epochs_finetune > 0:
        print(f"\n=== FASE 2: fine-tuning ({epochs_finetune} épocas) ===")
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(train_ds, validation_data=val_ds, epochs=epochs_finetune)

    # Persistir artefactos
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(ARTIFACT_PATH)
    CLASS_NAMES_PATH.write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nModelo guardado en {ARTIFACT_PATH}")
    print(f"Clases guardadas en {CLASS_NAMES_PATH} ({num_classes} clases)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena la CNN de alimentos (CU9).")
    parser.add_argument("--data", default=DEFAULT_DATA_DIR, help="Directorio del dataset")
    parser.add_argument("--epochs-frozen", type=int, default=10)
    parser.add_argument("--epochs-finetune", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Máximo de imágenes por clase (para acelerar; por defecto todas)",
    )
    parser.add_argument(
        "--limit-classes",
        type=int,
        default=None,
        help="Usar solo las primeras N clases (por defecto todas)",
    )
    args = parser.parse_args()
    train(
        data_dir=args.data,
        epochs_frozen=args.epochs_frozen,
        epochs_finetune=args.epochs_finetune,
        batch_size=args.batch_size,
        max_per_class=args.max_per_class,
        limit_classes=args.limit_classes,
    )
