"""
Обучение улучшенной модели оценки риска мошенничества и экспорт в TensorFlow Lite.

Улучшения по сравнению с версией 1:
  - Архитектура: Dense(32) → BatchNorm → Dropout(0.3) → Dense(16) → Dropout(0.2) → Sigmoid
  - Early Stopping: обучение останавливается когда val_auc перестаёт расти (patience=10)
  - class_weight: компенсация дисбаланса классов (fraud ~15 %)
  - Оптимальный порог классификации по F1 (сохраняется в scaler_params.json)
  - Полные метрики: Precision, Recall, F1, Confusion Matrix, ROC AUC
  - ROC-кривая сохраняется как PNG

Запуск: python train_and_export.py
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve, f1_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    FEATURE_NAMES, MODEL_TFLITE_PATH, SCALER_JSON_PATH,
    TRAIN_DATA_PATH, NUM_FEATURES, REPORT_PATH, ROC_CURVE_PATH,
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def find_optimal_threshold(y_true, y_prob):
    """Находит порог, максимизирующий F1-score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls),
        0.0
    )
    best_idx = np.argmax(f1_scores[:-1])   # последний элемент без порога
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def save_roc_curve(y_true, y_prob, path):
    """Сохраняет ROC-кривую как PNG (если matplotlib доступен)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, color="steelblue", lw=2,
                 label=f"ROC-кривая (AUC = {auc:.4f})")
        plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Случайный классификатор")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC-кривая модели обнаружения мошенничества")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"ROC-кривая сохранена: {path}")
    except ImportError:
        print("matplotlib не установлен — ROC-кривая не сохранена.")


def build_model(num_features):
    """
    Улучшенная архитектура MLP:
      Input(12) → Dense(32,ReLU) → BatchNorm → Dropout(0.3)
               → Dense(16,ReLU) → Dropout(0.2)
               → Dense(1,Sigmoid)
    """
    model = keras.Sequential([
        layers.Input(shape=(num_features,)),
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def main():
    # 1. Загрузка данных
    df = pd.read_csv(TRAIN_DATA_PATH)
    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    print(f"Датасет: {len(df)} строк, fraud={int(y.sum())} ({y.mean()*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    # 2. Нормализация (обязательно — на устройстве применяем те же mean/scale)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Веса классов (компенсация дисбаланса)
    n_fraud = int(y_train.sum())
    n_normal = len(y_train) - n_fraud
    weight_fraud = n_normal / n_fraud
    class_weight = {0: 1.0, 1: weight_fraud}
    print(f"class_weight: normal=1.0, fraud={weight_fraud:.2f}")

    # 4. Модель
    model = build_model(NUM_FEATURES)
    model.summary()

    # 5. Early Stopping — следим за val_auc, сохраняем лучшие веса
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_auc",
        patience=10,
        restore_best_weights=True,
        mode="max",
        verbose=1,
    )

    # 6. Обучение
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=100,
        batch_size=64,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=1,
    )

    # 7. Предсказания и метрики
    y_prob = model.predict(X_test_scaled, verbose=0).flatten()
    auc = roc_auc_score(y_test, y_prob)

    # Оптимальный порог по F1
    optimal_threshold, best_f1 = find_optimal_threshold(y_test, y_prob)
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)

    print(f"\n{'='*50}")
    print(f"Test ROC AUC:           {auc:.4f}")
    print(f"Оптимальный порог:      {optimal_threshold:.4f}  (F1={best_f1:.4f})")
    print(f"\nClassification Report (порог={optimal_threshold:.2f}):")
    print(classification_report(y_test, y_pred_optimal,
                                target_names=["Normal", "Fraud"], digits=4))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_optimal)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    # Также метрики при стандартном пороге 0.5
    y_pred_05 = (y_prob >= 0.5).astype(int)
    f1_05 = f1_score(y_test, y_pred_05)
    print(f"\nF1 при пороге 0.5:     {f1_05:.4f}")
    print(f"{'='*50}\n")

    # 8. Сохранение текстового отчёта
    report_lines = [
        "=== Отчёт об обучении модели ===\n",
        f"Датасет: {len(df)} строк, fraud={int(y.sum())} ({y.mean()*100:.1f}%)\n",
        f"Train: {len(X_train)}, Test: {len(X_test)}\n",
        f"Архитектура: Dense(32,BN,Drop0.3) → Dense(16,Drop0.2) → Sigmoid\n",
        f"Оптимизатор: Adam lr=5e-4, class_weight fraud={weight_fraud:.2f}\n",
        f"Остановлено на эпохе: {early_stop.stopped_epoch or 'не достигнут максимум'}\n\n",
        f"ROC AUC: {auc:.4f}\n",
        f"Оптимальный порог: {optimal_threshold:.4f} (F1={best_f1:.4f})\n\n",
        "Classification Report:\n",
        classification_report(y_test, y_pred_optimal,
                              target_names=["Normal", "Fraud"], digits=4),
        "\nConfusion Matrix:\n",
        f"  TN={cm[0,0]}  FP={cm[0,1]}\n",
        f"  FN={cm[1,0]}  TP={cm[1,1]}\n",
    ]
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print(f"Отчёт сохранён: {REPORT_PATH}")

    # 9. ROC-кривая
    save_roc_curve(y_test, y_prob, ROC_CURVE_PATH)

    # 10. Сохранение параметров нормализации + оптимального порога
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "optimal_threshold": optimal_threshold,
    }
    with open(SCALER_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(scaler_params, f, indent=2)
    print(f"Сохранено: {SCALER_JSON_PATH}")

    # 11. Экспорт в TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(MODEL_TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"Сохранено: {MODEL_TFLITE_PATH}  ({len(tflite_model)/1024:.1f} КБ)")

    # 12. Проверка через TFLite
    interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE_PATH)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    interpreter.set_tensor(inp[0]["index"], X_test_scaled[:1].astype(np.float32))
    interpreter.invoke()
    tflite_score = interpreter.get_tensor(out[0]["index"])[0][0]
    keras_score = float(y_prob[0])
    print(f"\nПроверка TFLite: Keras={keras_score:.4f}, TFLite={tflite_score:.4f} "
          f"(расхождение={abs(keras_score-tflite_score):.6f})")


if __name__ == "__main__":
    main()
