"""
Генерация синтетических учебных данных для модели оценки риска мошенничества.

Улучшения по сравнению с версией 1:
  - 5 000 образцов (было 2 000)
  - Реалистичный дисбаланс классов: ~15 % fraud, ~85 % normal (было 50/50)
  - 4 новых признака: is_emulator, is_usb_debugging, is_developer_options, is_vpn_active
  - Комбинированные правила разметки — учитывают взаимодействие признаков
  - Сниженный шум разметки: 3 % (было 5 %)

Запуск: python generate_synthetic_data.py
"""

import random
import math
import pandas as pd
from config import FEATURE_NAMES, TRAIN_DATA_PATH

RANDOM_SEED = 42
NUM_SAMPLES = 5000
FRAUD_RATIO = 0.15   # 15 % мошенников — реалистичное соотношение

random.seed(RANDOM_SEED)


def row_to_label(
    is_rooted, log_contact, wifi_count, has_biometric,
    sdk_norm, is_call, hour_norm, day_norm,
    is_emulator, is_usb_debug, is_dev_options, is_vpn
):
    """
    Улучшенная функция разметки: учитывает комбинации признаков,
    характерные для реальных схем мошенничества.
    """
    score = 0.0

    # --- Одиночные сильные сигналы ---
    if is_rooted:
        score += 0.35
    if is_emulator:
        score += 0.40   # эмулятор — очень сильный сигнал (фермы устройств)
    if is_vpn:
        score += 0.15   # скрытие реального местоположения

    # --- Комбинированные сигналы (сильнее одиночных) ---
    # Эмулятор + USB-отладка = автоматизированная атака
    if is_emulator and is_usb_debug:
        score += 0.20
    # Рут + режим разработчика = обход защиты
    if is_rooted and is_dev_options:
        score += 0.15
    # VPN + активный звонок ночью = социальная инженерия под прикрытием
    if is_vpn and is_call and hour_norm < 0.22:   # до 5 утра
        score += 0.20
    # Рут + мало контактов = "свежее" устройство-инструмент
    if is_rooted and log_contact < 1.0:
        score += 0.15

    # --- Слабые одиночные сигналы ---
    if log_contact < 1.0:           # 0–1 контакт
        score += 0.15
    if wifi_count > 20:             # много сетей → ферма/эмулятор
        score += 0.10
    if not has_biometric:
        score += 0.08
    if sdk_norm < 0.65:             # Android < 22 (старая версия)
        score += 0.08
    if is_usb_debug and not is_emulator:
        score += 0.06
    if is_dev_options and not is_rooted:
        score += 0.05
    if is_call:
        score += 0.04
    # Подозрительные временные окна
    if hour_norm < 0.13 or hour_norm > 0.96:   # 00:00–03:00 или 23:xx
        score += 0.06

    # Случайный шум (уменьшен до ±0.08)
    score += random.uniform(-0.08, 0.08)
    return 1 if score >= 0.5 else 0


def generate_fraud_sample():
    """Генерирует образец с профилем мошенника."""
    is_rooted = random.random() < 0.55
    contact_count = random.randint(0, 15)
    wifi_count = random.randint(5, 45)
    has_biometric = random.random() < 0.35
    sdk_int = random.randint(21, 31)
    is_call = random.random() < 0.25
    hour = random.choice(
        list(range(0, 5)) + list(range(0, 24))   # перевес на ночные часы
    )
    day = random.randint(0, 6)
    is_emulator = random.random() < 0.50
    is_usb_debug = random.random() < 0.60 if is_emulator else random.random() < 0.30
    is_dev_options = random.random() < 0.55 if is_rooted else random.random() < 0.25
    is_vpn = random.random() < 0.40
    return (is_rooted, contact_count, wifi_count, has_biometric,
            sdk_int, is_call, hour, day, is_emulator, is_usb_debug, is_dev_options, is_vpn)


def generate_normal_sample():
    """Генерирует образец с профилем обычного пользователя."""
    is_rooted = random.random() < 0.05
    contact_count = random.randint(20, 300)
    wifi_count = random.randint(1, 12)
    has_biometric = random.random() < 0.82
    sdk_int = random.randint(28, 34)
    is_call = random.random() < 0.08
    hour = random.randint(6, 23)   # днём и вечером
    day = random.randint(0, 6)
    is_emulator = random.random() < 0.02
    is_usb_debug = random.random() < 0.07
    is_dev_options = random.random() < 0.10
    is_vpn = random.random() < 0.15
    return (is_rooted, contact_count, wifi_count, has_biometric,
            sdk_int, is_call, hour, day, is_emulator, is_usb_debug, is_dev_options, is_vpn)


def generate_dataset():
    rows = []
    fraud_count = 0
    normal_count = 0

    for _ in range(NUM_SAMPLES):
        is_fraud_target = random.random() < FRAUD_RATIO

        if is_fraud_target:
            (is_rooted, contact_count, wifi_count, has_biometric,
             sdk_int, is_call, hour, day, is_emulator, is_usb_debug,
             is_dev_options, is_vpn) = generate_fraud_sample()
        else:
            (is_rooted, contact_count, wifi_count, has_biometric,
             sdk_int, is_call, hour, day, is_emulator, is_usb_debug,
             is_dev_options, is_vpn) = generate_normal_sample()

        log_contact = math.log1p(max(0, contact_count))
        wifi_capped = min(wifi_count, 50)
        sdk_norm = sdk_int / 34.0
        hour_norm = hour / 23.0
        day_norm = day / 6.0

        label = row_to_label(
            is_rooted, log_contact, wifi_capped, has_biometric,
            sdk_norm, is_call, hour_norm, day_norm,
            is_emulator, is_usb_debug, is_dev_options, is_vpn
        )

        # Шум разметки: 3 % (было 5 %)
        if random.random() < 0.03:
            label = 1 - label

        if label == 1:
            fraud_count += 1
        else:
            normal_count += 1

        rows.append({
            "is_rooted":            float(is_rooted),
            "log_contact_count":    log_contact,
            "wifi_networks_count":  float(wifi_capped),
            "has_biometric":        float(has_biometric),
            "sdk_version_norm":     sdk_norm,
            "is_call_active":       float(is_call),
            "hour_of_day_norm":     hour_norm,
            "day_of_week_norm":     day_norm,
            "is_emulator":          float(is_emulator),
            "is_usb_debugging":     float(is_usb_debug),
            "is_developer_options": float(is_dev_options),
            "is_vpn_active":        float(is_vpn),
            "label":                label,
        })

    df = pd.DataFrame(rows)
    df.to_csv(TRAIN_DATA_PATH, index=False)
    print(f"Сохранено {len(df)} строк в {TRAIN_DATA_PATH}")
    print(f"Fraud: {fraud_count} ({fraud_count/len(df)*100:.1f}%), "
          f"Normal: {normal_count} ({normal_count/len(df)*100:.1f}%)")


if __name__ == "__main__":
    generate_dataset()