# Константы для обучения и экспорта модели.
# Порядок признаков должен совпадать с FraudRiskScorer.kt.

NUM_FEATURES = 12

FEATURE_NAMES = [
    "is_rooted",            # 0 — рутованное устройство
    "log_contact_count",    # 1 — log(1 + кол-во контактов)
    "wifi_networks_count",  # 2 — кол-во Wi-Fi сетей (cap 50)
    "has_biometric",        # 3 — есть биометрия
    "sdk_version_norm",     # 4 — версия Android SDK / 34
    "is_call_active",       # 5 — активный звонок
    "hour_of_day_norm",     # 6 — час дня / 23
    "day_of_week_norm",     # 7 — день недели / 6
    "is_emulator",          # 8 — эмулятор (ферма устройств / тестовая среда)
    "is_usb_debugging",     # 9 — включена USB-отладка (ADB)
    "is_developer_options", # 10 — включены параметры разработчика
    "is_vpn_active",        # 11 — активен VPN
]

MODEL_TFLITE_PATH = "fraud_risk_model.tflite"
SCALER_JSON_PATH = "scaler_params.json"
TRAIN_DATA_PATH = "train_data.csv"
REPORT_PATH = "training_report.txt"
ROC_CURVE_PATH = "roc_curve.png"
