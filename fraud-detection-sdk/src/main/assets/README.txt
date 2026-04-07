Скопируй сюда файлы после обучения модели (см. ml/README_ML.md):

1. fraud_risk_model.tflite  — из папки ml/ после запуска train_and_export.py
2. scaler_params.json      — из папки ml/ после запуска train_and_export.py

Без этих файлов FraudRiskScorer не сможет загрузить модель и будет возвращать ошибку при evaluateRisk().
