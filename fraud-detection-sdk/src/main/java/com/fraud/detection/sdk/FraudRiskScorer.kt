package com.fraud.detection.sdk

import android.content.Context
import android.util.Log
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.ln
import kotlin.math.min

/** Градации риска по score (0.0..1.0). */
enum class RiskLevel {
    LOW,    // ниже нижней границы
    MEDIUM, // между границами
    HIGH    // выше оптимального порога
}

/**
 * Модуль оценки риска мошенничества на основе on-device ML-модели (TensorFlow Lite).
 *
 * Версия 2: 12 признаков (было 8), оптимальный порог из scaler_params.json.
 *
 * Порядок признаков (совпадает с ml/config.py):
 *  0  is_rooted
 *  1  log_contact_count
 *  2  wifi_networks_count
 *  3  has_biometric
 *  4  sdk_version_norm
 *  5  is_call_active
 *  6  hour_of_day_norm
 *  7  day_of_week_norm
 *  8  is_emulator
 *  9  is_usb_debugging
 * 10  is_developer_options
 * 11  is_vpn_active
 */
class FraudRiskScorer private constructor(private val context: Context) {

    companion object {
        private const val TAG = "FraudRiskScorer"
        private const val MODEL_FILENAME = "fraud_risk_model.tflite"
        private const val SCALER_FILENAME = "scaler_params.json"
        private const val NUM_FEATURES = 12
        private const val SCALE_EPSILON = 1e-8f
        private const val DEFAULT_THRESHOLD = 0.5f

        @Volatile
        private var instance: FraudRiskScorer? = null

        fun getInstance(context: Context): FraudRiskScorer {
            return instance ?: synchronized(this) {
                instance ?: FraudRiskScorer(context.applicationContext).also { instance = it }
            }
        }
    }

    private var interpreter: Interpreter? = null
    private var scalerMean: FloatArray? = null
    private var scalerScale: FloatArray? = null
    private var optimalThreshold: Float = DEFAULT_THRESHOLD
    private var loadError: String? = null

    init {
        loadModelAndScaler()
    }

    private fun loadModelAndScaler() {
        try {
            val modelBuffer = loadAssetFile(MODEL_FILENAME) ?: return
            interpreter = Interpreter(modelBuffer)
            loadScaler()
        } catch (e: Exception) {
            loadError = e.message
            Log.w(TAG, "Не удалось загрузить модель или scaler: ${e.message}")
        }
    }

    private fun loadAssetFile(filename: String): MappedByteBuffer? {
        return try {
            val fd = context.assets.openFd(filename)
            FileInputStream(fd.fileDescriptor).use { fis ->
                val channel = fis.channel
                channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Файл не найден в assets: $filename — выполни шаги из ml/README_ML.md")
            null
        }
    }

    private fun loadScaler() {
        try {
            context.assets.open(SCALER_FILENAME).use { input ->
                val json = input.bufferedReader().use { it.readText() }
                val obj = JSONObject(json)
                val meanArr = obj.getJSONArray("mean")
                val scaleArr = obj.getJSONArray("scale")
                scalerMean = FloatArray(NUM_FEATURES) { meanArr.getDouble(it).toFloat() }
                scalerScale = FloatArray(NUM_FEATURES) {
                    scaleArr.getDouble(it).toFloat().coerceAtLeast(SCALE_EPSILON)
                }
                // Загружаем оптимальный порог, найденный при обучении
                if (obj.has("optimal_threshold")) {
                    optimalThreshold = obj.getDouble("optimal_threshold").toFloat()
                        .coerceIn(0.1f, 0.9f)
                    Log.d(TAG, "Оптимальный порог: $optimalThreshold")
                }
            }
        } catch (e: Exception) {
            loadError = "Scaler: ${e.message}"
            Log.w(TAG, "Не удалось загрузить scaler: ${e.message}")
        }
    }

    /**
     * Формирует вектор из 12 признаков.
     * Порядок строго совпадает с ml/config.py — FEATURE_NAMES.
     */
    private fun buildFeatureVector(
        sdk: FraudDetectionSDK,
        hourOfDay: Int?,
        dayOfWeek: Int?
    ): FloatArray {
        val deviceInfo = sdk.getDeviceInfo()
        val sdkVersion = deviceInfo["sdk_version"]?.toIntOrNull() ?: 34
        val contactCount = sdk.getContactCount()
        val logContact = ln(1.0f + contactCount.coerceAtLeast(0).toFloat())
        val wifiCount = min(sdk.getWiFiNetworks().size, 50).toFloat()
        val hasBiometric = if (sdk.getBiometricInfo()["can_authenticate"] == true) 1f else 0f

        return floatArrayOf(
            /* 0  */ if (sdk.isDeviceRooted()) 1f else 0f,
            /* 1  */ logContact,
            /* 2  */ wifiCount,
            /* 3  */ hasBiometric,
            /* 4  */ (sdkVersion / 34.0f).coerceIn(0f, 1f),
            /* 5  */ if (sdk.isCallActive()) 1f else 0f,
            /* 6  */ ((hourOfDay ?: 0).coerceIn(0, 23) / 23.0f),
            /* 7  */ ((dayOfWeek ?: 0).coerceIn(0, 6) / 6.0f),
            /* 8  */ if (sdk.isEmulator()) 1f else 0f,
            /* 9  */ if (sdk.isUsbDebuggingEnabled()) 1f else 0f,
            /* 10 */ if (sdk.isDeveloperOptionsEnabled()) 1f else 0f,
            /* 11 */ if (sdk.isVpnActive()) 1f else 0f,
        )
    }

    /** Нормализует вектор признаков: (x - mean) / scale (как StandardScaler в Python). */
    private fun normalizeFeatures(raw: FloatArray): FloatArray {
        val mean = scalerMean ?: return raw
        val scale = scalerScale ?: return raw
        return FloatArray(NUM_FEATURES) { i -> (raw[i] - mean[i]) / scale[i] }
    }

    /**
     * Оценивает риск мошенничества по текущим данным SDK.
     * @return риск 0.0..1.0 или null, если модель не загружена
     */
    fun evaluateRisk(
        sdk: FraudDetectionSDK,
        hourOfDay: Int? = null,
        dayOfWeek: Int? = null
    ): Float? {
        if (interpreter == null || scalerMean == null || scalerScale == null) {
            Log.w(TAG, "Модель не загружена: $loadError")
            return null
        }
        val raw = buildFeatureVector(sdk, hourOfDay, dayOfWeek)
        val normalized = normalizeFeatures(raw)
        val inputBuffer = ByteBuffer.allocateDirect(NUM_FEATURES * 4).order(ByteOrder.nativeOrder())
        normalized.forEach { inputBuffer.putFloat(it) }
        inputBuffer.rewind()
        val outputBuffer = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder())
        interpreter!!.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()
        return outputBuffer.getFloat(0).coerceIn(0f, 1f)
    }

    /**
     * Возвращает градацию риска по числовому score.
     * Использует оптимальный порог из обучения для границы MEDIUM/HIGH.
     */
    fun getRiskLevel(score: Float): RiskLevel = when {
        score < optimalThreshold * 0.4f -> RiskLevel.LOW
        score < optimalThreshold -> RiskLevel.MEDIUM
        else -> RiskLevel.HIGH
    }

    /** Возвращает оптимальный порог, загруженный из модели. */
    fun getOptimalThreshold(): Float = optimalThreshold

    /** Проверяет, загружена ли модель. */
    fun isModelLoaded(): Boolean = interpreter != null && scalerMean != null && scalerScale != null
}