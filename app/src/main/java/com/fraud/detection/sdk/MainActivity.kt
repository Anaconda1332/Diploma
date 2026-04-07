package com.fraud.detection.sdk

import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.fraud.detection.sdk.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var fraudSDK: FraudDetectionSDK
    private lateinit var riskScorer: FraudRiskScorer
    private lateinit var binding: ActivityMainBinding

    companion object {
        private const val PERMISSION_REQUEST_CODE = 123
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        fraudSDK = FraudDetectionSDK.getInstance(this)
        riskScorer = FraudRiskScorer.getInstance(this)

        binding.calculateRiskButton.setOnClickListener {
            calculateRiskScore()
        }

        binding.checkPermissionsButton.setOnClickListener {
            checkAndRequestPermissions()
        }

        binding.refreshDataButton.setOnClickListener {
            refreshData()
        }

        binding.scanWifiButton.setOnClickListener {
            scanWiFiNetworks()
        }

        checkAndRequestPermissions()
    }

    private fun checkAndRequestPermissions() {
        val missingPermissions = fraudSDK.getMissingPermissions()
        if (missingPermissions.isEmpty()) {
            Toast.makeText(this, "All permissions are granted", Toast.LENGTH_SHORT).show()
            refreshData()
        } else {
            ActivityCompat.requestPermissions(
                this,
                missingPermissions.toTypedArray(),
                PERMISSION_REQUEST_CODE
            )
        }
    }

    private fun refreshData() {
        val deviceInfo = fraudSDK.getDeviceInfo()
        binding.deviceInfoTextView.text = deviceInfo.entries.joinToString("\n") { "${it.key}: ${it.value}" }

        val isRooted = fraudSDK.isDeviceRooted()
        binding.rootStatusTextView.text = if (isRooted) "Device is rooted" else "Device is not rooted"

        val isEmulator = fraudSDK.isEmulator()
        val isUsbDebug = fraudSDK.isUsbDebuggingEnabled()
        val isDevOptions = fraudSDK.isDeveloperOptionsEnabled()
        val isVpn = fraudSDK.isVpnActive()
        binding.securityIndicatorsTextView.text = buildString {
            appendLine("Emulator:          ${if (isEmulator) "YES ⚠" else "No"}")
            appendLine("USB Debugging:     ${if (isUsbDebug) "ON ⚠" else "Off"}")
            appendLine("Developer Options: ${if (isDevOptions) "ON ⚠" else "Off"}")
            append("VPN Active:        ${if (isVpn) "YES ⚠" else "No"}")
        }

        val biometricInfo = fraudSDK.getBiometricInfo()
        binding.biometricsTextView.text = biometricInfo.entries.joinToString("\n") { "${it.key}: ${it.value}" }

        val isCallActive = fraudSDK.isCallActive()
        binding.callStatusTextView.text = if (isCallActive) "Active" else "No active call"

        val contactCount = fraudSDK.getContactCount()
        binding.contactCountTextView.text = when (contactCount) {
            -1 -> "Permission not granted"
            else -> "Total contacts: $contactCount"
        }

        val wifiNetworks = fraudSDK.getWiFiNetworks()
        if (wifiNetworks.isEmpty()) {
            binding.wifiNetworksTextView.text = "No Wi-Fi networks found or permission not granted"
        } else {
            val wifiInfo = wifiNetworks.joinToString("\n\n") { network ->
                network.entries.joinToString("\n") { "${it.key}: ${it.value}" }
            }
            binding.wifiNetworksTextView.text = "Found ${wifiNetworks.size} networks:\n\n$wifiInfo"
        }
    }

    private fun calculateRiskScore() {
        if (!riskScorer.isModelLoaded()) {
            binding.riskScoreTextView.text = "Модель не загружена. Скопируйте fraud_risk_model.tflite и scaler_params.json в fraud-detection-sdk/src/main/assets/"
            Toast.makeText(this, "Модель не загружена", Toast.LENGTH_LONG).show()
            return
        }
        val hourOfDay = java.util.Calendar.getInstance().get(java.util.Calendar.HOUR_OF_DAY)
        val dayOfWeek = java.util.Calendar.getInstance().get(java.util.Calendar.DAY_OF_WEEK) - 1
        val score = riskScorer.evaluateRisk(fraudSDK, hourOfDay, dayOfWeek)
        if (score == null) {
            binding.riskScoreTextView.text = "Ошибка расчёта риска"
            Toast.makeText(this, "Ошибка расчёта", Toast.LENGTH_SHORT).show()
            return
        }
        val level = riskScorer.getRiskLevel(score)
        val threshold = riskScorer.getOptimalThreshold()
        val levelText = when (level) {
            RiskLevel.LOW -> "Низкий"
            RiskLevel.MEDIUM -> "Средний"
            RiskLevel.HIGH -> "Высокий"
        }
        binding.riskScoreTextView.text = buildString {
            appendLine("Score: %.3f".format(score))
            appendLine("Уровень риска: $levelText")
            append("Оптимальный порог: %.3f".format(threshold))
        }
        Toast.makeText(this, "Risk: %.2f — %s".format(score, levelText), Toast.LENGTH_SHORT).show()
    }

    private fun scanWiFiNetworks() {
        if (!fraudSDK.getMissingPermissions().isEmpty()) {
            Toast.makeText(this, "Please grant all permissions first", Toast.LENGTH_SHORT).show()
            return
        }

        val scanStarted = fraudSDK.startWiFiScan()
        if (scanStarted) {
            Toast.makeText(this, "Wi-Fi scan started", Toast.LENGTH_SHORT).show()
            // Даем время на сканирование и затем обновляем данные
            binding.wifiNetworksTextView.postDelayed({
                val wifiNetworks = fraudSDK.getWiFiNetworks()
                if (wifiNetworks.isEmpty()) {
                    binding.wifiNetworksTextView.text = "No Wi-Fi networks found or Wi-Fi is disabled"
                } else {
                    val wifiInfo = wifiNetworks.joinToString("\n\n") { network ->
                        network.entries.joinToString("\n") { "${it.key}: ${it.value}" }
                    }
                    binding.wifiNetworksTextView.text = "Found ${wifiNetworks.size} networks:\n\n$wifiInfo"
                }
            }, 2000) // 2 секунды на сканирование
        } else {
            binding.wifiNetworksTextView.text = "Failed to start Wi-Fi scan"
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                Toast.makeText(this, "All permissions granted", Toast.LENGTH_SHORT).show()
                refreshData()
            } else {
                Toast.makeText(this, "Some permissions were denied", Toast.LENGTH_SHORT).show()
            }
        }
    }
} 