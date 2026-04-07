package com.fraud.detection.sdk

import android.content.Context
import android.content.pm.PackageManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.wifi.WifiManager
import android.os.Build
import android.provider.ContactsContract
import android.provider.Settings
import android.telephony.TelephonyManager
import android.util.Log
import androidx.biometric.BiometricManager
import androidx.core.app.ActivityCompat
import java.io.File

class FraudDetectionSDK private constructor(private val context: Context) {

    companion object {
        private const val PERMISSION_READ_CONTACTS = android.Manifest.permission.READ_CONTACTS
        private const val PERMISSION_READ_PHONE_STATE = android.Manifest.permission.READ_PHONE_STATE
        private const val PERMISSION_ACCESS_FINE_LOCATION = android.Manifest.permission.ACCESS_FINE_LOCATION
        private const val PERMISSION_ACCESS_WIFI_STATE = android.Manifest.permission.ACCESS_WIFI_STATE
        private const val PERMISSION_CHANGE_WIFI_STATE = android.Manifest.permission.CHANGE_WIFI_STATE

        @Volatile
        private var instance: FraudDetectionSDK? = null

        fun getInstance(context: Context): FraudDetectionSDK {
            return instance ?: synchronized(this) {
                instance ?: FraudDetectionSDK(context.applicationContext).also { instance = it }
            }
        }
    }

    fun getDeviceInfo(): Map<String, String> {
        return mapOf(
            "manufacturer" to Build.MANUFACTURER,
            "model" to Build.MODEL,
            "os_version" to Build.VERSION.RELEASE,
            "sdk_version" to Build.VERSION.SDK_INT.toString(),
            "device" to Build.DEVICE,
            "product" to Build.PRODUCT,
            "brand" to Build.BRAND,
            "hardware" to Build.HARDWARE,
            "fingerprint" to Build.FINGERPRINT
        )
    }

    fun isDeviceRooted(): Boolean {
        val buildTags = Build.TAGS
        if (buildTags != null && buildTags.contains("test-keys")) {
            return true
        }

        val rootPaths = arrayOf(
            "/system/app/Superuser.apk",
            "/sbin/su",
            "/system/bin/su",
            "/system/xbin/su",
            "/data/local/xbin/su",
            "/data/local/bin/su",
            "/system/sd/xbin/su",
            "/system/bin/failsafe/su",
            "/data/local/su"
        )

        return rootPaths.any { File(it).exists() }
    }

    fun getBiometricInfo(): Map<String, Boolean> {
        val biometricManager = BiometricManager.from(context)
        return mapOf(
            "can_authenticate" to (biometricManager.canAuthenticate(BiometricManager.Authenticators.BIOMETRIC_STRONG) == BiometricManager.BIOMETRIC_SUCCESS),
            "has_fingerprint" to (biometricManager.canAuthenticate(BiometricManager.Authenticators.BIOMETRIC_STRONG) == BiometricManager.BIOMETRIC_SUCCESS),
            "has_face" to (biometricManager.canAuthenticate(BiometricManager.Authenticators.BIOMETRIC_STRONG) == BiometricManager.BIOMETRIC_SUCCESS)
        )
    }


    fun isCallActive(): Boolean {
        if (!hasPhoneStatePermission()) {
            return false
        }

        val telephonyManager = context.getSystemService(Context.TELEPHONY_SERVICE) as TelephonyManager
        return telephonyManager.callState == TelephonyManager.CALL_STATE_OFFHOOK
    }


    fun getContactCount(): Int {
        if (!hasContactsPermission()) {
            return -1
        }

        val cursor = context.contentResolver.query(
            ContactsContract.Contacts.CONTENT_URI,
            null,
            null,
            null,
            null
        )

        return cursor?.count ?: 0
    }

    fun getWiFiNetworks(): List<Map<String, String>> {
        if (!hasLocationPermission() || !hasWifiStatePermission()) {
            return emptyList()
        }

        return try {
            val wifiManager = context.getSystemService(Context.WIFI_SERVICE) as WifiManager
            
            if (!wifiManager.isWifiEnabled) {
                return emptyList()
            }

            // Инициируем сканирование если результаты устарели
            if (wifiManager.scanResults.isEmpty()) {
                if (hasChangeWifiStatePermission()) {
                    wifiManager.startScan()
                }
            }

            val scanResults = wifiManager.scanResults
            scanResults.map { scanResult ->
                mapOf(
                    "ssid" to (scanResult.SSID.ifEmpty { "Hidden Network" }),
                    "bssid" to scanResult.BSSID,
                    "capabilities" to scanResult.capabilities,
                    "frequency" to scanResult.frequency.toString(),
                    "level" to scanResult.level.toString(),
                    "timestamp" to scanResult.timestamp.toString()
                )
            }
        } catch (e: SecurityException) {
            // Логируем ошибку безопасности
            Log.w("FraudDetectionSDK", "Security exception in getWiFiNetworks: ${e.message}")
            emptyList()
        } catch (e: Exception) {
            // Логируем другие ошибки
            Log.e("FraudDetectionSDK", "Error in getWiFiNetworks: ${e.message}")
            emptyList()
        }
    }

    fun startWiFiScan(): Boolean {
        if (!hasLocationPermission() || !hasWifiStatePermission() || !hasChangeWifiStatePermission()) {
            return false
        }

        return try {
            val wifiManager = context.getSystemService(Context.WIFI_SERVICE) as WifiManager
            
            if (!wifiManager.isWifiEnabled) {
                return false
            }

            wifiManager.startScan()
        } catch (e: SecurityException) {
            // Логируем ошибку безопасности
            Log.w("FraudDetectionSDK", "Security exception in startWiFiScan: ${e.message}")
            false
        } catch (e: Exception) {
            // Логируем другие ошибки
            Log.e("FraudDetectionSDK", "Error in startWiFiScan: ${e.message}")
            false
        }
    }

    fun getMissingPermissions(): List<String> {
        val missingPermissions = mutableListOf<String>()

        if (!hasContactsPermission()) {
            missingPermissions.add(PERMISSION_READ_CONTACTS)
        }

        if (!hasPhoneStatePermission()) {
            missingPermissions.add(PERMISSION_READ_PHONE_STATE)
        }

        if (!hasLocationPermission()) {
            missingPermissions.add(PERMISSION_ACCESS_FINE_LOCATION)
        }

        if (!hasWifiStatePermission()) {
            missingPermissions.add(PERMISSION_ACCESS_WIFI_STATE)
        }

        if (!hasChangeWifiStatePermission()) {
            missingPermissions.add(PERMISSION_CHANGE_WIFI_STATE)
        }

        return missingPermissions
    }

    private fun hasContactsPermission(): Boolean {
        return ActivityCompat.checkSelfPermission(
            context,
            PERMISSION_READ_CONTACTS
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun hasPhoneStatePermission(): Boolean {
        return ActivityCompat.checkSelfPermission(
            context,
            PERMISSION_READ_PHONE_STATE
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun hasLocationPermission(): Boolean {
        return ActivityCompat.checkSelfPermission(
            context,
            PERMISSION_ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun hasWifiStatePermission(): Boolean {
        return ActivityCompat.checkSelfPermission(
            context,
            PERMISSION_ACCESS_WIFI_STATE
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun hasChangeWifiStatePermission(): Boolean {
        return ActivityCompat.checkSelfPermission(
            context,
            PERMISSION_CHANGE_WIFI_STATE
        ) == PackageManager.PERMISSION_GRANTED
    }

    /**
     * Определяет, запущено ли приложение на эмуляторе или в тестовой среде (ферма устройств).
     * Проверяет характерные строки в Build-параметрах.
     */
    fun isEmulator(): Boolean {
        return Build.FINGERPRINT.startsWith("generic")
            || Build.FINGERPRINT.startsWith("unknown")
            || Build.FINGERPRINT.contains("emulator")
            || Build.MODEL.contains("google_sdk", ignoreCase = true)
            || Build.MODEL.contains("Emulator", ignoreCase = true)
            || Build.MODEL.contains("Android SDK built for x86", ignoreCase = true)
            || Build.MANUFACTURER.contains("Genymotion", ignoreCase = true)
            || (Build.BRAND.startsWith("generic") && Build.DEVICE.startsWith("generic"))
            || Build.PRODUCT == "google_sdk"
            || Build.HARDWARE.contains("goldfish", ignoreCase = true)
            || Build.HARDWARE.contains("ranchu", ignoreCase = true)
    }

    /**
     * Проверяет, включена ли USB-отладка (ADB).
     * Активная отладка на устройстве — признак инструментального использования.
     */
    fun isUsbDebuggingEnabled(): Boolean {
        return Settings.Global.getInt(
            context.contentResolver,
            Settings.Global.ADB_ENABLED, 0
        ) > 0
    }

    /**
     * Проверяет, включён ли режим разработчика.
     * Комбинация с рутом или эмулятором резко повышает риск.
     */
    fun isDeveloperOptionsEnabled(): Boolean {
        return Settings.Global.getInt(
            context.contentResolver,
            Settings.Global.DEVELOPMENT_SETTINGS_ENABLED, 0
        ) > 0
    }

    /**
     * Проверяет, активен ли VPN-туннель.
     * VPN может использоваться для сокрытия реального местоположения при мошенничестве.
     */
    fun isVpnActive(): Boolean {
        return try {
            val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
            cm.allNetworks.any { network ->
                cm.getNetworkCapabilities(network)
                    ?.hasTransport(NetworkCapabilities.TRANSPORT_VPN) == true
            }
        } catch (e: Exception) {
            Log.w("FraudDetectionSDK", "isVpnActive error: ${e.message}")
            false
        }
    }
}