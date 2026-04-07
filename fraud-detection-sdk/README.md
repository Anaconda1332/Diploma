# Fraud Detection SDK

This SDK provides functionality to collect device environment data for fraud detection purposes in Android applications.

## Features

- Active call detection
- Contact list data collection
- Permission management

## Installation

Add the following to your app's `build.gradle.kts`:

```kotlin
dependencies {
    implementation(project(":fraud-detection-sdk"))
}
```

## Usage

1. Initialize the SDK in your Application class or Activity:

```kotlin
val fraudSDK = FraudDetectionSDK.getInstance(context)
```

2. Request required permissions:

```kotlin
val missingPermissions = fraudSDK.getMissingPermissions()
if (missingPermissions.isNotEmpty()) {
    ActivityCompat.requestPermissions(
        activity,
        missingPermissions.toTypedArray(),
        PERMISSION_REQUEST_CODE
    )
}
```

3. Use the SDK methods:

```kotlin
// Check if there's an active call
val isCallActive = fraudSDK.isCallActive()

// Get contact count
val contactCount = fraudSDK.getContactCount()
```

4. (Optional) ML risk score — on-device model (TensorFlow Lite). First copy `fraud_risk_model.tflite` and `scaler_params.json` from the `ml/` folder (after running the Python training scripts) into `fraud-detection-sdk/src/main/assets/`. See `ml/README_ML.md` for step-by-step training.

```kotlin
val riskScorer = FraudRiskScorer.getInstance(context)
val score = riskScorer.evaluateRisk(fraudSDK, hourOfDay = null, dayOfWeek = null)
if (score != null) {
    val level = riskScorer.getRiskLevel(score)
    // level: LOW (0.0-0.3), MEDIUM (0.3-0.7), HIGH (0.7-1.0)
}
```

## Required Permissions

Add the following permissions to your AndroidManifest.xml:

```xml
<uses-permission android:name="android.permission.READ_CONTACTS" />
<uses-permission android:name="android.permission.READ_PHONE_STATE" />
```

## Security Considerations

- The SDK requires sensitive permissions. Make sure to request them only when necessary.
- Handle permission results appropriately in your application.
- Consider implementing proper error handling for cases when permissions are not granted.

## License

[Your License Here] 