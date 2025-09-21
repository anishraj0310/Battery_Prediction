# ONNX Runtime Android — minimal setup & inference

This is a **reference snippet**, not a full Gradle project. Drop these into your app module.

## 1) Gradle

**Project `settings.gradle`** — ensure Maven Central:
```gradle
dependencyResolutionManagement {
  repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
  repositories {
    google()
    mavenCentral()
  }
}
```

**Module `app/build.gradle`** — add ONNX Runtime Mobile:
```gradle
dependencies {
  implementation "com.microsoft.onnxruntime:onnxruntime-android:1.18.0"
  // If you need NNAPI or GPU delegates, consider the 'onnxruntime-android-mobile' artifact
}
android {
  compileSdkVersion 34
  defaultConfig {
    minSdkVersion 24
    targetSdkVersion 34
    // Place model + preproc json in app/src/main/assets/
  }
}
```

## 2) Assets
Copy the following to `app/src/main/assets/`:
- `model.onnx` (or `model.int8.onnx` / `model.qlinear.onnx`)
- `preproc_export.json`

## 3) Kotlin sample (see `BatteryOrtClient.kt`)
```kotlin
val ort = BatteryOrtClient(context)
val features = mapOf(
  "battery_pct" to 42.0, "screen_on_hours" to 0.6, "brightness" to 0.7,
  "temperature_C" to 35.2, "capacity_health" to 0.9,
  "location_lat" to 28.61, "location_lon" to 77.21,
  "app_active" to "YouTube", "network_active" to 1, "fast_charge" to 1, "is_weekend" to 0
)
val (delta, tte) = ort.predict(features)
```

That returns a pair: `(Δbattery_next_hour_pct, time_to_5pct_hours)`.