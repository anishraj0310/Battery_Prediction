// Minimal ONNX Runtime client for Android to run the exported model
// Place this in your app module (e.g., app/src/main/java/your/pkg/BatteryOrtClient.kt)

package com.example.batterypredictor

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.FloatBuffer

class BatteryOrtClient(private val context: Context, private val seqLen: Int = 6) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val preproc: JSONObject
    private val inputDim: Int

    init {
        // Load model from assets
        val modelBytes = context.assets.open("model.int8.onnx").use { it.readBytes() }
        session = env.createSession(modelBytes, OrtSession.SessionOptions())

        // Load preprocessing configuration
        val json = context.assets.open("preproc_export.json").use { input ->
            BufferedReader(InputStreamReader(input)).readText()
        }
        preproc = JSONObject(json)

        // Calculate input dimension: numeric + categorical features
        val numFeatures: JSONArray<*> = preproc.getJSONArray("num_features")
        val catCategories: JSONArray<*> = preproc.getJSONArray("cat_categories")
        inputDim = numFeatures.length() + catCategories.length() // each categorical feature is one-hot encoded
    }

    fun predict(features: Map<String, Any>): BatteryPrediction {
        val vec = vectorize(preproc, features)

        // Create input tensor [1, seqLen, inputDim]
        val totalSize = seqLen * inputDim
        val buf = FloatBuffer.allocate(totalSize)
        repeat(seqLen) { buf.put(vec) }
        buf.rewind()

        val shape = longArrayOf(1L, seqLen.toLong(), inputDim.toLong())
        val inputTensor = OnnxTensor.createTensor(env, buf, shape)

        // Run inference
        val outputs = session.run(mapOf("input" to inputTensor))
        val deltaPct = (outputs[0].value as Array<FloatArray>)[0][0]
        val tteHours = (outputs[1].value as Array<FloatArray>)[0][0]

        outputs.close()
        inputTensor.close()

        return BatteryPrediction(deltaPct, tteHours)
    }

    private fun vectorize(preprocData: JSONObject, feats: Map<String, Any>): FloatArray {
        val numFeatures = preprocData.getJSONArray("num_features")
        val numMins = preprocData.getJSONArray("num_mins")
        val numScales = preprocData.getJSONArray("num_scales")
        val catFeatures = preprocData.getJSONArray("cat_features")
        val catCategories = preprocData.getJSONArray("cat_categories")

        // Process numeric features
        val nums = FloatArray(numFeatures.length())
        for (i in 0 until numFeatures.length()) {
            val key = numFeatures.getString(i)
            val value = (feats[key] as Number).toDouble()
            val min = numMins.getDouble(i)
            val scale = if (numScales.getDouble(i) == 0.0) 1.0 else numScales.getDouble(i)
            nums[i] = ((value - min) / scale).toFloat()
        }

        // Process categorical features (single integer encoding)
        val cats = FloatArray(catFeatures.length())
        for (i in 0 until catFeatures.length()) {
            val key = catFeatures.getString(i)
            val categoriesForFeature = catCategories.getJSONArray(i)
            val featureValue = feats[key]

            // Find index of the feature value in categories
            var index = 0
            for (j in 0 until categoriesForFeature.length()) {
                if (categoriesForFeature.get(j) == featureValue) {
                    index = j
                    break
                }
            }
            cats[i] = index.toFloat()
        }

        // Combine numeric and categorical features
        val result = FloatArray(nums.size + cats.size)
        nums.copyInto(result, 0, 0, nums.size)
        cats.copyInto(result, nums.size, 0, cats.size)

        return result
    }
}

// Data class for prediction results
data class BatteryPrediction(
    val batteryDropPct: Float,
    val timeTo5PctHours: Float
)

// Feature collection data class
data class BatteryFeatures(
    val batteryPct: Float,
    val screenOnHours: Float,
    val brightness: Float,
    val temperatureC: Float,
    val capacityHealth: Float,
    val locationLat: Double,
    val locationLon: Double,
    val appActive: String,
    val networkActive: Int,
    val fastCharge: Int,
    val isWeekend: Int
) {
    // Convert to feature map for prediction
    fun toFeatureMap(): Map<String, Any> = mapOf(
        "battery_pct" to batteryPct,
        "screen_on_hours" to screenOnHours,
        "brightness" to brightness,
        "temperature_C" to temperatureC,
        "capacity_health" to capacityHealth,
        "location_lat" to locationLat,
        "location_lon" to locationLon,
        "app_active" to appActive,
        "network_active" to networkActive,
        "fast_charge" to fastCharge,
        "is_weekend" to isWeekend
    )
}
