package com.example.batterypredictor

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import androidx.core.app.NotificationCompat
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import kotlinx.coroutines.delay

class BatteryMonitoringWorker(
    private val context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    companion object {
        const val CHANNEL_ID = "battery_prediction_channel"
        const val NOTIFICATION_ID = 1001
    }

    override suspend fun doWork(): Result {
        try {
            // Collect battery features
            val features = collectBatteryFeatures()

            // Load ONNX client and predict
            val onnxClient = try {
                BatteryOrtClient(context)
            } catch (e: Exception) {
                return Result.failure()
            }

            val prediction = onnxClient.predict(features.toFeatureMap())

            // Check if alert is needed
            if (prediction.batteryDropPct > 5.0f || prediction.timeTo5PctHours < 2.0f) {
                showWarningNotification(prediction, features.batteryPct)
            }

            return Result.success()

        } catch (e: Exception) {
            return Result.retry()
        }
    }

    private fun collectBatteryFeatures(): BatteryFeatures {
        val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val batteryIntent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))

        val batteryPct = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.LOLLIPOP) {
            batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY).toFloat()
        } else {
            batteryIntent?.let { intent ->
                val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
                val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
                val batteryLevel = level.toFloat() / scale.toFloat() * 100
                batteryLevel
            } ?: 75.0f // Default if unavailable
        }

        return BatteryFeatures(
            batteryPct = batteryPct,
            screenOnHours = 2.0f, // This would need usage stats permission for accuracy
            brightness = 0.8f, // Screen brightness
            temperatureC = 28.0f,
            capacityHealth = 0.95f,
            locationLat = 28.61,
            locationLon = 77.21,
            appActive = "System", // Active app name
            networkActive = 1,
            fastCharge = 0,
            isWeekend = if (java.util.Calendar.getInstance().get(java.util.Calendar.DAY_OF_WEEK) in arrayOf(1,7)) 1 else 0
        )
    }

    private fun showWarningNotification(prediction: BatteryPrediction, currentLevel: Float) {
        val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

        // Create notification channel for Android 8.0+
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Battery Predictions",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Notifications for battery level predictions"
                enableVibration(true)
            }
            notificationManager.createNotificationChannel(channel)
        }

        val title = "Battery Warning ⚠️"
        val message = "Battery ${currentLevel.toInt()}% | Drop: ${prediction.batteryDropPct.format(1)}% | ${prediction.timeTo5PctHours.format(1)}hrs left"

        val notification = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setContentTitle(title)
            .setContentText(message)
            .setStyle(NotificationCompat.BigTextStyle().bigText(message))
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .build()

        notificationManager.notify(NOTIFICATION_ID, notification)
    }

    private fun Float.format(digits: Int): String = "%.${digits}f".format(this)
}
