package com.example.batterypredictor

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.*
import android.widget.Toast
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import androidx.work.*
import java.util.concurrent.TimeUnit

class MainActivity : AppCompatActivity() {

    private lateinit var batteryManager: BatteryManager
    private lateinit var onnxClient: BatteryOrtClient

    companion object {
        private const val PERMISSION_REQUEST_CODE = 1001
        const val CHANNEL_ID = "battery_prediction_channel"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize ONNX client
        try {
            onnxClient = BatteryOrtClient(this)
        } catch (e: Exception) {
            Toast.makeText(this, "Error loading battery prediction model: ${e.message}", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        // Request permissions
        requestPermissions()

        // Create notification channel
        createNotificationChannel()

        // Start service
        startBatteryMonitoring()

        // Set UI content
        setContent {
            BatteryPredictionApp(onnxClient)
        }
    }

    private fun requestPermissions() {
        val permissions = arrayOf(
            Manifest.permission.POST_NOTIFICATIONS
        )

        val permissionsToRequest = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }.toTypedArray()

        if (permissionsToRequest.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, permissionsToRequest, PERMISSION_REQUEST_CODE)
        }
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val name = getString(R.string.prediction_channel)
            val descriptionText = "Notifications for battery predictions"
            val importance = NotificationManager.IMPORTANCE_DEFAULT
            val channel = NotificationChannel(CHANNEL_ID, name, importance).apply {
                description = descriptionText
            }
            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannel(channel)
        }
    }

    private fun startBatteryMonitoring() {
        val workRequest = PeriodicWorkRequestBuilder<BatteryMonitoringWorker>(
            15, TimeUnit.MINUTES, // Check every 15 minutes
            5, TimeUnit.MINUTES   // Allow 5 minutes flexibility
        ).build()

        WorkManager.getInstance(this).enqueueUniquePeriodicWork(
            "battery_monitoring",
            ExistingPeriodicWorkPolicy.UPDATE,
            workRequest
        )
    }
}

// ViewModel for managing battery prediction state
class BatteryViewModel(private val onnxClient: BatteryOrtClient) : ViewModel() {

    private val _batteryState = MutableStateFlow<BatteryState>(BatteryState.Loading)
    val batteryState: StateFlow<BatteryState> = _batteryState.asStateFlow()

    init {
        startPeriodicUpdates()
    }

    private fun startPeriodicUpdates() {
        viewModelScope.launch {
            while (true) {
                try {
                    // Collect current battery features
                    val features = collectBatteryFeatures()
                    val prediction = onnxClient.predict(features.toFeatureMap())

                    // Update state
                    _batteryState.value = BatteryState.Success(
                        BatteryData(
                            level = features.batteryPct,
                            prediction = prediction
                        )
                    )

                    // Check for alert conditions
                    if (prediction.batteryDropPct > 5.0 || prediction.timeTo5PctHours < 2.0) {
                        _batteryState.value = BatteryState.Warning(
                            BatteryData(
                                level = features.batteryPct,
                                prediction = prediction
                            )
                        )
                    }

                } catch (e: Exception) {
                    _batteryState.value = BatteryState.Error(e.message ?: "Unknown error")
                }

                delay(60_000) // Update every minute
            }
        }
    }

    private fun collectBatteryFeatures(): BatteryFeatures {
        val batteryManager = // TODO: Get battery manager
        val currentBatteryPct = 75.0f // TODO: Get actual battery level
        val screenOnTime = 2.0f // TODO: Collect screen on time
        val brightness = 0.8f // TODO: Get brightness
        val temperature = 28.0f // TODO: Get temperature
        val capacityHealth = 0.95f // TODO: Get capacity health

        return BatteryFeatures(
            batteryPct = currentBatteryPct,
            screenOnHours = screenOnTime,
            brightness = brightness,
            temperatureC = temperature,
            capacityHealth = capacityHealth,
            locationLat = 28.61, // Default location
            locationLon = 77.21,
            appActive = "Chrome", // TODO: Get active app
            networkActive = 1,
            fastCharge = 0,
            isWeekend = if (java.util.Calendar.getInstance().get(java.util.Calendar.DAY_OF_WEEK) in arrayOf(1,7)) 1 else 0
        )
    }
}

// UI State classes
sealed class BatteryState {
    data object Loading : BatteryState()
    data class Success(val data: BatteryData) : BatteryState()
    data class Warning(val data: BatteryData) : BatteryState()
    data class Error(val message: String) : BatteryState()
}

data class BatteryData(
    val level: Float,
    val prediction: BatteryPrediction
)

// Main composable UI
@Composable
fun BatteryPredictionApp(onnxClient: BatteryOrtClient) {
    val viewModel = viewModel<BatteryViewModel>(
        factory = BatteryViewModelFactory(onnxClient)
    )
    val batteryState = viewModel.batteryState.collectAsState()

    MaterialTheme {
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.background
        ) {
            BatteryScreen(batteryState.value)
        }
    }
}

// Factory for ViewModel
class BatteryViewModelFactory(private val onnxClient: BatteryOrtClient) : androidx.lifecycle.ViewModelProvider.Factory {
    override fun <T : androidx.lifecycle.ViewModel> create(modelClass: Class<T>): T {
        return BatteryViewModel(onnxClient) as T
    }
}

// Battery display screen
@Composable
fun BatteryScreen(state: BatteryState) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "🔋 Battery Predictor",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(bottom = 32.dp)
        )

        when (state) {
            is BatteryState.Loading -> {
                CircularProgressIndicator()
                Text("Loading battery prediction...")
            }

            is BatteryState.Success -> {
                BatteryCard(state.data, false)
            }

            is BatteryState.Warning -> {
                BatteryCard(state.data, true)
            }

            is BatteryState.Error -> {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = "⚠️ Error",
                            style = MaterialTheme.typography.headlineSmall,
                            color = MaterialTheme.colorScheme.error
                        )
                        Text(
                            text = state.message,
                            style = MaterialTheme.typography.bodyMedium,
                            modifier = Modifier.padding(top = 8.dp)
                        )
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(32.dp))

        Text(
            text = "Background monitoring enabled",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
        )
    }
}

@Composable
fun BatteryCard(data: BatteryData, isWarning: Boolean) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (isWarning) MaterialTheme.colorScheme.errorContainer else MaterialTheme.colorScheme.surface
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.Start
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = "Battery Level: ${data.level.toInt()}%",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold
                )
                if (isWarning) {
                    Text(
                        text = " ⚠️",
                        style = MaterialTheme.typography.headlineSmall
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            Text(
                text = "Prediction:",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold
            )

            Text(
                text = "Battery drop (next hour): ${data.prediction.batteryDropPct.format(2)}%",
                style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier.padding(top = 8.dp)
            )

            Text(
                text = "Time to 5%: ${data.prediction.timeTo5PctHours.format(1)} hours",
                style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier.padding(top = 4.dp)
            )

            if (isWarning) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 16.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.error)
                ) {
                    Text(
                        text = "⚠️ Low battery warning! Consider charging soon.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onError,
                        modifier = Modifier.padding(12.dp)
                    )
                }
            }
        }
    }
}

// Extension function for formatting floats
fun Float.format(digits: Int): String = "%.${digits}f".format(this)
