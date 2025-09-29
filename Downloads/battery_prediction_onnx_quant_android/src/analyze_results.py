"""
Including feature analysis, correlations, and detailed performance metrics.
"""
import pandas as pd
import numpy as np
import json
import onnxruntime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_and_preprocess(csv_path, preproc_path, seq_len=6):
    # Load preprocessing params
    with open(preproc_path) as f:
        preproc = json.load(f)
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Preprocess features
    for feat, min_val, scale in zip(preproc["num_features"], preproc["num_mins"], preproc["num_scales"]):
        df[feat] = (df[feat] - min_val) / scale
    
    # Convert categorical
    for feat, cats in zip(preproc["cat_features"], preproc["cat_categories"]):
        df[feat] = pd.Categorical(df[feat], categories=cats).codes
    
    return df

def get_predictions(model_path, df, preproc_path, seq_len=6):
    # Load preprocessing params
    with open(preproc_path) as f:
        preproc = json.load(f)
    
    # Initialize ONNX Runtime
    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Create sequences
    features = preproc["num_features"] + preproc["cat_features"]
    sequences = []
    actuals = []
    
    for i in range(len(df) - seq_len):
        seq = []
        for t in range(seq_len):
            row = df.iloc[i + t]
            vec = [row[f] for f in features]
            seq.append(vec)
        actuals.append([
            df.iloc[i + seq_len]["delta_next_hour_pct"],
            df.iloc[i + seq_len]["tte_5pct_hours"],
            df.iloc[i + seq_len]["energy_saved_by_alert"]
        ])
        sequences.append(seq)
    
    # Get predictions
    inputs = np.array(sequences, dtype=np.float32)
    predictions = session.run(None, {"input": inputs})[0]
    
    return np.array(predictions), np.array(actuals)

# Load data and get predictions
df = load_and_preprocess("data/synthetic_sessions.csv", "runs/gru/preproc_export.json")
predictions, actuals = get_predictions("runs/gru/model.onnx", df, "runs/gru/preproc_export.json")

# Calculate metrics
metrics = {
    "Battery Drop Rate (Next Hour)": {
        "MSE": mean_squared_error(actuals[:, 0], predictions[:, 0]),
        "MAE": mean_absolute_error(actuals[:, 0], predictions[:, 0]),
        "R2": r2_score(actuals[:, 0], predictions[:, 0])
    },
    "Time to 5% Battery": {
        "MSE": mean_squared_error(actuals[:, 1], predictions[:, 1]),
        "MAE": mean_absolute_error(actuals[:, 1], predictions[:, 1]),
        "R2": r2_score(actuals[:, 1], predictions[:, 1])
    },
    "Energy Saved by Alert": {
        "MSE": mean_squared_error(actuals[:, 2], predictions[:, 2]),
        "MAE": mean_absolute_error(actuals[:, 2], predictions[:, 2]),
        "R2": r2_score(actuals[:, 2], predictions[:, 2])
    }
}

# Print metrics table
print("\nModel Performance Metrics:")
print("-" * 50)
for target, metric_dict in metrics.items():
    print(f"\n{target}:")
    for metric_name, value in metric_dict.items():
        print(f"{metric_name}: {value:.4f}")

# Create visualization directory
import os
os.makedirs("analysis_results", exist_ok=True)

# Load original data for feature analysis
original_df = pd.read_csv("data/synthetic_sessions.csv")

# Feature Analysis
def analyze_features(df):
    # 1. Correlation Matrix for Numeric Features
    numeric_features = ["battery_pct", "screen_on_hours", "brightness", "temperature_C", 
                       "capacity_health", "location_lat", "location_lon", 
                       "delta_next_hour_pct", "tte_5pct_hours"]
    corr = df[numeric_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('analysis_results/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Distributions
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(numeric_features[:-2], 1):  # Excluding target variables
        plt.subplot(2, 4, i)
        sns.histplot(df[feature], kde=True)
        plt.title(f'{feature} Distribution')
        plt.xlabel(feature)
    plt.tight_layout()
    plt.savefig('analysis_results/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Categorical Feature Analysis
    cat_features = ["app_active", "network_active", "fast_charge", "is_weekend"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for i, feature in enumerate(cat_features):
        row = i // 2
        col = i % 2
        df[feature].value_counts().plot(kind='bar', ax=axes[row, col])
        axes[row, col].set_title(f'{feature} Distribution')
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel('Count')
    plt.tight_layout()
    plt.savefig('analysis_results/categorical_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Feature vs Target Relationships
    targets = ["delta_next_hour_pct", "tte_5pct_hours"]
    for target in targets:
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(numeric_features[:-2], 1):
            if feature != target:
                plt.subplot(2, 4, i)
                plt.scatter(df[feature], df[target], alpha=0.5)
                plt.xlabel(feature)
                plt.ylabel(target)
                plt.title(f'{feature} vs {target}')
        plt.tight_layout()
        plt.savefig(f'analysis_results/features_vs_{target}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    # 5. Battery Drain Patterns
    plt.figure(figsize=(15, 6))
    
    # Battery drain by app
    plt.subplot(1, 2, 1)
    sns.boxplot(x='app_active', y='delta_next_hour_pct', data=df)
    plt.title('Battery Drain by App')
    plt.xticks(rotation=45)
    
    # Battery drain by time (weekend vs weekday)
    plt.subplot(1, 2, 2)
    sns.boxplot(x='is_weekend', y='delta_next_hour_pct', data=df)
    plt.title('Battery Drain: Weekend vs Weekday')
    
    plt.tight_layout()
    plt.savefig('analysis_results/battery_drain_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Temperature Effects
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(df['temperature_C'], df['delta_next_hour_pct'], alpha=0.5)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Battery Drain Rate (%/hour)')
    plt.title('Temperature vs Battery Drain Rate')
    
    plt.subplot(1, 2, 2)
    plt.scatter(df['temperature_C'], df['tte_5pct_hours'], alpha=0.5)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Time to 5% (hours)')
    plt.title('Temperature vs Battery Life')
    
    plt.tight_layout()
    plt.savefig('analysis_results/temperature_effects.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Screen Usage Impact
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(df['screen_on_hours'], df['delta_next_hour_pct'], alpha=0.5)
    plt.xlabel('Screen On Time (hours)')
    plt.ylabel('Battery Drain Rate (%/hour)')
    plt.title('Screen Usage vs Battery Drain Rate')
    
    plt.subplot(1, 2, 2)
    plt.scatter(df['brightness'], df['delta_next_hour_pct'], alpha=0.5)
    plt.xlabel('Screen Brightness')
    plt.ylabel('Battery Drain Rate (%/hour)')
    plt.title('Screen Brightness vs Battery Drain Rate')
    
    plt.tight_layout()
    plt.savefig('analysis_results/screen_usage_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

# Energy Saving Analysis
def analyze_energy_savings(df):
    # Basic app and network analysis
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # First create separate detailed temperature analysis
    plt.figure(figsize=(15, 8))
    temp_ranges = pd.qcut(df['temperature_C'], q=10)  # Increased granularity
    efficiency_by_temp = df.groupby(temp_ranges)['delta_next_hour_pct'].agg(['mean', 'std', 'count']).abs()
    
    # Main temperature plot
    ax1 = plt.gca()
    efficiency_by_temp.plot(kind='bar', y='mean', yerr='std', capsize=5, 
                          color='skyblue', error_kw={'ecolor': 'red'}, ax=ax1)
    plt.title('Temperature Impact on Battery Efficiency', pad=20, fontsize=14, fontweight='bold')
    plt.xlabel('Temperature Range (°C)', labelpad=10, fontsize=12)
    plt.ylabel('Average Battery Drain (%/hour)', labelpad=10, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add sample count as text
    for i, count in enumerate(efficiency_by_temp['count']):
        ax1.text(i, ax1.get_ylim()[0], f'n={count}', 
                rotation=90, va='bottom', ha='center', fontsize=10)
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(pad=2.0)
    plt.savefig('analysis_results/temperature_efficiency_analysis.png', 
                dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    
    # Now create separate detailed brightness analysis
    plt.figure(figsize=(15, 8))
    
    # Create more detailed brightness ranges
    brightness_ranges = pd.qcut(df['brightness'], q=10)
    brightness_analysis = df.groupby(brightness_ranges).agg({
        'delta_next_hour_pct': ['mean', 'std', 'count'],
        'screen_on_hours': 'mean'
    })
    
    # Create twin axes for drain rate and screen time
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot battery drain
    brightness_analysis['delta_next_hour_pct']['mean'].abs().plot(
        kind='bar', yerr=brightness_analysis['delta_next_hour_pct']['std'], 
        capsize=5, color='lightcoral', ax=ax1, width=0.8,
        error_kw={'ecolor': 'darkred'}
    )
    
    # Plot screen time as a line
    brightness_analysis['screen_on_hours']['mean'].plot(
        color='darkblue', marker='o', ax=ax2, linewidth=2
    )
    
    # Customize the plot
    ax1.set_title('Screen Brightness Impact on Battery Life', pad=20, fontsize=14, fontweight='bold')
    ax1.set_xlabel('Brightness Level (Quantiles)', labelpad=10, fontsize=12)
    ax1.set_ylabel('Battery Drain (%/hour)', labelpad=10, fontsize=12, color='darkred')
    ax2.set_ylabel('Average Screen-on Time (hours)', labelpad=10, fontsize=12, color='darkblue')
    
    # Rotate x-labels
    plt.xticks(rotation=45, ha='right')
    
    # Add sample counts
    for i, count in enumerate(brightness_analysis['delta_next_hour_pct']['count']):
        ax1.text(i, ax1.get_ylim()[0], f'n={count}', 
                rotation=90, va='bottom', ha='center', fontsize=10)
    
    # Add grid
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust colors
    ax1.tick_params(axis='y', labelcolor='darkred')
    ax2.tick_params(axis='y', labelcolor='darkblue')
    
    plt.tight_layout(pad=2.0)
    plt.savefig('analysis_results/brightness_efficiency_analysis.png', 
                dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    
    # Create figure with more space between subplots for the main analysis
    plt.figure(figsize=(20, 14))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # 1. Energy consumption by app and network state
    plt.subplot(2, 2, 1)
    pivot_data = df.pivot_table(
        values='delta_next_hour_pct',
        index='app_active',
        columns='network_active',
        aggfunc='mean'
    ).abs()
    sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', fmt='.2f', cbar_kws={'label': 'Battery Drain (%/hour)'})
    plt.title('Energy Consumption by App and Network State', pad=20, fontsize=12, fontweight='bold')
    plt.xlabel('Network Active', labelpad=10)
    plt.ylabel('Application', labelpad=10)

    # 2. Battery drain distribution by time
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='is_weekend', y='delta_next_hour_pct', 
                hue='network_active', palette='Set3')
    plt.title('Battery Drain Distribution: Weekend vs Weekday', pad=20, fontsize=12, fontweight='bold')
    plt.xlabel('Weekend', labelpad=10)
    plt.ylabel('Battery Drain (%/hour)', labelpad=10)
    plt.legend(title='Network Active', title_fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')

    # 3. Temperature vs Energy Efficiency
    plt.subplot(2, 2, 3)
    temp_ranges = pd.qcut(df['temperature_C'], q=5)
    efficiency_by_temp = df.groupby(temp_ranges)['delta_next_hour_pct'].agg(['mean', 'std']).abs()
    ax = efficiency_by_temp.plot(kind='bar', y='mean', yerr='std', capsize=5, color='skyblue', error_kw={'ecolor': 'red'})
    plt.title('Energy Efficiency vs Temperature Ranges', pad=20, fontsize=12, fontweight='bold')
    plt.xlabel('Temperature Range (°C)', labelpad=10)
    plt.ylabel('Average Battery Drain (%/hour)', labelpad=10)
    plt.xticks(rotation=45, ha='right')

    # 4. Screen Brightness vs Energy Usage
    plt.subplot(2, 2, 4)
    brightness_ranges = pd.qcut(df['brightness'], q=5)
    sns.boxplot(data=df, x=brightness_ranges, y='delta_next_hour_pct', palette='YlOrRd')
    plt.title('Energy Usage vs Screen Brightness', pad=20, fontsize=12, fontweight='bold')
    plt.xlabel('Brightness Level (Quantiles)', labelpad=10)
    plt.ylabel('Battery Drain (%/hour)', labelpad=10)
    plt.xticks(rotation=45, ha='right')

    # Adjust layout with extra padding
    plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
    
    # Save with high quality and extra padding
    plt.savefig('analysis_results/energy_savings_analysis.png', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.5,
                facecolor='white',
                edgecolor='none')
    plt.close()

    # Calculate potential energy savings
    avg_drain = df['delta_next_hour_pct'].abs().mean()
    optimal_drain = df[df['brightness'] < df['brightness'].quantile(0.4)]['delta_next_hour_pct'].abs().mean()
    potential_savings = ((avg_drain - optimal_drain) / avg_drain) * 100

    print("\nEnergy Savings Analysis:")
    print("-" * 50)
    print(f"Average Battery Drain: {avg_drain:.2f}%/hour")
    print(f"Optimal Battery Drain: {optimal_drain:.2f}%/hour")
    print(f"Potential Energy Savings: {potential_savings:.1f}%")
    
    # App-specific analysis
    print("\nApp-specific Energy Consumption:")
    app_consumption = df.groupby('app_active')['delta_next_hour_pct'].agg(['mean', 'std']).abs()
    print(app_consumption)

# Run feature analysis
analyze_features(original_df)
analyze_energy_savings(original_df)

# Print feature statistics
print("\nFeature Statistics:")
print("-" * 50)
numeric_features = ["battery_pct", "screen_on_hours", "brightness", "temperature_C", 
                   "capacity_health", "delta_next_hour_pct", "tte_5pct_hours"]
print("\nNumeric Features:")
print(original_df[numeric_features].describe())

print("\nCategorical Features:")
cat_features = ["app_active", "network_active", "fast_charge", "is_weekend"]
for feat in cat_features:
    print(f"\n{feat} Distribution:")
    print(original_df[feat].value_counts(normalize=True))

# 1. Prediction vs Actual Scatter Plots
plt.figure(figsize=(15, 6))

# Battery Drop Rate
plt.subplot(1, 2, 1)
plt.scatter(actuals[:, 0], predictions[:, 0], alpha=0.5)
plt.plot([min(actuals[:, 0]), max(actuals[:, 0])], 
         [min(actuals[:, 0]), max(actuals[:, 0])], 'r--', label='Ideal')
plt.xlabel('Actual Battery Drop (%/hour)')
plt.ylabel('Predicted Battery Drop (%/hour)')
plt.title('Battery Drop Rate: Prediction vs Actual')
plt.legend()

# Time to 5%
plt.subplot(1, 2, 2)
plt.scatter(actuals[:, 1], predictions[:, 1], alpha=0.5)
plt.plot([min(actuals[:, 1]), max(actuals[:, 1])], 
         [min(actuals[:, 1]), max(actuals[:, 1])], 'r--', label='Ideal')
plt.xlabel('Actual Time to 5% (hours)')
plt.ylabel('Predicted Time to 5% (hours)')
plt.title('Time to 5%: Prediction vs Actual')
plt.legend()

plt.tight_layout()
plt.savefig('analysis_results/prediction_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Error Distribution
plt.figure(figsize=(15, 6))

# Battery Drop Rate Errors
plt.subplot(1, 2, 1)
errors_drop = predictions[:, 0] - actuals[:, 0]
plt.hist(errors_drop, bins=30, alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
plt.xlabel('Prediction Error (%/hour)')
plt.ylabel('Frequency')
plt.title('Battery Drop Rate Error Distribution')
plt.legend()

# Time to 5% Errors
plt.subplot(1, 2, 2)
errors_time = predictions[:, 1] - actuals[:, 1]
plt.hist(errors_time, bins=30, alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
plt.xlabel('Prediction Error (hours)')
plt.ylabel('Frequency')
plt.title('Time to 5% Error Distribution')
plt.legend()

plt.tight_layout()
plt.savefig('analysis_results/error_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Error vs Actual Value
plt.figure(figsize=(15, 6))

# Battery Drop Rate
plt.subplot(1, 2, 1)
plt.scatter(actuals[:, 0], errors_drop, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--', label='Zero Error')
plt.xlabel('Actual Battery Drop (%/hour)')
plt.ylabel('Prediction Error (%/hour)')
plt.title('Battery Drop Rate: Error vs Actual Value')
plt.legend()

# Time to 5%
plt.subplot(1, 2, 2)
plt.scatter(actuals[:, 1], errors_time, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--', label='Zero Error')
plt.xlabel('Actual Time to 5% (hours)')
plt.ylabel('Prediction Error (hours)')
plt.title('Time to 5%: Error vs Actual Value')
plt.legend()

plt.tight_layout()
plt.savefig('analysis_results/error_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()

# Print sample predictions and statistics
print("\nSample Predictions (First 5 records):")
print("-" * 50)
print("\nBattery Drop Rate Predictions (% per hour):")
for i in range(5):
    print(f"Actual: {actuals[i][0]:6.2f} | Predicted: {predictions[i][0]:6.2f}")

print("\nTime to 5% Battery Predictions (hours):")
for i in range(5):
    print(f"Actual: {actuals[i][1]:6.2f} | Predicted: {predictions[i][1]:6.2f}")

print("\nEnergy Saved by Alert Predictions:")
for i in range(5):
    print(f"Actual: {actuals[i][2]:6.2f} | Predicted: {predictions[i][2]:6.2f}")

# Additional Statistics
print("\nDetailed Statistics:")
print("-" * 50)
print("\nBattery Drop Rate:")
print(f"Mean Error: {np.mean(errors_drop):.4f} %/hour")
print(f"Error Std Dev: {np.std(errors_drop):.4f} %/hour")
print(f"Median Error: {np.median(errors_drop):.4f} %/hour")
print(f"90th Percentile Error: {np.percentile(np.abs(errors_drop), 90):.4f} %/hour")

print("\nTime to 5%:")
print(f"Mean Error: {np.mean(errors_time):.4f} hours")
print(f"Error Std Dev: {np.std(errors_time):.4f} hours")
print(f"Median Error: {np.median(errors_time):.4f} hours")
print(f"90th Percentile Error: {np.percentile(np.abs(errors_time), 90):.4f} hours")
