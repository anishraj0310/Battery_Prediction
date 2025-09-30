import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Expanded app options for diversity
apps = [
    'WhatsApp', 'YouTube', 'Chrome', 'Spotify', 'Instagram', 'Snapchat',
    'TikTok', 'Facebook', 'Twitter', 'Zoom', 'Amazon Prime', 'Netflix',
    'Games', 'Camera', 'Music Player', 'Maps', 'Others'
]

# Major Indian cities for location diversity
cities = {
    'Delhi': (28.6139, 77.2090),
    'Mumbai': (19.0760, 72.8777),
    'Bangalore': (12.9716, 77.5946),
    'Chennai': (13.0827, 80.2707),
    'Hyderabad': (17.3850, 78.4867),
    'Kolkata': (22.5726, 88.3639),
    'Pune': (18.5204, 73.8567),
    'Ahmedabad': (23.0225, 72.5714)
}

# Seasons and temperature ranges
seasons = {
    'Summer': (30, 45),  # Max 45°C
    'Winter': (10, 25),  # Min 10°C
    'Monsoon': (22, 35), # Moderate
    'Spring': (20, 32),  # Mild warm
    'Autumn': (15, 30)   # Cooling
}

def generate_battery_drop(battery_pct, screen_hours, brightness, temp,
                         capacity_health, app, network_active):
    """Advanced battery drop calculation"""
    base_drop = np.random.normal(2.0, 0.5)  # Base 2% per hour

    # Screen impact
    screen_impact = screen_hours * (brightness / 100) * 0.8
    if app in ['YouTube', 'Netflix', 'TikTok']:
        screen_impact *= 1.2  # Video apps drain more
    elif app in ['Games']:
        screen_impact *= 1.5  # Games drain battery

    # Temperature impact (non-linear)
    if temp > 35:
        temp_factor = 1 + (temp - 35) * 0.02
    elif temp < 15:
        temp_factor = 1.1
    else:
        temp_factor = 1.0

    # Capacity health impact
    health_impact = (100 - capacity_health) * 0.005

    # Network impact
    network_impact = 0.3 if network_active else 0.0

    # Random variance and battery level effect (faster drop at high levels)
    level_factor = 1.0 if battery_pct > 20 else 0.8  # Slower near empty

    total_drop = (base_drop + screen_impact) * temp_factor * (1 + health_impact) + network_impact
    total_drop *= level_factor

    return max(0.1, min(total_drop, 8.0))  # Cap between 0.1% and 8%

def generate_session(capacity_health, season):
    """Generate a single session with realistic battery behavior"""
    city = random.choice(list(cities.keys()))
    lat, lon = cities[city]

    # Adjust location slightly for realism
    lat += np.random.normal(0, 0.01)
    lon += np.random.normal(0, 0.01)

    app = random.choice(apps)
    network_active = random.choice([True, False])
    fast_charge = random.choice([True, False])
    is_weekend = random.choice([0, 1])

    # Temperature based on season and city
    season_temp_range = seasons[season]
    temp = np.random.uniform(season_temp_range[0], season_temp_range[1])

    # Battery starts higher on good health
    initial_battery = random.uniform(80, 100) if random.random() < 0.7 else random.uniform(20, 60)

    # Session duration based on activities
    if is_weekend:
        screen_hours = np.random.uniform(1, 12)
    else:
        screen_hours = np.random.uniform(0.5, 6)

    brightness = np.random.uniform(20, 100)  # More realistic brightness range

    # Simulate battery drain over session
    battery_pct = initial_battery
    elapsed_hours = 0
    drop_rate = generate_battery_drop(
        battery_pct, screen_hours, brightness, temp,
        capacity_health, app, network_active
    )

    # Final battery after session
    final_battery = max(0, battery_pct - drop_rate * (screen_hours / random.uniform(0.8, 1.2)))

    # Calculate targets
    delta_next_hour_pct = drop_rate * np.random.uniform(0.9, 1.1)

    # Time to 5% calculation
    if delta_next_hour_pct > 0:
        remaining_pct = battery_pct - 5
        if remaining_pct > 0:
            tte_5pct_hours = remaining_pct / delta_next_hour_pct
        else:
            tte_5pct_hours = 0.1  # Very low
    else:
        tte_5pct_hours = random.uniform(2, 50)  # Fallback

    # Theoretical energy saved by alert
    alert_efficiency = random.uniform(0.1, 0.5)
    energy_saved_by_alert = delta_next_hour_pct * alert_efficiency * random.uniform(0.8, 1.2)

    return {
        'battery_pct': round(battery_pct, 4),
        'screen_on_hours': round(screen_hours, 4),
        'brightness': round(brightness, 4),
        'temperature_C': round(temp, 4),
        'capacity_health': round(capacity_health, 4),
        'location_lat': round(lat, 6),
        'location_lon': round(lon, 6),
        'app_active': app,
        'network_active': 1 if network_active else 0,
        'fast_charge': 1 if fast_charge else 0,
        'is_weekend': is_weekend,
        'delta_next_hour_pct': round(delta_next_hour_pct, 4),
        'tte_5pct_hours': round(tte_5pct_hours, 4),
        'energy_saved_by_alert': round(energy_saved_by_alert, 4)
    }

# Generate diverse dataset
np.random.seed(42)
random.seed(42)

data = []

# Distribution: 50% degraded batteries (<90%), 30% good (90-95%), 20% excellent (95-100%)
battery_healths = []
for _ in range(2000):
    if random.random() < 0.5:
        battery_healths.append(random.uniform(50, 90))
    elif random.random() < 0.8:
        battery_healths.append(random.uniform(90, 95))
    else:
        battery_healths.append(random.uniform(95, 100))

# Seasonal distribution
seasons_list = list(seasons.keys())
season_weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # Summer most common

for i in range(5000):  # 5x more data
    capacity_health = battery_healths[i % len(battery_healths)]
    season = random.choices(seasons_list, weights=season_weights, k=1)[0]

    session = generate_session(capacity_health, season)
    data.append(session)

# Create DataFrame
df = pd.DataFrame(data)

# Add some correlation noise for realism
df['delta_next_hour_pct'] += np.random.normal(0, 0.2, len(df))
df['tte_5pct_hours'] += np.random.normal(0, 1, len(df))
df['energy_saved_by_alert'] += np.random.normal(0, 0.05, len(df))

# Ensure non-negative values
df['delta_next_hour_pct'] = df['delta_next_hour_pct'].clip(lower=0)
df['tte_5pct_hours'] = df['tte_5pct_hours'].clip(lower=0.1)
df['energy_saved_by_alert'] = df['energy_saved_by_alert'].clip(lower=0)

# Save to CSV
df.to_csv('../data/diverse_synthetic_sessions.csv', index=False)

print("Generated diverse synthetic dataset with 5,000 sessions")
print(f"Shape: {df.shape}")
print(f"Apps: {df['app_active'].unique()}")
print(f"Cities represented: {list(cities.keys())}")
print(f"Battery health range: {df['capacity_health'].min():.1f} - {df['capacity_health'].max():.1f}%")
print(f"Temperature range: {df['temperature_C'].min():.1f} - {df['temperature_C'].max():.1f}°C")
