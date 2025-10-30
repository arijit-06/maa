import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

# Sensor value ranges for different activity levels
ACTIVITY_PROFILES = {
    'sleeping': {
        'temp_offset': (-1.5, -0.5),      # Cooler than baseline
        'aqi_offset': (-10, 0),            # Lower than baseline
        'pressure_offset': (0, 0.5),
        'label': 'sleeping'
    },
    'low': {
        'temp_offset': (-0.5, 0.5),
        'aqi_offset': (-5, 5),
        'pressure_offset': (-0.2, 0.2),
        'label': 'normal'
    },
    'low_out': {
        'temp_offset': (-1.0, 0),
        'aqi_offset': (-15, -5),           # Much lower (nobody home)
        'pressure_offset': (0, 0.3),
        'label': 'away'
    },
    'mid': {
        'temp_offset': (0, 1.0),
        'aqi_offset': (0, 10),
        'pressure_offset': (-0.2, 0.2),
        'label': 'normal'
    },
    'high_mid': {
        'temp_offset': (0.5, 2.0),
        'aqi_offset': (5, 20),
        'pressure_offset': (-0.3, 0.1),
        'label': 'high_activity'
    },
    'high': {  # Cooking
        'temp_offset': (2.0, 4.0),
        'aqi_offset': (40, 80),            # Big spike
        'pressure_offset': (-0.5, 0),
        'label': 'cooking'
    },
    'gym': {
        'temp_offset': (-0.5, 0.5),
        'aqi_offset': (-15, -5),
        'pressure_offset': (0, 0.3),
        'label': 'away'
    },
    'home_workout': {
        'temp_offset': (1.0, 3.0),
        'aqi_offset': (15, 35),
        'pressure_offset': (-0.3, 0.1),
        'label': 'high_activity'
    },
    'agitated': {
        'temp_offset': (0, 1.5),
        'aqi_offset': (5, 15),
        'pressure_offset': (-0.2, 0.2),
        'label': 'chill'
    }
}

# Weekly schedule (your exact routine)
WEEKLY_SCHEDULE = {
    # Sunday (0)
    0: [
        (0, 11, 'sleeping'),
        (11, 12, 'agitated'),
        (12, 14, 'high'),
        (14, 17, 'sleeping'),
        (17, 21, 'low_out'),
        (21, 23, 'low'),
        (23, 24, 'sleeping'),
    ],
    # Monday (1) - Work day - Gym
    1: [
        (0, 7, 'sleeping'),
        (7, 9, 'high_mid'),
        (9, 19, 'low'),
        (19, 20.5, 'high'),
        (20.5, 21.5, 'gym'),
        (21.5, 23, 'low'),
        (23, 24, 'sleeping'),
    ],
    # Tuesday (2) - Work day - No gym
    2: [
        (0, 7, 'sleeping'),
        (7, 9, 'high_mid'),
        (9, 19, 'low'),
        (19, 20.5, 'high'),
        (20.5, 21, 'home_workout'),
        (21, 23, 'low'),
        (23, 24, 'sleeping'),
    ],
    # Wednesday (3) - Work day - Gym
    3: [
        (0, 7, 'sleeping'),
        (7, 9, 'high_mid'),
        (9, 19, 'low'),
        (19, 20.5, 'high'),
        (20.5, 21.5, 'gym'),
        (21.5, 23, 'low'),
        (23, 24, 'sleeping'),
    ],
    # Thursday (4) - Work day - No gym
    4: [
        (0, 7, 'sleeping'),
        (7, 9, 'high_mid'),
        (9, 19, 'low'),
        (19, 20.5, 'high'),
        (20.5, 21, 'home_workout'),
        (21, 23, 'low'),
        (23, 24, 'sleeping'),
    ],
    # Friday (5) - Work day - Gym
    5: [
        (0, 7, 'sleeping'),
        (7, 9, 'high_mid'),
        (9, 19, 'low'),
        (19, 20.5, 'high'),
        (20.5, 21.5, 'gym'),
        (21.5, 23, 'low'),
        (23, 24, 'sleeping'),
    ],
    # Saturday (6) - Work day - No gym
    6: [
        (0, 7, 'sleeping'),
        (7, 9, 'high_mid'),
        (9, 19, 'low'),
        (19, 20.5, 'high'),
        (20.5, 21, 'home_workout'),
        (21, 23, 'low'),
        (23, 24, 'sleeping'),
    ],
}

def get_baseline_environment(hour, minute, day_of_week):
    """
    Calculate baseline environmental conditions based on time of day
    This simulates natural daily patterns independent of user activity
    """
    
    hour_decimal = hour + minute / 60.0
    
    # ==== TEMPERATURE PATTERN ====
    # Natural daily cycle: coolest at 5 AM, warmest at 2-3 PM
    temp_cycle = 12 * math.sin((hour_decimal - 5) * math.pi / 12)  # Peak at 2 PM
    base_temp = 24.0 + temp_cycle * 0.25  # ±3°C variation
    
    # Night cooling (10 PM - 6 AM)
    if hour >= 22 or hour <= 6:
        base_temp -= 2.0
    
    # ==== AQI PATTERN ====
    # Morning traffic peak (7-9 AM): outdoor pollution enters when windows open
    # Evening traffic peak (6-9 PM): cooking + traffic
    # Lowest at night (12 AM - 5 AM)
    
    base_aqi = 50  # Clean baseline
    
    # Morning rush hour (7-9 AM)
    if 7 <= hour < 9:
        morning_peak = 20 * math.sin((hour - 7) * math.pi / 2)
        base_aqi += morning_peak
    
    # Evening rush hour (6-9 PM) - bigger spike
    elif 18 <= hour < 21:
        evening_peak = 30 * math.sin((hour - 18) * math.pi / 3)
        base_aqi += evening_peak
    
    # Daytime general increase (10 AM - 5 PM)
    elif 10 <= hour < 17:
        base_aqi += 10
    
    # Night time - cleanest air (12 AM - 6 AM)
    elif hour <= 6:
        base_aqi -= 10
    
    # Weekend: lower outdoor pollution (less traffic)
    if day_of_week in [0, 6]:  # Sunday or Saturday
        base_aqi -= 5
    
    # ==== PRESSURE PATTERN ====
    # Diurnal variation: higher in morning, lower in afternoon
    pressure_cycle = math.cos((hour_decimal - 4) * math.pi / 12)
    base_pressure = 1013.0 + pressure_cycle * 1.5  # ±1.5 hPa variation
    
    # Add random weather variation (simulate changing weather patterns)
    weather_noise = np.random.normal(0, 0.5)
    base_pressure += weather_noise
    
    return base_temp, base_aqi, base_pressure

def get_activity_for_time(day, hour_decimal):
    """Get activity type for given day and time"""
    schedule = WEEKLY_SCHEDULE[day]
    
    for start_hour, end_hour, activity in schedule:
        if start_hour <= hour_decimal < end_hour:
            return activity
    
    return 'sleeping'

def generate_sensor_reading(activity, hour, minute, day_of_week):
    """
    Generate realistic sensor readings combining:
    1. Natural daily environmental baseline
    2. User activity impact
    """
    
    # Get natural environmental baseline
    base_temp, base_aqi, base_pressure = get_baseline_environment(
        hour, minute, day_of_week
    )
    
    # Get activity profile
    profile = ACTIVITY_PROFILES[activity]
    
    # Add activity-specific offsets to baseline
    temp_offset = np.random.uniform(*profile['temp_offset'])
    aqi_offset = np.random.uniform(*profile['aqi_offset'])
    pressure_offset = np.random.uniform(*profile['pressure_offset'])
    
    # Final values = baseline + activity impact + noise
    temp = base_temp + temp_offset + np.random.normal(0, 0.3)
    aqi = base_aqi + aqi_offset + np.random.normal(0, 3)
    pressure = base_pressure + pressure_offset + np.random.normal(0, 0.2)
    
    # Ensure realistic ranges
    temp = max(15, min(35, temp))  # 15-35°C
    aqi = int(max(0, min(500, aqi)))  # 0-500 AQI
    pressure = max(1000, min(1025, pressure))  # 1000-1025 hPa
    
    return temp, aqi, pressure, profile['label']

def generate_weekly_data(num_weeks=4, samples_per_hour=60):
    """
    Generate training data for multiple weeks
    """
    data = []
    start_date = datetime(2025, 11, 1, 0, 0)
    
    minutes_per_sample = 60 // samples_per_hour
    total_minutes = num_weeks * 7 * 24 * 60
    
    print(f"╔════════════════════════════════════════╗")
    print(f"║  Enhanced Data Generator with Natural  ║")
    print(f"║  Daily Environmental Patterns          ║")
    print(f"╚════════════════════════════════════════╝")
    print(f"\nGenerating {num_weeks} weeks of data...")
    print(f"Sampling: Every {minutes_per_sample} minute(s)")
    print(f"Total samples: {total_minutes // minutes_per_sample:,}\n")
    
    # Temporary storage for rate calculation
    temp_data = []
    
    for minute in range(0, total_minutes, minutes_per_sample):
        current_time = start_date + timedelta(minutes=minute)
        
        day = current_time.weekday()
        day = (day + 1) % 7  # Adjust to Sunday=0
        
        hour = current_time.hour
        minute_of_hour = current_time.minute
        hour_decimal = hour + minute_of_hour / 60.0
        
        # Get activity
        activity = get_activity_for_time(day, hour_decimal)
        
        # Generate sensor readings with natural daily patterns
        temp, aqi, pressure, label = generate_sensor_reading(
            activity, hour, minute_of_hour, day
        )
        
        # Store for rate calculation
        temp_data.append({
            'temperature': temp,
            'aqi': aqi,
            'pressure': pressure
        })
        
        # Calculate rate of change
        temp_change = 0.0
        aqi_change = 0.0
        pressure_change = 0.0
        
        if len(temp_data) > 1:
            temp_change = temp - temp_data[-2]['temperature']
            aqi_change = aqi - temp_data[-2]['aqi']
            pressure_change = pressure - temp_data[-2]['pressure']
        
        data.append({
            'timestamp': current_time,
            'day_of_week': day,
            'hour': hour,
            'minute': minute_of_hour,
            'temperature': round(temp, 2),
            'pressure': round(pressure, 2),
            'aqi': aqi,
            'temp_change_rate': round(temp_change, 3),
            'aqi_change_rate': round(aqi_change, 2),
            'pressure_change_rate': round(pressure_change, 3),
            'activity_type': activity,
            'label': label
        })
        
        # Progress indicator
        if minute % (60 * 24) == 0 and minute > 0:
            day_num = minute // (60 * 24)
            week_num = (day_num // 7) + 1
            day_of_week = day_num % 7
            days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
            print(f"  Week {week_num}, Day {day_num + 1} ({days[day_of_week]}) - {len(data):,} samples")
    
    print(f"\n✓ Generation complete!")
    return pd.DataFrame(data)

def add_derived_features(df):
    """Add additional features for better learning"""
    print("\nAdding derived features...")
    
    # Rolling averages
    df['temp_avg_15min'] = df['temperature'].rolling(window=15, min_periods=1).mean()
    df['aqi_avg_15min'] = df['aqi'].rolling(window=15, min_periods=1).mean()
    df['temp_avg_1hr'] = df['temperature'].rolling(window=60, min_periods=1).mean()
    df['aqi_avg_1hr'] = df['aqi'].rolling(window=60, min_periods=1).mean()
    
    # Time-based features
    df['is_night'] = df['hour'].apply(lambda x: 1 if (x >= 22 or x <= 6) else 0)
    df['is_morning'] = df['hour'].apply(lambda x: 1 if (6 <= x <= 9) else 0)
    df['is_cooking_hours'] = df['hour'].apply(lambda x: 1 if (19 <= x <= 21) else 0)
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [0, 6] else 0)
    df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (18 <= x <= 20) else 0)
    
    # Cyclic encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    print("✓ Features added")
    return df

def print_statistics(df):
    """Print dataset statistics"""
    print("\n" + "="*60)
    print("ENHANCED DATASET STATISTICS")
    print("="*60)
    
    print(f"\nTotal samples: {len(df):,}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Sampling rate: 1 sample per minute")
    
    print("\n--- Label Distribution ---")
    for label, count in df['label'].value_counts().items():
        percentage = (count / len(df)) * 100
        print(f"{label:15s}: {count:6,} ({percentage:5.1f}%)")
    
    print("\n--- Sensor Value Ranges ---")
    print(f"Temperature: {df['temperature'].min():.1f}°C to {df['temperature'].max():.1f}°C (avg: {df['temperature'].mean():.1f}°C)")
    print(f"AQI: {df['aqi'].min()} to {df['aqi'].max()} (avg: {df['aqi'].mean():.0f})")
    print(f"Pressure: {df['pressure'].min():.1f} to {df['pressure'].max():.1f} hPa (avg: {df['pressure'].mean():.1f} hPa)")
    
    print("\n--- Average Values by Time of Day ---")
    hourly_avg = df.groupby('hour')[['temperature', 'aqi', 'pressure']].mean()
    print("\nMorning (7-9 AM) - Traffic Peak:")
    print(hourly_avg.loc[7:9].round(2))
    print("\nAfternoon (2-4 PM) - Warmest:")
    print(hourly_avg.loc[14:16].round(2))
    print("\nEvening (6-9 PM) - Cooking + Traffic:")
    print(hourly_avg.loc[18:21].round(2))
    print("\nNight (2-4 AM) - Coolest/Cleanest:")
    print(hourly_avg.loc[2:4].round(2))
    
    print("\n--- Average Values by Label ---")
    print(df.groupby('label')[['temperature', 'aqi', 'pressure']].mean().round(2))

def main():
    """Main execution"""
    print("\n╔════════════════════════════════════════╗")
    print("║  Behavioral Training Data Generator    ║")
    print("║  WITH Natural Daily Patterns           ║")
    print("╚════════════════════════════════════════╝\n")
    
    # Generate data
    df = generate_weekly_data(num_weeks=4, samples_per_hour=60)
    
    # Add features
    df = add_derived_features(df)
    
    # Statistics
    print_statistics(df)
    
    # Save
    output_file = 'behavioral_training_data_natural.csv'
    print(f"\nSaving to file...")
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Data saved to: {output_file}")
    print(f"✓ Total features: {len(df.columns)}")
    print(f"✓ Ready for model training!")
    
    # Preview natural patterns
    print("\n--- Morning AQI Pattern (Monday 6-10 AM) ---")
    morning = df[(df['day_of_week'] == 1) & (df['hour'].isin([6, 7, 8, 9, 10]))]
    print(morning.groupby('hour')[['aqi', 'temperature']].mean().round(2))
    
    print("\n--- Cooking vs Natural Evening AQI (7-8 PM) ---")
    print("With cooking activity:")
    cooking = df[(df['hour'] == 19) & (df['label'] == 'cooking')]
    print(f"  AQI: {cooking['aqi'].mean():.0f} (natural baseline + cooking spike)")
    
    print("\n" + "="*60)
    print("ENHANCED DATA GENERATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
