import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def analyze_time_series(time_series_data, session_id):
    """Analyze and plot NDVI time series data"""
    dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in time_series_data]
    ndvi_means = [item['ndvi_mean'] for item in time_series_data]
    vegetation_percentages = [item['healthy_vegetation_percentage'] for item in time_series_data]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot NDVI mean over time
    ax1.plot(dates, ndvi_means, 'g-o', linewidth=2, markersize=6)
    ax1.set_title('NDVI Mean Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('NDVI Mean', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add trend line
    if len(dates) > 2:
        z = np.polyfit(range(len(dates)), ndvi_means, 1)
        p = np.poly1d(z)
        ax1.plot(dates, p(range(len(dates))), "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.4f})')
        ax1.legend()
    
    # Plot healthy vegetation percentage over time
    ax2.plot(dates, vegetation_percentages, 'b-s', linewidth=2, markersize=6)
    ax2.set_title('Healthy Vegetation Coverage Over Time', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Healthy Vegetation (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add trend line
    if len(dates) > 2:
        z = np.polyfit(range(len(dates)), vegetation_percentages, 1)
        p = np.poly1d(z)
        ax2.plot(dates, p(range(len(dates))), "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.4f}%)')
        ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'time_series_{session_id}.png'
    plot_path = os.path.join('static/results', plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def detect_anomalies(time_series_data, threshold=2):
    """Detect anomalies in NDVI time series using statistical methods"""
    ndvi_values = [item['ndvi_mean'] for item in time_series_data]
    
    if len(ndvi_values) < 3:
        return []
    
    mean_ndvi = np.mean(ndvi_values)
    std_ndvi = np.std(ndvi_values)
    
    anomalies = []
    for i, (data_point, ndvi_val) in enumerate(zip(time_series_data, ndvi_values)):
        z_score = abs(ndvi_val - mean_ndvi) / std_ndvi
        if z_score > threshold:
            anomalies.append({
                'index': i,
                'date': data_point['date'],
                'ndvi_mean': ndvi_val,
                'z_score': z_score,
                'type': 'high' if ndvi_val > mean_ndvi else 'low'
            })
    
    return anomalies

def calculate_seasonal_trends(time_series_data):
    """Calculate seasonal trends in vegetation health"""
    if len(time_series_data) < 4:
        return None
    
    # Group by month
    monthly_data = {}
    for item in time_series_data:
        date = datetime.strptime(item['date'], '%Y-%m-%d')
        month = date.month
        if month not in monthly_data:
            monthly_data[month] = []
        monthly_data[month].append(item['ndvi_mean'])
    
    # Calculate monthly averages
    monthly_averages = {}
    for month, values in monthly_data.items():
        monthly_averages[month] = np.mean(values)
    
    return monthly_averages
