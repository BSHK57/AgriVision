from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import json
from utils.ndvi import calculate_ndvi, create_ndvi_heatmap
from utils.preprocess import preprocess_image, simulate_classification
from utils.time_series import analyze_time_series

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Store time series data in memory (in production, use a database)
time_series_data = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'rgb_image' not in request.files:
            return jsonify({'error': 'No RGB image uploaded'}), 400
        
        rgb_file = request.files['rgb_image']
        nir_file = request.files.get('nir_image')
        
        if rgb_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rgb_filename = f'rgb_{timestamp}_{rgb_file.filename}'
        rgb_path = os.path.join(app.config['UPLOAD_FOLDER'], rgb_filename)
        rgb_file.save(rgb_path)
        
        nir_path = None
        if nir_file and nir_file.filename != '':
            nir_filename = f'nir_{timestamp}_{nir_file.filename}'
            nir_path = os.path.join(app.config['UPLOAD_FOLDER'], nir_filename)
            nir_file.save(nir_path)
        
        return jsonify({
            'success': True,
            'rgb_path': rgb_path,
            'nir_path': nir_path,
            'timestamp': timestamp
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/compute_ndvi', methods=['POST'])
def compute_ndvi():
    try:
        data = request.json
        rgb_path = data['rgb_path']
        nir_path = data.get('nir_path')
        
        # Load images
        rgb_image = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        if nir_path:
            nir_image = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Simulate NIR from RGB (for demo purposes)
            nir_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        # Calculate NDVI
        ndvi = calculate_ndvi(rgb_image, nir_image)
        
        # Create NDVI heatmap
        heatmap_path = create_ndvi_heatmap(ndvi, rgb_image, data['timestamp'])
        
        # Calculate statistics
        ndvi_stats = {
            'mean': float(np.mean(ndvi)),
            'std': float(np.std(ndvi)),
            'min': float(np.min(ndvi)),
            'max': float(np.max(ndvi)),
            'healthy_vegetation_percentage': float(np.sum(ndvi > 0.3) / ndvi.size * 100)
        }
        
        return jsonify({
            'success': True,
            'heatmap_path': heatmap_path,
            'stats': ndvi_stats
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classify_land_use', methods=['POST'])
def classify_land_use():
    try:
        data = request.json
        rgb_path = data['rgb_path']
        
        # Load and preprocess image
        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = preprocess_image(image)
        
        # Simulate classification (in production, use trained CNN model)
        classification_result = simulate_classification(processed_image)
        
        # Create classification visualization
        viz_path = create_classification_visualization(
            image, classification_result, data['timestamp']
        )
        
        return jsonify({
            'success': True,
            'classification': classification_result,
            'visualization_path': viz_path
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add_time_series', methods=['POST'])
def add_time_series():
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        date = data['date']
        ndvi_stats = data['ndvi_stats']
        
        if session_id not in time_series_data:
            time_series_data[session_id] = []
        
        time_series_data[session_id].append({
            'date': date,
            'ndvi_mean': ndvi_stats['mean'],
            'healthy_vegetation_percentage': ndvi_stats['healthy_vegetation_percentage']
        })
        
        # Sort by date
        time_series_data[session_id].sort(key=lambda x: x['date'])
        
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/plot_time_series', methods=['POST'])
def plot_time_series():
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        if session_id not in time_series_data or len(time_series_data[session_id]) < 2:
            return jsonify({'error': 'Insufficient time series data'}), 400
        
        plot_path = analyze_time_series(time_series_data[session_id], session_id)
        
        return jsonify({
            'success': True,
            'plot_path': plot_path
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_classification_visualization(image, classification_result, timestamp):
    """Create visualization of land use classification"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Classification result
    class_colors = {
        'Forest': [0, 1, 0],
        'Urban': [0.5, 0.5, 0.5],
        'Water': [0, 0, 1],
        'Agricultural': [1, 1, 0],
        'Bareland': [0.8, 0.4, 0.2]
    }
    
    # Create a simple classification map (simulation)
    h, w = image.shape[:2]
    class_map = np.zeros((h, w, 3))
    
    # Simulate different regions
    regions = [
        ('Forest', (0, h//3, 0, w//2)),
        ('Agricultural', (h//3, 2*h//3, 0, w//2)),
        ('Urban', (2*h//3, h, 0, w//2)),
        ('Water', (0, h//2, w//2, w)),
        ('Bareland', (h//2, h, w//2, w))
    ]
    
    for class_name, (y1, y2, x1, x2) in regions:
        if class_name in class_colors:
            class_map[y1:y2, x1:x2] = class_colors[class_name]
    
    ax2.imshow(class_map)
    ax2.set_title('Land Use Classification')
    ax2.axis('off')
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=class_colors[cls], label=cls) 
                      for cls in class_colors.keys()]
    ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
    
    plt.tight_layout()
    
    # Save plot
    viz_filename = f'classification_{timestamp}.png'
    viz_path = os.path.join('static/results', viz_filename)
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return viz_path

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
