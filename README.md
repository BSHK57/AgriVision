# AgriVision
AgriVision is a web application for crop health monitoring and land use classification using satellite imagery. It provides NDVI (Normalized Difference Vegetation Index) analysis, land use classification, and time series visualization for agricultural monitoring.

## Features
- Upload RGB and NIR satellite images
- Compute NDVI and visualize as heatmaps
- Simulate land use classification (demo)
- Track and plot NDVI trends over time
- Demo data generation for quick testing

## Project Structure
```
app.py                  # Main Flask application
requirements.txt        # Python dependencies
scripts/
  setup_demo_data.py    # Script to generate demo satellite images
static/
  results/              # Stores generated analysis images (NDVI, classification, time series)
  uploads/              # Stores uploaded user images
  demo_data/            # (Created by setup_demo_data.py) Demo images for testing
templates/
  index.html            # Main web UI template
utils/
  ndvi.py               # NDVI calculation and visualization
  preprocess.py         # Image preprocessing and classification simulation
  time_series.py        # NDVI time series analysis and plotting
```

## Installation
1. **Clone the repository**
2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage
1. **(Optional) Generate demo data:**
   ```powershell
   python scripts/setup_demo_data.py
   ```
2. **Run the application:**
   ```powershell
   python app.py
   ```
3. **Open your browser:**
   Go to [http://localhost:5000](http://localhost:5000)
4. **Upload images:**
   - Use demo images from `static/demo_data/` or your own satellite images.
   - Optionally, provide a date for time series analysis.

## How It Works
- **NDVI Calculation:** Uses RGB and (optionally) NIR images to compute NDVI, visualized as a heatmap.
- **Land Use Classification:** Simulates classification using a mock model (for demo; replace with a real model for production).
- **Time Series:** Tracks NDVI statistics over time and plots trends.

## Dependencies
See `requirements.txt` for all dependencies. Key packages:
- Flask
- numpy
- opencv-python
- Pillow
- matplotlib
- scikit-learn
- tensorflow, torch (for future/real model support)

## Notes
- Uploaded and generated images are saved in `static/results/` and `static/uploads/`.
- Demo data is created in `static/demo_data/`.
- For production, use a database for time series data instead of in-memory storage.

## License
This project is for demonstration and educational purposes.
