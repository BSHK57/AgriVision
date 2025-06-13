import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def create_demo_satellite_images():
    """Create demo satellite images for testing"""
    
    # Create directories
    os.makedirs('static/demo_data', exist_ok=True)
    
    # Create a synthetic RGB satellite image
    height, width = 512, 512
    
    # Create different land use regions
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Forest (green)
    rgb_image[0:height//3, 0:width//2] = [34, 139, 34]  # Forest green
    
    # Agricultural land (yellow-green)
    rgb_image[height//3:2*height//3, 0:width//2] = [154, 205, 50]  # Yellow green
    
    # Urban area (gray)
    rgb_image[2*height//3:height, 0:width//2] = [128, 128, 128]  # Gray
    
    # Water (blue)
    rgb_image[0:height//2, width//2:width] = [30, 144, 255]  # Dodger blue
    
    # Bareland (brown)
    rgb_image[height//2:height, width//2:width] = [160, 82, 45]  # Saddle brown
    
    # Add some noise for realism
    noise = np.random.normal(0, 10, (height, width, 3))
    rgb_image = np.clip(rgb_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Save RGB image
    rgb_pil = Image.fromarray(rgb_image)
    rgb_pil.save('static/demo_data/demo_rgb.png')
    
    # Create synthetic NIR image (higher values for vegetation)
    nir_image = np.zeros((height, width), dtype=np.uint8)
    
    # Forest - high NIR
    nir_image[0:height//3, 0:width//2] = 200
    
    # Agricultural - medium-high NIR
    nir_image[height//3:2*height//3, 0:width//2] = 150
    
    # Urban - low NIR
    nir_image[2*height//3:height, 0:width//2] = 50
    
    # Water - very low NIR
    nir_image[0:height//2, width//2:width] = 20
    
    # Bareland - low NIR
    nir_image[height//2:height, width//2:width] = 60
    
    # Add noise
    nir_noise = np.random.normal(0, 5, (height, width))
    nir_image = np.clip(nir_image.astype(np.float32) + nir_noise, 0, 255).astype(np.uint8)
    
    # Save NIR image
    nir_pil = Image.fromarray(nir_image, mode='L')
    nir_pil.save('static/demo_data/demo_nir.png')
    
    print("Demo satellite images created successfully!")
    print("RGB image: static/demo_data/demo_rgb.png")
    print("NIR image: static/demo_data/demo_nir.png")
    
    # Create a visualization showing the demo data
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(rgb_image)
    axes[0].set_title('Demo RGB Satellite Image')
    axes[0].axis('off')
    
    axes[1].imshow(nir_image, cmap='gray')
    axes[1].set_title('Demo NIR Image')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('static/demo_data/demo_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Demo overview saved: static/demo_data/demo_overview.png")

def create_time_series_demo_data():
    """Create multiple demo images for time series analysis"""
    
    dates = ['2023-03-01', '2023-06-01', '2023-09-01', '2023-12-01']
    vegetation_health = [0.3, 0.7, 0.8, 0.4]  # Seasonal variation
    
    for i, (date, health) in enumerate(zip(dates, vegetation_health)):
        height, width = 512, 512
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Vary vegetation color based on health
        vegetation_color = [int(34 * health), int(139 * health), int(34 * health)]
        
        # Create image with seasonal variation
        rgb_image[0:height//2, 0:width//2] = vegetation_color
        rgb_image[height//2:height, 0:width//2] = [154, 205, 50]  # Agricultural
        rgb_image[0:height//2, width//2:width] = [30, 144, 255]  # Water
        rgb_image[height//2:height, width//2:width] = [128, 128, 128]  # Urban
        
        # Add noise
        noise = np.random.normal(0, 10, (height, width, 3))
        rgb_image = np.clip(rgb_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Save image
        rgb_pil = Image.fromarray(rgb_image)
        rgb_pil.save(f'static/demo_data/demo_rgb_{date}.png')
        
        print(f"Created time series image for {date}")

if __name__ == "__main__":
    create_demo_satellite_images()
    create_time_series_demo_data()
    print("\nDemo data setup complete!")
    print("\nTo test the application:")
    print("1. Run: python app.py")
    print("2. Open http://localhost:5000 in your browser")
    print("3. Upload the demo images from static/demo_data/")
