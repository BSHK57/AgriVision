import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def calculate_ndvi(rgb_image, nir_image):
    """
    Calculate NDVI from RGB and NIR images
    NDVI = (NIR - RED) / (NIR + RED)
    """
    # Extract red channel from RGB image
    red_channel = rgb_image[:, :, 0].astype(np.float32)
    nir_channel = nir_image.astype(np.float32)
    
    # Normalize to 0-1 range
    red_channel = red_channel / 255.0
    nir_channel = nir_channel / 255.0
    
    # Calculate NDVI
    numerator = nir_channel - red_channel
    denominator = nir_channel + red_channel
    
    # Avoid division by zero
    ndvi = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    # Clip values to valid NDVI range [-1, 1]
    ndvi = np.clip(ndvi, -1, 1)
    
    return ndvi

def create_ndvi_heatmap(ndvi, original_image, timestamp):
    """Create NDVI heatmap visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # NDVI heatmap
    im = axes[1].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[1].set_title('NDVI Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # NDVI overlay on original image
    # Create alpha mask based on NDVI values
    alpha = np.abs(ndvi)
    alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min())
    
    # Create colored overlay
    ndvi_colored = plt.cm.RdYlGn((ndvi + 1) / 2)  # Normalize to 0-1 for colormap
    ndvi_colored[:, :, 3] = alpha * 0.7  # Set alpha channel
    
    axes[2].imshow(original_image)
    axes[2].imshow(ndvi_colored)
    axes[2].set_title('NDVI Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    heatmap_filename = f'ndvi_heatmap_{timestamp}.png'
    heatmap_path = os.path.join('static/results', heatmap_filename)
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return heatmap_path

def classify_vegetation_health(ndvi):
    """Classify vegetation health based on NDVI values"""
    health_map = np.zeros_like(ndvi, dtype=int)
    
    # Classification thresholds
    health_map[ndvi < 0] = 0      # No vegetation/water
    health_map[(ndvi >= 0) & (ndvi < 0.2)] = 1   # Poor vegetation
    health_map[(ndvi >= 0.2) & (ndvi < 0.4)] = 2  # Moderate vegetation
    health_map[(ndvi >= 0.4) & (ndvi < 0.6)] = 3  # Good vegetation
    health_map[ndvi >= 0.6] = 4   # Excellent vegetation
    
    return health_map
