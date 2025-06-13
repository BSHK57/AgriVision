import numpy as np
import cv2
from PIL import Image

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for CNN model input"""
    # Resize image
    image_resized = cv2.resize(image, target_size)
    
    # Normalize pixel values to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch

def simulate_classification(processed_image):
    """
    Simulate CNN-based land use classification
    In production, this would use a trained model like ResNet on EuroSAT dataset
    """
    # Simulate model prediction with random probabilities
    # In reality, this would be: model.predict(processed_image)
    
    classes = ['Forest', 'Urban', 'Water', 'Agricultural', 'Bareland', 'Permanent Crop', 'Pasture']
    
    # Generate realistic-looking probabilities
    np.random.seed(42)  # For reproducible results
    probabilities = np.random.dirichlet(np.ones(len(classes)) * 2)
    
    # Create classification result
    classification_result = {
        'predicted_class': classes[np.argmax(probabilities)],
        'confidence': float(np.max(probabilities)),
        'all_probabilities': {cls: float(prob) for cls, prob in zip(classes, probabilities)}
    }
    
    return classification_result

def extract_patches(image, patch_size=64, stride=32):
    """Extract patches from image for detailed analysis"""
    patches = []
    h, w = image.shape[:2]
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    
    return np.array(patches)

def augment_image(image):
    """Apply data augmentation techniques"""
    augmented_images = []
    
    # Original image
    augmented_images.append(image)
    
    # Horizontal flip
    augmented_images.append(cv2.flip(image, 1))
    
    # Rotation
    center = (image.shape[1]//2, image.shape[0]//2)
    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        augmented_images.append(rotated)
    
    return augmented_images
