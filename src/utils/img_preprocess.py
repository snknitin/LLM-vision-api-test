import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io


def enhance_image(pil_image):
    """Apply multiple enhancement techniques to improve image quality"""
    # Convert to numpy array for OpenCV operations
    img_np = np.array(pil_image)

    # Convert to grayscale for certain operations
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) > 2 else img_np

    # Check if image is low contrast
    min_val, max_val, _, _ = cv2.minMaxLoc(gray)
    contrast_range = max_val - min_val

    enhanced_img = pil_image

    # Apply enhancements based on image quality assessment
    if contrast_range < 100:  # Low contrast image
        # Increase contrast
        enhancer = ImageEnhance.Contrast(enhanced_img)
        enhanced_img = enhancer.enhance(1.5)

        # Increase sharpness
        enhancer = ImageEnhance.Sharpness(enhanced_img)
        enhanced_img = enhancer.enhance(1.5)

    # Check if image is too dark
    if np.mean(gray) < 100:
        # Increase brightness
        enhancer = ImageEnhance.Brightness(enhanced_img)
        enhanced_img = enhancer.enhance(1.3)

    # Apply a light sharpening filter to all images
    enhanced_img = enhanced_img.filter(ImageFilter.SHARPEN)

    return enhanced_img


def detect_shipping_box(image_bytes):
    """
    Use OpenCV to attempt to detect shipping boxes in the image.
    Falls back to LLM-based detection if OpenCV methods fail.
    """
    try:
        # Convert image bytes to numpy array for OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Canny edge detector
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area to find potential boxes
        img_area = img.shape[0] * img.shape[1]
        min_area_ratio = 0.05  # Box should be at least 5% of the image

        potential_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > img_area * min_area_ratio:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Check if shape is rectangular (aspect ratio check)
                aspect_ratio = float(w) / h
                if 0.5 < aspect_ratio < 2.0:  # Typical box aspect ratios
                    # Normalize coordinates
                    x1 = x / img.shape[1]
                    y1 = y / img.shape[0]
                    x2 = (x + w) / img.shape[1]
                    y2 = (y + h) / img.shape[0]

                    potential_boxes.append([x1, y1, x2, y2])

        # If we found potential boxes, return the largest one
        if potential_boxes:
            # Sort by area (descending)
            potential_boxes.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
            return {
                "box_detected": True,
                "box_bounding_box": potential_boxes[0],
                "detection_method": "opencv"
            }

        # If OpenCV detection failed, return None (will use LLM for detection)
        return {
            "box_detected": False,
            "box_bounding_box": None,
            "detection_method": "none"
        }

    except Exception as e:
        print(f"Error in shipping box detection: {str(e)}")
        return {
            "box_detected": False,
            "box_bounding_box": None,
            "detection_method": "error"
        }


def crop_to_box(pil_image, bbox):
    """Crop the image to the detected box"""
    if not bbox:
        return pil_image

    width, height = pil_image.size
    x1, y1, x2, y2 = bbox

    # Convert normalized coordinates to pixel values
    x1, x2 = int(x1 * width), int(x2 * width)
    y1, y2 = int(y1 * height), int(y2 * height)

    # Add a small margin around the box (5% of dimensions)
    margin_x = int((x2 - x1) * 0.05)
    margin_y = int((y2 - y1) * 0.05)

    # Ensure we don't go outside image boundaries
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(width, x2 + margin_x)
    y2 = min(height, y2 + margin_y)

    # Crop the image
    cropped_img = pil_image.crop((x1, y1, x2, y2))
    return cropped_img


def assess_image_quality(pil_image):
    """Assess the quality of the image"""
    # Convert to numpy array
    img_np = np.array(pil_image)

    # Convert to grayscale if needed
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) > 2 else img_np

    # Calculate Laplacian variance (measure of image sharpness)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Calculate contrast
    min_val, max_val, _, _ = cv2.minMaxLoc(gray)
    contrast = max_val - min_val

    # Calculate brightness
    brightness = np.mean(gray)

    # Determine quality based on these metrics
    if laplacian_var > 100 and contrast > 80 and 50 < brightness < 200:
        quality = "high"
    elif laplacian_var > 50 and contrast > 40 and 30 < brightness < 220:
        quality = "medium"
    else:
        quality = "low"

    return {
        "quality": quality,
        "sharpness": laplacian_var,
        "contrast": contrast,
        "brightness": brightness
    }


def preprocess_delivery_image(image_file):
    """Main function to preprocess delivery images"""
    # Read image file
    image_bytes = image_file.getvalue()
    pil_image = Image.open(io.BytesIO(image_bytes))

    # Assess original image quality
    quality_assessment = assess_image_quality(pil_image)

    # Detect shipping box
    box_detection = detect_shipping_box(image_bytes)

    # Apply enhancements based on image quality
    if quality_assessment["quality"] != "high":
        enhanced_image = enhance_image(pil_image)
    else:
        enhanced_image = pil_image

    # If box was detected with OpenCV, crop to that box
    if box_detection["box_detected"]:
        cropped_image = crop_to_box(enhanced_image, box_detection["box_bounding_box"])
    else:
        # If no box detected, use the enhanced full image
        cropped_image = enhanced_image

    return {
        "original_image": pil_image,
        "enhanced_image": enhanced_image,
        "processed_image": cropped_image,
        "quality_assessment": quality_assessment,
        "box_detection": box_detection
    }