import streamlit as st
import os
import json
import base64
from PIL import Image, ImageDraw
import io
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


def encode_image(image_bytes):
    """Encode image bytes to base64 string for API request"""
    return base64.b64encode(image_bytes).decode("utf-8")


def analyze_package_with_gpt4o(image_file):
    """Use GPT-4o to analyze package compliance"""
    # Read and encode the image
    image_bytes = image_file.getvalue()
    base64_image = encode_image(image_bytes)
    pil_image = Image.open(io.BytesIO(image_bytes))

    # Prepare headers and payload for OpenAI API
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are a package compliance analyst. Analyze delivery package images for retail branding compliance."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
                        Analyze this delivery package image for compliance with the following rules:
                        1. Packages should not have visible retail branding (like Walmart, Target, Amazon, etc.)
                        2. Packaging tape should not have visible retail branding

                        For any non-compliance:
                        1. Provide a compliance score from 0-100 (100 being fully compliant)
                        2. Identify what makes it non-compliant (box, tape, or both)
                        3. Describe the specific violations
                        4. Provide bounding box coordinates [x1, y1, x2, y2] around each violation 
                           (coordinates should be normalized from 0-1 as ratios of image dimensions)

                        Format your response as a JSON object with these fields:
                        {
                            "compliance_score": int,
                            "is_compliant": boolean,
                            "violations": [
                                {
                                    "type": "box|tape",
                                    "description": "string",
                                    "brand_detected": "string",
                                    "bounding_box": [x1, y1, x2, y2]
                                }
                            ],
                            "summary": "string"
                        }

                        Return ONLY the JSON object, nothing else.
                        """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": 1000
    }

    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()

        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"]

        # Handle if the response is wrapped in markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)
        return result, pil_image

    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return {
            "compliance_score": 0,
            "is_compliant": False,
            "violations": [],
            "summary": f"Error processing image: {str(e)}"
        }, pil_image


def draw_bounding_boxes(pil_image, violations):
    """Draw bounding boxes on the image based on violations"""
    img = pil_image.copy()
    draw = ImageDraw.Draw(img)

    width, height = img.size

    for violation in violations:
        bbox = violation.get("bounding_box")
        if not bbox or len(bbox) != 4:
            continue

        # Convert normalized coordinates to pixel values
        x1, y1, x2, y2 = bbox
        x1, x2 = int(x1 * width), int(x2 * width)
        y1, y2 = int(y1 * height), int(y2 * height)

        # Choose color based on violation type
        color = "red" if violation.get("type") == "box" else "yellow"

        # Draw rectangle
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)

        # Add label
        brand = violation.get("brand_detected", "")
        draw.text((x1, y1 - 15), f"{violation.get('type', 'Unknown')}: {brand}", fill=color)

    return img


def main():
    st.set_page_config(page_title="Package Compliance Checker (GPT-4o)", layout="wide")

    st.title("📦 Package Compliance Checker (GPT-4o)")
    st.write("Upload images of delivered packages to check for brand compliance issues")

    # File uploader for multiple images
    uploaded_files = st.file_uploader("Upload package images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for img_file in uploaded_files:
            st.subheader(f"Analyzing: {img_file.name}")

            col1, col2 = st.columns(2)

            with col1:
                # Create a placeholder for the image
                original_img = Image.open(img_file)
                st.image(original_img, caption="Original Image", use_container_width=True)

            # Analyze image
            with st.spinner("Analyzing image with GPT-4o..."):
                result, pil_image = analyze_package_with_gpt4o(img_file)

            # Display results
            with col2:
                compliance_color = "green" if result.get("is_compliant", False) else "red"
                st.markdown(
                    f"### Compliance Score: <span style='color:{compliance_color};'>{result.get('compliance_score', 0)}/100</span>",
                    unsafe_allow_html=True)

                st.markdown(f"**Summary:** {result.get('summary', 'No summary available')}")

                violations = result.get("violations", [])
                if not result.get("is_compliant", True) and violations:
                    st.markdown("### Violations Detected:")

                    for idx, violation in enumerate(violations):
                        with st.expander(
                                f"Violation #{idx + 1}: {violation.get('type', 'Unknown')} - {violation.get('brand_detected', 'Unknown')}"):
                            st.write(violation.get("description", "No description provided"))

                    # Draw and display image with bounding boxes
                    annotated_img = draw_bounding_boxes(pil_image, violations)
                    st.image(annotated_img, caption="Violations Highlighted", use_container_width=True)
                else:
                    st.success("✅ No compliance issues detected!")

            st.markdown("---")


if __name__ == "__main__":
    main()