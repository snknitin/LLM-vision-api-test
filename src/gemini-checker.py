import streamlit as st
import os
import json
import base64
from PIL import Image, ImageDraw
import io
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_image_data(image_file):
    """Process uploaded image file into format needed for Gemini API"""
    if image_file is None:
        return None

    image_data = image_file.getvalue()
    image = Image.open(io.BytesIO(image_data))

    return {
        "data": image_data,
        "mime_type": image_file.type,
        "pil_image": image
    }


def analyze_package_compliance(image_data):
    """Use Gemini to analyze package compliance"""
    model = genai.GenerativeModel('gemini-pro-vision')

    prompt = """
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

    try:
        response = model.generate_content([prompt, image_data["pil_image"]])

        # Extract the JSON response
        response_text = response.text
        # Some models may wrap the JSON in ```json ``` markdown - handle that case
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)
        return result
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return {
            "compliance_score": 0,
            "is_compliant": False,
            "violations": [],
            "summary": f"Error processing image: {str(e)}"
        }


def draw_bounding_boxes(image_data, violations):
    """Draw bounding boxes on the image based on violations"""
    img = image_data["pil_image"].copy()
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
    st.set_page_config(page_title="Package Compliance Checker", layout="wide")

    st.title("📦 Package Compliance Checker")
    st.write("Upload images of delivered packages to check for brand compliance issues")

    # File uploader for multiple images
    uploaded_files = st.file_uploader("Upload package images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for img_file in uploaded_files:
            st.subheader(f"Analyzing: {img_file.name}")

            col1, col2 = st.columns(2)

            # Process image
            image_data = get_image_data(img_file)

            with col1:
                st.image(image_data["pil_image"], caption="Original Image", use_container_width=True)

            # Analyze image
            with st.spinner("Analyzing image..."):
                result = analyze_package_compliance(image_data)

            # Display results
            with col2:
                compliance_color = "green" if result["is_compliant"] else "red"
                st.markdown(
                    f"### Compliance Score: <span style='color:{compliance_color};'>{result['compliance_score']}/100</span>",
                    unsafe_allow_html=True)

                st.markdown(f"**Summary:** {result['summary']}")

                if not result["is_compliant"] and result["violations"]:
                    st.markdown("### Violations Detected:")

                    for idx, violation in enumerate(result["violations"]):
                        with st.expander(
                                f"Violation #{idx + 1}: {violation.get('type', 'Unknown')} - {violation.get('brand_detected', 'Unknown')}"):
                            st.write(violation.get("description", "No description provided"))

                    # Draw and display image with bounding boxes
                    annotated_img = draw_bounding_boxes(image_data, result["violations"])
                    st.image(annotated_img, caption="Violations Highlighted", use_container_width=True)
                else:
                    st.success("✅ No compliance issues detected!")

            st.markdown("---")


if __name__ == "__main__":
    main()