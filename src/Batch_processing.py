import streamlit as st
import pandas as pd
import os
import time
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime


# Add this section to the main app after importing the necessary functions

def batch_processing_tab():
    st.header("Batch Processing")
    st.write("Upload multiple images for batch processing")

    uploaded_files = st.file_uploader("Upload package images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if not uploaded_files:
        st.info("Please upload images to begin batch processing")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"Total images: {len(uploaded_files)}")
        max_workers = st.slider("Concurrent workers", min_value=1, max_value=10, value=3)

    with col2:
        model_choice = st.radio("Select model", ["Gemini Pro Vision", "GPT-4o", "Claude 3 Sonnet"])

    if st.button("Start Batch Processing"):
        # Create a timestamp for this batch
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = f"results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)

        # Create CSV for results
        csv_path = os.path.join(results_dir, "compliance_results.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'compliance_score', 'is_compliant', 'violation_count', 'summary']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        # Setup progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()

        # Function to process a single image
        def process_image(img_file, index):
            status_text.text(f"Processing {index + 1}/{len(uploaded_files)}: {img_file.name}")

            try:
                # Choose the right analysis function based on model selection
                if model_choice == "Gemini Pro Vision":
                    image_data = get_image_data(img_file)
                    result = analyze_package_compliance(image_data)
                elif model_choice == "GPT-4o":
                    result, _ = analyze_package_with_gpt4o(img_file)
                else:  # Claude
                    # You would need to implement this function similar to others
                    result = {"compliance_score": 0, "is_compliant": False, "violations": [],
                              "summary": "Claude implementation not shown in this example"}

                # Save the annotated image
                if not result.get("is_compliant", True) and result.get("violations", []):
                    if model_choice == "Gemini Pro Vision":
                        pil_image = image_data["pil_image"]
                    else:
                        pil_image = Image.open(img_file)

                    annotated_img = draw_bounding_boxes(pil_image, result.get("violations", []))
                    img_save_path = os.path.join(results_dir, f"annotated_{img_file.name}")
                    annotated_img.save(img_save_path)

                # Write to CSV
                with open(csv_path, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=['filename', 'compliance_score', 'is_compliant',
                                                                 'violation_count', 'summary'])
                    writer.writerow({
                        'filename': img_file.name,
                        'compliance_score': result.get('compliance_score', 0),
                        'is_compliant': result.get('is_compliant', False),
                        'violation_count': len(result.get('violations', [])),
                        'summary': result.get('summary', '')
                    })

                # Update progress
                progress_bar.progress((index + 1) / len(uploaded_files))

                return result

            except Exception as e:
                st.error(f"Error processing {img_file.name}: {str(e)}")
                return {
                    "compliance_score": 0,
                    "is_compliant": False,
                    "violations": [],
                    "summary": f"Error: {str(e)}"
                }

        # Process images in parallel
        all_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_image, file, i) for i, file in enumerate(uploaded_files)]
            for future in futures:
                all_results.append(future.result())

        # Display summary
        status_text.text("Processing complete!")

        # Read back the CSV with pandas for display
        df = pd.read_csv(csv_path)

        with results_container:
            st.subheader("Processing Results")
            st.write(f"Results saved to: {results_dir}")

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Images Processed", len(df))
            with col2:
                compliant_count = df[df['is_compliant'] == True].shape[0]
                st.metric("Compliant Images", compliant_count)
            with col3:
                avg_score = df['compliance_score'].mean()
                st.metric("Average Compliance Score", f"{avg_score:.1f}")

            # Display dataframe
            st.dataframe(df)

            # Download link for CSV
            with open(csv_path, 'r') as file:
                st.download_button(
                    label="Download CSV Report",
                    data=file,
                    file_name=f"compliance_report_{timestamp}.csv",
                    mime="text/csv"
                )


# Add this to your main function
def enhanced_main():
    st.set_page_config(page_title="Package Compliance Checker", layout="wide")

    st.title("📦 Package Compliance Checker")

    tab1, tab2 = st.tabs(["Single Image Analysis", "Batch Processing"])

    with tab1:
        # Your existing single image analysis code here
        pass

    with tab2:
        batch_processing_tab()


if __name__ == "__main__":
    enhanced_main()