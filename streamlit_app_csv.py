import streamlit as st
from PIL import Image
from paddleocr import PaddleOCR
import numpy as np
import pandas as pd

# Initialize PaddleOCR
ocr = PaddleOCR(lang='en')

# Create a Streamlit app
st.title("PaddleOCR Image Upload")

# Use st.file_uploader to upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    image = np.array(image)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # ...

    # Perform OCR on the uploaded image
    with st.spinner("Performing OCR..."):
        ocr_results = ocr.ocr(image)

    # Display OCR results
    st.subheader("OCR Results:")

    # Initialize lists to store bounding boxes, text, and probabilities
    boxes = [box[0] for box in ocr_results]
    text = [text[1][0] for text in ocr_results]
    probabilities = [prob[1][1] for prob in ocr_results]

    # Create a DataFrame
    df = pd.DataFrame({'Boxes': boxes, 'Texts': text, 'Probabilities': probabilities})

    # Display the DataFrame
    st.write(df)

    # Save the DataFrame as a CSV file
    if st.button("Save as CSV"):
        df.to_csv("ocr_results.csv", index=False)
        st.success("CSV file saved successfully!")

    # ...

        
    
# Run the Streamlit app
if __name__ == "__main__":
    st.write("To upload an image, use the file uploader above.")
