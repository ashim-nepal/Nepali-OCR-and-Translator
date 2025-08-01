# app.py

import streamlit as st
import easyocr
from transformers import MarianMTModel, MarianTokenizer
from PIL import Image
import tempfile

# Load OCR and translation models
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ne', 'en'])  # Nepali + English

@st.cache_resource
def load_translation_model():
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ne-en")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ne-en")
    return tokenizer, model

ocr_reader = load_ocr()
tokenizer, translation_model = load_translation_model()

# Streamlit UI
st.title("📄 Nepali ↔ English OCR Translator")
st.markdown("Upload an image with **Nepali or English** text. This tool will extract the text and translate it.")

uploaded_file = st.file_uploader("Upload a document image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        ocr_result = ocr_reader.readtext(temp_file.name, detail=0)
    
    raw_text = " ".join(ocr_result)
    st.subheader("🔍 OCR Result")
    st.code(raw_text)

    if raw_text:
        st.subheader("🌐 Translated Text (Nepali → English)")
        try:
            inputs = tokenizer(raw_text, return_tensors="pt", padding=True, truncation=True)
            translated = translation_model.generate(**inputs)
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            st.success(translated_text)
        except Exception as e:
            st.error("Translation failed. Try with simpler text or shorter input.")
