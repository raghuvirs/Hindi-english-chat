import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MBartForConditionalGeneration, MBart50TokenizerFast
import google.generativeai as genai
import os

# header
st.title("Hindi-English Translation and AI Content Generator")


os.environ["GOOGLE_API_KEY"] = "AIzaSyBJhMiJBjVAZbRok3MLLDPWz_AJ2gieF-s"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

#translation models
@st.cache_resource
def load_models():
    # Hindi to English 
    tokenizer_hi2en = AutoTokenizer.from_pretrained("TestZee/FineTuned-hindi-to-english-V8")
    model_hi2en = AutoModelForSeq2SeqLM.from_pretrained("TestZee/FineTuned-hindi-to-english-V8")

    # English to Hindi 
    model_en2hi = MBartForConditionalGeneration.from_pretrained("SnypzZz/Llama2-13b-Language-translate")
    tokenizer_en2hi = MBart50TokenizerFast.from_pretrained("SnypzZz/Llama2-13b-Language-translate", src_lang="en_XX")

    return tokenizer_hi2en, model_hi2en, model_en2hi, tokenizer_en2hi

tokenizer_hi2en, model_hi2en, model_en2hi, tokenizer_en2hi = load_models()

#user input
hindi_input = st.text_area("Enter Hindi text for translation:", value="संयुक्त राष्ट्र के नेता कहते हैं कि सीरिया में कोई सैन्य समाधान नहीं है")

if st.button("Translate and Generate AI Content"):
    # Hindi to English translation
    with st.spinner("Translating Hindi to English..."):
        batch = tokenizer_hi2en([hindi_input], return_tensors="pt")
        generated_ids = model_hi2en.generate(**batch)
        hi2en_text = tokenizer_hi2en.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # st.write(f"Translated text (Hindi to English): {hi2en_text}")

    with st.spinner("Generating AI content..."):
        model = genai.GenerativeModel("gemini-1.5-flash")
        gen_text = model.generate_content(hi2en_text)
        # st.write(f"Generated AI Content: {gen_text.text}")

    #English to Hindi translation
    with st.spinner("Translating AI content back to Hindi..."):
        model_inputs = tokenizer_en2hi(gen_text.text, return_tensors="pt")
        generated_tokens = model_en2hi.generate(
            **model_inputs,
            forced_bos_token_id=tokenizer_en2hi.lang_code_to_id["hi_IN"]
        )
        en2hi_text = tokenizer_en2hi.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        st.write(f"{en2hi_text}")

st.write("This app translates text from Hindi to English, generates AI content, and translates it back to Hindi.")
