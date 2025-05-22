import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect
import torch
import os
from io import StringIO

# Проверка доступности CUDA
if torch.cuda.is_available():
    device = 0  # Использовать GPU если доступно
else:
    device = -1  # Использовать CPU если GPU недоступен

# Установка директории для кэша моделей
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')

# Настройки страницы
st.set_page_config(
    page_title="Нейро-переводчик",
    layout="wide"
)

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #2e6fdf;
        color: white;
    }
    .stTextArea>div>div>textarea {
        background-color: black;
        color: white;
    }
    .stTextArea>div>div>textarea::placeholder {
        color: #666666;
    }
    </style>
    """, unsafe_allow_html=True)

st.title(" Нейро-переводчик")
st.write("Переводите текст между различными языками с помощью нейронных сетей")

# Языки
LANGUAGES = {
    "Автоопределение": "auto",
    "Английский": "en",
    "Русский": "ru",
    "Французский": "fr",
    "Немецкий": "de",
    "Испанский": "es",
    "Китайский": "zh",
    "Японский": "ja"
}

# Обнаружение языков
LANG_CODES_TO_NAMES = {v: k for k, v in LANGUAGES.items()}

@st.cache_resource
def load_translation_pipeline(source_lang, target_lang):
    if source_lang == "auto":
        # Для автоопределения используем модель, которая поддерживает перевод с нескольких языков
        model_name = f"Helsinki-NLP/opus-mt-mul-{target_lang}"  # Используем мультиязычную модель
        try:
            translator = pipeline("translation", model=model_name, device=device)
            return translator, None
        except Exception as e:
            # Если мультиязычная модель недоступна, пробуем использовать модель для перевода с английского
            model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
            try:
                translator = pipeline("translation", model=model_name, device=device)
                return translator, None
            except Exception as e:
                return None, str(e)
    else:
        # Для конкретной пары языков используем соответствующую модель
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        try:
            translator = pipeline("translation", model=model_name, device=device)
            return translator, None
        except Exception as e:
            return None, str(e)

def translate_text(text, translator, source_lang="auto", max_length=1024):
    try:
        if source_lang == "auto":
            # Сначала определяем язык
            detected_lang = detect_language(text)
            if detected_lang:
                # Если язык определен, используем соответствующую модель
                model_name = f"Helsinki-NLP/opus-mt-{detected_lang}-{target_lang}"
                try:
                    translator = pipeline("translation", model=model_name, device=device)
                except:
                    # Если модель не найдена, используем мультиязычную модель
                    pass
        
        result = translator(text, max_length=max_length)
        return result[0]['translation_text'], None
    except Exception as e:
        return None, str(e)

#Определение языка введенного текста.
def detect_language(text):
    try:
        detected = detect(text)
        return detected if detected in LANG_CODES_TO_NAMES else None
    except:
        return None

def translate_file(file_content, translator):
    try:
        # подсчёт символов
        chunks = [file_content[i:i+1024] for i in range(0, len(file_content), 1024)]
        translated_chunks = []
        
        for chunk in chunks:
            translation, error = translate_text(chunk, translator)
            if error:
                return None, error
            translated_chunks.append(translation)
        
        return "\n".join(translated_chunks), None
    except Exception as e:
        return None, str(e)


# Выбор ввода
input_mode = st.radio("Выберите режим ввода:", ["Текст", "Файл"])

if input_mode == "Текст":
    # выбор языков
    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.selectbox("С какого языка:", list(LANGUAGES.keys()), index=0)
    with col2:
        target_lang = st.selectbox("На какой язык:", list(LANGUAGES.keys())[1:], index=0)

    # счётчик букв
    input_text = st.text_area("Введите текст для перевода:", height=150)
    st.caption(f"Количество символов: {len(input_text)}/1024")

    # автоопределение языка
    if input_text and source_lang == "Автоопределение":
        detected_lang_code = detect_language(input_text)
        if detected_lang_code and detected_lang_code in LANG_CODES_TO_NAMES:
            st.info(f"Определен язык: {LANG_CODES_TO_NAMES[detected_lang_code]}")
            source_lang = LANG_CODES_TO_NAMES[detected_lang_code]

    if st.button("Перевести"):
        if input_text:
            # коды языков
            source_code = LANGUAGES[source_lang]
            target_code = LANGUAGES[target_lang]
            
            # проверка языков
            if source_code == target_code:
                st.error("Выбраны одинаковые языки. Пожалуйста, выберите разные языки для перевода.")
                st.stop()
            
            if len(input_text) > 1024:
                st.warning("Текст слишком длинный. Пожалуйста, ограничьтесь 1024 символами.")
            else:
                with st.spinner("Загрузка модели перевода..."):
                    # Load translation pipeline
                    translator, load_error = load_translation_pipeline(source_code, target_code)
                    
                    if translator:
                        with st.spinner("Перевод..."):
                            translation, trans_error = translate_text(input_text, translator, source_code)
                            if translation:
                                st.text_area("Перевод:", value=translation, height=150)
                                st.success("Перевод завершен!")
                            else:
                                st.error(f"Ошибка перевода: {trans_error}")
                    else:
                        st.error(f"Не удалось загрузить модель перевода: {load_error}")
        else:
            st.warning("Пожалуйста, введите текст для перевода.")
else:
    # загрузка файла
    uploaded_file = st.file_uploader("Загрузите файл для перевода", type=['txt'])
    target_lang = st.selectbox("На какой язык перевести:", list(LANGUAGES.keys())[1:], index=0)
    
    if uploaded_file is not None:
        # считывание файла
        file_content = uploaded_file.getvalue().decode("utf-8")
        
        # автоопределение языка
        detected_lang_code = detect_language(file_content)
        if detected_lang_code and detected_lang_code in LANG_CODES_TO_NAMES:
            st.info(f"Определен язык: {LANG_CODES_TO_NAMES[detected_lang_code]}")
            source_lang = LANG_CODES_TO_NAMES[detected_lang_code]
        else:
            source_lang = "Английский"
        
        if st.button("Перевести файл"):
            # коды языков
            source_code = LANGUAGES[source_lang]
            target_code = LANGUAGES[target_lang]
            
            # проверка языков
            if source_code == target_code:
                st.error("Выбраны одинаковые языки. Пожалуйста, выберите разные языки для перевода.")
                st.stop()
            
            with st.spinner("Загрузка модели перевода..."):
                translator, load_error = load_translation_pipeline(source_code, target_code)
                
                if translator:
                    with st.spinner("Перевод файла..."):
                        translation, trans_error = translate_file(file_content, translator)
                        if translation:
                            st.text_area("Перевод:", value=translation, height=300)
                            
                            # создание кнопки
                            st.download_button(
                                label="Скачать переведенный файл",
                                data=translation,
                                file_name=f"translated_{uploaded_file.name}",
                                mime="text/plain"
                            )
                            st.success("Перевод завершен!")
                        else:
                            st.error(f"Ошибка перевода: {trans_error}")
                else:
                    st.error(f"Не удалось загрузить модель перевода: {load_error}")

# как пользоватся
with st.expander("ℹ️ Информация об использовании"):
    st.write("""
    - Выберите режим ввода: текст или файл
    - Для текстового режима:
        - Выберите исходный и целевой языки из выпадающих списков
        - Введите текст, который хотите перевести (максимум 1024 символа)
        - Система автоматически определит язык введенного текста
        - Нажмите кнопку "Перевести"
    - Для файлового режима:
        - Загрузите текстовый файл
        - Выберите целевой язык
        - Система автоматически определит исходный язык
        - Скачайте переведенный файл нажав кнопку "Скачать переведенный файл"
    """)