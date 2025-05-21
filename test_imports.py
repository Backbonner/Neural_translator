import streamlit
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect
import torch
import sentencepiece

print("All imports successful!") 