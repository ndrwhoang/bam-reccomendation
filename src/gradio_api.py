import os
import configparser

import torch
import gradio as gr
from transformers import AutoTokenizer

from src.inference.custom_pipeline import RecommendationPipeline
from src.inference.inference_utils import load_model, load_tokenizer


def greet(text):
    return pipeline(text)

config = configparser.ConfigParser()
config.read(os.path.normpath('configs/base_config.ini'))
model = load_model(config)
tokenizer = load_tokenizer(config)
pipeline = RecommendationPipeline(
    model=model,
    tokenizer=tokenizer
)
input_textbox = gr.Textbox(label='Description', placeholder='bam description', lines=2)


demo = gr.Interface(fn=greet, inputs='text', outputs='text')
demo.launch()