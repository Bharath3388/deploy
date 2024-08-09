from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
from googletrans import Translator
from gtts import gTTS
import os
from flask import Flask, render_template, request, send_file

# Initialize the model and components
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)
    
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    output_ids = model.generate(pixel_values, **gen_kwargs)
    
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

app = Flask(__name__)

def translate_text(text, dest_language):
    translator = Translator()
    translated_text = translator.translate(text, dest=dest_language)
    return translated_text.text

def text_to_speech(text, language):
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("static/caption_audio.mp3")

@app.route('/', methods=['GET'])
def index():
    return render_template('main.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = './static/' + imagefile.filename
    imagefile.save(image_path)
    
    selected_language = request.form['language']
    caption = predict_step([image_path])
    
    if caption:
        caption_text = caption[0]
    else:
        caption_text = "No caption generated."
    
    translated_caption = translate_text(caption_text, selected_language)
    text_to_speech(translated_caption, selected_language)
    
    return render_template('display3.html', prediction=translated_caption, path=image_path)

@app.route('/caption_audio')
def caption_audio():
    return send_file("static/caption_audio.mp3", as_attachment=True)

