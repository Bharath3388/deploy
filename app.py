from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
from googletrans import Translator
from gtts import gTTS
import os
from flask import Flask, render_template, request, send_file
import tempfile

# Initialize Flask application
app = Flask(__name__)

# Initialize the model and components once
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define generation parameters
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        with Image.open(image_path) as i_image:
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")
            images.append(i_image)
    
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    output_ids = model.generate(pixel_values, **gen_kwargs)
    
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

def translate_text(text, dest_language):
    translator = Translator()
    translated_text = translator.translate(text, dest=dest_language)
    return translated_text.text

def text_to_speech(text, language):
    # Use a temporary file for storing audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(tmp_file.name)
        return tmp_file.name

@app.route('/', methods=['GET'])
def index():
    return render_template('main.html')

@app.route('/', methods=['POST'])
def predict():
    # Get the uploaded image file
    imagefile = request.files['imagefile']
    
    # Use a temporary file for image storage
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(imagefile.filename)[1]) as tmp_image:
        imagefile.save(tmp_image.name)
        image_path = tmp_image.name
    
    # Get the selected language from the form
    selected_language = request.form['language']
    
    # Generate caption for the uploaded image
    caption = predict_step([image_path])
    
    caption_text = caption[0] if caption else "No caption generated."
    
    # Translate the generated caption
    translated_caption = translate_text(caption_text, selected_language)
    
    # Convert caption to speech and get the file path
    audio_path = text_to_speech(translated_caption, selected_language)
    
    return render_template('display3.html', prediction=translated_caption, path=image_path, audio_path=audio_path)

@app.route('/caption_audio')
def caption_audio():
    audio_path = request.args.get('audio_path')
    if audio_path and os.path.exists(audio_path):
        return send_file(audio_path, as_attachment=True)
    return "Audio file not found", 404
