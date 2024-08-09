{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/bharathmogalapu/Library/Python/3.9/lib/python/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020\n",
      "  warnings.warn(\n",
      "/Users/bharathmogalapu/Library/Python/3.9/lib/python/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.\n",
      "  warnings.warn(\n",
      "/Users/bharathmogalapu/Library/Python/3.9/lib/python/site-packages/transformers/models/vit/feature_extraction_vit.py:28: FutureWarning: The class ViTFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use ViTImageProcessor instead.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[31m\u001b[1mWARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\u001b[0m\n",
      " * Running on http://127.0.0.1:6969\n",
      "\u001b[33mPress CTRL+C to quit\u001b[0m\n",
      "127.0.0.1 - - [05/Aug/2024 16:56:18] \"GET / HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [05/Aug/2024 16:56:18] \"GET / HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [05/Aug/2024 16:56:18] \"\u001b[33mGET /main.css HTTP/1.1\u001b[0m\" 404 -\n",
      "127.0.0.1 - - [05/Aug/2024 16:56:18] \"GET /static/image.png HTTP/1.1\" 200 -\n"
     ]
    }
   ],
   "source": [
    "from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer\n",
    "import torch\n",
    "from PIL import Image\n",
    "from googletrans import Translator\n",
    "from gtts import gTTS\n",
    "import os\n",
    "from flask import Flask, render_template, request, send_file\n",
    "\n",
    "model = VisionEncoderDecoderModel.from_pretrained(\"nlpconnect/vit-gpt2-image-captioning\")\n",
    "feature_extractor = ViTFeatureExtractor.from_pretrained(\"nlpconnect/vit-gpt2-image-captioning\")\n",
    "tokenizer = AutoTokenizer.from_pretrained(\"nlpconnect/vit-gpt2-image-captioning\")\n",
    "\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "model.to(device)\n",
    "\n",
    "max_length = 16\n",
    "num_beams = 4\n",
    "gen_kwargs = {\"max_length\": max_length, \"num_beams\": num_beams}\n",
    "\n",
    "def predict_step(image_paths):\n",
    "    images = []\n",
    "    for image_path in image_paths:\n",
    "        i_image = Image.open(image_path)\n",
    "        if i_image.mode != \"RGB\":\n",
    "            i_image = i_image.convert(mode=\"RGB\")\n",
    "        images.append(i_image)\n",
    "    \n",
    "    pixel_values = feature_extractor(images=images, return_tensors=\"pt\").pixel_values\n",
    "    pixel_values = pixel_values.to(device)\n",
    "    \n",
    "    output_ids = model.generate(pixel_values, **gen_kwargs)\n",
    "    \n",
    "    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)\n",
    "    preds = [pred.strip() for pred in preds]\n",
    "    return preds\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "def translate_text(text, dest_language):\n",
    "    translator = Translator()\n",
    "    translated_text = translator.translate(text, dest=dest_language)\n",
    "    return translated_text.text\n",
    "\n",
    "def text_to_speech(text, language):\n",
    "    tts = gTTS(text=text, lang=language, slow=False)\n",
    "    tts.save(\"static/caption_audio.mp3\")\n",
    "\n",
    "@app.route('/', methods=['GET'])\n",
    "def index():\n",
    "    return render_template('main.html')\n",
    "\n",
    "@app.route('/', methods=['POST'])\n",
    "def predict():\n",
    "    # Get the uploaded image file\n",
    "    imagefile = request.files['imagefile']\n",
    "    # Save the image file\n",
    "    image_path = './static/' + imagefile.filename\n",
    "    imagefile.save(image_path)\n",
    "    # Get the selected language from the form\n",
    "    selected_language = request.form['language']\n",
    "    # Generate caption for the uploaded image\n",
    "    caption = predict_step([image_path])\n",
    "    if caption:\n",
    "        caption_text = caption[0]  # Extract the string from the list\n",
    "    else:\n",
    "        caption_text = \"No caption generated.\"\n",
    "\n",
    "    # Translate the generated caption\n",
    "    translated_caption = translate_text(caption_text, selected_language)\n",
    "    # Convert caption to speech\n",
    "    text_to_speech(translated_caption, selected_language)\n",
    "    return render_template('display3.html', prediction=translated_caption, path=image_path)\n",
    "\n",
    "# Route to serve the audio file\n",
    "@app.route('/caption_audio')\n",
    "def caption_audio():\n",
    "    return send_file(\"static/caption_audio.mp3\", as_attachment=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
