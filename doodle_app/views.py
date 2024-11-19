from django.shortcuts import render, redirect
from django.http import JsonResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from django.views.decorators.csrf import csrf_exempt
import base64
from PIL import Image
from io import BytesIO

# Load the pre-trained model (ensure the model path is correct)
model = load_model('D:\Doodle\model.h5')
CLASSES = ['apple', 'banana', 'calculator', 'candle', 'circle', 'cloud', 'donut', 'fish', 'flower', 'hexagon', 'house', 'ladder', 'pizza', 'square', 'sword', 'watermelon', 'wheel']

def landing_page(request):
    return render(request, 'landing.html')

def canvas_page(request):
    return render(request, 'canvas.html')

def contact(request):
    return render(request, 'contact.html')

def ai(request):
    return render(request, 'ai.html')
import os

import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import uuid
from datetime import datetime

# Your model and class list
# model = ...  # Define your model
# CLASSES = [...]  # Define your class labels

@csrf_exempt
def classify_doodle(request):
    if request.method == 'POST':
        try:
            # Get the image data from the request
            data = request.POST.get('image')
            
            if not data:
                return JsonResponse({'error': 'No image data provided'}, status=400)

            # Remove the data URL prefix (e.g., 'data:image/png;base64,')
            header, encoded = data.split(',', 1)
            
            # Decode the base64 image
            img_bytes = base64.b64decode(encoded)
            
            # Open the image using PIL
            img = Image.open(BytesIO(img_bytes))
            
            # Ensure the image is in RGB mode
            img = img.convert('RGB')
            
            # Convert to grayscale
            img = img.convert('L')
            
            # Resize the image to 28x28 pixels
            img = img.resize((28, 28), Image.LANCZOS)
            
            # Convert image to numpy array and normalize
            img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
            
            # Predict the class using the model

            
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            
            return JsonResponse({
                'class': CLASSES[predicted_class],
                'confidence': float(prediction[0][predicted_class])
            })
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)
