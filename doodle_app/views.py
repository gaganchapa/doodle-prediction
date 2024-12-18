from django.shortcuts import render, redirect
from django.http import JsonResponse
import numpy as np
import tensorflow as tf
from django.views.decorators.csrf import csrf_exempt
import base64
from PIL import Image
from io import BytesIO

def landing_page(request):
    return render(request, 'landing.html')

def canvas_page(request):
    return render(request, 'canvas.html')

def contact(request):
    return render(request, 'contact.html')

def ai(request):
    return render(request, 'ai.html')




model_path = 'model50.keras'

model = tf.keras.models.load_model(model_path)



class_names = ['airplane',
 'bicycle',
 'bird',
 'birthday_cake',
 'candle',
 'car',
 'chair',
 'cloud',
 'fish',
 'mountain',
 'octopus',
 'smiley_face',
 'table',
 'tree',
 'umbrella']
WIDTH, HEIGHT = 28, 28

def preprocess_image(img):
    # Convert the image to grayscale
    img = img.convert('L')  # 'L' mode ensures grayscale

    # Resize the image to match the model's input dimensions (28x28)
    img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)

    # Invert the image so the background is black and the drawing is white
    img = Image.fromarray(255 - np.array(img))

    # Normalize pixel values to range [0, 1]
    img_array = np.array(img) / 255.0

    # Reshape the array to add the batch and channel dimensions (1, 28, 28, 1)
    img_array = img_array.reshape((1, HEIGHT, WIDTH, 1))
    return img_array

@csrf_exempt
def classify_doodle(request):
    if request.method == 'POST':
        try:
            # Extract data from request
            data = request.POST.get('image')
            expected = request.POST.get('expected')
            if not data or not expected:
                return JsonResponse({'error': 'Image or expected object missing'}, status=400)

            # Decode and process the image
            header, encoded = data.split(',', 1)
            img_bytes = base64.b64decode(encoded)
            img = Image.open(BytesIO(img_bytes))
            img_array = preprocess_image(img)

            # Predict the class using the ML model
            if model is None:
                return JsonResponse({'error': 'Model is not loaded'}, status=500)

            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)

            # Check if prediction matches the expected object
            if class_names[predicted_class] == expected:
                return JsonResponse({
                    'success': True,
                    'class': class_names[predicted_class],
                    'confidence': float(prediction[0][predicted_class])
                })
            else:
                return JsonResponse({
                    'success': False,
                    'class': class_names[predicted_class],
                    'confidence': float(prediction[0][predicted_class])
                })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)



import random
from django.http import JsonResponse

def get_random_object(request):
    random_object = random.choice(class_names)
    return JsonResponse({'object': random_object})



