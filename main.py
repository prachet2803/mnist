import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your saved MNIST model
model = tf.keras.models.load_model('mnist_digit_classification_model.h5')

# Define the prediction function
def predict_digit(image):
    # Convert to grayscale and resize to 28x28
    image = Image.fromarray(image).convert('L').resize((28, 28))
    image = np.array(image) / 255.0  # Normalize the image
    image = image.reshape(1, 28, 28, 1)  # Reshape to match model input
    
    # Get the model's prediction
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)
    return predicted_digit

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.inputs.Sketchpad(shape=(28, 28)),
    outputs="label",
)

# Launch the interface
interface.launch()
