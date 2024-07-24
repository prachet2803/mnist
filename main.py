import gradio as gr
import tensorflow as tf
import numpy as np

# Load your saved MNIST model
model = tf.keras.models.load_model("mnist_digit_classification_model.h5")

# Define the prediction function
def predict_digit(image):
    # Preprocess the image to match the input format of the model
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image = np.array(image)  # Convert to numpy array
    image = image / 255.0  # Normalize the image
    image = image.reshape(1, 28, 28, 1)  # Reshape to match model input
    
    # Get the model's prediction
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)
    return predicted_digit

# Create the Gradio interface
interface = gr.Interface(fn=predict_digit,
                         inputs=gr.inputs.Sketchpad(shape=(28, 28)),
                         outputs=gr.outputs.Label(num_top_classes=1))

# Launch the interface
interface.launch()
