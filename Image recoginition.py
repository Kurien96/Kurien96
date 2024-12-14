import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, pipeline

# Load the pre-trained YOLOv5 model (change to a larger model if needed)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # Use yolov5m for better detection

# Load and display an image
image_path = r"C:\Users\kkkos\Desktop\busy-pedestrian-street-city-with-people-crossing-road-cars-parked-side-road-buildings-background_14117-356710.jpg"  # Use a raw string for the path
image = Image.open(image_path)
plt.imshow(image)
plt.axis('off')
plt.show()

# Run the object detection model on the image
results = yolo_model(image)

# Display results
results.print()  # print the detected objects
results.show()  # display the image with the bounding boxes

# Load the image captioning model
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Preprocess the image
inputs = feature_extractor(images=image, return_tensors="pt")

# Generate caption
pixel_values = inputs.pixel_values
output_ids = caption_model.generate(pixel_values, max_length=16, num_beams=4)
captions = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Caption:", captions)

# Load a pre-trained text generation model (with token authentication if needed)
# Replace <YOUR_TOKEN> with your actual Hugging Face token
# Load a pre-trained text generation model (without token if not needed)
chatbot = pipeline("text-generation", model="gpt-2")  # Remove use_auth_token if the model is public


# Function to generate a response based on image data
def generate_response(image_path, user_input):
    # 1. Detect objects in the image using YOLO
    image = Image.open(image_path)
    results = yolo_model(image)
    detected_objects = results.pandas().xyxy[0]['name'].tolist()  # List of detected object names

    # 2. Get image caption
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values
    output_ids = caption_model.generate(pixel_values, max_length=16, num_beams=4)
    captions = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 3. Generate response based on objects and caption
    context = f"The image contains: {', '.join(detected_objects)}. Caption: {captions}"
    response = chatbot(f"{user_input} {context}")

    return response[0]['generated_text']

# Example usage
user_query = "Can you describe what's happening here?"
response = generate_response(image_path, user_query)
print("Response:", response)
