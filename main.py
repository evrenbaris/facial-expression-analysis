# Import required libraries
from fer import FER
import matplotlib.pyplot as plt
from google.colab import files

# Upload an image
print("Please upload an image file:")
uploaded = files.upload()

# Get the uploaded file name
image_path = list(uploaded.keys())[0]

# Load and analyze the image
image = plt.imread(image_path)
detector = FER(mtcnn=True)
emotion, score = detector.top_emotion(image)

# Print the detected emotion and score
print(f"Detected Emotion: {emotion} with a confidence score of {score}")

# Display the image with the emotion result
plt.imshow(image)
plt.title(f"Emotion: {emotion} (Confidence: {score})")
plt.axis('off')
plt.show()
