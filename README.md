# AI Pneumonia Detector Workshop
### KISJ Medical Club 4-Week Fast-Track Program for AI in Medicine

![Workshop Poster](assets/workshop-poster.png)

This repository contains the full lecture notes and Google Colab notebook content for the **AI Pneumonia Detector Workshop**, a 4-week hands-on program hosted by the **Medical Club at Korea International School, Jeju Campus (KISJ)**.

As the **Medical Club President**, I designed and tutored this workshop for students interested in artificial intelligence, computer vision, and medicine. The workshop guided students through a complete beginner-friendly medical AI workflow: loading chest X-ray data, building a CNN model, training it, saving it, and running inference on unseen test images.

> **Important note:** This project is for educational purposes only. It is not intended for clinical diagnosis, medical decision-making, or professional healthcare use.

---

## Program Overview

- **Organizer:** KISJ Medical Club
- **School:** Korea International School, Jeju Campus
- **Instructor:** Danu Kim, Medical Club President
- **Format:** 4-week fast-track workshop
- **Session Length:** 30 minutes per week
- **Target Audience:** Medical Club members and students interested in AI and medicine
- **Main Tool:** Google Colab
- **Main Topic:** Chest X-ray pneumonia classification using a CNN

---

## Curriculum Summary

| Week | Topic | Main Focus |
|---|---|---|
| Week 1 | Vision Principles & Data Check | Google Colab, datasets, PneumoniaMNIST, normal vs. pneumonia X-ray visualization |
| Week 2 | AI Model Design - CNN Basics | Conv2D, MaxPooling, Flatten, Dense, Sequential CNN model construction |
| Week 3 | Intensive Training & Loss | Epoch, loss, accuracy, model training, training graphs, model saving |
| Week 4 | System Finalization & Test Inference | Loading a saved model, test evaluation, prediction confidence, inference on unseen images |

---

## Repository Structure

```
AI-Pneumonia-Detector-Workshop/
|
├── README.md
├── assets/
│   └── workshop-poster.png
|
├── medicalclub-week-1.docx
├── medicalclub-week-1.ipynb
├── medicalclub-week-2.docx
├── medicalclub-week-2.ipynb
├── medicalclub-week-3.docx
├── medicalclub-week-3.ipynb
├── medicalclub-week-4.docx
└── medicalclub-week-4.ipynb
```

---

## Tools and Technologies

- Python
- Google Colab
- TensorFlow / Keras
- Convolutional Neural Networks (CNNs)
- MedMNIST / PneumoniaMNIST
- NumPy
- Matplotlib

---

## Dataset

This workshop uses **PneumoniaMNIST**, a lightweight chest X-ray image dataset prepared for machine learning practice. Each image is assigned a binary label:

- `0` = Normal
- `1` = Pneumonia

Reference:

> J. Yang, R. Shi, D. Wei, Z. Liu, L. Zhao, B. Ke, H. Pfister, and B. Ni, “MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification,” *Scientific Data*, vol. 10, no. 1, Art. no. 41, 2023.

---

## Complete Workshop Content

The following sections include the full lecture-note content and the on-screen Colab notebook walkthrough for each week.

---

## Week 1: Introduction to Google Colab and Exploring Chest X-ray Images

### Lecture Note

#### Week 1 Lecture Note
Topic: Introduction to Google Colab and Exploring Chest X-ray Images
#### 1. Objective
In this lesson, students will learn what Google Colab is, practice running code in Colab, understand the meaning of a dataset, load the PneumoniaMNIST dataset, and display one normal chest X-ray image and one pneumonia chest X-ray image.
#### 2. What is Google Colab?
Google Colab is an online coding notebook that allows us to write and run Python code in a web browser. It is free to use, requires no installation, is easy to share, and is widely used for AI and data science projects.
**Important parts of Google Colab:**
Code cell: a box where you write code
Run button: the button that executes the code in a cell
Output area: the place where results appear after the code is run
#### 3. What is PneumoniaMNIST?
PneumoniaMNIST is a dataset of chest X-ray images prepared for machine learning practice. Each image is assigned a label:
0 = normal
1 = pneumonia
**Reference:**
J. Yang, R. Shi, D. Wei, Z. Liu, L. Zhao, B. Ke, H. Pfister, and B. Ni, “MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification,” Scientific Data, vol. 10, no. 1, Art. no. 41, 2023.
#### 4. First Practice in Colab
First, go to the Colab and click New Notebook. A new notebook will open. This is your Colab file. You can rename it by clicking the notebook title at the top of the page.
In this activity, we will download the dataset, find one image labeled as normal, find one image labeled as pneumonia, and display both images side by side.

### Colab Notebook Walkthrough

#### 0. First Practice in Colab
```python
print("Hello, Colab!")
```
```python
x = 3
y = 5
print(x + y)
```
#### 1. Loading the Dataset
```python
# !pip install medmnist installs the dataset package
!pip install medmnist

# import matplotlib.pyplot as plt imports a tool for showing images
import matplotlib.pyplot as plt

# from medmnist import PneumoniaMNIST imports the dataset
from medmnist import PneumoniaMNIST
```
#### 2. Downloading the Training Dataset
```python
# train_dataset stores the dataset
# split='train' means we are loading the training set
# download=True means Colab downloads it for us
train_dataset = PneumoniaMNIST(split='train', download=True)
```
#### 3. Checking the Number of Images
```python
# tells us how many training images are in the dataset.
print("Number of training images:", len(train_dataset))
```
#### 4. Looking at One Example
```python
import numpy as np

# image is the chest X-ray image
# label is the correct answer for that image
image, label = train_dataset[0]

# Convert the PIL Image to a NumPy array to access the .shape attribute
image_np = np.array(image)

print("Image shape:", image_np.shape)
print("Label:", label)
```
#### 5. Displaying One Image
```python
# plt.imshow() displays the image
# cmap='gray' shows the image in black and white
plt.imshow(image_np.squeeze(), cmap='gray')

plt.title(f"Label: {label[0]}")
plt.axis('off')

# plt.show() displays the final result
plt.show()
```
#### 6. Displaying a Normal Image
```python
# Find a normal image (label 0)
normal_image = None
normal_label = None

# Iterate through the dataset
for i in range(len(train_dataset)):
    image, label = train_dataset[i]

    # Check if the label is '0' (normal)
    if label[0] == 0:
        normal_image = np.array(image)
        normal_label = label
        break # Stop after finding the first normal image

if normal_image is not None:
    print(f"Found a normal image with label: {normal_label}")

    # Display the normal image
    plt.imshow(normal_image.squeeze(), cmap='gray')
    plt.title(f"Label: {normal_label[0]} (Normal)")
    plt.axis('off')
    plt.show()
else:
    print("Could not find a normal image (label 0) in the training dataset.")
```


---

## Week 2: CNN Basics and Sequential Model Design

### Lecture Note

#### Week 2 Lecture Note
Topic: Understand the idea of CNNs and build a small Sequential model in Google Colab
#### 1. Objectives
- (1) Explain what a CNN does in simple language: it learns useful patterns from images.
- (2) Introduce the role of the main layers: Conv2D, MaxPooling, Flatten, and Dense.
- (3) Help students assemble a basic Sequential model without writing everything from scratch.
#### 2. Key Concepts
- (1) What is a CNN?
A Convolutional Neural Network (CNN) is a deep learning model designed for image analysis. It detects small visual patterns first, then combines them to make a final prediction such as “Pneumonia” or “Normal.”
In simple terms, the process is: find features → reduce size → flatten information → make a prediction
- (2) Conv2D with Filter / Kernel
A filter (or kernel) is like a small moving window. It scans the image step by step and responds when it finds useful patterns such as edges, brightness changes, or textures.
Instead of understanding the whole image at once, the model examines small regions and extracts important features gradually.
- (3) Pooling
Pooling reduces the size of the feature maps while keeping the most important information. This makes the model more efficient and helps it focus on stronger signals. MaxPooling keeps the strongest responses and removes less important details, reducing computation.
- (4) Dense Layer
After the model extracts image features, the Dense layer combines them and makes the final decision.
Before this step, the 2D feature maps are converted into a 1D form so the model can use all collected clues to classify the image as Normal or Pneumonia.
- (5) Compile: The compile step tells the model how to learn.
Loss: measures how wrong the prediction is
Optimizer: updates the model to reduce error
Metrics: shows performance during training

### Colab Notebook Walkthrough

#### 1. Loading the Dataset
```python
# !pip install medmnist installs the dataset package
!pip install medmnist

# import matplotlib.pyplot as plt imports a tool for showing images
import matplotlib.pyplot as plt

# from medmnist import PneumoniaMNIST imports the dataset
from medmnist import PneumoniaMNIST
```
#### 2. Downloading the Training Dataset
```python
# train_dataset stores the dataset
# split='train' means we are loading the training set
# download=True means Colab downloads it for us
train_dataset = PneumoniaMNIST(split='train', download=True)
```
#### 3. Checking the Number of Images
```python
# tells us how many training images are in the dataset.
print("Number of training images:", len(train_dataset))
```
#### 4. Import TensorFlow / Keras
```python
# Import TensorFlow, the main deep learning library
import tensorflow as tf

# Import Keras layers and model tools for building a neural network.
from tensorflow.keras import layers, models
```
#### 5. Build a Simple CNN Model
```python
# Create a CNN model by stacking layers in order.
model = models.Sequential([
    # Define the input shape: 28x28 grayscale image (1 channel)
    layers.Input(shape=(28, 28, 1)),

    # Apply 16 filters of size 3x3 to detect simple image features.
    layers.Conv2D(16, (3, 3), activation='relu'),

    # Reduce the feature map size while keeping important information.
    layers.MaxPooling2D((2, 2)),

    # Apply 32 filters to learn more complex patterns.
    layers.Conv2D(32, (3, 3), activation='relu'),

    # Downsample again to make the model more efficient.
    layers.MaxPooling2D((2, 2)),

    # Convert 2D feature maps into a 1D vector.
    layers.Flatten(),

    # Fully connected layer to combine extracted features.
    layers.Dense(64, activation='relu'),

    # Output one probability for binary classification.
    layers.Dense(1, activation='sigmoid')
])
```
#### 6. Show Model Structure
```python
# Display the model architecture, output shapes, and number of parameters in each layer.
model.summary()
```
14. Compile the Model
- loss: how wrong the model is
- optimizer: how the model updates itself
- metrics: what performance we want to monitor
```python
model.compile(
    optimizer='adam',           # Use Adam to update the model weights during training.
    loss='binary_crossentropy', # Use binary cross-entropy because this is a two-class problem.
    metrics=['accuracy']        # Track accuracy to see how often the model predicts correctly.
)
```
#### 15. Practice: change the numbers
```python
model = tf.keras.Sequential([
    # Define the input shape: 28x28 grayscale image (1 channel)
    layers.Input(shape=(28, 28, 1)),

    # TODO 1: Change the number of filters in the first convolutional layer from 16 to 32.
    layers.Conv2D(32 , 3, activation='relu'),
    layers.MaxPooling2D(),

    # TODO 1: Change the number of filters in the second convolutional layer from 32 to 64.
    layers.Conv2D(32 , 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    # TODO 3: Change the number of units in the Dense layer from 64 to 128.
    layers.Dense(128  , activation='relu'),

    layers.Dense(1, activation='sigmoid')
])

model.summary()
```


---

## Week 3: Training the CNN and Checking Performance

### Lecture Note

#### Week 3 Lecture Note
Topic: Training the CNN and Checking Performance in Google Colab
#### 1. Objectives
- (1) Understand what it means to train a model; Learn the meaning of epoch, loss, and accuracy.
- (2) Run model.fit() to start actual learning; Observe how the model improves over time,
#### 2. Key Concepts
- (1) What is an Epoch?
An epoch means one full round of learning using the whole training dataset once. An epoch is like one full study session over all practice questions.
Epoch 1 = the model sees all training images one time
Epoch 2 = the model sees them all again
Epoch 3 = one more full round
- (2) What is Loss?
Loss shows how wrong the model is: high loss = the model is making larger mistakes; low loss = the model is making smaller mistakes
Loss is not the same as accuracy; the model tries to minimize loss during training
- (3) What is Accuracy?
Accuracy shows how often the model predicts correctly; accuracy = 0.85 means 85% correct
Loss = how wrong. Accuracy = how often correct
- (4) Saving the Model
Saving the model means storing what the AI has learned so you can use it later without training again. In your class, after model.fit(), the CNN has adjusted many internal numbers called weights. So when you run: model.save("pneumonia_cnn_week3.h5")
It includes the model structure, the learned weights, and often the training configuration.
If you save it, you do not need to retrain from the beginning

### Colab Notebook Walkthrough

#### 1. Loading the Dataset
```python
# !pip install medmnist installs the dataset package
!pip install medmnist

# import matplotlib.pyplot as plt imports a tool for showing images
import matplotlib.pyplot as plt

# from medmnist import PneumoniaMNIST imports the dataset
from medmnist import PneumoniaMNIST
```
#### 2. Downloading the Training Dataset
```python
# train_dataset stores the dataset
# split='train' means we are loading the training set
# download=True means Colab downloads it for us
train_dataset = PneumoniaMNIST(split='train', download=True)
```
#### 3. Checking the Number of Images
```python
# tells us how many training images are in the dataset.
print("Number of training images:", len(train_dataset))
```
#### 4. Import TensorFlow / Keras
```python
# Import TensorFlow, the main deep learning library
import tensorflow as tf

# Import Keras layers and model tools for building a neural network.
from tensorflow.keras import layers, models
```
#### 5. Build a Simple CNN Model
```python
# Create a CNN model by stacking layers in order.
model = models.Sequential([
    # Define the input shape: 28x28 grayscale image (1 channel)
    layers.Input(shape=(28, 28, 1)),

    # Apply 16 filters of size 3x3 to detect simple image features.
    layers.Conv2D(16, (3, 3), activation='relu'),

    # Reduce the feature map size while keeping important information.
    layers.MaxPooling2D((2, 2)),

    # Apply 32 filters to learn more complex patterns.
    layers.Conv2D(32, (3, 3), activation='relu'),

    # Downsample again to make the model more efficient.
    layers.MaxPooling2D((2, 2)),

    # Convert 2D feature maps into a 1D vector.
    layers.Flatten(),

    # Fully connected layer to combine extracted features.
    layers.Dense(64, activation='relu'),

    # Output one probability for binary classification.
    layers.Dense(1, activation='sigmoid')
])
```
#### 6. Show Model Structure
```python
# Display the model architecture, output shapes, and number of parameters in each layer.
model.summary()
```
14. Compile the Model
- loss: how wrong the model is
- optimizer: how the model updates itself
- metrics: what performance we want to monitor
```python
model.compile(
    optimizer='adam',           # Use Adam to update the model weights during training.
    loss='binary_crossentropy', # Use binary cross-entropy because this is a two-class problem.
    metrics=['accuracy']        # Track accuracy to see how often the model predicts correctly.
)
```
15. Convert dataset to NumPy arrays
- NumPy makes handling large image datasets faster, easier, and compatible with AI models.
```python
import numpy as np

# Convert all training images to NumPy arrays and normalize pixel values to the range [0, 1]
x_train = np.array([np.array(img, dtype=np.float32) / 255.0 for img, label in train_dataset])

# Extract the label for each training image
y_train = np.array([label[0] for img, label in train_dataset])

# Add a channel dimension so the image shape becomes (28, 28, 1)
# This is needed because CNN models expect height, width, and channel
x_train = x_train[..., np.newaxis]

# Print the shapes of the image data and label data
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
```
15. Train the model
- training images = about 4,700, batch size = 32
- then: 4700 % 32 ≈ 148.
- So Colab shows about 148 steps to finish one full epoch.
```python
# Train the CNN model using the training images and labels
history = model.fit(
    x_train, y_train,   # input images and their correct labels
    epochs=30,           # repeat training 5 times over the whole dataset
    batch_size=64       # use 32 images at a time in each training step
)
```
#### 16. Visualize loss
```python
# Create a new figure for the loss graph
plt.figure(figsize=(6,4))

# Plot the training loss recorded after each epoch
plt.plot(history.history['loss'], label='train loss')

# Label the x-axis as Epoch
plt.xlabel('Epoch')

# Label the y-axis as Loss
plt.ylabel('Loss')

# Add a title to the graph
plt.title('Training Loss')

# Show the legend
plt.legend()
plt.show()
```
#### 17. Visualize accuracy
```python
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='train accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.show()
```
18. Save the model

- After running this cell, check the Files icon (left) in Colab.
- You should see "my_model.keras" there.
- That means the trained model was saved successfully.
```python
# Save the trained model to a file named "my_model.keras"
model.save('my_model.keras')

# Print a message to confirm that the model has been saved
print("Model saved!")
```


---

## Week 4: Testing the Saved Model and Making Predictions

### Lecture Note

#### Week 4 Lecture Note
Topic: Testing the Saved Model and Making Predictions
#### 1. Objectives
- (1) Understand what prediction means in AI and load the model saved in Week 3
- (2) Test the model on unseen images and see whether the model predicts Normal or Pneumonia
- (3) Understand how AI can help in medical image analysis.
#### 2. Key Concepts
- (1) What is Inference (or Prediction)?
Inference means using a trained model to make a prediction.
**For example:**
input: a new chest X-ray image
output: Normal or Pneumonia
- (2) Why Use a Saved Model?
Because the model has already learned in Week 3. So instead of training again, we can:
load the saved model
use it immediately
test it on images
- (3) Confidence / Probability
The model output is often a number between 0 and 1. Example:
0.10 → likely Normal
0.85 → likely Pneumonia
**A simple rule:**
below 0.5 → Normal
0.5 or above → Pneumonia
- (4) Real-World Meaning
You can connect this to medical applications. AI does not replace doctors, but it can help detect patterns quickly and support medical decision-making.
We built a complete AI medical imaging project. First, we explored the data. Then we built a CNN, trained it, saved it, and finally used it to predict new chest X-ray images.

### Colab Notebook Walkthrough

#### 1. Loading the Dataset
```python
# !pip install medmnist installs the dataset package
!pip install medmnist

# import matplotlib.pyplot as plt imports a tool for showing images
import matplotlib.pyplot as plt

# from medmnist import PneumoniaMNIST imports the dataset
from medmnist import PneumoniaMNIST
```
#### 2. Downloading the Training Dataset
```python
# train_dataset stores the dataset
# split='train' means we are loading the training set
# download=True means Colab downloads it for us
train_dataset = PneumoniaMNIST(split='train', download=True)
```
#### 3. Checking the Number of Images
```python
# tells us how many training images are in the dataset.
print("Number of training images:", len(train_dataset))
```
#### 4. Import TensorFlow / Keras
```python
# Import TensorFlow, the main deep learning library
import tensorflow as tf

# Import Keras layers and model tools for building a neural network.
from tensorflow.keras import layers, models
```
#### 5. Build a Simple CNN Model
```python
# Create a CNN model by stacking layers in order.
model = models.Sequential([
    # Define the input shape: 28x28 grayscale image (1 channel)
    layers.Input(shape=(28, 28, 1)),

    # Apply 16 filters of size 3x3 to detect simple image features.
    layers.Conv2D(16, (3, 3), activation='relu'),

    # Reduce the feature map size while keeping important information.
    layers.MaxPooling2D((2, 2)),

    # Apply 32 filters to learn more complex patterns.
    layers.Conv2D(32, (3, 3), activation='relu'),

    # Downsample again to make the model more efficient.
    layers.MaxPooling2D((2, 2)),

    # Convert 2D feature maps into a 1D vector.
    layers.Flatten(),

    # Fully connected layer to combine extracted features.
    layers.Dense(64, activation='relu'),

    # Output one probability for binary classification.
    layers.Dense(1, activation='sigmoid')
])
```
#### 6. Show Model Structure
```python
# Display the model architecture, output shapes, and number of parameters in each layer.
model.summary()
```
14. Compile the Model
- loss: how wrong the model is
- optimizer: how the model updates itself
- metrics: what performance we want to monitor
```python
model.compile(
    optimizer='adam',           # Use Adam to update the model weights during training.
    loss='binary_crossentropy', # Use binary cross-entropy because this is a two-class problem.
    metrics=['accuracy']        # Track accuracy to see how often the model predicts correctly.
)
```
15. Convert dataset to NumPy arrays
- NumPy makes handling large image datasets faster, easier, and compatible with AI models.
```python
import numpy as np

# Convert all training images to NumPy arrays and normalize pixel values to the range [0, 1]
x_train = np.array([np.array(img, dtype=np.float32) / 255.0 for img, label in train_dataset])

# Extract the label for each training image
y_train = np.array([label[0] for img, label in train_dataset])

# Add a channel dimension so the image shape becomes (28, 28, 1)
# This is needed because CNN models expect height, width, and channel
x_train = x_train[..., np.newaxis]

# Print the shapes of the image data and label data
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
```
15. Train the model
- training images = about 4,700, batch size = 32
- then: 4700 % 32 ≈ 148.
- So Colab shows about 148 steps to finish one full epoch.

- You can change the epochs (1 to 10) and batch_size (8, 64, 128)
```python
# Train the CNN model using the training images and labels
history = model.fit(
    x_train, y_train,   # input images and their correct labels
    epochs=30,           # repeat training 5 times over the whole dataset
    batch_size=32       # use 32 images at a time in each training step
)
```
#### 16. Visualize loss
```python
# Create a new figure for the loss graph
plt.figure(figsize=(6,4))

# Plot the training loss recorded after each epoch
plt.plot(history.history['loss'], label='train loss')

# Label the x-axis as Epoch
plt.xlabel('Epoch')

# Label the y-axis as Loss
plt.ylabel('Loss')

# Add a title to the graph
plt.title('Training Loss')

# Show the legend
plt.legend()
plt.show()
```
#### 17. Visualize accuracy
```python
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='train accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.show()
```
18. Save the model

- After running this cell, check the Files icon (left) in Colab.
- You should see "my_model.keras" there.
- That means the trained model was saved successfully.
```python
# Save the trained model to a file named "my_model.keras"
model.save('my_model.keras')

# Print a message to confirm that the model has been saved
print("Model saved!")
```
### Week 4 Colab Notebook ####
#### 19. Load the saved model
```python
from tensorflow.keras.models import load_model

# Load the trained model saved in Week 3
model = load_model("my_model.keras")

# Print a message to confirm that the model was loaded
print("Model loaded!")
```
#### 20. Load test dataset
```python
# Load the test split of the PneumoniaMNIST dataset
test_dataset = PneumoniaMNIST(split='test', download=True)

# tells us how many test images are in the dataset.
print("Number of test images:", len(test_dataset))
```
#### 21. Convert test data to NumPy arrays
```python
# Convert test images to NumPy arrays and normalize pixel values to [0, 1]
x_test = np.array([np.array(img, dtype=np.float32) / 255.0 for img, label in test_dataset])

# Extract the labels for the test images
y_test = np.array([label[0] for img, label in test_dataset])

# Add a channel dimension so the image shape becomes (28, 28, 1)
x_test = x_test[..., np.newaxis]

# Print the shapes of the test images and labels
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)
```
#### 22. Evaluate the saved model
```python
# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(x_test, y_test)

# Print the test results
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
```
23. Predict one image (the 1st image in the test dataset)
- You can change the index number to choose another image
- e.g., index = 2, index = 20
```python
# Select the first test image
index = 0
sample_image = x_test[index]
true_label = y_test[index]

# Display the test image
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.title("Chest X-ray Test Image")
plt.axis('off')
plt.show()
```
#### 24. Predict one image
```python
# Add one more dimension because the model expects a batch
sample_input = np.expand_dims(sample_image, axis=0)

# Predict the probability
prediction = model.predict(sample_input)[0][0]

# Convert probability to class
predicted_label = 1 if prediction >= 0.5 else 0

# Print prediction results
print("Prediction score:", prediction)
print("Predicted label:", predicted_label)
print("True label:", true_label)
```
#### 25. Show the result as text
```python
# Convert numeric labels to class names
label_names = ["Normal", "Pneumonia"]

print("Predicted class:", label_names[predicted_label])
print("True class:", label_names[true_label])
```
#### 26. Try several images
```python
# Show predictions for the first 5 test images
label_names = ["Normal", "Pneumonia"]

for i in range(10):
    image = x_test[i]
    true_label = y_test[i]

    input_image = np.expand_dims(image, axis=0)
    prediction = model.predict(input_image, verbose=0)[0][0]
    predicted_label = 1 if prediction >= 0.5 else 0

    plt.figure(figsize=(3,3))
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Pred: {label_names[predicted_label]} / True: {label_names[true_label]}")
    plt.axis('off')
    plt.show()
```

---

## Final Workflow Learned by Students

By the end of the workshop, students completed the following AI project workflow:

1. Open and use Google Colab
2. Install and import the MedMNIST package
3. Load the PneumoniaMNIST chest X-ray dataset
4. Inspect the number of images and labels
5. Display normal and pneumonia X-ray examples
6. Build a CNN using TensorFlow/Keras
7. Compile the model with an optimizer, loss function, and accuracy metric
8. Convert image data into NumPy arrays
9. Normalize image pixels to `[0, 1]`
10. Train the CNN model
11. Visualize loss and accuracy
12. Save the trained model
13. Load the saved model
14. Evaluate the model on test data
15. Predict whether unseen X-ray images show normal or pneumonia cases
16. Interpret prediction confidence
17. Discuss the role and limits of AI in medical imaging

---

## Instructor Role

**Danu Kim**  
Medical Club President  
Korea International School, Jeju Campus

Responsibilities:

- Designed the 4-week workshop curriculum
- Prepared weekly lecture notes and Colab notebooks
- Tutored students during Medical Club sessions
- Explained AI concepts in beginner-friendly language
- Guided students through coding, training, and inference
- Connected computer vision concepts to real-world medical applications

---

## Educational Purpose and Disclaimer

This project was created as a student-led educational workshop to introduce AI in medicine. Although it uses chest X-ray data and pneumonia classification as the central example, the workshop model is not a medical device and should not be used for diagnosis or treatment decisions.

AI can support pattern recognition and decision-making, but clinical interpretation must be performed by qualified medical professionals.
