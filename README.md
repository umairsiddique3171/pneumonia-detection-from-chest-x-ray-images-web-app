## pneumonia_detection_from_chest_x-ray_images_web_app_using_opencv_sklearn_and_streamlit

This project aims to develop a web application for pneumonia detection from chest X-ray images using OpenCV, scikit-learn, and Streamlit.

## App Preview


## Introduction

Pneumonia is a prevalent and potentially life-threatening condition that requires timely diagnosis and treatment. Chest X-ray imaging is a common method for diagnosing pneumonia. This project leverages machine learning techniques to automate the process of pneumonia detection from chest X-ray images. Most probable classification criteria is that patients with pneumonia have foggy images of exposing lungs unclearly due to bacteria clogging where as normal patients have x-rays images with organs clearly visible. Medical Specialists also mostly use this criteria for classification.

## Pneumonia Classifier
Training pneumonia classifier involved training different Machine Learning models using GridSearchCV for pneumonia detection from chest x-ray images. Several Preprocessing techniques were applied to the data such as Image Resizing (224,224), Gray Scale Conversion, GaussainBlur and CLAHE (Contrast-Limited Adaptive Histogram Equalization). You can access the classifier source code [here](https://github.com/umairsiddique3171/Machine-Learning-Projects/tree/main/pneumonia_detection_from_chest_x-ray_images).

## App Features

- Web-based interface for uploading chest X-ray images.
- Utilizes a trained machine learning model (Random Forest) for pneumonia detection.
- Provides confidence scores for the detected pneumonia cases.
- Interactive and user-friendly interface using Streamlit.

## Technologies Used

- Python
- OpenCV
- scikit-learn
- Streamlit

## Project Structure

The project is structured as follows:

- `app.py`: Main application script containing Streamlit UI and image processing logic.
- `utils.py`: Utility functions for image processing, model loading, and background setting.
- `classifier_training`: Directory containing pneumonia classifier source code and trained models.
- `background_img.jpg`: Background image used in the web interface.

## Usage

To run the application locally, follow these steps:

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd pneumonia_detection_web_app
   ```
2. **Create and activate a virtual environment:**
   ```
   python -m venv env
   .\env\Scripts\activate
   ```
3. **Install the required dependencies:**

   ```
   pip install -r requirements.txt
   ```
4. **Run the application:**
   ```
   streamlit run app.py
   ```
5. **Open a web browser and navigate to the provided URL.**

## Acknowledgements
Dataset for training pneumonia classifier was available on kaggle, which can be found [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download).

## License

This project is licensed under the [MIT License](https://github.com/umairsiddique3171/pneumonia_detection_from_chest_x-ray_images_web_app_using_opencv_sklearn_and_streamlit/blob/main/LICENSE).
