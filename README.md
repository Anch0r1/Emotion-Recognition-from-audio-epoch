
# Emotion Recognition from Audio

This project aims to train a deep learning model to classify human emotions from .wav audio files. It uses both audio and prosody features to improve emotion recognition accuracy.

## Key Functionalities

* **Data Loading:** Efficient loading and preprocessing of audio data for model training.
* **Model Training:** Training of separate models for audio and prosody features, as well as a fusion model.
* **Fusion of Modalities:** Combining audio and prosody features to enhance emotion recognition.
* **Evaluation:** Evaluating the performance of the trained models.

## Dataset
The dataset used is the [RAVDESS Emotional Speech Audio (Kaggle)](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio?resource=download)
> Note: Dataset files are expected to be placed under `data/archive/`.


## Repository Structure

The repository contains the following main files and directories:

* `emotion_dataset.py`: Defines the dataset for loading audio data and extracting spectrograms.
* `updated_emotion_dataset.py`: Defines the dataset for extracting prosody features (pitch, loudness, and tempo) from audio.
* `fusion_dataset.py`: Defines the dataset for fusing audio and prosody features.
* `model.py`:  Contains the implementation of the Convolutional Neural Network (CNN) model for audio feature extraction.
* `prosodytrain.py`: Contains the implementation and training of the prosody model.
* `fusion_model.py`: Contains the implementation of the model that fuses audio and prosody features.
* `dataloader.py`: Contains code for creating DataLoaders for the CNN model (if separated from `train.py`).
* `train.py`: Script to train the CNN model.
* `eval.py`: Script to evaluate the CNN model's performance.
* `load_prosody.py`: Utility code for loading the prosody model.
* `evalprosody.py`: Script to evaluate the prosody model's performance.
* `feature_extractor_cnn.py`: Utility code for extracting features from the CNN model. 
* `prosody_feature_extractor.py`: Utility code for extracting features from the prosody model. 
* `train_fusion.py`: Script to train the fusion model.
* `evaluate_fusion.py`: Script to evaluate the fusion model's performance.
* `requirements.txt`: Lists the Python dependencies required to run the project.

## Results
<img width="668" height="218" alt="image" src="https://github.com/user-attachments/assets/e8a30b31-fb59-485a-af26-1e9cb90c2335" />

[Click here to view Project report](Project_Report.pdf)

### Fusion Model confusion matrix
<img width="940" height="728" alt="2785086a667d5ace8afb67018263c7c196751b3a" src="https://github.com/user-attachments/assets/13faf039-f68d-4c3b-9bef-0a75dc4ed085" />


## Dependencies

To run this project, you'll need to install the following Python libraries. You can install them using pip:

```bash
pip install -r requirements.txt


