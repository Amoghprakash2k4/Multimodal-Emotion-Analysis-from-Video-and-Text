# Multimodal Emotion Analysis Using Machine Learning

## Overview
This project aims to perform emotion classification using a multimodal approach by analyzing video clips from the TV show **Friends**. It incorporates audio, visual, and textual modalities to detect and classify human emotions using machine learning techniques.

## Key Highlights

- **Adaptability**: Rapidly implemented and evaluated multiple machine learning frameworks (including classical ML and deep learning) to identify the most effective emotion classification models.
- **Feedback-Oriented Development**: Actively sought feedback from professors and peers to refine the emotion classification pipeline, demonstrating a commitment to continuous improvement and the adoption of best practices.

## Dataset

- **Source**: Short clips from the TV show *Friends*.
- **Modalities**:
  - **Visual**: Facial expressions and body language.
  - **Audio**: Voice tone, pitch, and speech patterns.
  - **Text**: Subtitles or transcripts derived from the video clips.

## Methodology

1. **Data Preprocessing**:
   - Extracted frames from video clips for visual input.
   - Extracted audio tracks and converted them into spectrograms/features.
   - Transcribed audio to text (using automatic speech recognition).
   
2. **Feature Engineering**:
   - Used pre-trained models for facial emotion detection (e.g., OpenFace, FER+).
   - Extracted MFCCs and spectrogram features for audio.
   - Applied BERT-based embeddings for textual data.

3. **Modeling**:
   - Trained and evaluated separate models for each modality.
   - Combined features for multimodal fusion using ensemble methods and neural networks.

4. **Evaluation**:
   - Accuracy, F1-score, confusion matrix.
   - Cross-validation to ensure model generalizability.

## Tools & Technologies

- **Languages**: Python
- **Libraries**: scikit-learn, TensorFlow/PyTorch, OpenCV, librosa, Hugging Face Transformers
- **Other Tools**: Google Colab, Jupyter Notebook, Matplotlib/Seaborn

## How to Run

1. Clone this repository.
2. Install required dependencies using `requirements.txt`.
3. Place your video clips in the `data/clips/` directory.
4. Run the `main.py` script to start preprocessing and model training.

```bash
pip install -r requirements.txt
python main.py
```

## Future Work

- Incorporate real-time emotion detection.
- Extend the dataset for diverse emotional contexts.
- Fine-tune transformer models with emotion-specific datasets.

## Acknowledgements

Special thanks to professors and peers for insightful feedback throughout the project.

---
