# 🌿 PlantVillage Disease Classification with Transfer Learning

This project leverages **transfer learning** using various state-of-the-art CNN architectures to classify plant diseases from the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease). It supports different **image preprocessing techniques** (Color, Grayscale, and Segmented) and **training/validation splits** (e.g., 70-30, 60-40), providing a comprehensive benchmarking pipeline.

Authors:
 - [Ľubomír Švec](mailto:lubomir.svec@student.tuke.sk)
 - [Martin Saxa](mailto:martin.saxa@student.tuke.sk)

## 🚀 Models Used
The following pre-trained models from `tensorflow.keras.applications` are supported:
- InceptionV3
- InceptionResNetV2
- MobileNetV2
- EfficientNetB0 (not accurate)

These models are fine-tuned using features extracted from the PlantVillage images.

## 🖼️ Image Types Supported
- **Color**: Original RGB images.
- **Grayscale**: Converted to grayscale and back to RGB to retain channel dimensions.
- **Segmented**: Simple HSV-based segmentation to isolate leaf regions.

## 📁 Dataset
- **PlantVillage** from `tensorflow_datasets`.
- Automatically downloaded and preprocessed.
- Configurable splits:
  - `80-20`: 80% train / 20% test
  - `70-30`: 70% train / 30% test
  - `60-40`: 60% train / 40% test

## 🧠 Features
- Dynamic image preprocessing pipeline.
- Transfer learning with frozen convolutional base.
- Automated training, evaluation, and model saving for all combinations.
- Outputs performance (accuracy/loss) to `out.csv`.
- Automatically skips training if a model is already trained and saved.

## 📊 Output
Each model combination is evaluated and logged to `out.csv` in the following format:

| Model           | Image Type | Split   | Test Accuracy | Test Loss |
|----------------|------------|---------|----------------|-----------|
| InceptionV3     | Color      | 70-30   | 0.9876         | 0.0342    |
| MobileNetV2     | Grayscale  | 60-40   | 0.9543         | 0.0921    |
| ...             | ...        | ...     | ...            | ...       |

## 🧪 Running the Code

1. **Install dependencies**:
   ```bash
   pip install tensorflow tensorflow-datasets
   ```

2. **Run the script**:
   ```bash
   python main.py
   ```

> 💡 The script handles GPU memory growth and clears sessions between runs to avoid memory issues.

## 🧼 Leaf Segmentation (Simple HSV)
The `segment_leaf()` function isolates the leaf region by applying an HSV color threshold mask, removing the background.

## 📦 Output Files
- Trained models: `final_model_<Model>_<ImageType>_<Split>.keras`
- Best validation checkpoint: `best_model_<Model>_<ImageType>_<Split>.keras`
- Results CSV: `out.csv`

## 🧹 Memory Management
- Uses `tf.keras.backend.clear_session()` and Python's `gc.collect()` to minimize GPU memory usage and clean up after each model run.
- Implemented due to need to run on not-state-of-art hardware.

## 📍 Known Limitations
- Segmentation is basic and may not generalize well outside the dataset.
- Only sparse categorical cross-entropy is used (assumes integer labels).
- EfficientNetB0 - Not working yet :(

## 🙌 Acknowledgments
- [PlantVillage Dataset](https://www.tensorflow.org/datasets/catalog/plant_village) - It was used for easer manipulation. Original files
are in this repo for reference.
- Researchers and developers of the pre-trained models used.
---
