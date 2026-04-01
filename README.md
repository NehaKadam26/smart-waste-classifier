# ♻️ Smart Waste Classifier

> AI-powered waste classification using deep learning — upload an image, get instant disposal guidance.

🌐 **Live Demo:** [smart-waste-classifier-26.streamlit.app](https://smart-waste-classifier-26.streamlit.app)

Smart Waste Classifier is a computer vision web app that identifies 12 categories of household and industrial waste from a single image. Built on MobileNetV2 with transfer learning, it achieves **90.42% validation accuracy** and provides actionable disposal instructions for each category — helping users make the right recycling and waste decisions effortlessly.

The app is designed to be simple: upload a photo, and within seconds you get the waste category, the correct bin, preparation tips, and a confidence score.

---

## 📸 Screenshots

### Upload Page
![Upload](screenshots/01_upload.png)

### Cardboard Detection
![Cardboard](screenshots/02_cardboard.png)

### Glass Detection
![Glass](screenshots/03_glass.png)

### Metal Detection
![Metal](screenshots/04_metal.png)

### Textile Detection
![Textile](screenshots/05_textile.png)

### Battery Detection
![Battery](screenshots/06_battery.png)

---

## 🎯 Model Performance

| Metric | Score |
|---|---|
| Training Accuracy | 93.04% |
| Validation Accuracy | 90.42% |
| Architecture | MobileNetV2 (Transfer Learning) |
| Dataset Classes | 12 |
| Epochs | 10 |
| Input Size | 224 × 224 px |

The model uses MobileNetV2 pretrained on ImageNet as a frozen feature extractor, with a custom classification head trained on the waste dataset. Class weights were applied during training to handle dataset imbalance across categories.

---

## 🗂️ Waste Categories

| Category | Disposal |
|---|---|
| 🔋 Battery | Hazardous Waste Drop-off |
| 🌿 Biological | Compost / Organic Bin |
| 🍺 Brown Glass | Glass Recycling |
| 📦 Cardboard | Paper & Cardboard Bin |
| 👕 Clothes | Textile Recycling |
| 🍾 Green Glass | Glass Recycling |
| 🥫 Metal | Metal Recycling |
| 📄 Paper | Paper & Cardboard Bin |
| 🧴 Plastic | Plastic Recycling |
| 👟 Shoes | Textile / Shoe Recycling |
| 🗑️ Trash | General Waste |
| 🥛 White Glass | Glass Recycling |

---

## 🚀 Run Locally

**Requirements:** Python 3.11
```bash
git clone https://github.com/NehaKadam26/smart-waste-classifier.git
cd smart-waste-classifier
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 🏗️ Project Structure
```
smart-waste-classifier/
├── app/
│   └── app.py                  # Streamlit web app
├── data/
│   ├── train/                  # Training images (12 classes)
│   └── val/                    # Validation images
├── model/
│   ├── model.h5                # Trained MobileNetV2 model
│   └── class_indices.json      # Class name → index mapping
├── notebooks/
│   ├── 01_setup_and_data.ipynb # Data preparation & splitting
│   └── 02_train_model.ipynb    # Model training & evaluation
├── screenshots/                # App screenshots
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | TensorFlow / Keras |
| Architecture | MobileNetV2 (pretrained on ImageNet) |
| Web App | Streamlit |
| Image Processing | Pillow, NumPy |
| Language | Python 3.11 |


