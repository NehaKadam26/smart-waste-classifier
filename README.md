# ♻️ Smart Waste Classifier

A machine learning web app that classifies waste into 12 categories and provides disposal instructions. Built using transfer learning on MobileNetV2 and deployed with Streamlit.

🌐 **Live Demo:** [smart-waste-classifier-26.streamlit.app](https://smart-waste-classifier-26.streamlit.app)

A lot of waste ends up in the wrong bin simply because people don't know where it goes. Upload a photo of any waste item and the model tells you its category, the correct bin, and how to prepare it for disposal. Trained on 12 waste categories using MobileNetV2 with transfer learning, achieving 90.42% validation accuracy.
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

The base MobileNetV2 model was pretrained on ImageNet and used as a frozen feature extractor. A custom classification head was added on top and trained on the waste dataset. Class weights were used during training to handle the imbalance across categories.

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

