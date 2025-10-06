# 🧠 MoR-MBTI-PREDICTION  
**Multimodal Personality Prediction System based on Mixture-of-Recursions (MoR) Architecture)**

---

## 🌟 Overview
This project is a **multimodal MBTI personality prediction system** integrating:
- **Visual emotion recognition** from facial expressions, and  
- **Text-based personality inference** using a **Mixture-of-Recursions (MoR)** model.  

The system analyzes **video frames**, **speech-transcribed text**, and **typed responses** to infer a user’s MBTI type in real-time through a Flask web interface.

---

## 🧩 Key Features
- 🎥 Emotion Recognition via CBAM-Enhanced CNN (EfficientNet-B3)
- 💬 MoR-based Text Personality Inference (Transformer alternative)
- 🔄 Multimodal Fusion combining emotional and linguistic features
- 🧾 Automatic result storage (`~/MBTI_Results`)
- 🌐 Flask REST API for text, speech, and image analysis

---

## ⚙️ Architecture

### Overall Pipeline
```
[Web Camera] → [Face Detection (MediaPipe)] → [Emotion CNN (CBAM + EfficientNet)]  
[User Text/Speech] → [Text Preprocessing] → [MoR Language Model]  
              ↓  
       [Feature Fusion + Rule-based Weighting]  
              ↓  
           [MBTI Type + Confidence Scores]
```

### Model Highlights
| Component | Description |
|------------|-------------|
| **MoR Text Model** | Adaptive recursion depth per token for efficient reasoning |
| **Emotion Model** | EfficientNet-B3 + CBAM for attention-enhanced emotion recognition |
| **Integration** | Weighted fusion of emotion and MBTI text probabilities |

---

## 🧪 API Endpoints
| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/api/process_frame` | POST | Analyze base64 image for emotion |
| `/api/process_text` | POST | Analyze text for MBTI |
| `/api/process_speech` | POST | Analyze transcribed speech |
| `/api/final_results` | POST | Fuse multimodal results and return final MBTI |
| `/test` | GET | Server health check |

---

## 🛠 Installation
```bash
git clone https://github.com/HuaTNA/MoR_MBTI_PREDICTION.git
cd MoR_MBTI_PREDICTION
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Place models in:
```
models/
 ├── emotion/improved_emotion_model.pth
 └── text/ml/model.pkl, vectorizer.pkl, label_encoder.pkl, config.json
```

Run the app:
```bash
python app.py
```
Access at **http://localhost:5000/**

---

## 📊 Example Output
```json
{
  "mbti_type": "ENFP",
  "dimension_scores": {"I/E": [0.35, 0.65], "S/N": [0.4, 0.6], "T/F": [0.25, 0.75], "J/P": [0.45, 0.55]}
}
```

---

## 🧠 Authors
**Hua Tan**  
University of Toronto, Industrial Engineering & AI Minor  
📍 Toronto, ON  

**Kevin Li**  
University of Toronto, Computer Engineering  
📍 Toronto, ON  

---

## 📜 License
MIT License

---

## 🧭 Future Work
- Integrate Vision-MoR backbone  
- Extend to dialogue-level multimodal reasoning  
- Cloud deployment (GCP / AWS)
