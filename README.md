# MedArchive Solutions - Medical Transcription Classification System

A machine learning system for automated classification of medical transcriptions into 13 specialty categories using TF-IDF vectorization and Softmax Regression, with deployment on Google Cloud Vertex AI.

## 📋 Project Overview

MedArchive Solutions automates the classification of medical transcriptions across 13 medical specialties, reducing manual routing time and improving operational efficiency in medical documentation workflows.

### Supported Medical Specialties

- Cardiovascular / Pulmonary
- ENT - Otolaryngology
- Gastroenterology
- Hematology - Oncology
- Nephrology
- Neurology
- Neurosurgery
- Obstetrics / Gynecology
- Ophthalmology
- Orthopedic
- Pediatrics - Neonatal
- Psychiatry / Psychology
- Radiology

## 🎯 Key Features

- **Automated Classification**: 80% accuracy on validation set
- **Multi-class Classification**: 13 medical specialty categories
- **Production-Ready Pipeline**: End-to-end preprocessing and prediction
- **Cloud Deployment**: Deployed on Google Cloud Vertex AI
- **Comprehensive Analysis**: EDA, classification modeling, and clustering insights

## 📊 Model Performance

### Classification Metrics (Validation Set)

| Metric | Score |
|--------|-------|
| Overall Accuracy | 80.0% |
| Macro F1-Score | 0.763 |
| Weighted F1-Score | 0.802 |
| Cross-validation F1 | 0.742 ± 0.025 |

### Per-Specialty Performance (Top 5)

| Specialty | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Ophthalmology | 1.000 | 1.000 | 1.000 |
| Obstetrics / Gynecology | 0.923 | 1.000 | 0.960 |
| Cardiovascular / Pulmonary | 0.926 | 0.935 | 0.930 |
| Gastroenterology | 0.885 | 0.793 | 0.836 |
| Nephrology | 0.923 | 0.750 | 0.828 |

### Business Impact

- **Daily Processing**: ~1,000 medical documents
- **Estimated Misrouted Documents**: 200/day (20% error rate)
- **Time Savings**: ~80% reduction vs manual classification
- **Automation Rate**: 80% of documents correctly classified

## 🛠️ Technical Architecture

### Machine Learning Pipeline

1. **Data Preprocessing**
   - TF-IDF Vectorization (5,000 features)
   - Unigram and bigram extraction
   - Stop word removal
   - Feature normalization

2. **Classification Model**
   - Algorithm: Softmax Regression (Multinomial Logistic Regression)
   - Regularization: L2 (C=10.0)
   - Class weighting: Balanced
   - Solver: L-BFGS

3. **Unsupervised Clustering**
   - Algorithm: K-Means (k=20)
   - Purpose: Sub-specialty discovery and validation
   - Quality: 9 high-purity clusters identified
   - Automation potential: 28.9% for high-confidence routing

### Technology Stack

- **Python**: 3.x
- **scikit-learn**: 1.3.0 (ML models)
- **pandas**: 2.0.3 (Data manipulation)
- **numpy**: 1.24.3 (Numerical computing)
- **joblib**: 1.3.2 (Model serialization)
- **Google Cloud Vertex AI**: Model deployment

## 📁 Project Structure

```
MedArchive-Solutions-Classification-System/
├── notebooks/
│   ├── 1_eda_and_preprocessing.ipynb    # Exploratory data analysis
│   ├── 2_classification_modeling.ipynb  # Classification model training
│   └── 3_clustering_analysis.ipynb      # Clustering analysis
├── artifacts/
│   ├── unified_pipeline.joblib          # Complete ML pipeline
│   ├── best_softmax_model.joblib        # Trained classifier
│   ├── tfidf_vectorizer.joblib          # TF-IDF vectorizer
│   ├── preprocessed_data.joblib         # Processed datasets
│   ├── model_evaluation_results.joblib  # Evaluation metrics
│   ├── clustering_results.joblib        # Clustering analysis
│   └── requirements.txt                 # Python dependencies
├── test_vertex_endpoint.py              # Vertex AI endpoint testing
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager
- Google Cloud account (for deployment)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MedArchive-Solutions-Classification-System
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r artifacts/requirements.txt
```

### Usage

#### Local Prediction

```python
import joblib

# Load the unified pipeline
pipeline = joblib.load('artifacts/unified_pipeline.joblib')

# Predict on raw text
text = "Patient presents with chest pain and shortness of breath. ECG shows ST elevation."
prediction = pipeline.predict([text])[0]
confidence = pipeline.predict_proba([text])[0].max()

print(f"Predicted Specialty: {prediction}")
print(f"Confidence: {confidence:.3f}")
```

#### Vertex AI Endpoint Testing

```bash
python test_vertex_endpoint.py
```

This script tests the deployed Vertex AI endpoint with sample medical transcriptions across different specialties.

### Training the Model

Follow the notebooks in order:

1. **EDA and Preprocessing** (`1_eda_and_preprocessing.ipynb`)
   - Load dataset from Hugging Face
   - Explore data characteristics
   - Preprocess text with TF-IDF
   - Save preprocessed data

2. **Classification Modeling** (`2_classification_modeling.ipynb`)
   - Train Softmax Regression model
   - Hyperparameter tuning with GridSearchCV
   - Evaluate model performance
   - Create unified pipeline

3. **Clustering Analysis** (`3_clustering_analysis.ipynb`)
   - Discover natural document groupings
   - Identify sub-specialties
   - Validate classification results

## 📈 Dataset

- **Source**: Hugging Face (`hpe-ai/medical-cases-classification-tutorial`)
- **Total Documents**: 2,464
- **Training Set**: 1,724 documents (70%)
- **Validation Set**: 370 documents (15%)
- **Test Set**: 370 documents (15%)
- **Classes**: 13 medical specialties
- **Class Imbalance**: 14.8:1 ratio (Cardiovascular/Radiology)

### Data Characteristics

- **Average Transcription Length**: 503 words (3,319 characters)
- **Word Count Range**: 1 - 2,460 words
- **Features**: Medical terminology, procedures, diagnoses, symptoms
- **Specialty-Specific Keywords**: Strong correlation with medical specialty

## 🔍 Key Findings

### From EDA

1. **Class Imbalance**: Cardiovascular/Pulmonary (742 cases) vs Radiology (50 cases)
2. **Text Variability**: High variability in documentation styles across specialties
3. **Specialty Patterns**: Each specialty has distinct terminology patterns
4. **Feature Importance**: Top features include patient, procedure, anatomical terms

### From Classification

1. **Best Performers**: Ophthalmology, Obstetrics/Gynecology, Cardiovascular
2. **Challenging Specialties**: Neurology, Radiology, Neurosurgery
3. **Common Misclassifications**: Neurology ↔ Neurosurgery, Orthopedic ↔ Neurology
4. **Production Readiness**: Model meets accuracy threshold but needs improvement in 6 specialties

### From Clustering

1. **Optimal Clusters**: 20 natural groupings discovered
2. **High-Purity Clusters**: 9 clusters suitable for automated routing
3. **Sub-Specialties**: 4 potential sub-specialties identified
4. **Automation Potential**: 28.9% of documents for high-confidence routing

## 🚀 Deployment

### Google Cloud Vertex AI

The model is deployed on Google Cloud Vertex AI for production inference:

- **Project ID**: `forge-ml-project`
- **Endpoint ID**: `2909108755490668544`
- **Region**: `us-central1`
- **Model Type**: Custom scikit-learn model
- **Input**: Raw medical transcription text
- **Output**: Predicted medical specialty

### Deployment Steps

1. Upload model artifact to Google Cloud Storage
2. Import model to Vertex AI Model Registry
3. Deploy model to Vertex AI Endpoint
4. Test endpoint with sample predictions

## 📊 Business Value

### Operational Efficiency

- **Manual Processing Time**: ~2 minutes per document
- **Automated Processing Time**: ~0.1 minutes per document
- **Daily Time Savings**: ~1,520 minutes (25+ hours)
- **Monthly Savings**: ~550 hours

### Quality Improvements

- **Consistency**: Standardized classification criteria
- **Accuracy**: 80% correct classification vs human variability
- **Speed**: Real-time classification
- **Scalability**: Handles 1,000+ documents per day

### Cost Reduction

- Reduced manual routing workload by 80%
- Faster document processing and specialty assignment
- Improved resource allocation for medical staff
- Lower operational costs

## ⚠️ Limitations & Future Improvements

### Current Limitations

1. **Class Imbalance**: 6 specialties have suboptimal performance
2. **Rare Specialties**: Limited training data for Radiology, Psychiatry
3. **Cross-Specialty Cases**: Difficulty with overlapping medical domains
4. **Low Silhouette Score**: Clustering shows moderate separation

### Planned Improvements

1. **Data Augmentation**: Collect more examples for rare specialties
2. **Ensemble Methods**: Combine multiple models for better accuracy
3. **Deep Learning**: Explore transformer-based models (BERT, BioBERT)
4. **Active Learning**: Implement human-in-the-loop for edge cases
5. **Multi-label Classification**: Support documents with multiple specialties
6. **Confidence Thresholding**: Route low-confidence predictions to human review

## 📝 Recommendations

### For Production Deployment

1. **Confidence Thresholding**: Implement human review for predictions <70% confidence
2. **Monitoring**: Track model performance continuously in production
3. **Feedback Loop**: Collect misclassification feedback for retraining
4. **A/B Testing**: Compare automated vs manual routing performance
5. **Specialty-Specific Models**: Train specialized models for problematic classes

### For Model Improvement

1. Focus on 6 critical specialties with precision/recall <0.7
2. Balance training data across all specialties
3. Incorporate domain-specific medical ontologies
4. Use ensemble methods (Random Forest, XGBoost, Neural Networks)
5. Retrain quarterly with new medical documentation

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Shyamantha Perera**
- **Uvindu Sanketh**
- **Thisura Liyanage**
- **Sangeeth Sulochana**
- **Rashmika Kavishan**
- **Glen Perera**

## 🙏 Acknowledgments

- Dataset: Hugging Face (`hpe-ai/medical-cases-classification-tutorial`)
- Medical transcriptions from various healthcare providers
- Google Cloud Vertex AI for deployment infrastructure
- scikit-learn community for ML tools

## 📞 Contact

For questions, feedback, or collaboration opportunities, please open an issue in this repository.

---

**Note**: This system is designed for educational and operational efficiency purposes. It should be used as a decision-support tool alongside human medical professionals, not as a replacement for medical expertise.
