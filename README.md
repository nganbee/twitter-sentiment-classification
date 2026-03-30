# Twitter Sentiment Analysis
## Key Features
* **Model Benchmarking:** Established a robust baseline using traditional Machine Learning (`TF-IDF` + `SVC`) and compared its performance against a Deep Learning architecture (`BiLSTM`) to find the optimal balance between speed and accuracy.
* **Production-Ready API:** The highest-performing model is served via a fast, asynchronous RESTful API using `FastAPI` and `Uvicorn`.
* **Clean Code Architecture:** Modular source code (`src/`) for data preprocessing, training, and inference, strictly separated from exploratory research (`notebooks/`).
* **Integrated Web Interface:** Includes a custom frontend (`templates/`, `static/`) to interact with the API and visualize sentiment predictions in real-time.

## Model Evaluation & Selection

To ensure the best performance for the API, we benchmarked two distinct approaches:

| Model | Feature Extraction | Architecture Type |
| :--- | :--- | :--- |
| **SVC (Baseline)** | TF-IDF | Traditional ML | 
| **BiLSTM** | Word Embeddings | Deep Learning |

## Tech Stack
* **Deep Learning & ML:** TensorFlow/Keras (BiLSTM), Scikit-learn (SVC, TF-IDF), Pandas, NumPy.
* **Backend Framework:** FastAPI, Uvicorn, Pydantic.

## Project Structure

```text
├── data/              # Raw and processed datasets (ignored in Git)
├── models/            # Saved model weights and tokenizers
├── notebooks/         # Jupyter notebooks for EDA and model prototyping
│   ├── 01_EDA.ipynb
│   ├── 02_svcModel.ipynb
│   └── 03_lstmModel.ipynb
├── src/               # Core source code for the ML pipeline
│   ├── BiLSTM/        
│   │   └── train.py   # Training script for BiLSTM model
│   ├── evaluate.py    # Model evaluation metrics
│   ├── inference.py   # Script to load model and run predictions
│   ├── preprocess.py  # Text cleaning and tokenization
│   └── train.py       # Main training script
├── static/            # CSS/JS for the web interface
├── templates/         # HTML templates for the frontend UI
├── main.py            # FastAPI application and endpoint definitions
├── requirements.txt   # Project dependencies
└── README.md
```

## Model Evaluation and Comparision

| Model | Accuracy | Macro F1-Score | Weighted F1-Score
| :--- | :--- | :--- | :--- |
| **SVC (Baseline)** | 92% | 0.92 | 0.92 | 
| **BiLSTM** | 95% | 0.94 | 0.95 |

- Deep Learning Advantage: The BiLSTM model achieves approximately a 3% accuracy improvement over the SVC baseline. In a production environment, this margin is critical for significantly reducing misclassifications when processing large-scale Twitter datasets, which are typically characterized by high noise levels and heavy use of slang.

- Robustness: BiLSTM demonstrates superior stability across all target classes (0, 1, 2). Most notably, it maintains a high F1-score of 0.93 on the most challenging category (Class 2), compared to 0.90 achieved by the SVC model.

- Deployment Strategy:

    - Priority on Speed & Simplicity: SVC is the optimal choice due to its low computational overhead and rapid inference times.

    - Priority on Maximum Accuracy: BiLSTM is the indispensable solution for capturing deep semantic nuances and context within the text.

## How to run
### Option 1: Run locally
1. **Clone the repository**
```bash
git clone https://github.com/nganbee/twitter-sentiment-classification.git
cd twitter-sentiment-classification
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up Environment Variables**  
Create a `.env` file in the root directory and add your API keys (required API in the `.env.example`)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate
```

4. **Run the Demo web**
```bash
uvicorn main:app --reload
```

### Option 2: Run with Docker
1. **Clone the repository**
```bash
git clone https://github.com/nganbee/twitter-sentiment-classification.git
cd twitter-sentiment-classification
```
2. **Build and run with Docker**
```bash
docker-compose up --build
```

3. **Access the web demo**  

- API: http://localhost:8000