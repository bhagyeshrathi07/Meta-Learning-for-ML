# 🤖 AutoML Model Comparator & Code Generator

> **A Modern, Full-Stack AutoML Platform for Classification & Regression.** 
> *Featuring Python Code Generation, Glassmorphism UI, and Resource Profiling.*

This application allows users to upload any dataset (CSV), automatically detects the task type (**Classification** or **Regression**) and trains multiple state-of-the-art machine learning models in parallel.

It goes beyond simple metrics by providing a **Resource-Aware Leaderboard** (CPU/RAM efficiency), **Interactive Visualizations** (ROC Curves/Confusion Matrices/Scatter Plots), and a unique **Code Generation Engine** that writes clean, reproducible Python scripts for your winning model.

* **GitHub Repository:** [https://github.com/bhagyeshrathi07/AutoML](https://github.com/bhagyeshrathi07/AutoML)
---

## 🌟 Key Features

### 🧠 Dual-Mode Machine Learning
* **🎯 Classification:** Logistic Regression, Random Forest, SVM, KNN, XGBoost, Decision Tree
    * *Metrics:* Accuracy, F1 Score, Precision, Recall, ROC-AUC
    * *Visuals:* Interactive ROC Curves & Confusion Matrices
* **📈 Regression:** Linear Regression, Random Forest, XGBoost, Decision Tree, SVM (SVR/SGD)
    * *Metrics:* R² Score, RMSE, MAE
    * *Visuals:* Actual vs. Predicted Scatter Plots with Perfect Fit lines

### 🎯 Intelligent Task Detection
* **Automatic Classification/Regression Detection:** Analyzes your target column and automatically determines whether your problem is classification or regression
* **Smart Model Recommendations:** Automatically selects the most appropriate models for your detected task type
* **Manual Override:** Users can override automatic detection if needed

### ⚡ Performance & Optimization
* **Expanded Hyperparameter Tuning:** Uses `RandomizedSearchCV` with an expanded search space (e.g., `gamma` for SVM, `subsample` for XGBoost, `min_samples_leaf` for Trees).
* **Smart Model Switching:** Automatically switches from computationally expensive models (SVM/SVR) to optimized equivalents (SGD) when dataset rows exceed 2,000.
* **Hardware Profiling:** A custom context manager tracks **Peak RAM (MB)** and **CPU Usage (%)** for every specific model training run.
* **Stratified Sampling:** Maintains class distribution when downsampling large classification datasets

### 🎨 Modern UX & "Glassmorphism" UI
* **Instant Data Stats:** Uses an intelligent "Chunk Reader" to estimate row counts for large files (1GB+) in milliseconds without freezing the browser.
* **Aesthetic Design:** Features a custom CSS **Glassmorphism** interface, animated gradients, and a polished Dark/Light mode.
* **Sortable Leaderboard:** Rank models by Accuracy/R2, but also by Training Time or RAM efficiency.
* **Real-time Progress:** Live progress bar and logs during model training

### 📜 Reproducibility
* **Download Model:** Export the serialized `.pkl` file for immediate deployment.
* **Generate Script (New):** Click one button to generate a clean, standalone `train_model.py` script pre-filled with the **exact hyperparameters** of the winning model.
* **Full Pipeline Code:** Generated scripts include complete preprocessing, training, and evaluation code

---

## 🏗 Architecture

```mermaid
graph TD
    Client[React Frontend] <-->|Multipart File / JSON| API[Flask API]
    API --> Pipeline[ML Pipeline]
    
    subgraph Backend
        Pipeline --> TypeDetect["Task Detection (Class vs Reg)"]
        TypeDetect --> Cleaner["Smart Cleaner & Sampling"]
        Cleaner --> Pre["Preprocessing & Scaling"]
        Pre --> Tuning[Hyperparameter Search]
        Tuning --> Monitor[Resource Monitor Thread]
        Monitor --> Models["XGBoost / RF / SVM / etc"]
        
        Models --> Results["Metrics & Charts"]
        Results --> CodeGen[Code Generator Engine]
    end
    
    Results --> Leaderboard[JSON Response]
    CodeGen --> Script["Download model(.pkl) / .py Script"]
```

---

## 🛠 Tech Stack

### **Backend (Python)**
* **Flask:** REST API & Background Threading
* **Scikit-Learn:** Models, Pipelines, Imputation, and Metrics
* **XGBoost:** Optimized Gradient Boosting
* **Joblib:** Model serialization
* **Psutil:** Real-time hardware resource tracking
* **Pandas & Numpy:** Data processing and vectorization

### **Frontend (React)**
* **React 18:** Hooks-based UI architecture
* **Custom CSS:** Glassmorphism, gradients, and animations (No standard Bootstrap theme)
* **Recharts:** Responsive visualizations (ROC & Scatter)
* **PapaParse:** Worker-based CSV parsing for large files
* **Axios:** Async polling and file uploads

---

## 🚀 Installation & Setup

### 1. Prerequisites
* Python 3.8+
* Node.js 14+ & npm
* *(macOS Users only)*: `brew install libomp` (Required for XGBoost)

### 2. Backend Setup
Navigate to the backend folder and set up the Python environment.

```bash
cd automl-app/backend

# Create Virtual Environment
python3 -m venv .venv

# Activate Environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup
Open a new terminal, navigate to the frontend folder, and install Node modules.

```bash
cd automl-app/frontend

# Install dependencies
npm install
```

---

## ▶️ Usage Guide

### 1. Start the Server
In your **Backend** terminal:
```bash
python app.py
```
*Server runs on `http://127.0.0.1:5000` with threaded task execution.*

### 2. Launch the UI
In your **Frontend** terminal:
```bash
npm start
```
*App opens at `http://localhost:3000`*

### 3. Run the Pipeline
1.  **Upload:** Drop your CSV file. The app instantly estimates rows/cols
2.  **Select Target:** Choose your prediction target from the dropdown
3.  **Task Detection:** The app auto-suggests Classification or Regression (you can override this)
4.  **Model Selection:** Review auto-selected models or manually adjust the selection
5.  **Launch:** Click "🚀 Launch Experiment" and watch the real-time logs as models train in parallel

### 4. Analyze, Export, Reproduce
1.  **Sort:** Click "RAM" or "Time" headers to find efficient models, or "Accuracy/R2" for best performance.
2.  **Visualize:** See the Confusion Matrix (Classification) or Actual vs. Predicted scatter (Regression).
3.  **Get Code:** Click **"📜 Generate Python Script"** to download a `.py` file that reproduces that exact model.

---

## 🔐 API Key Authentication

The application is protected by API key authentication to prevent unauthorized access.

### Setup
1. Create a `.env` file in the `backend/` folder:
```bash
cd automl-app/backend
echo "AUTOML_API_KEY=your-secret-key-here" > .env
```

2. Replace `your-secret-key-here` with a strong, unique key.

3. Users must enter this API key in the login screen to access the application.

> **Note:** The `.env` file is excluded from git. Never commit your API key.

---

## 🐳 Docker Deployment

Deploy the application as a single Docker container where Flask serves both the API and the React frontend.

### Prerequisites
* Docker installed on your server

### Build the Image
```bash
cd automl-app
docker build -t automl .
```

### Run the Container
```bash
docker run -d \
  -p 5000:5000 \
  -e AUTOML_API_KEY=your-secret-key-here \
  --name automl \
  automl
```

The application will be available at `http://your-server-ip:5000`

### Useful Commands
```bash
docker logs automl       # View application logs
docker stop automl       # Stop the container
docker start automl      # Start the container
docker rm automl         # Remove the container
```

---

## 📂 Project Structure

```text
automl-app/
├── backend/
│   ├── app.py               # API Routes (Upload, Status, Download)
│   ├── pipeline.py          # ML Logic, Sampling, Stratification
│   ├── codegen.py           # Python Script Generator Logic
│   ├── monitor.py           # Resource (CPU/RAM) Tracker
│   ├── models/              # Temp storage for .pkl files
│   └── uploads/             # Temp storage for CSVs
│
├── frontend/
│   ├── src/
│   │   ├── App.js           # Main Logic, Polling, & Glass Layout
│   │   ├── App.css          # Glassmorphism & Animation Styles
│   │   ├── Leaderboard.js   # Sorting Logic & Results Table
│   │   ├── RegressionChart.js # Scatter Plot
│   │   ├── ROCChart.js      # ROC Visualization
│   │   └── ConfusionMatrix.js 
│   └── public/              # Static Assets
│
└── README.md                # Documentation
```

---

## 🧠 Advanced Logic Explained

### ⚡ Smart Downsampling & Stratification
* **Classification:** For datasets >100,000 rows, the app performs **Stratified Sampling** to maintain class ratios (e.g., preserving minority class representation)
* **Regression:** Uses Random Sampling (Stratification is not applicable for continuous targets).
* **Large Data Handling:** If rows > 2,000, the pipeline swaps slow `SVC/SVR` for `SGDClassifier/SGDRegressor` to prevent server timeouts while maintaining accuracy.

### 📉 Imbalance Handling
* **Class Weights:** Automatically calculates `class_weight='balanced'` for most of the models.
* **XGBoost:** Dynamically calculates `scale_pos_weight` (Negative/Positive ratio) to boost minority class detection.

### 🖥️ Real-Time Resource Monitoring
The `ResourceMonitor` runs in a separate thread context, sampling memory and CPU usage every 0.2 seconds. This provides accurate metrics showing exactly how much RAM and CPU each model architecture consumed during training, helping users choose production-friendly models.

### 🔧 Hyperparameter Optimization
* **RandomizedSearchCV:** Explores hyperparameter space efficiently
* **Expanded Search Spaces:** 
  - SVM: C, kernel, gamma parameters
  - Trees: max_depth, min_samples_split, min_samples_leaf
  - XGBoost: learning_rate, n_estimators, subsample, colsample_bytree
* **Adaptive Tuning:** Adjusts n_iter based on search space size

---

## 📊 Supported Models

### Classification Models
1. **Logistic Regression** - Fast linear baseline with L1/L2 regularization
2. **Random Forest** - Ensemble learning with class balancing
3. **Support Vector Machine (SVM)** - Kernel-based classification (auto-switches to SGD for large datasets)
4. **K-Nearest Neighbors (KNN)** - Instance-based learning
5. **XGBoost** - Gradient boosting with scale_pos_weight optimization
6. **Decision Tree** - Interpretable tree-based classifier

### Regression Models
1. **Linear Regression** - Simple linear baseline
2. **Random Forest** - Ensemble regression with variance reduction
3. **Support Vector Regression (SVR)** - Kernel-based regression (auto-switches to SGD for large datasets)
4. **XGBoost** - Gradient boosting for regression
5. **Decision Tree** - Interpretable tree-based regressor

---

## 🎨 UI Features

### Dark/Light Mode
Toggle between themes with persistent localStorage saving

### Glassmorphism Design
* Frosted glass effect cards
* Gradient backgrounds
* Smooth animations and transitions
* Responsive design for all screen sizes

### Real-Time Feedback
* Progress bar with percentage
* Live training logs with monospace font
* Emoji-enhanced status messages
* Animated loading states

---

## 📈 Metrics & Visualizations

### Classification Metrics
* **Accuracy:** Overall correctness
* **F1 Score:** Harmonic mean of precision and recall
* **Precision:** True positive rate
* **Recall:** Sensitivity measure
* **ROC-AUC:** Area under receiver operating characteristic curve
* **Confusion Matrix:** Visual breakdown of predictions

### Regression Metrics
* **R² Score:** Coefficient of determination
* **RMSE:** Root mean squared error
* **MAE:** Mean absolute error
* **Scatter Plot:** Actual vs. predicted with perfect fit line
---

## 🔮 Future Roadmap

* [ ] **Feature Engineering:** Automated interaction terms and polynomial features.
* [ ] **Ensemble Builder:** Option to stack the top 3 models into a Voting Classifier.
* [ ] **Deployment API:** One-click generation of a `predict.py` Flask wrapper.
* [ ] **Ensemble Builder:** One-click stacking of top 3 models into voting classifier/regressor
* [ ] **Neural Networks:** Add TensorFlow/PyTorch models for deep learning
* [ ] **Time Series Support:** Add ARIMA, Prophet, and LSTM models
* [ ] **Cloud Integration:** Direct deployment to AWS/GCP/Azure
* [ ] **Explainability:** SHAP values and feature importance visualization
* [ ] **Automated Feature Selection:** Remove redundant/irrelevant features
* [ ] **Multi-target Support:** Handle multiple target columns simultaneously

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request to the [GitHub repository](https://github.com/bhagyeshrathi07/AutoML).


## 📄 License

This project is open source and available under the MIT License.

## 🔗 Links

* **GitHub Repository:** [https://github.com/bhagyeshrathi07/AutoML](https://github.com/bhagyeshrathi07/AutoML)
* **Issues & Feature Requests:** [GitHub Issues](https://github.com/bhagyeshrathi07/AutoML/issues)

---

## 👥 Authors

Built by the AutoML Team:
 - Bhagyesh Rathi
 - Hriday Ampavatina
 - Akshay Kumar
 - Ronak Patel
