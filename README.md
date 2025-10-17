# Machine Learning Projects in Python

A comprehensive collection of machine learning projects covering classification, regression, clustering, dimensionality reduction, and model optimization techniques. Each project includes complete implementations with detailed notebooks demonstrating end-to-end machine learning workflows.

## 📚 Table of Contents

- [Classification Projects](#classification-projects)
- [Regression Projects](#regression-projects)
- [Clustering Projects](#clustering-projects)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Model Optimization](#model-optimization)
- [Data Analysis and Preprocessing](#data-analysis-and-preprocessing)
- [Text Classification](#text-classification)
- [Blog Tutorials](#blog-tutorials)

---

## Classification Projects

### Support Vector Machines (SVM)

**Cancer Cell Prediction** | `Classification/cancer-prediction/`

- Binary classification of cell samples as benign or malignant
- Implements SVM with RBF and linear kernels
- Demonstrates handling non-numerical data and label encoding
- Evaluation using confusion matrix, F1-score, and Jaccard index
- **Dataset**: Cell samples with morphological features

**Breast Cancer Prediction** | `Classification/evaluation-techniques/`

- Comprehensive evaluation techniques for classification models
- Uses scikit-learn's built-in breast cancer dataset
- Compares multiple evaluation metrics and visualization techniques

### Logistic Regression

**Customer Churn Prediction** | `Classification/customer-churn-prediction/`

- Predicts customer churn in telecommunications
- Feature scaling and regularization techniques
- Coefficient interpretation for business insights
- Handles class imbalance with appropriate metrics
- **Dataset**: ChurnData.csv

**Softmax (Multinomial Logistic) Regression** | `Classification/Softmax(Multinomial Logistic) Regression/`

- Multi-class classification using softmax regression
- Compares One-vs-Rest and One-vs-One strategies
- Decision boundary visualization
- **Dataset**: Iris dataset

### Decision Trees

**Patient Drug Prediction** | `Classification/drug-prediction/`

- Predicts appropriate medication based on patient characteristics
- Label encoding for categorical features
- Decision tree visualization and interpretation
- Achieves 98.33% accuracy
- **Dataset**: drug200.csv

### K-Nearest Neighbors (KNN)

**Customer Service Tier Prediction** | `Classification/service-tier-prediction/`

- Classifies customers into service tier categories
- Demonstrates distance metrics and K value selection
- **Dataset**: teleCust1000t.csv

### Ensemble Methods

**Random Forest Evaluation** | `Classification/california-housing/`

- House price prediction using Random Forest regression
- Feature importance analysis
- Residual analysis and model diagnostics
- Comprehensive evaluation metrics (MAE, MSE, RMSE, R²)
- **Dataset**: California Housing dataset

**Credit Card Fraud Detection** | `Classification/fraud-detection/`

- Handles severely imbalanced datasets (99.8% vs 0.2%)
- Implements Decision Trees and SVM
- Sample weighting and stratified splitting
- ROC-AUC evaluation for imbalanced data
- **Dataset**: creditcard.csv (284,807 transactions)

### Multi-Class Classification

**Obesity Risk Prediction** | `Classification/obesity-risk-prediction/`

- Multi-class classification for obesity risk levels
- Feature engineering and preprocessing
- **Dataset**: Obesity_level_prediction_dataset.csv

---

## Regression Projects

### Linear Regression

**CO2 Emission Prediction** | `Simple and Mutiple Linear Regression/co2-emission/`

- Simple and multiple linear regression comparison
- Predicts vehicle CO2 emissions from engine characteristics
- Feature relationship analysis and coefficient interpretation
- Demonstrates improvement from single to multiple features
- **Dataset**: FuelConsumptionCo2.csv

**Housing Price Prediction** | `Simple and Mutiple Linear Regression/housing_prediction/`

- California housing price prediction
- Multiple feature regression analysis
- Model evaluation and residual analysis
- **Dataset**: housing.csv

**Real Estate Price Prediction** | `Simple and Mutiple Linear Regression/real-estate-price-prediction/`

- Real estate market analysis and price prediction
- Feature engineering for property characteristics
- **Dataset**: real_estate_data.csv

**Taxi Tip Prediction** | `Simple and Mutiple Linear Regression/taxi-tip-prediction/`

- Predicts taxi tip amounts from trip characteristics
- Large-scale dataset analysis
- **Dataset**: yellow_tripdata_2019-06.csv

### Regularization

**Regularization Techniques** | `Simple and Mutiple Linear Regression/regularization/`

- Ridge, Lasso, and Elastic Net regularization
- Prevents overfitting in linear models
- Demonstrates bias-variance tradeoff

---

## Clustering Projects

### K-Means Clustering

**Customer Segmentation** | `Clustering/customer-segmentation/`

- Unsupervised customer grouping for marketing
- K-Means++ initialization strategy
- Feature normalization and cluster interpretation
- Business applications and insights
- **Dataset**: Cust_Segmentation.csv

**Evaluating K-Means** | `Clustering/evaluating-kmeans/`

- Optimal cluster number selection
- Elbow method, Silhouette score, and Davies-Bouldin index
- Comprehensive cluster quality assessment

### Density-Based Clustering

**DBSCAN vs HDBSCAN** | `Clustering/DBSCAN-HDBSCAN/`

- Compares density-based clustering algorithms
- Geospatial clustering of museum locations across Canada
- Handles clusters of arbitrary shapes and identifies outliers
- Coordinate system transformations and basemap overlays
- **Dataset**: ODCAF_v1.0.csv (Canadian cultural facilities)

---

## Dimensionality Reduction

### Principal Component Analysis (PCA)

**PCA Implementation** | `Dimensionalty Reduction Algorithms/PCA/`

- Reduces high-dimensional data while preserving variance
- Explained variance analysis and component interpretation
- Visualizing Iris dataset in 2D
- Both conceptual understanding and practical application

### Non-Linear Methods

**t-SNE and UMAP** | `Dimensionalty Reduction Algorithms/t-SNE and UMAP/`

- Advanced visualization techniques for high-dimensional data
- Compares t-SNE and UMAP performance
- Parameter tuning (perplexity, min_dist)
- Understanding local vs global structure preservation

---

## Model Optimization

### Hyperparameter Tuning

**GridSearchCV** | `GridSearchCV/`

- Systematic hyperparameter optimization
- Cross-validation for robust evaluation
- Parallelization for efficiency
- SVM hyperparameter tuning example (C, gamma, kernel)
- **Dataset**: Iris dataset

### Machine Learning Pipelines

**Pipeline with GridSearchCV** | `Machine Learning Pipeline/`

- Prevents data leakage with proper workflows
- Combines preprocessing and modeling
- Stratified K-Fold cross-validation
- End-to-end workflow automation
- **Dataset**: Iris dataset

---

## Data Analysis and Preprocessing

### Exploratory Data Analysis

**Adult Census Income Analysis** | `Preprocessing and EDA/`

- Comprehensive EDA workflow demonstration
- Missing value detection and handling strategies
- Statistical analysis and visualization techniques
- Cross-tabulation and relationship analysis
- Feature encoding (one-hot encoding)
- Normalization and standardization
- Correlation analysis and feature selection
- **Dataset**: adult-modified-09-13-2025.csv (32,561 samples)

---

## Text Classification

### Document Classification

**Newsgroup Classification with KNN and Rocchio** | `Document Classification/`

- Custom KNN implementation from scratch
- Euclidean vs Cosine distance for text data
- TF-IDF weighting for improved accuracy
- Rocchio (nearest centroid) method
- Comparison of custom vs scikit-learn implementations
- Multiple classifier comparison (KNN, Decision Trees, Naive Bayes, LDA)
- **Dataset**: Newsgroups (800 training, 200 test documents, 5,500 features)

**Census Data Classification** | Part of `Document Classification/`

- Predictive modeling on census data
- Feature normalization and scaling
- Hyperparameter tuning for KNN
- Overfitting analysis
- Comparison of multiple algorithms

---

## Blog Tutorials

Detailed tutorial blog posts are available in the [blogs](https://joneshshrestha.com/blog/):

1. **Cancer Prediction with SVM** - Understanding kernel functions and medical ML
2. **Customer Churn with Logistic Regression** - Regularization and business applications
3. **CO2 Prediction with Linear Regression** - Simple to multiple regression
4. **Customer Segmentation with K-Means** - Unsupervised learning for business
5. **Drug Prediction with Decision Trees** - Interpretable healthcare ML
6. **Credit Card Fraud Detection** - Handling imbalanced datasets
7. **DBSCAN vs HDBSCAN** - Density-based clustering for geospatial data
8. **PCA for Dimensionality Reduction** - Understanding variance and components
9. **Hyperparameter Tuning with GridSearchCV** - Systematic optimization
10. **Machine Learning Pipelines** - Professional workflow automation
11. **EDA and Preprocessing** - Comprehensive data preparation guide
12. **Document Classification** - Text classification with KNN and Rocchio
13. **Softmax Regression** - Multi-class classification strategies
14. **t-SNE and UMAP** - Visualizing high-dimensional data
15. **Random Forest Evaluation** - Regression model diagnostics

---

## Technologies Used

- **Python 3.x**
- **scikit-learn** - Core ML library
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **seaborn** - Statistical visualization
- **plotly** - Interactive visualizations
- **UMAP** - Dimensionality reduction
- **HDBSCAN** - Hierarchical density-based clustering
- **geopandas** - Geospatial data analysis

---

## Project Structure

```
machine-learning-python/
├── Classification/           # Classification projects
├── Clustering/              # Clustering projects
├── Simple and Multiple Linear Regression/  # Regression projects
├── Dimensionalty Reduction Algorithms/     # PCA, t-SNE, UMAP
├── Document Classification/  # Text classification
├── GridSearchCV/            # Hyperparameter tuning
├── Machine Learning Pipeline/  # Pipeline implementations
├── Preprocessing and EDA/   # Data analysis workflows
├── blogs/                   # Tutorial blog posts
└── Cheetsheet/             # Reference materials
```

---

## Key Learning Outcomes

### Classification

- Binary and multi-class classification
- Kernel methods (SVM)
- Ensemble methods (Random Forest)
- Handling imbalanced datasets
- Model evaluation metrics

### Regression

- Simple and multiple linear regression
- Regularization techniques
- Feature engineering
- Residual analysis
- Model diagnostics

### Clustering

- Partitioning methods (K-Means)
- Density-based methods (DBSCAN, HDBSCAN)
- Cluster evaluation
- Unsupervised learning applications

### Dimensionality Reduction

- Linear methods (PCA)
- Non-linear methods (t-SNE, UMAP)
- Visualization techniques
- Feature extraction

### Best Practices

- Data preprocessing and cleaning
- Train/test splitting strategies
- Cross-validation techniques
- Preventing data leakage
- Hyperparameter optimization
- Model interpretation

---

## Getting Started

Each project folder contains:

- Jupyter notebooks with complete implementations
- Datasets (or links to datasets)
- Detailed comments and explanations
- Visualizations and results

To run any project:

```bash
# Clone the repository
git clone https://github.com/yourusername/machine-learning-python.git

# Navigate to a project folder
cd machine-learning-python/Classification/cancer-prediction

# Open the notebook
jupyter notebook cancer_cell_prediction.ipynb
```

---

## Contributing

These projects represent hands-on learning from various machine learning concepts. Each notebook includes:

- Step-by-step implementation
- Explanation of concepts and techniques
- Insights from actual experimentation
- Best practices and common pitfalls

---

## License

This project is licensed under the terms included in the LICENSE file.

---

## Acknowledgments

Projects developed through practical machine learning experience, combining academic learning with real-world problem-solving. Special focus on understanding not just how algorithms work, but when and why to use them.
