\# Online Shoppers Purchasing Intention – Machine Learning Classification



\## 1. Problem Statement



The objective of this project is to develop and compare five machine learning classification algorithms to predict whether an online shopping session will result in a purchase.



The project evaluates Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Support Vector Machine (SVM) using multiple classification performance metrics.



\## 2. Dataset Description



\*\*Dataset:\*\* Online Shoppers Purchasing Intention Dataset  

\*\*Source:\*\* \[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)



\- Total instances: 12,330

\- Input features: 17

\- Target variable: `Revenue`

\- Target type: Binary classification

\- Missing values: None



The dataset contains information about online shopping sessions, including page visits, page values, visitor type, traffic type, operating system, browser, and other session-related attributes.



The target variable `Revenue` indicates whether the shopping session resulted in a purchase.



\## 3. Machine Learning Algorithms



The following five classification algorithms were implemented and evaluated:



1\. Logistic Regression

2\. Decision Tree Classifier

3\. K-Nearest Neighbors (KNN)

4\. Gaussian Naive Bayes

5\. Support Vector Machine (SVM)



\### Data Preprocessing



\- The dataset was divided into training and testing sets using an 80:20 split.

\- Stratified sampling was used to maintain the target class distribution.

\- Numerical features were standardized using `StandardScaler`.

\- Categorical features were encoded using `OneHotEncoder`.

\- Preprocessing and classification were combined using Scikit-learn pipelines.



The experiments used a fixed random state of 42 for reproducibility.



\## 4. Model Performance Comparison



The models were evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).



| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |

|---|---:|---:|---:|---:|---:|---:|

| Decision Tree | 0.8418 | 0.8706 | 0.4935 | 0.7984 | \*\*0.6100\*\* | \*\*0.5412\*\* |

| Support Vector Machine | 0.8560 | 0.8856 | 0.5258 | 0.7199 | 0.6077 | 0.5319 |

| Logistic Regression | 0.8500 | \*\*0.8962\*\* | 0.5107 | 0.7487 | 0.6072 | 0.5330 |

| K-Nearest Neighbors | \*\*0.8747\*\* | 0.7998 | \*\*0.6714\*\* | 0.3743 | 0.4807 | 0.4389 |

| Gaussian Naive Bayes | 0.6736 | 0.7932 | 0.2937 | 0.7880 | 0.4279 | 0.3234 |



\## 5. Model-wise Observations



\### Logistic Regression

\- Achieved the highest AUC of 0.8962.

\- Demonstrated good ability to distinguish between purchasing and non-purchasing sessions.

\- Provided a balanced performance across the evaluated metrics.



\### Decision Tree

\- Achieved the highest F1 Score of 0.6100 and MCC of 0.5412.

\- Recorded the highest recall of 0.7984.

\- Demonstrated strong performance in identifying purchase sessions.



\### K-Nearest Neighbors

\- Achieved the highest accuracy of 0.8747 and precision of 0.6714.

\- However, its recall was comparatively low at 0.3743.

\- It missed a larger proportion of actual purchase sessions.



\### Gaussian Naive Bayes

\- Achieved a recall of 0.7880.

\- Its accuracy, precision, F1 Score, and MCC were comparatively low.

\- It showed weaker overall classification performance among the evaluated models.



\### Support Vector Machine

\- Achieved an accuracy of 0.8560 and an AUC of 0.8856.

\- Recorded an F1 Score of 0.6077.

\- Demonstrated competitive and balanced classification performance.



\## 6. Overall Comparison and Best Model



Based on the experimental results, \*\*Decision Tree\*\* is selected as the overall preferred model because it achieved the highest F1 Score (0.6100) and MCC (0.5412), along with the highest recall (0.7984).



However, the best model can depend on the evaluation objective:



\- Highest Accuracy: K-Nearest Neighbors

\- Highest AUC: Logistic Regression

\- Highest F1 Score: Decision Tree

\- Highest MCC: Decision Tree



Therefore, Decision Tree provides the strongest overall balance based on F1 Score and MCC for this experiment.



\## 7. Streamlit Web Application



A Streamlit application was developed to evaluate the five saved classification models.



Application features:



\- Select one of the five trained models.

\- Upload a test dataset in CSV format.

\- Generate predictions using the selected saved model.

\- Display Accuracy, AUC, Precision, Recall, F1 Score, and MCC.

\- Display a confusion matrix.

\- Display a classification report.



The application loads previously saved models and evaluates uploaded test data. It does not retrain the models.



\*\*Live Application:\*\* \[Add your deployed Streamlit Community Cloud URL here]



\## 8. Repository



\*\*GitHub Repository:\*\* \[Add your public GitHub repository URL here]



The repository contains the Jupyter Notebook, Streamlit application, saved models, test dataset, model comparison results, and dependency file.



\## 9. Project Structure



```text

ML\_Assignment\_2/

│

├── ML\_Assignment\_2.ipynb

├── app.py

├── requirements.txt

├── model\_comparison.csv

├── test\_data.csv

│

└── models/

&#x20;   ├── logistic\_regression.joblib

&#x20;   ├── decision\_tree.joblib

&#x20;   ├── knn.joblib

&#x20;   ├── naive\_bayes.joblib

&#x20;   └── support\_vector\_machine.joblib

```



\## 10. Installation and Execution



\### Install dependencies



```bash

pip install -r requirements.txt

```



\### Run the Streamlit application



```bash

python -m streamlit run app.py

```



The application will open in a browser. Select a model and upload `test\_data.csv` to evaluate its performance.



\## 11. Conclusion



This project demonstrates an end-to-end classification workflow, including dataset exploration, preprocessing, model development, performance evaluation, model persistence, and deployment through Streamlit.



The experimental comparison shows that different algorithms perform differently depending on the evaluation metric. Decision Tree achieved the best F1 Score and MCC, while Logistic Regression achieved the highest AUC and KNN achieved the highest accuracy.



The project provides an interactive application for evaluating all five saved models without requiring model retraining.

