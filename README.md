# **Explainable Phishing URL Detection Using Machine Learning and SHAP**


## **1. Project Explanation**

### **Background**

Phishing websites try to trick users into giving away passwords, financial details or other sensitive information. A phishing attack usually starts with a link, so the URL itself is often the first thing a user or a security system sees. Machine learning models can learn patterns in URLs that are linked to phishing. However, many high-performing models act as "black boxes", which makes it hard for analysts to understand or trust their decisions.

### **Research Problem**

Detecting phishing URLs accurately is only part of the problem. Security analysts also need to know **why** a model flags a URL as phishing. This project investigates how well machine learning models can separate phishing URLs from legitimate URLs using only features taken from the URL text. It then uses explainable AI to show which URL characteristics drive those predictions.

### **What the Project Does**

- Loads, inspects and cleans a labelled phishing URL dataset
- Extracts 17 lexical and structural features from each URL
- Explores the features through exploratory data analysis (EDA)
- Trains and compares five classifiers, including a neural network
- Tunes the models and compares baseline and tuned performance
- Explains model predictions with **SHAP (SHapley Additive exPlanations)**
- Checks how much the results depend on the protocol (HTTP or HTTPS)
- Answers the three research questions in a written conclusion

### **Methodology Summary**

1. **Data preparation:** load both CSV files, inspect columns, labels, missing values and duplicates, combine the two files, standardise labels and remove duplicate URLs. This leaves 504,932 unique URLs.
2. **Label encoding:** `legitimate = 0`, `phishing = 1`.
3. **Feature engineering:** extract 17 URL-based indicators, including:
   - URL length
   - counts of dots, hyphens, digits, slashes and special characters
   - HTTPS and HTTP usage
   - hostname, path and query lengths
   - hostname dot count and an approximate subdomain count
4. **Exploratory data analysis:** class balance, feature distributions by class, HTTPS usage and feature correlations.
5. **Baseline modelling:** train Logistic Regression, Decision Tree, Random Forest, XGBoost and a neural network (multi-layer perceptron) on a stratified 80/20 train and test split. Class imbalance is handled with balanced class weights and, for XGBoost, `scale_pos_weight`.
6. **Hyperparameter tuning:** tune every model with 3-fold stratified cross-validation scored on F1. Grid search is used for Logistic Regression and the Decision Tree, and randomised search for the Random Forest, XGBoost and the neural network. The searches run on a stratified sample of 100,000 training URLs, and each best model is then retrained on the full training set.
7. **Evaluation:** compare baseline and tuned models using accuracy, precision, recall, F1-score and ROC-AUC, alongside classification reports, confusion matrices and ROC curves.
8. **Explainability:** apply SHAP to the tuned XGBoost model on 5,000 test URLs to rank the features, show the direction of their effects and explain individual predictions.
9. **Robustness check:** retrain the tuned models without the HTTPS and HTTP features, and then without `url_length` as well, because the protocol can still be worked out from the URL length.
10. **Conclusion:** answer the three research questions and set out the limitations and future work.

### **Research Questions**

- **RQ1:** How accurately can machine learning classifiers distinguish phishing URLs from legitimate URLs using the Kaitholikkal and Arthi phishing URL dataset?
- **RQ2:** Which lexical and structural URL characteristics are identified by SHAP as the most influential factors in distinguishing phishing URLs from legitimate URLs?
- **RQ3:** Which machine learning model provides the best performance for phishing URL detection in terms of accuracy, precision, recall, F1-score and ROC-AUC?


## **2. Aim**

To build and evaluate machine learning models that detect phishing URLs from lexical and structural URL features, and to use SHAP to explain which features most influence their predictions.

## **3. Objectives**

1. Load, inspect and clean the Kaitholikkal and Arthi phishing URL dataset.
2. Engineer lexical and structural features from raw URLs.
3. Carry out exploratory data analysis to compare phishing and legitimate URLs.
4. Train baseline classifiers: Logistic Regression, Decision Tree, Random Forest, XGBoost and a neural network (multi-layer perceptron).
5. Tune the hyperparameters of each classifier and retrain the tuned models.
6. Evaluate and compare baseline and tuned models using accuracy, precision, recall, F1-score and ROC-AUC.
7. Apply SHAP to identify and explain the most influential URL features.
8. Assess how much performance depends on protocol features through a robustness check.
9. Report the findings, limitations and directions for future work.


### **Schedule**

1. **Project setup:** topic agreement, dataset selection and project form submission
2. **Data preparation:** loading, cleaning, feature engineering and EDA
3. **Modelling:** baseline training, hyperparameter tuning and evaluation
4. **Explainability:** SHAP analysis and robustness check
5. **Write-up:** report writing, review and final submission

## **4. Project Files**

| File | Contents |
|---|---|
| `Explainable_Phishing_URL_Detection.ipynb` | Final notebook with the full project, from data preparation to the conclusion. All outputs are saved in it. |
| `URL dataset.csv` | Main labelled dataset, 450,176 URLs |
| `Phishing URLs.csv` | Extra phishing URLs, 54,807 URLs |
| `requirements.txt` | Python libraries and the versions the notebook was tested with |
| `README.md` | This file |

## **5. Dataset**

### **Overview**

- **Name:** Phishing URL Dataset
- **Authors:** Jishnu K. S. Kaitholikkal and Arthi B.
- **Repository:** Mendeley Data
- **Published:** 2 April 2024
- **Version:** 1
- **DOI:** [10.17632/vfszbj9b36.1](https://doi.org/10.17632/vfszbj9b36.1)
- **Link:** [https://data.mendeley.com/datasets/vfszbj9b36/1](https://data.mendeley.com/datasets/vfszbj9b36/1)

### **Description**

The dataset contains URLs gathered from publicly available online sources, including PhishTank and Majestic Million, among others. Each URL is labelled as either **phishing** or **legitimate**. The dataset page does not specify an exact collection date range.

The data is **tabular and textual**. Each record holds a raw URL and its label. The raw URLs are processed into numerical features for modelling.

### **Files Used**

| CSV file | Rows | URL column | Label column | Label values | Role in project |
|---|---:|---|---|---|---|
| `URL dataset.csv` | 450,176 | `url` | `type` | `legitimate`, `phishing` | Main labelled dataset |
| `Phishing URLs.csv` | 54,807 | `url` | `Type` | `Phishing` | Extra phishing URLs, combined with the main dataset |

### **Structure**

- `URL dataset.csv` contains **345,738 legitimate** and **104,438 phishing** URLs.
- `Phishing URLs.csv` contains only phishing URLs, so it cannot be used on its own for binary classification. 48 of its URLs also appear in the main file.
- The two files are combined into one working dataset of 504,983 rows. After duplicate URLs are removed, **504,932 unique URLs** remain: **345,738 legitimate (68.47%)** and **159,194 phishing (31.53%)**.
- The stratified split gives **403,945 training URLs** and **100,987 test URLs**.


### **Where the Data Comes From**

Both CSV files are included in this folder, unchanged from the [Mendeley Data page](https://data.mendeley.com/datasets/vfszbj9b36/1). They are shared under the **CC BY 4.0** licence, which allows copying and reuse as long as the authors are credited.

### **Running the Notebook**

On a computer with Python 3.12:

1. Open a terminal in this folder.
2. Install the libraries with `pip install -r requirements.txt`.
3. Start Jupyter with `jupyter notebook` and open `Explainable_Phishing_URL_Detection.ipynb`.
4. Run the cells in order. The notebook reads the two CSV files from this folder.

On Google Colab, upload the notebook and both CSV files, then run the cells in order. Colab already has most of the libraries, and the notebook installs `xgboost` 3.1.2 and `shap` 0.50.0 itself, so the results stay the same on each run.

A full run takes a few minutes. It also saves the cleaned dataset as `phishing_urls_processed.csv` in this folder (about 84 MB).

## **6. Key Results**

All scores are from the test set of 100,987 URLs, which is not used in training or tuning.

### **Baseline and Tuned Models**

| Model | Baseline F1-score | Tuned F1-score | Tuned accuracy | Tuned precision | Tuned recall | Tuned ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8775 | 0.8777 | 0.9252 | 0.9055 | 0.8515 | 0.9549 |
| Decision Tree | 0.9396 | 0.9401 | 0.9626 | 0.9491 | 0.9313 | 0.9862 |
| Random Forest | 0.9472 | 0.9489 | 0.9684 | 0.9686 | 0.9301 | 0.9924 |
| XGBoost | 0.9458 | 0.9464 | 0.9668 | 0.9647 | 0.9288 | 0.9914 |
| Neural Network | 0.9443 | 0.9455 | 0.9668 | 0.9796 | 0.9137 | 0.9902 |

The tuned Random Forest is the best model on accuracy, F1-score and ROC-AUC. XGBoost and the neural network are less than 0.004 behind on F1-score. The neural network has the highest precision and the Decision Tree the highest recall. Tuning changed the scores very little, with a largest F1-score gain of 0.0017.

### **SHAP Findings**

For the tuned XGBoost model, `uses_http` is the most influential feature with a mean absolute SHAP value of 2.51. It is followed by `path_length` (0.84), `hostname_dot_count` (0.71), `hostname_length` (0.64), `digit_count` (0.51) and `uses_https` (0.43). A URL that starts with `http://` is pushed strongly towards phishing.

### **Robustness Check**

| Model | F1-score, all features | F1-score, no protocol features | F1-score, no protocol features or URL length |
|---|---:|---:|---:|
| Logistic Regression | 0.8777 | 0.6720 | 0.6704 |
| Decision Tree | 0.9401 | 0.8556 | 0.8529 |
| Random Forest | 0.9489 | 0.9060 | 0.8796 |
| XGBoost | 0.9464 | 0.8797 | 0.8675 |
| Neural Network | 0.9455 | 0.9342 | 0.8662 |

Every URL in the dataset that starts with `http://` is phishing (103,659 URLs), while 13.84% of the `https://` URLs are. Once all protocol information is removed, the best accuracy falls from 96.8% to 92.5% and the best F1-score from 0.949 to 0.880. The Random Forest stays in first place.

### **Limitations**

- The protocol separates the classes far more cleanly in this dataset than it would in real traffic. The same applies to the `www.` prefix, which 99.8% of the legitimate URLs have and only 9.59% of the phishing URLs.
- About 67% of the test URLs share a hostname with a URL in the training set, which makes the test easier than classifying an unseen website.
- The models use 17 simple counts and lengths only. Tuning ran on a 100,000 URL sample, and all results come from one train and test split.
- SHAP was applied to XGBoost only, so the ranking describes that model.

The scores above come from one run of the notebook. A new run with different library versions can change the last decimal place.

## **7. Ethics and Data Use**

> **Warning:** This dataset contains real phishing URLs. **Do not open, visit or click any URL from the dataset.** All analysis is done on the URL text only.

- The dataset is publicly available on Mendeley Data under the CC BY 4.0 licence.
- The data is used only for academic research within this university project.
- The two CSV files in this folder are unchanged copies of the original files, included with credit to the authors as the licence requires.
- The dataset does not contain direct personal records. However, some URLs may include query strings or identifiers, so raw URLs are not published in reports beyond what is needed for illustration.
- The models are research prototypes and are not intended for use as a production security tool.
- The original dataset authors are credited in all outputs.

## **8. Key References**

- Kaitholikkal, J. K. S. and Arthi, B. (2024) *Phishing URL Dataset*, Mendeley Data, Version 1. DOI: [10.17632/vfszbj9b36.1](https://doi.org/10.17632/vfszbj9b36.1)
- Lundberg, S. M. and Lee, S.-I. (2017) 'A Unified Approach to Interpreting Model Predictions', *Advances in Neural Information Processing Systems*.
- Chen, T. and Guestrin, C. (2016) 'XGBoost: A Scalable Tree Boosting System', *Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.
- Breiman, L. (2001) 'Random Forests', *Machine Learning*.
- Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python', *Journal of Machine Learning Research*.
