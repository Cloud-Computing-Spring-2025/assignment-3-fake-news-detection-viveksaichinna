# Assignment-5-FakeNews-Detection

# Assignment #5: Fake News Detection using Spark MLlib

##  Dataset used 
- `fake_news_sample.csv`
- Contains news articles with:
  - `id`: Unique ID
  - `title`: Title of the news article
  - `text`: Content of the article
  - `label`: Label ('FAKE' or 'REAL')

---

## Project Tasks

### Task 1: Load & Basic Exploration
- Loaded `fake_news_sample.csv` into a Spark DataFrame.
- Created a temporary view named `news_data`.
- Displayed:
  - First 5 rows
  - Total number of articles
  - Distinct labels
- Output saved: `task1_output.csv`

### Task 2: Text Preprocessing
- Lowercased the `text` column.
- Tokenized the text into individual words.
- Removed stopwords using Spark’s `StopWordsRemover`.
- Output saved: `task2_output.csv`

### Task 3: Feature Extraction
- Applied `HashingTF` and `IDF` to extract TF-IDF features from filtered words.
- Converted `label` into numerical format using `StringIndexer`:
  - FAKE → 0.0
  - REAL → 1.0
- Output saved: `task3_output.csv`

### Task 4: Model Training
- Split the dataset into 80% training and 20% testing sets.
- Trained a Logistic Regression model using Spark MLlib.
- Generated predictions on the test set.
- Output saved: `task4_output.csv`

### Task 5: Model Evaluation
- Evaluated model performance using:
  - **Accuracy**
  - **F1 Score**
- Output saved: `task5_output.csv`

---

## How to Run the Code

### 1. Requirements
- Python 3.x
- PySpark
- Pandas

Install using pip if needed:
```bash
pip install pyspark 
pip install faker 
spark submit --version 

Run the script by 
spark-submit fakenews.py

```



After running, you will find:
	•	task1_output.csv
	•	task2_output.csv
	•	task3_output.csv
	•	task4_output.csv
	•	task5_output.csv

Thats it for the Assignment
