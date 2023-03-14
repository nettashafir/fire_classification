# The Classifire (Sorry)

## Description
A classifier of fire types, Trained on the dataset "1.88 Million US Wildfires" from Kaggle. The classifier is Gradient Boosting of (xgboost)[https://xgboost.readthedocs.io/en/stable/index.html] package

A full description of the project, including explanation about the feature engineering and the performances of other models on that problem are discussed in detail in the `final_report.pdf`.

This project was submitted for the course **Applied Competitive Lab in Data Science (67818)** of the Hebrew University of Jerusalem.


## Usage
Download the dataset [from Kaggle](https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires/discussion) as a `.csv` file.<br>

Download all the `.py` and `.ipynb` files to the same directory as the dataset. Open `main.ipynb` and replace the paths of train and test datasets. You can assist the function `get_work_data` in the `utils.py` file for split the full dataset to train and test. <br>

A full description of the code's modules appeares at the end of the file `final_report.pdf`.




