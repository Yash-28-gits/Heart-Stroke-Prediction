# Heart Stroke Prediction

This project focuses on building predictive models for heart stroke detection to enhance clinical decision-making. The notebook `stroke.ipynb` contains the code for data preprocessing, model building, and evaluation.

## Dataset

The dataset used in this project is the `healthcare-dataset-stroke-data.csv`. It contains various medical attributes that are used to predict the likelihood of a heart stroke.

## Requirements

To run the notebook, you need to have the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- keras

You can install the required libraries using the following command:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras
```

## Usage

1. Clone the repository and navigate to the project directory.
2. Open the `stroke.ipynb` notebook in Jupyter Notebook or JupyterLab.
3. Run the cells in the notebook to preprocess the data, build the models, and evaluate their performance.

## Project Structure

- `stroke.ipynb`: The main notebook containing the code for data preprocessing, model building, and evaluation.
- `healthcare-dataset-stroke-data.csv`: The dataset used for the project.

## Data Preprocessing

The data preprocessing steps include:

- Loading the dataset
- Dropping unnecessary columns
- Handling missing values
- Encoding categorical variables
- Normalizing numerical features

## Model Building

The notebook explores various machine learning and deep learning models, including:

- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machine (SVM)
- Neural Networks

## Evaluation

The models are evaluated using various metrics such as accuracy, precision, recall, and F1-score. The performance of the models is compared to select the best model for heart stroke prediction.

## Conclusion

This project demonstrates the process of building and evaluating predictive models for heart stroke detection. The best-performing model can be used to assist healthcare professionals in making informed decisions.

## References

- [Dataset Source](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)
