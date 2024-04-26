# Machine-Learning-Models-for-Financial-and-Medical-Data-Analysis


This repository documents the application of multiple machine learning models to predict outcomes from both financial and medical datasets. The project involves the use of Random Forest and Lasso Regression for financial data from the NYSE, and Decision Tree Classification for medical data from the breast cancer dataset.
Project Overview:

The project begins with the application of machine learning models to the NYSE dataset to predict estimated shares outstanding, followed by an analysis of the breast cancer dataset to predict diagnoses.

## Project Structure:

### Random Forest for Financial Data:
- Loaded the NYSE dataset, removed specific features, and split the data into training and testing sets (70% train, 30% test
- Built a basic Random Forest model with default parameters and computed the Mean Squared Error (MSE
- Adjusted the min_samples_split parameter and compared results to understand model sensitivity and performance under different configurations.

### Feature Importance Analysis:
- Employed "Mean Decrease in Impurity" and "Permutation Feature Importance" methods to determine variable importance within the Random Forest model.
- Explained the methodologies behind each importance metric and compared their results to glean deeper insights into feature significance

### Lasso Regression Comparison:
- Implemented Lasso Regression on the same financial dataset split to compare its performance with the Random Forest model, analyzing differences in prediction accuracy and model suitability.

### Decision Tree Classification for Medical Data:
- Built a Decision Tree Classifier using the breast cancer dataset to predict diagnoses, split similarly to the previous models-
- Analyzed the model's performance through a confusion matrix, providing insights into model accuracy and error types
- Visualized the decision tree to identify key variables influencing diagnoses and discussed potential benefits of pruning the tree to optimize model performance.

### Model Evaluation and Interpretations:
- Each model's effectiveness was critically assessed, with detailed interpretations of their outcomes and comparative analyses to identify optimal modeling approaches for each dataset.

## Technologies Used:
- Python
- Pandas and NumPy for data manipulation
- Scikit-learn for building and evaluating machine learning models
- Matplotlib and Seaborn for visualization

## Key Insights:
- Demonstrated the effectiveness of different machine learning models in analyzing and predicting outcomes from diverse datasets.
- Provided a comparative insight into model performances, highlighting how different parameters and modeling techniques affect prediction accuracy and interpretability.
