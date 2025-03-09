# phase3
SyriaTel Customer Churn Prediction

Project Overview

This project aims to predict customer churn for SyriaTel, a telecommunication company, using customer data. The goal is to assist stakeholders in understanding the factors leading to customer churn and develop strategies to improve customer retention. This analysis uses machine learning models to predict whether a customer will discontinue their services, helping SyriaTel take proactive actions.

Business Objective
Understanding and predicting customer churn is crucial for telecom companies as it directly impacts their revenue and market share. By analyzing churn patterns, companies can identify contributing factors (e.g., poor service quality, high prices, lack of personalized offers) and develop strategies to improve customer satisfaction and reduce churn rates. Accurate churn prediction also enables better resource allocation and targeted marketing efforts.

Dataset
The dataset used in this analysis comes from a publicly available Kaggle dataset. It contains 3,333 customer entries, each with 21 columns (20 feature columns and 1 target column). Some key features include:

account length: Duration of the customer's account with SyriaTel (in months)
international plan: Whether the customer has an international calling plan (Yes/No)
customer service calls: Number of customer service calls made by the customer
churn: Target variable indicating whether the customer churned (1) or not (0)
Important Note: Features like phone number, state, and area code were removed from the analysis as they were deemed irrelevant for churn prediction or would limit the generalization of the model.

Data Preprocessing
Before fitting models, the data underwent the following preprocessing steps:

Encoding categorical variables (e.g., 'international plan', 'voice mail plan')
Normalizing numeric features to ensure they have a similar scale
Addressing class imbalance (14.5% churn) using techniques like oversampling or adjusting class weights
Data Splitting
The dataset was split into training and testing sets to avoid data leakage and ensure the testing data remains untouched for evaluating the model's performance.

Modeling
The following models were tested to predict customer churn:

1. Logistic Regression
A simple logistic regression model was trained as the baseline.
Results: Poor performance due to its inability to capture data complexities.
2. Decision Tree
A decision tree classifier was tested to better capture patterns in the data.
Results: Overfitting the training data. Good performance on training data but poor generalization on testing data.
3. Random Forest
A Random Forest model was tested to overcome the limitations of individual decision trees and improve model generalization.
Results: Performed well, capturing more complex patterns in the data and showed better generalization to unseen data.
4. Extreme Gradient Boosting
This was the best model above all the tuned models. The best Model is XGBoost had an accuracy of 93.2143% Precision 86.3636% Recall of 54.2857% F1 Score of 66.6667% and ROC-AUC of 87.4869
 
5. Tuned Random Forest and Logististic regression
Hyperparameters of the Random Forest model were tuned for better performance.
Tuned Parameters:
max_depth: 15
min_samples_leaf: 1
min_samples_split: 2
n_estimators: 200
Results: Achieved improved performance, particularly in recall, making it better at identifying churned customers.
Evaluation Metrics
To evaluate model performance, the following metrics were used:

Accuracy: Proportion of correct predictions.
Precision: Correct positive predictions relative to all positive predictions.
Recall: Correct positive predictions relative to all actual positive instances.
F1-Score: The balance between precision and recall.
ROC Curve & AUC: To visualize performance and evaluate the model’s ability to distinguish between churned and non-churned customers.
Results
Random Forest:

Highest testing accuracy and precision.
High recall, capturing a significant portion of churned customers.
Highest F1-score, balancing precision and recall.
AUC of 0.905, indicating excellent performance.
Tuned Random Forest:

Improved recall at the cost of a slight decrease in precision, leading to a better balance of false positives and false negatives.
Best model with AUC of 0.954, indicating near-perfect prediction capabilities.
Conclusion
The Random Forest model, especially the tuned version, outperformed Logistic Regression and Decision Tree models in predicting customer churn. The tuned model provided the best balance between precision and recall, minimizing false positives while effectively identifying churned customers.

Key features influencing churn include:

Total_charge
Total_minutes
avg_charge_per_minnute
avg_minutes_per_cal
Number vmail Messages
international plan
The model achieved an accuracy of 93.21%, with a recall of 86.36%, meaning it captured slightly 86% of actual churn cases, which is critical for retention strategies.

Future Work
While the Extreme Gradient Boosting model provides good results, there is room for improvement:

Further tuning of hyperparameters may improve model performance.
Exploring more advanced models like Neural Networks.
Incorporating additional features like customer feedback and satisfaction scores could potentially improve prediction accuracy.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Thanks to Kaggle for providing the dataset.
Special thanks to the stakeholders at SyriaTel for providing the context and objective for this analysis.
How to Use This Repository
Clone the repository:

bash
Copy
git clone https://github.com/JaelKirwa/phase3/blob/main/SyriaTel%20Phase3.ipynb
Install required dependencies:

bash
Copy
pip install -r requirements.txt
Run the Jupyter Notebook or Python script to train and evaluate the models.
