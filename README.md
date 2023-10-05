** -Churn-prediction for sprint-
### **Week 1:CHURN PREDICTION FOR SPRINT**

###  **INTRODUCTION.**
A company's objective is to estimate customer survival function and customer hazard function to gain knowledge of customer churn over the time of customer tenure. Their second objective is to demonstrate how survival analysis techniques are used to identify the customers who are at high risk of churn and when they will churn.

### **QUESTION:**
Imagine you're working with Sprint, one of the biggest telecom companies in the USA. They're keen on figuring out how many customers might decide to leave them in the coming months. Luckily, they've got a bunch of past data about when customers have left before, as well as info about who these customers are, what they've bought, and other things like that.
So, if you were in charge of predicting customer churn, how would you go about using machine learning to make a good guess about which customers might leave? What steps would you take to create a machine-learning model that can predict if someone's going to leave or not?

### **SOLUTION:**
In the telecommunications industry, customer churn occurs when a customer stops conducting business with a company. This decision includes actions like canceling a subscription to a service, not renewing a contract, or seizing all purchases and usage of a product or solution.
### **Churn prediction** is a company’s ability to spot customers at risk of churning before they actually do. Companies invest time and resources into building effective churn prediction strategies to retain revenue and minimize losses from all fronts. 
The end goal of churn prediction is to understand when exactly a customer is expected to churn. From there, companies can prioritize at-risk customers that are valuable for growth and execute personalized strategies to reengage those customers again. 
### **A churn prediction model** is built with machine learning to predict churn with an algorithm training with patterns in historical data. 
You can predict churn by looking at how customers behave when interacting with your product and representatives from your team. Companies can collect various data points that indicate frustration, dissatisfaction, or a change in heart in their customers. The algorithm is fed with these data points to analyze how specific input factors may lead to churn as an output. 
Companies use churn prediction models instead of relying solely on human-driven decision-making because this technology helps: 
Identify microtrends and subtle patterns that humans can easily miss
Scale data analysis across thousands or millions of data points simultaneously
Standardize churn prediction factors and apply them consistently to all customers with less risk of bias and human error
Reduce resource expenditure on manual and repetitive data analysis work

### **Detailed step by step to create an xgboost machine learning model for predicting customer churn at sprint for the next six months**

### **Benefits to utilizing xgboost.**
Performance: XGBoost has a strong track record of producing high-quality results in various machine learning tasks, especially in Kaggle competitions, where it has been a popular choice for winning solutions
Scalability: XGBoost is designed for efficient and scalable training of machine learning models, making it suitable for large datasets.
Customizability: XGBoost has a wide range of hyperparameters that can be adjusted to optimize performance, making it highly customizable.

### **Step 1:Definition of churn event.**
The broad definition of churn is the action in which a customer’s telecommunications service is canceled. This includes both service-provider-initiated churn and customer-initiated churn. Customer-initiated churn is defined by a series of cancel reason codes. For example, unacceptable call quality, more favorable competitor’s pricing plan, misinformation given by sales, customer expectations not met, billing problems, moving, change in business, and so on.

### **Step 2: Data collection.**
Demographic Data – Demographic data is from a third-party vendor. In this study, the following are examples of customer-level demographic information: Primary household member’s age, Gender, and marital status.
Customer Internal Data – Customer internal data is from the company’s data warehouse. It consists of two parts. The first part is about customer information like market channels, plan types, bill agencies, and so on. The second part of the customer is the customer’s telecommunications usage data. Examples of customer usage variables are Weekly average call counts, Percentage change of minutes, Share of domestic/international revenue Customer Contact Records.
```
```python
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

 data = {
    'customer_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'age': [40, 29, 61, 38, 29, 45, 68, 48, 39, 24],
    'gender': ['Male', 'Male', 'Bigender', 'Female', 'Female', 'Female', 'Female', 'Non-binary', 'Female', 'Male'],
    'monthly_payment': [424.84, 836.32, 373.56, 50.04, 53.9, 619.31, 645.95, 610.18, 639.42, 804.5],
    'contract_length': [9, 24, 7, 10, 8, 7, 16, 10, 18, 17],
    'data_usage': [78.87, 86.77, 42.49, 41.22, 62.11, 80.13, 75.99, 44.03, 6.03, 4.84],
    'voice_usage': [423.45, 219.58, 46.7, 653.6, 337.39, 326.12, 866.99, 674.33, 396.49, 754.62],
    'roaming_usage': [6.93, 92.69, 44.93, 27.57, 82.09, 47.03, 43.08, 2.88, 47.56, 79.09],
    'customer_service_calls': [2, 7, 7, 5, 2, 6, 10, 8, 2, 6],
    'churn': [False, True, True, False, True, False, False, False, False, True]
}

df = pd.DataFrame(data)

```
### **Step 3:Data Preprocessing.**
After data collection, we preprocess the data focusing on addressing missing values, outliers, and duplicate records. Explanatory data analysis was conducted to prepare the data for the survival analysis. An univariate frequency analysis was used to pinpoint value distributions, missing values, and outliers. The variable transformation was conducted for some necessary numerical variables to reduce the level of skewness using one-hot encoding. Encode categorical variables using techniques like one-hot encoding or label encoding. Split the data into features (X) and the target variable (y), where y represents churn labels.

Here is how I would do it
```
```python
plt.figure(figsize=(8, 5))
sns.histplot(df['age'], bins=10, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

```python
plt.figure(figsize=(6, 4))
df['gender'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

```python
plt.figure(figsize=(8, 5))
sns.boxplot(x='churn', y='monthly_payment', data=df, palette='pastel')
plt.title('Monthly Payment vs. Churn')
plt.xlabel('Churn')
plt.ylabel('Monthly Payment')
plt.xticks([0, 1], ['No Churn', 'Churn'])
plt.show()

```
### **Step 4:Feature Engineering and Selection.**
We'll now identify and select the most influential features important for predicting churn.
The specific features to include are:
The rate of change in monthly internet usage.
Contract removal history.
Billing patterns and fluctuations
Customer engagement.
Customer demographics, product & pricing.
```
```python
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

```
### **Step 5:Model selection- xgboost.**
we are choosing the XGBoost algorithm since it has a strong track record of producing high-quality results in various machine learning tasks, especially in Kaggle competitions, where it has been a popular choice for winning solutions. XGBoost is also designed for efficient and scalable training of machine learning models, making it suitable for large datasets.
```
```python
 Convert categorical variables into numerical using one-hot encoding
df = pd.get_dummies(df, columns=['gender'], drop_first=True)

# Split the data into features (X) and target variable (y)
X = df.drop(['customer_id', 'churn'], axis=1)
y = df['churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### **Step 6:Training the model**
After selecting xgboost as a suitable model, we will need to train the model using a set of preprocessed data. This data accurately represents the data you want your churn prediction model;xgboost, to work with moving forward. Xgboost then learns the relationships between different variables and how they contribute to the likeliness of churn.
We will also proceed to perform hyperparameter tuning for xgboost with a focus on max depth, min child weight, number of iterations, beta, gamma, percent of columns considered, and percent of samples considered. It's computationally infeasible to tune all of these simultaneously in a huge grid search.
```
```python
import xgboost as xgb

```python
# Create an XGBoost classifier
xgb_classifier = xgb.XGBClassifier(**best_params)
# Train the model on the training data
xgb_classifier.fit(X_train, y_train)

```python
# Perform Grid Search CV for hyperparameter tuning
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

```python
# Get the best hyperparameters
best_params = grid_search.best_params_

```
### **Step 7:Model Integration**
When your xgboost delivers the right outcomes, you can integrate it into your organization’s technology stack. The model should then d sync with your tools to monitor, analyze, and act on product and customer data. This includes customer support software, customer relationship management (CRM) platforms, digital adoption platforms, and project management tools.


### **Step 8:Model evaluation.**
A well-integrated tcustom reports, filter through user segments, and deploy targeted content and optimizations. If xgboost identifies high churn risk for specific customers, you should be able to view what actions led to that classification and make informed decisions about mitigation strategies.
```
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

```python
# Make predictions on the test data
y_pred = xgb_classifier.predict(X_test)

```python
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

```python
# Display classification report and confusion matrix
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

```
### **Step 9:Monitoring and refinement**
As your customer base grows and your product becomes more complex, the variables influencing churn might change over time. Your team should factor in regular checkpoints or milestones to assess the quality of your xgboost algorithm and identify opportunities for improvement. You may need to retrain the xgboost model with a different set of data or optimize the model by factoring in more customer data.

### **Step 10:Feedback Loop**
Continuously gather feedback from customer retention efforts and monitor churn rates over time. Use this feedback to refine the xgboost model and improve its accuracy.

### **Step 11:Regular Model Maintenance**
Periodically reevaluate and update the xgboost model to ensure its effectiveness as customer behavior and market dynamics change.

### **CONCLUSION.**
If you’re wondering how to predict churn, data analytics is a great place to start. By following the steps outlined in this guide, you can predict when a customer is likely to leave and take action accordingly. Remember, it’s much more expensive to acquire new customers than it is to keep existing ones happy. So if you can prevent even a few customers from leaving, you’ll be doing your business a big favor!

