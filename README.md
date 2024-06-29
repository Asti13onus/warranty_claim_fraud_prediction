Warranty Claims Fraud Prediction
Overview
The aim of this data science project is to predict the authenticity of warranty claims by analyzing various factors such as region, product category, claim value, and more. The dataset used for this project was sourced from Kaggle and comprises 358 rows and 21 columns.

## Data Dictionary
| Column Name         | Description                                     |
|---------------------|-------------------------------------------------|
| Unnamed: 0          | Index                                           |
| Region              | Region of the claim                             |
| State               | State of the claim                              |
| Area                | Area of the claim                               |
| City                | City of the claim                               |
| Consumer_profile    | Consumer profile (Business/Personal)            |
| Product_category    | Product category (Household/Entertainment)      |
| Product_type        | Product type (AC/TV)                            |
| AC_1001_Issue       | Issue with AC component 1 (0 - No issue/No component, 1 - repair, 2 - replacement) |
| AC_1002_Issue       | Issue with AC component 2 (0 - No issue/No component, 1 - repair, 2 - replacement) |
| AC_1003_Issue       | Issue with AC component 3 (0 - No issue/No component, 1 - repair, 2 - replacement) |
| TV_2001_Issue       | Issue with TV component 1 (0 - No issue/No component, 1 - repair, 2 - replacement) |
| TV_2002_Issue       | Issue with TV component 2 (0 - No issue/No component, 1 - repair, 2 - replacement) |
| TV_2003_Issue       | Issue with TV component 3 (0 - No issue/No component, 1 - repair, 2 - replacement) |
| Claim_Value         | Claim value in INR                              |
| Service_Center      | Service center code                             |
| Product_Age         | Product age in days                             |
| Purchased_from      | Purchased from (Dealer, Manufacturer, Internet) |
| Call_details        | Call duration                                   |
| Purpose             | Purpose of the call                             |
| Fraud               | Fraudulent (1) or Genuine (0) Conclusion       |

Conclusion
Exploratory data analysis revealed that warranty claims are more common in southern India, specifically Andhra Pradesh and Tamil Nadu.
- Fraudulent claims are especially prevalent in metropolitan areas like Hyderabad and Chennai.
- The information comprises claims for two products: televisions and air conditioners. When purchased for personal use, TVs have a higher warranty claim rate than air conditioners.
- Fraudulent claims for air conditioners can arise even when the AC parts are in good working order.
- Fraudulent claims for televisions can arise with or without component faults.
- Fraudulent claims are more common when lodged through the manufacturer.
- Fraudulent claims typically have larger claim values than legitimate ones.
- Service center 13 had the most false claims, although having less overall warranty claims.
- Fraudulent claims are more common when the customer service contact lasts less than 3-4 minutes.

Machine learning models such as Decision Tree Classifier, Random Forest Classifier, and Logistic Regression were used for prediction. These models produced high accuracy rates of 91-92%. However, due to the small number of fraudulent claims and dataset size, the models had lower recall scores for false claims. This problem can be solved by gathering additional data.

This initiative has the potential to help identify and prevent warranty claims fraud, saving costs and increasing the efficiency of warranty claim processing.
