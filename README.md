# Portugese-Bank-Telemarketing-Solution

<br>
## Problem Statement :
<br>
The Portuguese Bank had run a telemarketing campaign in the past, making sales calls for a term-deposit product. Whether a prospect had bought the product or not is mentioned in the column named 'response'.
<br>
The marketing team wants to launch another campaign, and they want to learn from the past one. You, as an analyst, decide to build a supervised model in Python and achieve the following goals:
<br>
·   Reduce the marketing cost by X% and acquire Y% of the prospects (compared to random calling), where X and Y are to be maximized
<br>
·   Present the financial benefit of this project to the marketing team

<br>

## Solution Description and Results :

### Results :

#### i) Reduction in Marketing Cost X = 50 %

#### ii) Acquired Percentage of the Prospects Y = 98.877 %

### Description :
<br>
Step 1 : Imported the Following Libraries - > pandas , numpy , matplotlib ,scipy
<br>
Step 2 : Imported the Dataset and Created the Matrix of Features and Vector of Dependent Variables
<br>
Step 3 : Data Preprocessing -> Label Encoding and OneHot Encoding of Categorical Data
<br>
Step 4 : Splitting the Dataset into Training and Test Set
<br>
Step 5 : Feature Scaling
<br>
Step 6 : Making the Confusion Matrix
<br>
Step 7 : Defining the CAP (Cumulative Accuracy Profile) Curve Function
<br>
#### Step 8 : Generating the Cumulative Accuracy Profile Curve
<br>
In the Cumulative Accuracy Profile Curve Shown in the figure above , we can see three different curves. 
<br>
i)The blue line represents the random model ie., when we send out the marketing scheme to     all the customers.
<br>
ii)The grey lin represents the ideal model ie., when we send out the marketing scheme to      only those customers who will eventually give a positive response in the future ie., 
100% conversion rate (ideal).
<br>
#### iii) The red curve represents our model. Here we reduce the marketing costs by 50 % and achieve 98.877 % of the prospects compared to random calling.
<br>
## Summary 
<br>
### As we can see the marketing team can now target a specific set of customers based on this model and reduce 50% of the total expenditure (random model) and achieve 98.877 % of the prospects compared to the random calling method.