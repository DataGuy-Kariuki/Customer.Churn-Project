# Building a machine learning model in R involves a structured workflow.
#load the dataset
#We will use the Telco customer churn dataset , which is ideal for churn prediction tasks.

# Install the necessary packages if you haven't already
#install.packages("xgboost")
#install.packages("caret")  # For data splitting and training
#install.packages("dplyr")   # For data manipulation
#install.packages("ggplot2") # For visualization

# Load the libraries
library(xgboost)
library(caret)
library(dplyr)
library(ggplot2)

#load the dataset

telco_data<- read.csv("https://raw.githubusercontent.com/DataGuy-Kariuki/Customer.Churn-Project/refs/heads/main/Telco-Customer-Churn.csv")

#view the first few rows and summary statistics
head(telco_data)

str(telco_data)

summary(telco_data)

telco_data <-telco_data[,-1]

## Understanding Key Variables and Identifying Data Types

# Customer Demographics: gender, SeniorCitizen, Partner, and Dependents are categorical, where SeniorCitizen is binary (0 or 1).
# Customer Account Information: Contract, PaperlessBilling, and PaymentMethod are categorical, while tenure is numerical and ranges from 0 to 72.
# Service Information: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies are categorical.
# Charges: MonthlyCharges is numerical, and TotalCharges should also be numerical but currently has some missing values (NA's).

# We’ll convert relevant columns to categorical (factor) types in R and handle missing values in TotalCharges.
#Total charges has 11 missing values.

#Handling missing values

# Convert TotalCharges to numeric
telco_data$TotalCharges <- as.numeric(telco_data$TotalCharges)

# Remove missing values or alternatively, fill with median or mean
#telco_data <- telco_data[!is.na(telco_data$TotalCharges), ]
#Go with median imputation
# OR to impute with median
telco_data$TotalCharges[is.na(telco_data$TotalCharges)] <- median(telco_data$TotalCharges, na.rm = TRUE)

#confirm if there is any missing values present in that columns
sum(is.na(telco_data$TotalCharges))

# Exploratory Data Analysis (EDA) Using ggplot2
# EDA helps us understand the structure and potential relationships in the data.

#Distribution of all numerical variables present

##Histogram of tenure
ggplot(telco_data,aes(x= tenure))+
  geom_histogram(fill = "skyblue", bins =30)+
  labs(title = "Distribution of Tenure", x = "Tenure", y = "Count")

#Histogram for MonthlyCharges

ggplot(telco_data,aes(x=MonthlyCharges))+
  geom_histogram(fill = "salmon", bins =30)+
  labs(title = "Distribution of Monthly Charges", x = "Monthly Charges" , y= "count")

#Histogram of the Totalchrges

ggplot(telco_data,aes(x= TotalCharges))+
  geom_histogram(fill = "lightgreen", bins =30)+
  labs(title = 'Distribution of Total Charges', x ="Total charges", y ="count")

#Categorical Variables vs. Target (Churn)
#This helps us see which services or demographics are associated with higher churn rates.

# churn by gender 

# Corrected Churn by Gender visualization
ggplot(telco_data, aes(x = gender, fill = factor(Churn))) +
  geom_bar(position = "fill") +  # "fill" scales bars to relative proportion within each gender
  labs(title = "Churn Rate by Gender", x = "Gender", y = "Proportion") +
  scale_y_continuous(labels = scales::percent_format()) +
  scale_fill_manual(values = c("No" = "skyblue", "Yes" = "salmon"), name = "Churn")

# Churn by Contract Type
ggplot(telco_data, aes(x = Contract, fill = factor(Churn))) +
  geom_bar(position = "fill") +
  labs(title = "Churn by Contract Type", x = "Contract Type", y = "Proportion")

# 4. Data Preparation for Machine Learning
# With insights from EDA, we can finalize data preparation:
# Convert Categorical Variables: Convert all relevant columns to factors.
# Scale Numerical Features: This helps models like gradient boosting perform better, especially when there are features with different scales.


# Convert relevant columns to factors
categorical_vars <- c("gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection","TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "Churn")

telco_data[categorical_vars] <- lapply(telco_data[categorical_vars], as.factor)
# Convert Churn to numeric (0 and 1)
telco_data$Churn <- ifelse(telco_data$Churn == "Yes", 1, 0)

# Normalize/Scale numerical features
telco_data$tenure <- scale(telco_data$tenure)
telco_data$MonthlyCharges <- scale(telco_data$MonthlyCharges)
telco_data$TotalCharges <- scale(telco_data$TotalCharges)

#Data Splitting
#Splitting the data into training, testing, and validation sets is important for unbiased model evaluation. A common approach is:

#Training Set: Used to train the model.
#Testing Set: Used for tuning and optimizing model parameters.
#Validation Set: Used for final model evaluation.
#Using the caret package helps split the data consistently across these subsets.


# Set a seed for reproducibility
set.seed(123)

# Split data into 70% training and 30% for test and validation
trainIndex <- createDataPartition(telco_data$Churn, p = 0.7, list = FALSE)
train_data <- telco_data[trainIndex, ]
temp_data <- telco_data[-trainIndex, ]

# Split remaining data into test (15%) and validation (15%)
testIndex <- createDataPartition(temp_data$Churn, p = 0.5, list = FALSE)
test_data <- temp_data[testIndex, ]
validation_data <- temp_data[-testIndex, ]


#4. Model Selection and Training with Gradient Boosting
#Gradient Boosting is an ensemble technique that builds multiple decision trees to improve prediction accuracy. Here, we use xgboost, a popular R package for gradient boosting.

#Prepare Data for xgboost
#xgboost requires the data to be in a special matrix format called a DMatrix. We use model.matrix to encode categorical variables automatically.

# Convert training data to DMatrix format
train_matrix <- model.matrix(Churn ~ . - 1, data = train_data)
train_label <- train_data$Churn
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)

# Prepare test data for early stopping and evaluation
test_matrix <- model.matrix(Churn ~ . - 1, data = test_data)
test_label <- test_data$Churn
dtest <- xgb.DMatrix(data = test_matrix, label = test_label)

#Train the xgboost Model
#Now, we specify model parameters and train the model. Common parameters include:
#objective: Defines the task (binary classification here).
#eval_metric: Sets the evaluation metric (AUC for classification).
#eta: Learning rate, which controls how quickly the model adapts.
#max_depth: The depth of each tree; deeper trees capture more detail but can overfit.

# Rebuild DMatrix
train_matrix <- model.matrix(Churn ~ . - 1, data = train_data)
train_label <- train_data$Churn
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)

test_matrix <- model.matrix(Churn ~ . - 1, data = test_data)
test_label <- test_data$Churn
dtest <- xgb.DMatrix(data = test_matrix, label = test_label)

# Define model parameters again if needed
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train the model again
set.seed(123)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain, eval = dtest),
  early_stopping_rounds = 10,
  print_every_n = 10
)

# Make predictions
predictions <- predict(xgb_model, dtest)
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

# Create confusion matrix
confusion_matrix <- table(Predicted = predicted_classes, Actual = test_label)
print(confusion_matrix)



# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", accuracy, "\n")

# Plot confusion matrix
library(caret)
confusionMatrix(factor(predicted_classes), factor(test_label), positive = "1")

# Assume you have already run your model and predicted classes
# predicted_classes <- ifelse(predictions > 0.5, 1, 0)  # Example thresholding

# 1. Create the confusion matrix
confusion_matrix <- table(Predicted = predicted_classes, Actual = test_label)

# 2. Convert it to a data frame
confusion_df <- as.data.frame(confusion_matrix)

# 3. Plot the confusion matrix
library(ggplot2)

ggplot(confusion_df, aes(x = Predicted, y = Actual)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "white", high = "blue") +
  geom_text(aes(label = Freq), vjust = 1) +
  labs(title = "Confusion Matrix", x = "Predicted", y = "Actual") +
  theme_minimal()

# Calculate accuracy for various thresholds
thresholds <- seq(0, 1, by = 0.05)
accuracy_values <- sapply(thresholds, function(thresh) {
  predicted_classes <- ifelse(predictions > thresh, 1, 0)
  sum(predicted_classes == test_label) / length(test_label)
})

# Create a data frame for ggplot
accuracy_df <- data.frame(Threshold = thresholds, Accuracy = accuracy_values)

# Plot accuracy vs threshold
ggplot(accuracy_df, aes(x = Threshold, y = Accuracy)) +
  geom_line(color = "blue") +
  labs(title = "Accuracy vs. Threshold", x = "Threshold", y = "Accuracy") +
  theme_minimal()


# Save the model to a file
saveRDS(xgb_model, file = "xgb_model.rds")


