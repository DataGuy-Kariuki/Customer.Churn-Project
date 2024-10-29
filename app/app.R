# Install necessary packages if you haven't already
# install.packages("shiny")
# install.packages("xgboost")
# install.packages("caret")
# install.packages("dplyr")
# install.packages("ggplot2")

# Load the required libraries
library(shiny)
library(xgboost)
library(caret)
library(dplyr)
library(ggplot2)

# Load your trained model here if you saved it
# Load the model from the file
# Set the working directory
setwd("C:/Users/XaviourAluku.BERRY/Documents/model7/model7") # Update this path to your desired directory
xgb_model <- readRDS("xgb_model.rds") 


# Define UI
ui <- fluidPage(
    titlePanel("Customer Churn Prediction"),
    
    sidebarLayout(
        sidebarPanel(
            fileInput("file", "Upload CSV File", accept = c(".csv")),
            actionButton("predict", "Predict Churn")
        ),
        
        mainPanel(
            tableOutput("predictions"),
            plotOutput("confusionPlot")
        )
    )
)

# Define server logic
server <- function(input, output) {
    
    observeEvent(input$predict, {
        req(input$file)
        
        # Load the dataset
        telco_data <- read.csv(input$file$datapath)
        
        # Preprocess the data (similar to your model)
        telco_data <- telco_data[,-1]  # Remove the first column if it's an index
        telco_data$TotalCharges <- as.numeric(telco_data$TotalCharges)
        telco_data$TotalCharges[is.na(telco_data$TotalCharges)] <- median(telco_data$TotalCharges, na.rm = TRUE)
        
        # Convert to factors
        categorical_vars <- c("gender", "Partner", "Dependents", "PhoneService", "MultipleLines", 
                              "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                              "TechSupport", "StreamingTV", "StreamingMovies", "Contract", 
                              "PaperlessBilling", "PaymentMethod", "Churn")
        telco_data[categorical_vars] <- lapply(telco_data[categorical_vars], as.factor)
        telco_data$Churn <- ifelse(telco_data$Churn == "Yes", 1, 0)
        
        # Scale numerical features
        telco_data$tenure <- scale(telco_data$tenure)
        telco_data$MonthlyCharges <- scale(telco_data$MonthlyCharges)
        telco_data$TotalCharges <- scale(telco_data$TotalCharges)
        
        # Prepare data for predictions
        pred_matrix <- model.matrix(Churn ~ . - 1, data = telco_data)
        dpred <- xgb.DMatrix(data = pred_matrix)
        
        # Make predictions
        predictions <- predict(xgb_model, dpred)
        predicted_classes <- ifelse(predictions > 0.5, 1, 0)
        
        # Output predictions
        output$predictions <- renderTable({
            telco_data$Predicted_Churn <- predicted_classes
            head(telco_data[, c("CustomerID", "Predicted_Churn")])  # Display customer ID and prediction
        })
        
        # Confusion matrix
        output$confusionPlot <- renderPlot({
            confusion_matrix <- table(Predicted = predicted_classes, Actual = telco_data$Churn)
            confusion_df <- as.data.frame(confusion_matrix)
            
            ggplot(confusion_df, aes(x = Predicted, y = Actual)) +
                geom_tile(aes(fill = Freq), color = "white") +
                scale_fill_gradient(low = "white", high = "blue") +
                geom_text(aes(label = Freq), vjust = 1) +
                labs(title = "Confusion Matrix", x = "Predicted", y = "Actual") +
                theme_minimal()
        })
    })
}

# Run the application 
shinyApp(ui = ui, server = server)

