# Load rsconnect package
library(rsconnect)

# Set account info (replace with your actual details)
rsconnect::setAccountInfo(name='kamwanaanalyst',
                          token='F2BECBC274AB97308196FFE5102B6C63',
                          secret='Thjz9T/h9u3BTvOvxaeNkmEyBwc6N4xX7kzBl9Ah')

# Deploy the app from the directory where app.R is saved
rsconnect::deployApp(appDir = "C:/Users/XaviourAluku.BERRY/Documents/model7/model7",  # replace with your app directory path
                     appName = "ChurnPredictionApp",         # name your app
                     appTitle = "Customer Churn Prediction") # title displayed on shinyapps.io

