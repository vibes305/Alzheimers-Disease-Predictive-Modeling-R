library(dplyr);library(caret);library(MASS)
library(prettyR)
library(ggplot2)
library(cluster)
library(factoextra)
library('faraway')
library('ISLR')
library('corrplot')
library(randomForest)
library(Boruta)

#Read the dataset for analysis
data <- read.csv("project data.csv")

#Check the data types of all variables
glimpse(data)

#Pre-Analysis

#As M/F is Chr converting it into numeric
data$M.F <- as.factor(data$M.F)
data$M.F <- as.numeric(data$M.F)

factor(data$M.F)
table(data$M.F)

#Before removing totally 373 observations
#Female 1 Male 2
#Remove rows , where Group value is Converted and missing values.

data<- data[data$Group != "Converted",]

#After removing totally 363 observations , So 10 rows had value Converted.


#Remove missing values

data_new <- na.omit(data)

#After removing missing values 317 observations



#Task 1 - Descriptive Statistics 


#Numerical representations

Demented_val <- data_new %>% filter(data_new$Group=="Demented") 
Non_Demented_val <- data_new %>% filter(data_new$Group=="Nondemented")

summary(Demented_val)
summary(Non_Demented_val)

# Graphical representations

#Boxplot

ggplot(data_new , aes(x=Group, y= MMSE,color=Group)) + geom_boxplot() + labs(title = "MMSE by group")

#Scatterplot

ggplot(data_new , aes(x=CDR,y=nWBV, color=Group)) + geom_point() + labs(title= "eTIV vs CDR")



#Task 2 K-means Clustering Algorithm  


#Only numeric variables must be used while clustering
data_numeric <- select(data_new, where(is.numeric))

#scaled data ensures all variables in similar scale units.
data1 <- scale(data_numeric)


#Determining the optimal number of clusters
fviz_nbclust(data1, kmeans, method = "wss")+ geom_vline(xintercept = 3, linetype = 2)


#Clustering
#Using set.seed(), So that everytime same points are used for clustering.
set.seed(123)

kmeans3 <- kmeans(data1, centers = 3, nstart = 20)
kmeans3

kmeans2 <- kmeans(data1, centers = 2, nstart = 20)


#To visualise the results the fviz_cluster function can be used:
f1 <- fviz_cluster(kmeans2, geom = "point", data = data1) + ggtitle("k = 2")
f2 <- fviz_cluster(kmeans3, geom = "point", data = data1) + ggtitle("k = 3")

#Display Cluster plots
library(gridExtra)
grid.arrange(f1, f2, nrow = 2)

#To check classification 
table(kmeans3$cluster,data_new$Group)
table(kmeans2$cluster,data_new$Group)



#Use the aggregate() function to find the mean of the variables in each cluster:
aggregate(data1, by=list(cluster=kmeans3$cluster), mean)



#Task4 Fit logistic regression model 

data_new$Group <- as.factor(data_new$Group)
#data_new$Group <- as.numeric(data_new$Group)



# Split the data into training and test set
set.seed(123)

training.samples <- data_new$Group %>% 
  createDataPartition(p = 0.7, list = FALSE)

train.data  <- data_new[training.samples, ]
test.data <- data_new[-training.samples, ]

#Training the model

#Model1 with accuracy 0.989 based on all predictor confirmed by Boruta method
model1 <- glm( Group ~ ., data = train.data, family = binomial)

#Model2 with accuracy 1 based on CDR as confirmed by RFE with Cross validation
model2 <- glm( Group ~ CDR, data = train.data, family = binomial)

#Model3 with accuracy 0.957 based on Step forward model of all variables
model3 <-glm(Group~ CDR +  M.F + EDUC + Age + ASF + nWBV,data=train.data,family=binomial)

#Model4 with accuracy 0.98 based on Step forward model of only intercept 
model4 <-glm(Group~ CDR +  M.F + eTIV + EDUC ,data=train.data,family=binomial)

#Model5 with accuracy 0.62 based on least important predictors confirmed by Boruta method
model5<-glm(Group~ Age + SES + M.F,data=train.data,family=binomial)

#Model6 with accuracy 1 based on most important predictors confirmed by Boruta method
model6 <-glm(Group~ CDR + MMSE + nWBV + ASF,data=train.data,family=binomial)


#Prediction by model

probabilities <- model1 %>% predict(test.data, type = "response")


contrasts(test.data$Group)

predicted.classes <- ifelse(probabilities > 0.5, "Nondemented", "Demented")

#Accuracy of the model
mean(predicted.classes == test.data$Group)


#Task5 Feature Selection method
#To find most important features.


#Method 1
#Wrapper variable selection methods


Group_factor <- as.factor(data_new$Group)
Group_numeric <- as.numeric(Group_factor)

#Model with intercept only
model1<-lm(Group_numeric~1,data=data_new[,-1])

#Step forward
step1_For <-step(model1,scope = ~ M.F + Age + EDUC+SES + MMSE + CDR + eTIV + nWBV + ASF,
            method='forward')

#Model with all variables
model2<-lm(Group_numeric~.,data=data_new[,-1])

#Step forward
step1_For_all <-step(model2,method="forward")

#Based on the output of this method Model3 and Model4  were executed.
#CDR +  M.F + EDUC + Age + ASF + nWBV was confirmed by Model with intercept only
#CDR +  M.F + eTIV + EDUC was confirmed by Model with all variables



#Method 2 : Boruta

boruta1 <- Boruta(Group_numeric~., data=data_new, doTrace=1)
decision<-boruta1$finalDecision
signif <- decision[boruta1$finalDecision %in% c("Confirmed")]
print(signif)
plot(boruta1, xlab="", main="Variable Importance")
attStats(boruta1)

#Based on the output of this method Model1 Model5 , Model6  were executed.
#Boruta also confirmed that all predictors are good(Model1)
#Boruta output suggests CDR,MMSE,nwBV & ASF as Most important predictors(Model6).
#Boruta output suggests SES,Age & M.F as Least important predictors(Model5).



#Method3 RFE with Cross validation

# Define the control parameters for RFE
control <- rfeControl(functions = rfFuncs, method = "cv", number = 5)

# Perform feature selection using RFE
rfe_results <- rfe(x = data_new[, -1], y = data_new$Group, sizes = c(1:ncol(data_new) - 1), rfeControl = control)

# Print the results
print(rfe_results)

#As this method says CDR is top predictor Model2 was executed












