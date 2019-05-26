#Algoritmo: Para a realização do treinamento foi utilizado o algoritmo "XGBoost" (eXtreme Gradient Boosting)
#Dataset: Utilizado o data set train_sample disponibilizado no site https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data
# o mesmo possui a descrição de cada campo.
#Referências utilizadas para o desenvolvimento deste projeto: 
# https://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html
# https://rpubs.com/dalekube/XGBoost-Iris-Classification-Example-in-R
# https://cran.r-project.org/web/packages/xgboost/vignettes/xgboost.pdf

#install.packages("data.table")
#install.packages("dplyr")
#install.packages("xgboost")
#install.packages("DiagrammeR")
#install.packages("caret")
#install.packages("e1071")

library(data.table)
library(dplyr)
library(xgboost)
library(caret)

#setwd("D:/DSA/Projeto01/AdTracking-Fraud-Detection")
df <- fread('Data/train_sample.csv')
head(df)
str(df)
summary(df)

#Função feita para adicionar as colunas "click_minute" e "click_hour".
addFeatures <- function(dataFrame) {
  dataFrame$click_minute <- as.integer(format(as.POSIXct(strptime(df$click_time,"%Y-%m-%d %H:%M:%S",tz="")) ,format = "%M"))
  dataFrame$click_hour <- as.integer(format(as.POSIXct(strptime(df$click_time,"%Y-%m-%d %H:%M:%S",tz="")) ,format = "%H"))
  return(dataFrame)
}

#Função feita para remover as colunas "attributed_time" e "click_time".
removeUnnusedFeatures <- function(dataFrame) {
  dataFrame$attributed_time = NULL
  dataFrame$click_time = NULL
  return(dataFrame)
}

df <- addFeatures(df)
df <- removeUnnusedFeatures(df)
head(df)

#Objeto utilizado para dividir o dataset em 70% treino e 30% para teste, mantendo a proporção.
index <- sample(1:nrow(df), size = 0.7 * nrow(df))

x_train <- df[index,-"is_attributed"]
y_train <- df[index,"is_attributed"]
x_test <- df[-index,-"is_attributed"]
y_test <- df[-index,"is_attributed"]

#Objeto criado para conseguir colocar o dataset de treino no modelo para treinar.
trainMatrix <- xgb.DMatrix(as.matrix(x_train), 
                           label = y_train$is_attributed)
#Objeto criado para conseguir colocar o dataset de teste no modelo para teste.
testMatrix <- xgb.DMatrix(as.matrix(x_test), 
                          label = y_test$is_attributed)

#A configuração dos parâmetros foi baseado no kernel disponibilizado pelo Pranav Pandya. (https://www.kaggle.com/pranav84/single-xgboost-in-r-histogram-optimized-version)
params <- list(objective   = "binary:logistic", 
               grow_policy = "lossguide",
               tree_method = "hist",
               eval_metric = "auc", 
               max_leaves  = 7, 
               scale_pos_weight = 99.7,
               eta = 0.1, 
               max_depth = 4, 
               subsample = 0.7, 
               min_child_weight = 0,
               colsample_bytree = 0.7, 
               random_state = 84)

#?xgb.train
modelo2 <- xgb.train(data = trainMatrix, params = params,
                   watchlist = list(valid = testMatrix), 
                   nthread = 4, nrounds = 2000,
                   early_stopping_rounds = 100, verbose = 0)
head(modelo2)
summary(modelo2)

#Realizando novas previsões com base nos 30% que foram designados para teste.
pred2 <- predict(modelo2, testMatrix)

#Considerando as previsões acima de 0.7 para identificar se o ip irá efetuar o dowload depois de clicar no anúncio.
pred_transformated <- as.factor(as.numeric(pred2>0.7))

#Matriz de confusão para identificar a acurácia do algoritmo.
confusionMatrix(data = pred_transformated, reference = as.factor(y_test$is_attributed))

#Função utilizada para identificar o grau de importancia das features utilizadas no treinamento.
xgb.importance(model = modelo2)