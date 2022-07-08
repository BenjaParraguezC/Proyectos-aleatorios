#Cargamos la base de datos
setwd("~/Trabajos R/Trabajo Mineria")
DataTrain <- read.csv(file = 'trainData.csv')
DataTest<-read.csv(file = 'evalData.csv')

DataTrain #La observamos

#Verificamos si tenemos valores NA o bien valores Null

sum(is.na(DataTrain))
sum(is.na(DataTest))

sum(is.null(DataTrain))
sum(is.null(DataTest))


#Como podemos notar no existen valores NA en los dataframes tanto para Train como para test.

#Transformamos los valores de no show , a valores binarios.
DataTrain$noshow <-  replace(DataTrain$noshow, DataTrain$noshow < 4, 0)
DataTrain$noshow <-  replace(DataTrain$noshow, DataTrain$noshow > 3, 1)

library(dplyr)
#Comenzamos a eliminar columnas que no contienen informaci�n relevante para el modelo, como lo son la ID, Date, Origin, destination
#Departure time y fligth_number, mediante la libreria dplyr, como tambi�n revenues_usd.

DataTrain <- select(DataTrain,-id,-origin,-destination,-departure_time,-date,-fligth_number,-revenues_usd)

#Realizamos el mismo filtrado para nuestros datos de Test (debido a que tienen la mismas columnas.)

DataTest <- select(DataTest,-id,-origin,-destination,-departure_time,-date,-fligth_number,-revenues_usd)

#Eliminamos las filas repetidas con dplyr
distinct(DataTrain)
distinct(DataTest)

#Realizamos boxplot para cada una de las variables que nos permitan identificar los posibles outliers.

summary(DataTrain) #Permite entender como se comportan las variables.

boxplot(DataTrain$distance, horizontal = T)
boxplot(DataTrain$denied_boarding, horizontal = T)
boxplot(DataTrain$pax_midlow, horizontal = T)
boxplot(DataTrain$pax_high, horizontal = T)

boxplot(DataTrain$pax_midhigh, horizontal = T)
boxplot(DataTrain$pax_low, horizontal = T)
boxplot(DataTrain$pax_freqflyer, horizontal = T)

boxplot(DataTrain$group_bookings, horizontal = T)
boxplot(DataTrain$out_of_stock, horizontal = T)

boxplot(DataTrain$dom_cnx, horizontal = T)

boxplot(DataTrain$int_cnx, horizontal = T)
boxplot(DataTrain$p2p, horizontal = T)

boxplot(DataTrain$capacity, horizontal = T)
boxplot(DataTrain$bookings, horizontal = T)


#Analizando cada boxplot, identificamos que no se identifican posiblex outliers o puntos at�picos de caracter menor a cero


#Analizando la matriz de correlaci�n, tambien identificamos posibles variables significativas.
library(corrplot)

mcorr_data <- cor(DataTrain)
corrplot(mcorr_data,method="circle")


#Ahora con la libreria Caret, podemos identificar las variables que m�s aportan para la predicci�n del noshow.
#Esto debido a que se entrena un modelo utilizando rpart.
#Esto se demora, asi que hay que tener paciencia - menos de 1 hora eso si.
library(caret)
set.seed(100)
rPartMod <- train(noshow ~ ., data=DataTrain, method="rpart")
rpartImp <- varImp(rPartMod)
print(rpartImp)

#Tomamos las variables mas significativas y volvemos a filtrar.


#Realizamos el mismo filtrado para nuestros datos de Test (debido a que tienen la mismas columnas.)

DataTest <- select(DataTest,capacity,distance,pax_low,p2p,int_cnx,group_bookings,pax_midlow,bookings,dom_cnx)

DataTrain <- select(DataTrain,capacity,distance,pax_low,p2p,int_cnx,group_bookings,pax_midlow,bookings,dom_cnx,noshow)

DataTrain <- scale(DataTrain) #Escalamos los datos, para que las variables trabajen en la misma m�trica.
DataTrain <-  as.data.frame(DataTrain) #Lo transformamos a dataframe.


#--------------------------------------------------------------- Entrenamiento

library(caTools)
set.seed(0) #Es necesario colocar una semilla inicial para poder predecir.
split = sample.split(DataTrain$noshow, SplitRatio = 0.7) #Dividimos la data en 70-30, como tambien mencionamos el target.
training_set = subset(DataTrain, split == TRUE) #separamos el train 
test_set = subset(DataTrain, split == FALSE)



library(e1071)
#Modelo Naive Bayes.

model_naive <- naiveBayes(x = training_set %>% select(-noshow), #predictor
                          y = training_set$noshow, #target
                          laplace = 1)  #Ajustamos los hiperparametros.

pred_label_naive <- predict(model_naive, test_set, type = "class") #Predecimos utilizando naivae bayes
head(data.frame(actual = test_set$noshow, prediction = pred_label_naive)) #Visualizamos los resultados

levels(pred_label_naive)[1] <- 0
levels(pred_label_naive)[2] <- 1

#Es necesario realizar un par de ajustes para poder visualizar de mejor manera.
test_set$noshow <- ifelse (test_set$noshow > 0.5,1,0)

#Realizamos la matriz de confusion
library(caret)
mat1 <- confusionMatrix(data = pred_label_naive, reference = as.factor(test_set$noshow), positive = "1", mode = "everything")
mat1


#10 Variables, 0.7 split, escalado los datos, no aplicar factor, F1-0.7619, Accuracy 0.6468, laplace 1.
#10 Variables, 0.75 split, escalado los datos, no aplicar factor, F1-0.7607, Accuracy 0.6461, laplace 1.
#10 Variables, 0.75 split, escalado los datos, no aplicar factor, F1-0.7615, Accuracy 0.6463 , laplace 3.
#10 Variables, 0.7 split, escalado los datos, no aplicar factor, F1-0.7619, Accuracy 0.6463 , laplace 3.

#15 Variables, 0.7 split, escalado los datos, no aplicar factor, F1-0.7750 , Accuracy 0.6504 , laplace 3.
#15 Variables, 0.65 split, escalado los datos, no aplicar factor, F1-0.7751 , Accuracy 0.6507 , laplace 3.
#15 Variables, 0.7 split, escalado los datos, no aplicar factor, F1-0.7750 , Accuracy 0.6504 , laplace 1.


#Curva ROC- Area bajo la curva ROC.

library(pROC)

prob_survive <- predict(model_naive, test_set, type = "raw")

levels(prob_survive)[1] <- 0
levels(prob_survive)[2] <- 1

#Es necesario realizar un par de ajustes para poder visualizar de mejor manera.

# prepare dataframe for  ROC
data_roc <- data.frame(prob = prob_survive[,2], # probability of positive class(survived)
                       labels = as.numeric(test_set$noshow == "1")) #get the label as the test data who survived
head(data_roc)

naive_roc <- ROCR::prediction(data_roc$prob, data_roc$labels) 

plot(performance(naive_roc, "tpr", "fpr"), #tpr = true positive rate, fpr = false positive rate
     main = "ROC")
abline(a = 0, b = 1)

auc_n <- performance(naive_roc, measure = "auc")
auc_n@y.values #Test regular.

#----------------------------------------------------------------------------------------------------

#Arbol de decision - Es necesario "resetear" nuestro training_set - test_Set , sino trabajar� con los datos anteriores.

library(caTools)
set.seed(0) #Es necesario colocar una semilla inicial para poder predecir.
split = sample.split(DataTrain$noshow, SplitRatio = 0.7) #Dividimos la data en 70-30, como tambien mencionamos el target.
training_set = subset(DataTrain, split == TRUE) #separamos el train 
test_set = subset(DataTrain, split == FALSE)

library(partykit) 

model_dt <- ctree(noshow~ .,training_set)

pred_test_dt <- predict(model_dt, newdata = test_set, 
                        type = "response")
head(data.frame(actual = test_set$noshow, prediction = pred_test_dt))

pred_test_dt <- ifelse (pred_test_dt > 0,1,0)
test_set$noshow <- ifelse (test_set$noshow > 0,1,0)


mat2 <- confusionMatrix(data = as.factor(pred_test_dt), reference = as.factor(test_set$noshow), mode ='everything')
mat2

#7 variables, 0.7 split, F1 :0.5239,  Accuracy : 0.6193 - datos escalables.

#Aplicaci�n de ROC-AUC

prob_survive_dt <- predict(model_dt, test_set, type = "node")

prob_survive_dt[,2]

data_roc1 <- data.frame(prob = prob_survive_dt[2], # probability of positive class(survived)
                        labels = as.numeric(test_set$noshow == 1)) #get the label as the test data who survived

dt_roc <- ROCR::prediction(data_roc1$prob, data_roc1$labels) 

# ROC curve
plot(performance(dt_roc, "tpr", "fpr"), #tpr = true positive rate, fpr = false positive rate
     main = "ROC")
abline(a = 0, b = 1)

dt_auc <- performance(dt_roc, measure = "auc")
dt_auc@y.values


#--------------------------------------------------------------------------------------------------------
 

library(caTools)
set.seed(0)
split = sample.split(DataTrain$noshow, SplitRatio = 0.7)
training_set = subset(DataTrain, split == TRUE) #separamos el train 
test_set = subset(DataTrain, split == FALSE)

  
#Regresion logistica.   

regresion_log <- lm(noshow ~ ., data=training_set)
predict_log <- predict(regresion_log, test_set, type = "response")
head(data.frame(actual = test_set$noshow, prediction = predict_log))

predict_log <- ifelse (predict_log > 0.5,1,0)
test_set$noshow <- ifelse (test_set$noshow > 0.5,1,0)

conf_log <- confusionMatrix(as.factor(test_set$noshow), as.factor(predict_log), mode = 'everything')
conf_log


#10 Variables, 0.7 split, escalado los datos,F1-0.51430  , Accuracy 0.3513
#15 Variables, 0.7 split, escalado los datos,F1-0.51442  , Accuracy 0.3519   

#-------------------------------------------------------

#Intento de Knn, no me da por temas de rendimiento.


library(caTools)
set.seed(0)
split = sample.split(DataTrain$noshow, SplitRatio = 0.7)
training_set = subset(DataTrain, split == TRUE) #separamos el train 
test_set = subset(DataTrain, split == FALSE)


library(kknn)


control <- trainControl(method = "cv",   # cross validation
                        number = 10,     # 10 k-folds or number 
)



fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  repeats = 10,
  classProbs = TRUE,
  summaryFunction = multiClassSummary)

knncv <- train(noshow~., 
               data=training_set,
               method = "knn", 
               trControl=control)


print(knncv)


PredictionKnn <- predict(knn.fit, newdata =test_set)

head(data.frame(actual = test_set$noshow, prediction = PredictionKnn))

confusionMatrix(PredictionKnn, as.factor(test_set$noshow))



#-------------------------------------------------------

#Intento de SVM, problema: los vectores de svm pueden llegar a ser muy grandes en tama�o > 10Gb, es necesario reajustar
#demasiado los datos del procesamiento inicial, como tambien la disminuci�n de las variables.


library(caTools)
set.seed(0)
split = sample.split(DataTrain$noshow, SplitRatio = 0.7)
training_set = subset(DataTrain, split == TRUE) #separamos el train 
test_set = subset(DataTrain, split == FALSE)


library(e1071)
regressor_svr = svm(formula = factor(training_set$noshow) ~ .,
                    data = training_set,
                    type = 'C-classification',
                    kernel = 'linear')

y_pred_svr = predict(regressor_svr,  newdata = test_set) #

confusionMatrix(factor(y_pred_svr), 
                factor(test_set$TARGET))

#----------------------------------------------------------


#Seleccion modelo: seleccionamos el modelo naive bayes por que tiene un buen rendimiento en las m�tricas de desempe�o
#(B�sicamente se comparan los modelos mediante F1-Score)
#como tambien un buen valor en el area bajo la curva ROC. #Profundizar


#-----------------------------------------------------

#Predecimos con nuestro modelo, con el dataset final.
Resultado <- predict(model_naive, DataTest)

Resultado

levels(Resultado) #Es necesario cambiar los valores de los factores a 0 y 1.

levels(Resultado)[1] <- 0
levels(Resultado)[2] <- 1

Resultado

Resultado_Final <- as.data.frame(Resultado) #Transformamos a dataframe para posteriormente enviarlo a un .csv

library(readr)
write.table(Resultado_Final,col.names=FALSE, file= 'Resultado.csv',row.names = FALSE)
