# Diabetes-prediction-and-prevention

DIABETES PREDICTION MODEL 
                                 
(Using Python, HTML, CSS, ML)
From building model to implementing using it in backend of a website

BY 
BHAVLEEN KAUR
Data Scientist 
bhavleenkaur2021@gmail.com

ABSTRACT
Whole information is generated through health care trade and in straightforward wordings it’s tough to analyse or work on that to extract data mistreatment 
ancient ways. Data processing is that the concrete support pillar to any or all variety of analysation in any field. This Report conclude all criteria to 
know data processing techniques that helps in stunning variety once it involves hindrance and management. little doubt during this that variety of information scientists,
analysts square measure operating wonderful within the field  of data/knowledge mining to extract helpful information from the huge number of data. 
Machine learning is Associate in Nursing highlighted scientific field in information science that deals  with the ways in which during which machines learn from expertise. 
The aim of this project is to develop a system which might perform early prediction of polygenic disease for a patient with a better accuracy by combining the results of 
various machine learning techniques. This project aims to predict polygenic disease by ways including: supply regression, Random Forest Model. This project additionally aims
to propose a good technique for earlier detection of the polygenic ( in our case its diabetes ) disease.
Let’s begin!!
***************************************************************************************************************************************************************************
***************************************************************************************************************************************************************************
1.INTRODUCTION

1.1.Data Mining
Diabetes is neither systematic nor random , person to person it will vary. Trendy Technology and advanced ways square measure serving to doctors to cure diabetic patients. 
Combination of statistics, machine learning, computing and info technology. the information integration, selection , cleaning , transformation, mining ,pattern analysis , and
information presentation square measure the steps concerned in data processing. These ways not solely predicts even forestall the diseases.

1.2.Machine Learning
Machine learning is that the scientific field addressing the ways in which during which machines learn from expertise. For several scientists, the term “machine learning” 
is the image of the term “artificial intelligence”, on condition that the likelihood of learning is that the main characteristic of Associate in Nursing entity referred to
as intelligent within the broadest sense of the word. The aim of machine learning is that the construction of pc systems which will adapt and learn from their expertise.

1.3.Supervised Learning
In supervised learning, the system should “learn” inductively a operate referred to as target operate, which is Associate in Nursing expression of a model describing the 
data. The target operate is employed to predict the value of a variable, referred to as variable quantity or output variable, from a collection of variables, called 
independent variables or input variables or characteristics or options( can be said as features). The set of doable input values of the operate, i.e. its domain, are
called instances. Every case is represented by a collection of characteristics (attributes or features). A set of all cases, that the output variable price is thought, 
is called coaching information or training data or examples. So as to infer the best target operate, the educational system, given a training set, takes into thought 
different functions, referred to as hypothesis and denoted by h. In supervised learning, there are 2 styles of learning tasks: classification and regression. Classification 
models try and predict distinct categories, such as e.g. blood teams, whereas regression models predict numerical values. a number of the foremost common techniques are 
call Trees (DT), Rule Learning, and Instance based mostly Learning (IBL), like k-Nearest Neighbours (k-NN), Genetic Algorithms (GA), Artificial Neural Networks (ANN), 
and Support Vector Machines (SVM).

****************************************************************************************************************************************************************************
****************************************************************************************************************************************************************************
2.EXISTING METHOD
So talking about existing method, Clinical methods as well as biological preidictors were used to determine whether a patient is diabetic or not. Fasting glucose which 
also include its squared term is the major predictice of all the features of dataset we have. Second most preferable attribute is BMI i.e. Body mass index or we can use
more acuurately the waist circumference for the betterment of accuracy. In Men factors which matters for predictions are fasing glucose, BMI, smoking status, waist 
circumferences, and GGT. For ladies it can be summarised with the factors having fasting glucose, BMI, diabetes in the family, and triglycerides. 

Currently if we see in machine learning consideration so there are some algorithm like 
Naïve Bayes 
SVM( support vector machine )
J48
JRip
PNN(probabilistic neural network)
ANN these features/techniques are used get more and more accuracy in the model.

******************************************************************************************************************************************************************************
******************************************************************************************************************************************************************************
3.PROPOSED METHOD

3.1.Classification
Classification is one in all the foremost necessary decision making techniques in several globe problem. During this work, the main objective is to classify the info as 
diabetic or non-diabetic and improve the classification accuracy. For many classification downside, the upper variety of samples chosen however it doesn’t results in higher 
classification accuracy. In several cases, the performance of algorithm is high within the context of speed however the accuracy of knowledge classification is low. 
The main objective of our model is to realize high accuracy. Classification accuracy may be increase if we tend to use much of the information set for training and few 
data sets for testing. This survey has analyzed varied classification techniques for classification of diabetic and non-diabetic information. Thus, it's as certained that  
techniques like Support Vector Machine, Logistic Regression, and Artificial Neural Network are most suitable for implementing the polygenic disease prediction system.

3.2.Feature Selection
Feature choice strategies will scale back the quantity of attributes, which may avoid the redundant options. There are several feature choice strategies. During this study, 
we have a tendency to used PCA and minimum redundancy most connectedness (mRMR) to cut back the spatial property if needed.

3.3.Random Forest
RF could be a classification in order to use several call trees. This rule projected by Breiman (Breiman, 2001). RF could be a multifunctional machine learning technique. 
It will perform the tasks of prediction and regression. Additionally, RF is predicated on textile and it plays a vital role in ensemble machine learning (Breiman, 2001; Lin 
et al., 2014; Svetnik et al.,2015). RF has been utilized in many biomedicine analysis (Zhao et al., 2014; Liao dynasty etal., 2016).
RF generates several call trees, that is extremely totally different from call tree rule (Pal, 2005). once the RF is predicting a replacement object supported some attributes, 
every tree in RF can provide its own classification result and ‘vote,’ then the general output of the forest are going to be the biggest range of taxonomy. within the 
regression downside, the RF output is that the average price of output of all call trees.

3.4.Logistic Regression
In statistics provision regression may be a regression model wherever the variable quantity is categorical, particularly binary dependent variable-that is, where it will 
take solely 2 values, "0" and "1", which represent outcomes like pass/fail, win/lose, alive/dead or healthy/sick. Provision regression is employed in numerous fields, 
together with machine learning, most medical fields, and social sciences. For instance, the Trauma and Injury Severity Score (TRISS), which is widely accustomed predict 
mortality in dislocated patients, was originally developed using provision regression. Many different medical scales accustomed assess severity of a patient are developed 
with the help of provision regression. The technique also can be utilized in engineering, particularly for predicting the chance of failure of a given method, system or 
product. It is also utilized in promoting applications like prediction of a customer's propensity to get a product or halt a subscription. In political economy it is 
accustomed predict the chance of an individual's selecting to be in the labor pool, and a business application is getting ready to predict the chance of a house owner 
defaulting on a mortgage. Conditional random fields, associate extension of logistic regression to serial information, are utilized in natural language process. During 
this project, Logistic regression was accustomed predict whether or not a patient suffer from polygenic disease like diabetes, supported nine(9) discovered characteristics 
of the patient.

********************************************************************************************************************************************************************************
********************************************************************************************************************************************************************************
IMPLEMENTATION
Technically, I used JUPYTER NOTEBOOK, SPYDER, VISUAL STUDIO CODE to complete my whole project.
For Customer segmentation that was totally introductory I simply used NOTEBOOK to perform hierarical clustering and implementes some basic algorithm like apriori, 
kmeans on it.

For diabetes prediction I used JUPYTER NOTEBOOK for analysis, SPYDER for building model as well as for deployement purpose , VSCODE for designing HTML PAGES.
Particularly for analysation part I worked on from finding the problems to having a solution in my hand to make model better in any case.
Why I used Deployement?
I used Deployement for merging my model with front end of simply designed webpages in order to undersatand and show that the model is working perfectly 
fine when we give any type of input whether person is diabetic or not it is giving output according to the data given to the classifier.

Regarding implementation I explaines A to Z in my video briefly as well  as ppt which is attached to my linkedin.
Ps: I also implemented this same model with Luzhou dataset for my final satisfaction for the model and it working came up fine .

********************************************************************************************************************************************************************************
********************************************************************************************************************************************************************************
RESULTS
Diabetes mellitus may be a disease, which may cause several complications. A way to specifically predict and diagnose this unwellness with the help of machine 
learning is worthy enough to get an accurate way to predict and prevent. Consistent reading with all higher experiments, it is  tend to found the accuracy with 
PCA isn't much good, and also the results of the all options and with usage of  mRMR have higher results. The result, that solely used abstinence aldohexose or 
we can say glucose, features a higher performance particularly in Luzhou dataset( another dataset on which I used my proposed model) . It means the higher glucose 
is that the most significant index for predict, however solely using abstinence aldohexose cannot attain the most effective result, thus if wish to predict accurately, 
we'd like a lot of indexes. Additionally, by comparison the results of 3 classifications, we will notice there's not a lot of distinction among random forest, call tree 
and neural network, however random forests square measure clearly higher than the another classifiers in some ways. The most effective result for this type of diabetes 
dataset that I used is 0.8084, and also the best performance for Luzhou is 0.7721, which may indicate machine learning are often used for prediction polygenic disease, 
however finding appropriate attributes, classifier and data processing technique square measure  are vital. Due to information , we tend to cannot predict the sort of 
polygenic disease, thus in future we tend to aim to predicting form of polygenic disease and exploring the proportion of every indicator, which can improve the accuracy 
of predicting polygenic disease.

********************************************************************************************************************************************************************************
********************************************************************************************************************************************************************************
CONCLUSION
Machine learning has the good ability to revolutionize the polygenic disorder risk prediction with the help of advanced machine ways and availability of enormous quantity 
of medicine and dataset. Detection of polygenic disorder in its early stages is that the key for treatment. This work has described a machine learning approach to predicting 
diabetes levels. The technique may additionally facilitate researchers to develop associate in Nursing correct and effective tool that will reach at the table of clinicians 
to assist them make higher call regarding the sickness.

ps: go through ppt,video that is on linkedin for proper steps needed.
THANKYOU!!

