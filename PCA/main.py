import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error, classification_report, confusion_matrix, accuracy_score, recall_score,f1_score , classification_report
from sklearn import svm
from math import sqrt

warnings.filterwarnings('ignore')

gps1 = pd.read_csv("C:/Users/Admin/Downloads/google-play-store-apps/googleplaystore.csv") #change path
gps2 = pd.read_csv("C:/Users/Admin/Downloads/google-play-store-apps/googleplaystore_user_reviews.csv") #change path

gps = pd.merge(gps1, gps2, how='inner', on=['App'])
gps['Rating'] = gps['Rating'].astype(str).astype(float)

gps['Reviews'] = gps['Reviews'].apply(lambda x: x.replace('3.0M', '3000000')) #exists one review with 3.0M
gps['Reviews'] = gps['Reviews'].apply(lambda x: int(x))

gps = gps[gps['Installs'] != 'Free']
gps = gps[gps['Installs'] != 'Paid']
gps['Installs'] = gps['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
gps['Installs'] = gps['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
gps['Installs'] = gps['Installs'].apply(lambda x: int(x))


gps['Price'] = gps['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
gps['Price'] = gps['Price'].apply(lambda x: float(x))


#Size
def change_size(size):
    if 'M' in size:
        x = size[:-1]
        x = float(x)*1000000
        return(x)
    elif 'k' == size[-1:]:
        x = size[:-1]
        x = float(x)*1000
        return(x)
    else:
        return None

gps["Size"] = gps["Size"].map(change_size)


#Convert Categories to integer
CategoryString = gps["Category"]
categoryVal = gps["Category"].unique()
categoryValCount = len(categoryVal)
category_dict = {}
for i in range(0,categoryValCount):
    category_dict[categoryVal[i]] = i
gps["Category_c"] = gps["Category"].map(category_dict).astype(int)


#Convert Type to binary
def type_cat(types):
    if types == 'Free':
        return 0
    else:
        return 1

gps['Type'] = gps['Type'].map(type_cat)


#Convert content rating to integer
RatingL = gps['Content Rating'].unique()
RatingDict = {}
for i in range(len(RatingL)):
    RatingDict[RatingL[i]] = i
gps['Content Rating'] = gps['Content Rating'].map(RatingDict).astype(int)


#Convert genres to integer
GenresL = gps.Genres.unique()
GenresDict = {}
for i in range(len(GenresL)):
    GenresDict[GenresL[i]] = i
gps['Genres_c'] = gps['Genres'].map(GenresDict).astype(int)

#Convert App to integer
App = gps.App.unique()
AppDict = {}
for i in range(len(App)):
    AppDict[App[i]] = i
gps['App_c'] = gps['App'].map(AppDict).astype(int)

#Remove App because now have App_c which is integer
gps.drop(labels = ['Last Updated','Current Ver','Android Ver','Genres','Category','App','Translated_Review'], axis = 1, inplace = True)


def sentiment_convert(sentiment):
    if sentiment == 'Positive':
        return 0
    elif sentiment == 'Negative':
        return 1
    elif sentiment == 'Neutral':
        return 2
    else:
        return None

gps['Sentiment'] = gps['Sentiment'].map(sentiment_convert)


###FILTER METHOD####
# Reviews feature is correlation
# plt.figure(figsize=(12,10))
# cor = gps.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()


#################LISTWISE######################
gps.dropna(subset=['Sentiment'],how='any',inplace=True)
gps.dropna(subset=['Sentiment_Polarity'],how='any',inplace=True)
gps.dropna(subset=['Sentiment_Subjectivity'],how='any',inplace=True)
gps.dropna(subset=['Size'],how='any',inplace=True)
gps.dropna(subset=['Rating'],how='any',inplace=True)
gps.info()

####RFE (Recursive Feature Elimination) WITH LISTWISE####
# model = LinearRegression()
# cols = list(['Rating', 'Reviews', 'Size', 'Type', 'Price', 'Sentiment', 'Sentiment_Polarity', 'Sentiment_Subjectivity', 'Category_c', 'Content_Rating_c', 'Genres_c'])
# newGps = gps.drop(labels = ['Installs'], axis = 1);
# X = np.array(newGps.iloc[:, :]);
# y = np.array(gps['Installs']);
# #no of features
# nof_list=np.arange(1,12)
# high_score=0
# #Variable to store the optimum features
# nof=0
# score_list =[]
# for n in range(len(nof_list)):
#     X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
#     model = LinearRegression()
#     rfe = RFE(model,nof_list[n])
#     X_train_rfe = rfe.fit_transform(X_train,y_train)
#     X_test_rfe = rfe.transform(X_test)
#     model.fit(X_train_rfe,y_train)
#     score = model.score(X_test_rfe,y_test)
#     score_list.append(score)
#     if(score>high_score):
#         high_score = score
#         nof = nof_list[n]
# print("Optimum number of features: %d" %nof)
# print("Score with %d features: %f" % (nof, high_score))



# model = LinearRegression()
# #Initializing RFE model
# rfe = RFE(model, 11) #optimum number
# #Transforming data using RFE
# X_rfe = rfe.fit_transform(X,y)
# #Fitting the data to model
# model.fit(X_rfe,y)
# temp = pd.Series(rfe.support_,index = cols)
# selected_features_rfe = temp[temp==True].index
# print(selected_features_rfe)

########Embedded Method (Lasso) WITH LISTWISE#########
cols = list(['Rating', 'Reviews', 'Size', 'Type', 'Price', 'Sentiment', 'Sentiment_Polarity', 'Sentiment_Subjectivity', 'Category_c', 'Content_Rating_c', 'Genres_c'])
newGps = gps.drop(labels = ['Installs'], axis = 1);
X = np.array(newGps.iloc[:, :]);
y = np.array(gps['Installs']);
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = cols)
imp_coef = coef.sort_values()
plt.figure(figsize=(12,10))
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()

################PAIRWISE#######################
# gps.info()
# gps['Sentiment'].mean()
# gps['Sentiment_Polarity'].mean()
# gps['Sentiment_Subjectivity'].mean()
# gps['Size'].mean()
# gps['Rating'].mean()
# gps.info()
################Dropping Entire Column##########
# gps.info()
# gps.drop(labels = ['Sentiment', 'Sentiment_Polarity', 'Sentiment_Subjectivity', 'Size', 'Rating'], axis = 1, inplace = True)
# gps.info()
##################MEAN#########################
# gps.info()
# mean_imputer = SimpleImputer(strategy='mean')# strategy can also be mean or median
# gps.iloc[:,:] = mean_imputer.fit_transform(gps)
# gps.info()

##################MEDIAN#########################
gps.info()
mean_imputer = SimpleImputer(strategy='median')# strategy can also be mean or median
gps.iloc[:,:] = mean_imputer.fit_transform(gps)
gps.info()

##################MODE############################
# gps.info();
# print(gps['Sentiment'].mode()[0])
# print(gps['Sentiment_Polarity'].mode()[0])
# print(gps['Sentiment_Subjectivity'].mode()[0])
# print(gps['Size'].mode()[0])
# print(gps['Rating'].mode()[0])
# gps['Sentiment'] = gps['Sentiment'].fillna(gps['Sentiment'].mode()[0])
# gps['Sentiment_Polarity'] = gps['Sentiment_Polarity'].fillna(gps['Sentiment_Polarity'].mode()[0])
# gps['Sentiment_Subjectivity'] = gps['Sentiment_Subjectivity'].fillna(gps['Sentiment_Subjectivity'].mode()[0])
# gps['Size'] = gps['Size'].fillna(gps['Size'].mode()[0])
# gps['Rating'] = gps['Rating'].fillna(gps['Rating'].mode()[0])

##################CONSTANT#########################
# train_constant = gps.copy()
# gps.info()
# mean_imputer = SimpleImputer(strategy='constant')# strategy can also be mean or median
# gps.iloc[:,:] = mean_imputer.fit_transform(gps)
# gps.info()

#################LOCF######################
# gps.info();
# gps.fillna(method='ffill',inplace=True)
# gps.info()

#################NOCB######################
# gps.info()
# gps.fillna(method='bfill',inplace=True)
# gps.dropna(subset=['Size'],how='any',inplace=True)
# gps.info()

#################Linear interpolation######################
# train_li = gps.copy()
# gps.info()
# gps.interpolate(limit_direction="both",inplace=True)
# gps.info()

###################KNN####################

# train_knn = gps.copy(deep=True)
# imputer = KNNImputer(n_neighbors=3, weights='uniform');
# train_knn['Sentiment'] = imputer.fit_transform(train_knn[['Sentiment']])


##################MICE####################
# gps.info()
# mice_imputer = IterativeImputer()
# gps['Sentiment'] = mice_imputer.fit_transform(gps[['Sentiment']])
# gps['Sentiment_Polarity'] = mice_imputer.fit_transform(gps[['Sentiment_Polarity']])
# gps['Sentiment_Subjectivity'] = mice_imputer.fit_transform(gps[['Sentiment_Subjectivity']])
# gps['Size'] = mice_imputer.fit_transform(gps[['Size']])
# gps['Rating'] = mice_imputer.fit_transform(gps[['Rating']])
# gps.info()



####KNeighborsClassifier####
newGps = gps.drop(labels = ['Installs'], axis = 1)
X = np.array(newGps.iloc[:, :]);
Y = np.array(gps['Installs']);

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=10);

k_values = [i for i in range(1,100,2)]
k_acc_scores = [];

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    cv_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    k_acc_scores.append(cv_scores.mean())

optimal_k = k_values[k_acc_scores.index(max(k_acc_scores))]
scores = []
for n in k_values:
    knn.set_params(n_neighbors=n)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
print(scores) #find optimal k in range(1,100)

plt.plot(k_values,k_acc_scores)
plt.xlabel("Vrijednost k")
plt.ylabel('Koeficijent determinacije')
plt.show()


###########ROOT_MEAN_SQUARED_ERROR#######
newGps = gps.drop(labels = ['Installs'], axis = 1)
X = np.array(newGps.iloc[:, :]);
Y = np.array(gps['Installs']);

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=10);

knn = KNeighborsClassifier(n_neighbors=9) #depends of optimal k

knn.fit(X_train, y_train)
pred_y = knn.predict(X_test)

print(sqrt(mean_squared_error(y_test,pred_y)))



############SVM#####
newGps = gps.drop(labels = ['Installs'], axis = 1)
X = np.array(newGps.iloc[:, :]);
y = np.array(gps['Installs']);
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

sv1 = svm.SVC(kernel='rbf')
sv1.fit(X_train,y_train)
result = sv1.predict(X_test)


print(accuracy_score(y_test, result))
print(sqrt(mean_squared_error(y_test,result)))

