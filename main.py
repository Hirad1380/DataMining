import matplotlib
import pandas as pd
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import skfuzzy as fuzz

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, adjusted_rand_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import silhouette_score
from sklearn.svm import SVC
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics









df = pd.read_csv("C:/Users/acer/Desktop/diabetes_prediction_dataset.csv")




# ---------------------------------------Display the first 5 data records----------------------------
pd.set_option('display.max_columns', None)
print(df.head())


# ---------------------------------------Basic data description----------------------------
# pd.set_option('display.max_columns', None)
# print(df.describe())



# ---------------------------------------Showing other genders----------------------------
# x = df[df['gender'] == 'Other']
# print(x.value_counts())
# print(x.index)


# ---------------------------------------Gender chart with Other gender----------------------------
# gender_counts = df['gender'].value_counts()
# plt.bar(gender_counts.index, gender_counts.values)
# print(gender_counts.values)
# print(gender_counts.index)
# plt.show()


# ---------------------------------------Remove Other Gender----------------------------
df = df[df['gender'] != 'Other']



# ---------------------------------------Checking that all other genders are cleared----------------------------
x = df[df['gender'] == 'Other']
print(x.value_counts())
print(x.index)





# ---------------------------------------Number of duplicating rows----------------------------
duplicate_rows_data = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_data.shape)


# ---------------------------------------Clear the number of duplicate rows----------------------------
df = df.drop_duplicates()



# ---------------------------------------Number of specific data for each feature----------------------------
for column in df.columns:
    num_distinct_values = len(df[column].unique())
    print(f"{column} : {num_distinct_values} distinct values")


# ---------------------------------------Number of null data----------------------------
print(df.isnull().sum())


# ---------------------------------------Gender chart without Other gender----------------------------
# gender_counts = df['gender'].value_counts()
# plt.bar(gender_counts.index, gender_counts.values, color=['pink', 'blue'])
# print(gender_counts.values)
# print(gender_counts.index)
# plt.savefig('plot1')


# ---------------------------------------Age chart----------------------------
# plt.hist(df['age'], bins=100, edgecolor='black')
# plt.title('Age Distribution')
# plt.xlabel('Age')
# plt.ylabel('Count')
# plt.savefig('plot2')




# ---------------------------------------Number of data for available ages----------------------------
# age_counts = df['age'].value_counts().all
# pd.set_option('display.max_rows', None)
# print(age_counts)


# ---------------------------------------BMI chart----------------------------
# sns.histplot(df['bmi'], bins=45, kde=True)
# plt.title('BMI Distribution')
# plt.savefig('plot3')



# ---------------------------------------Hypertension chart----------------------------
# for col in ['hypertension']:
#     sns.countplot(x=col, data=df)
#     plt.title(f'{col} Distribution')
#     plt.savefig('plot4')



# ---------------------------------------Heart Disease chart----------------------------
# for col in ['heart_disease']:
#     sns.countplot(x=col, data=df)
#     plt.title(f'{col} Distribution')
#     plt.savefig('plot5')



# ---------------------------------------Diabetes chart----------------------------
# for col in ['diabetes']:
#     sns.countplot(x=col, data=df)
#     plt.title(f'{col} Distribution')
#     plt.savefig('plot6')



# ---------------------------------------Smoking History chart and number of each category----------------------------
# sns.countplot(x='smoking_history', data=df)
# smoking_history = df['smoking_history'].value_counts()
# print(smoking_history)
# plt.title('Smoking History Distribution')
# plt.savefig('plot7')



# ---------------------------------------BMI vs Diabetes chart----------------------------
# sns.boxplot(x='diabetes', y='bmi', data=df)
# plt.title('BMI vs Diabetes')
# plt.savefig('plot8')


# ---------------------------------------Age vs Diabetes chart----------------------------
# sns.boxplot(x='diabetes', y='age', data=df)
# plt.title('Age vs Diabetes')
# plt.savefig('plot9')



# ---------------------------------------Gender vs Diabetes chart----------------------------
# sns.countplot(x='gender', hue='diabetes', data=df)
# plt.title('Gender vs Diabetes')
# plt.savefig('plot10')



# ---------------------------------------HbA1c level vs Diabetes chart----------------------------
# sns.boxplot(x='diabetes', y='HbA1c_level', data=df)
# plt.title('HbA1c level vs Diabetes')
# plt.savefig('plot11')


# ---------------------------------------Blood Glucose Level vs Diabetes chart----------------------------
# sns.boxplot(x='diabetes', y='blood_glucose_level', data=df)
# plt.title('Blood Glucose Level vs Diabetes')
# plt.savefig('plot12')






# ---------------------------------------Types of graphs (high execution time)----------------------------
# sns.pairplot(df, hue='diabetes')
# plt.savefig('plot13')



# ---------------------------------------Age vs BMI chart----------------------------
# sns.scatterplot(x='age', y='bmi', hue='diabetes', data=df)
# plt.title('Age vs BMI')
# plt.savefig('plot14')



# ---------------------------------------Diabetes chart----------------------------
# sns.countplot(x='diabetes', data=df)
# plt.title('Diabetes Distribution')
# plt.savefig('plot15')



# ---------------------------------------BMI vs Diabetes split by Gender chart----------------------------
# sns.violinplot(x='diabetes', y='bmi', hue='gender', split=True, data=df)
# plt.title('BMI vs Diabetes split by Gender')
# plt.savefig('plot16')


# ---------------------------------------BMI Distribution by Diabetes Status and Gender chart----------------------------
# sns.boxplot(x='diabetes', y='bmi', hue='gender', data=df)
# plt.title('BMI Distribution by Diabetes Status and Gender')
# plt.savefig('plot17')


# ---------------------------------------Age Distribution by Diabetes Status and Gender chart----------------------------
# sns.boxplot(x='diabetes', y='age', hue='gender', data=df)
# plt.title('Age Distribution by Diabetes Status and Gender')
# plt.savefig('plot18')


# ---------------------------------------Reducing the number of smoking history categories----------------------------
def recategorize_smoking(smoking_status):
    if smoking_status in ['never', 'No Info']:
        return 'non-smoker'
    elif smoking_status == 'current':
        return 'current'
    elif smoking_status in ['ever', 'former', 'not current']:
        return 'past_smoker'
df['smoking_history'] = df['smoking_history'].apply(recategorize_smoking)
print(df['smoking_history'].value_counts())


# ---------------------------------------Encoding Gender and smoking_history----------------------------
data = df.copy()
def perform_one_hot_encoding(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df.drop(column_name, axis=1), dummies], axis=1)
    return df

data = perform_one_hot_encoding(data, 'gender')
data = perform_one_hot_encoding(data, 'smoking_history')




# ---------------------------------------Correlation Matrix Heatmap chart----------------------------
# correlation_matrix = data.corr()
# plt.figure(figsize=(21, 21))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
# plt.title("Correlation Matrix Heatmap")
# plt.savefig('plot19')



# ---------------------------------------Correlation with Diabetes chart----------------------------
# corr = data.corr()
# plt.figure(figsize=(17,10))
# target_corr = corr['diabetes'].drop('diabetes')
# target_corr_sorted = target_corr.sort_values(ascending=False)
# sns.set(font_scale=0.9)
# sns.set_style("white")
# sns.set_palette("PuBuGn_d")
# sns.heatmap(target_corr_sorted.to_frame(), cmap="coolwarm", annot=True, fmt='.2f')
# plt.title('Correlation with Diabetes')
# plt.savefig('plot20')




smoking_history_dict = {'non-smoker':0, 'past_smoker':1, 'current':2}
df['smoking_history'] = df.smoking_history.map(smoking_history_dict)

gender_dict = {'Female':0, 'Male':1}
df['gender'] = df.gender.map(gender_dict)

pd.set_option('display.max_columns', None)
print(df.head())

print(df.isnull().sum())









# ---------------------------------------Decision Tree----------------------------
# X = df[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
#        'bmi', 'HbA1c_level', 'blood_glucose_level']]
# y = df["diabetes"]
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# Tree1 = DecisionTreeClassifier().fit(x_train, y_train)
# y_predTree = Tree1.predict(x_test)
# cm = confusion_matrix(y_test, y_predTree)
# cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
#                                  index=['Predict Positive:1', 'Predict Negative:0'])
#
# sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='mako')
# plt.title('Decision Tree')
# plt.show()
# print(classification_report(y_test,y_predTree))
# accTree = accuracy_score(y_test, y_predTree)
# print("Accuracy : " ,  accTree)
# plt.figure(figsize=(15, 10))
# plot_tree(Tree1, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], filled=True, rounded=True)
# plt.savefig('plot100', dpi=1000)
# tree_rules = export_text(Tree1, feature_names=list(X.columns))
# print("Decision Tree Rules:")
# print(tree_rules)
# y_prob = Tree1.predict_proba(x_test)[:, 1]
# fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# roc_auc = auc(fpr, tpr)
# print(f"Area Under the ROC Curve (AUROC): {roc_auc:.2f}")
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()



# ---------------------------------------C4.5 Decision Tree----------------------------
# X = df[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
# y = df["diabetes"]
#
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# Tree1 = DecisionTreeClassifier(criterion="entropy").fit(x_train, y_train)
# y_predTree = Tree1.predict(x_test)
# cm = confusion_matrix(y_test, y_predTree)
# cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
#                                  index=['Predict Positive:1', 'Predict Negative:0'])
#
# sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='mako')
# plt.title('C4.5 Decision Tree')
# plt.show()
# print(classification_report(y_test,y_predTree))
# accTree = accuracy_score(y_test, y_predTree)
# print("Accuracy C4.5 : ", accTree)
# plt.figure(figsize=(15, 10))
# plot_tree(Tree1, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], filled=True, rounded=True)
# tree_rules = export_text(Tree1, feature_names=list(X.columns))
# plt.savefig('plot101', dpi=1000)
# print("C4.5 Decision Tree Rules:")
# print(tree_rules)
# y_prob = Tree1.predict_proba(x_test)[:, 1]
# fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# roc_auc = auc(fpr, tpr)
# print(f"Area Under the ROC Curve (AUROC): {roc_auc:.2f}")
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()



# ---------------------------------------CART Decision Tree----------------------------
# X = df[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
#        'bmi', 'HbA1c_level', 'blood_glucose_level']]
# y = df["diabetes"]
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# Tree1 = DecisionTreeClassifier(criterion='gini').fit(x_train, y_train)
# y_predTree = Tree1.predict(x_test)
# cm = confusion_matrix(y_test, y_predTree)
# cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
#                                  index=['Predict Positive:1', 'Predict Negative:0'])
#
# sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='mako')
# plt.title('CART Decision Tree')
# plt.show()
# print(classification_report(y_test,y_predTree))
# accTree = accuracy_score(y_test, y_predTree)
# print("Accuracy CART: " ,  accTree)
# plt.figure(figsize=(15, 10))
# plot_tree(Tree1, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], filled=True, rounded=True)
# plt.savefig('plot102', dpi=1000)
# tree_rules = export_text(Tree1, feature_names=list(X.columns))
# print("Decision Tree Rules:")
# print(tree_rules)
# y_prob = Tree1.predict_proba(x_test)[:, 1]
# fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# roc_auc = auc(fpr, tpr)
# print(f"Area Under the ROC Curve (AUROC): {roc_auc:.2f}")
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()




# ---------------------------------------SVM----------------------------
scaler = StandardScaler()
scaled = scaler.fit(df.drop('diabetes',axis=1)).transform(df.drop('diabetes',axis=1))
df_scaled = pd.DataFrame(scaled, columns=df.columns[:-1])
df_scaled.head()
x = df_scaled
y = df['diabetes']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# svc=SVC(probability=True)
# svc.fit(x_train, y_train)
# y_pred=svc.predict(x_test)
# print('Model accuracy : {0:0.7f}'. format(accuracy_score(y_test, y_pred)))
# cm = confusion_matrix(y_test, y_pred)
# cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
#                                  index=['Predict Positive:1', 'Predict Negative:0'])

# sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='mako')
# plt.title('Model')
# plt.show()
# print(classification_report(y_test,y_pred))


# y_prob = svc.predict_proba(x_test)[:, 1]
# fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# roc_auc = auc(fpr, tpr)
# print(f"Area Under the ROC Curve (AUROC): {roc_auc:.2f}")
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()



# ---------------------------------------linear kernel SVM----------------------------
# linear_classifier= SVC(kernel='linear', probability=True).fit(x_train,y_train)
# y_pred = linear_classifier.predict(x_test)
# print('linear kernel : {0:0.7f}'. format(accuracy_score(y_test, y_pred)))
# cm = confusion_matrix(y_test, y_pred)
# cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
#                                  index=['Predict Positive:1', 'Predict Negative:0'])

# sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='mako')
# plt.title('linear kernel with SVM')
# plt.show()
# print(classification_report(y_test,y_pred))
#
# y_prob = linear_classifier.predict_proba(x_test)[:, 1]
# fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# roc_auc = auc(fpr, tpr)
# print(f"Area Under the ROC Curve (AUROC): {roc_auc:.2f}")
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()



# ---------------------------------------RBF kernel Nonlinear SVM----------------------------
# rbf_classifier = SVC(kernel='rbf', probability=True)
# rbf_classifier.fit(x_train, y_train)
# y_pred_rbf = rbf_classifier.predict(x_test)
# print('Nonlinear SVM with RBF kernel accuracy: {0:0.7f}'.format(accuracy_score(y_test, y_pred_rbf)))
# cm_rbf = confusion_matrix(y_test, y_pred_rbf)
# cm_matrix_rbf = pd.DataFrame(data=cm_rbf, columns=['Actual Positive:1', 'Actual Negative:0'],
#                              index=['Predict Positive:1', 'Predict Negative:0'])
# sns.heatmap(cm_matrix_rbf, annot=True, fmt='d', cmap='mako')
# plt.title('Nonlinear SVM with RBF kernel')
# plt.show()
# print(classification_report(y_test, y_pred_rbf))
#
# y_prob = rbf_classifier.predict_proba(x_test)[:, 1]
# fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# roc_auc = auc(fpr, tpr)
# print(f"Area Under the ROC Curve (AUROC): {roc_auc:.2f}")
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()


# ---------------------------------------Polynomial kernel Nonlinear SVM----------------------------
# poly_classifier = SVC(kernel='poly', degree=3, probability=True)
# poly_classifier.fit(x_train, y_train)
# y_pred_poly = poly_classifier.predict(x_test)
# print('Nonlinear SVM with Polynomial kernel accuracy: {0:0.7f}'.format(accuracy_score(y_test, y_pred_poly)))
# cm_poly = confusion_matrix(y_test, y_pred_poly)
# cm_matrix_poly = pd.DataFrame(data=cm_poly, columns=['Actual Positive:1', 'Actual Negative:0'],
#                               index=['Predict Positive:1', 'Predict Negative:0'])
# sns.heatmap(cm_matrix_poly, annot=True, fmt='d', cmap='mako')
# plt.title('Nonlinear SVM with Polynomial kernel')
# plt.show()
# print(classification_report(y_test, y_pred_poly))
#
# y_prob = poly_classifier.predict_proba(x_test)[:, 1]
# fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# roc_auc = auc(fpr, tpr)
# print(f"Area Under the ROC Curve (AUROC): {roc_auc:.2f}")
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()




# ---------------------------------------Sigmoid kernel SVM----------------------------
# sigmoid_classifier = SVC(kernel='sigmoid', probability=True)
# sigmoid_classifier.fit(x_train, y_train)
# y_pred_sigmoid = sigmoid_classifier.predict(x_test)
# print('SVM with Sigmoid kernel accuracy: {0:0.3f}'.format(accuracy_score(y_test, y_pred_sigmoid)))
# cm_sigmoid = confusion_matrix(y_test, y_pred_sigmoid)
# cm_matrix_sigmoid = pd.DataFrame(data=cm_sigmoid, columns=['Actual Positive:1', 'Actual Negative:0'],
#                                   index=['Predict Positive:1', 'Predict Negative:0'])
# sns.heatmap(cm_matrix_sigmoid, annot=True, fmt='d', cmap='mako')
# plt.title('SVM with Sigmoid kernel')
# plt.show()
# print(classification_report(y_test, y_pred_sigmoid))
#
# y_prob = sigmoid_classifier.predict_proba(x_test)[:, 1]
# fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# roc_auc = auc(fpr, tpr)
# print(f"Area Under the ROC Curve (AUROC): {roc_auc:.2f}")
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()





# ---------------------------------------KNN----------------------------
# X = df[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
#        'bmi', 'HbA1c_level', 'blood_glucose_level']]
# y = df["diabetes"]
#
# scaler = StandardScaler()
# scaled = scaler.fit(df.drop('diabetes', axis=1)).transform(df.drop('diabetes', axis=1))
# df_scaled = pd.DataFrame(scaled, columns=df.columns[:-1])
# X = df_scaled
# y = df['diabetes']
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# knn = KNeighborsClassifier()
# knn.fit(x_train, y_train)
# y_prob = knn.predict_proba(x_test)[:, 1]
# fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# roc_auc = auc(fpr, tpr)
# print(f"Area Under the ROC Curve (AUROC): {roc_auc:.2f}")
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()
# KnnPredictions = knn.predict(x_test)
# KnnAccuracy = accuracy_score(y_test, KnnPredictions)
# print("Knn Accuracy: {:.3f}".format(KnnAccuracy))
# print(confusion_matrix(y_test, KnnPredictions))
# sns.heatmap(confusion_matrix(y_test, KnnPredictions),fmt ="d", annot = True, cmap = 'Blues')
# plt.show()
# print(classification_report(y_test, KnnPredictions))





# ---------------------------------------Naive Bayes----------------------------
# X = df[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
#        'bmi', 'HbA1c_level', 'blood_glucose_level']]
# y = df["diabetes"]
# scaler = StandardScaler()
# scaled = scaler.fit(df.drop('diabetes', axis=1)).transform(df.drop('diabetes', axis=1))
# df_scaled = pd.DataFrame(scaled, columns=df.columns[:-1])
# X = df_scaled
# y = df['diabetes']
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# NB = GaussianNB()
# NB.fit(x_train, y_train)
# y_prob = NB.predict_proba(x_test)[:, 1]
# fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# roc_auc = auc(fpr, tpr)
# print(f"Area Under the ROC Curve (AUROC): {roc_auc:.2f}")
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()
# NBPredictions = NB.predict(x_test)
# NBAccuracy = accuracy_score(y_test, NBPredictions)
# print("Naive Bais Accuracy: {:.3f}".format(NBAccuracy))
# print(confusion_matrix(y_test, NBPredictions))
# sns.heatmap(confusion_matrix(y_test, NBPredictions),fmt ="d", annot = True, cmap = 'Reds')
# plt.show()
# print(classification_report(y_test, NBPredictions ))





# ---------------------------------------K-Mean----------------------------
X = df[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
       'bmi', 'HbA1c_level', 'blood_glucose_level']]
y = df["diabetes"]
# Kmean = cluster.KMeans(n_clusters=2, n_init=10)
# Kmean = Kmean.fit(X)
# SSE = Kmean.inertia_
# df['Cluster'] = Kmean.labels_
# print(f"SSE K-Means: {SSE}")
# plt.scatter(X['age'], X['bmi'], c=df['Cluster'], cmap='viridis')
# plt.title('K-Means Clustering')
# plt.xlabel('Age')
# plt.ylabel('bmi')
# plt.show()
#
# conf_matrix = confusion_matrix(y, df['Cluster'])
# print("Confusion Matrix:")
# print(conf_matrix)
# precision = precision_score(y, df['Cluster'])
# recall = recall_score(y, df['Cluster'])
# f1 = f1_score(y, df['Cluster'])
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1 Score: {f1:.2f}")
# accuracy = accuracy_score(y, df['Cluster'])
# print(f"Accuracy: {accuracy}")
#
# silhouette_avg = silhouette_score(X, df['Cluster'])
# print(f"Silhouette Score: {silhouette_avg}")
# fpr, tpr, thresholds = roc_curve(y, df['Cluster'])
# roc_auc = auc(fpr, tpr)
# plt.figure(figsize=(10, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()
#
# diabetes = df["diabetes"]
# ari = adjusted_rand_score(diabetes, df['Cluster'])
# print(f"Adjusted Rand Index: {ari}")




# ---------------------------------------Fuzzy C-mean----------------------------
# diabetes = df["diabetes"]
# cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
#     X.T, 10, 2, error=0.005, maxiter=100, init=None
# )
# cluster_membership = pd.DataFrame(u.T, columns=[f'Cluster_{i+1}' for i in range(10)])
# df['Cluster'] = cluster_membership.idxmax(axis=1)
# binary_labels = (cluster_membership['Cluster_1'] > 0.5).astype(int)
# fpr, tpr, thresholds = roc_curve(diabetes, binary_labels)
# roc_auc = auc(fpr, tpr)
# print(f"Area Under the ROC Curve (AUROC): {roc_auc:.2f}")
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()
# sse = 0.0
# for i in range(len(X)):
#     for j in range(10): #c
#         sse += u[j, i] ** 2 * ((X.iloc[i] - cntr[j])**2).sum()
#
# print(f"SSE Fuzzy C-Mean: {sse}")
# cluster_membership = pd.DataFrame(u.T, columns=[f'Cluster_{i+1}' for i in range(10)])#c
# df['Cluster'] = cluster_membership.idxmax(axis=1)
# plt.scatter(X['age'], X['bmi'], c=df['Cluster'].astype('category').cat.codes, cmap='viridis')
# plt.title('Fuzzy C-Means Clustering')
# plt.xlabel('Age')
# plt.ylabel('BMI')
# plt.show()




# ---------------------------------------Density-based----------------------------
# diabetes = df["diabetes"]
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# epsilon = 0.5
# min_samples = 2
# dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
# dbscan.fit(X_scaled)
# df['dbscan_cluster'] = dbscan.labels_
# # plt.scatter(X['age'], X['bmi'], c=df['dbscan_cluster'], cmap='viridis')
# # plt.title('DBSCAN Clustering')
# # plt.xlabel('AGE')
# # plt.ylabel('BMI')
# # plt.show()
# binary_labels = (df['dbscan_cluster'] == df['dbscan_cluster'].value_counts().idxmax()).astype(int)
# fpr, tpr, thresholds = roc_curve(diabetes, binary_labels)
# roc_auc = auc(fpr, tpr)
# print(f"Area Under the ROC Curve (AUROC): {roc_auc:.2f}")
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()


# ---------------------------------------Hierarchical-Agglomerative----------------------------
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# train_df, test_df = train_test_split(df, test_size=0.3, random_state=1)
# agg_cluster = AgglomerativeClustering(n_clusters=2)
# labels = agg_cluster.fit_predict(x_test)
# test_df['Cluster'] = labels
# fpr, tpr, thresholds = roc_curve(y_test, labels)
# roc_auc = auc(fpr, tpr)
# print(f"Area Under the ROC Curve (AUROC): {roc_auc:.2f}")
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()
# plt.figure(figsize=(10, 20))
# linkage_matrix = linkage(x_test, method='ward', metric='euclidean')
# dendrogram(linkage_matrix, truncate_mode='level', p=2)
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('Data points')
# plt.ylabel('Distance')
# plt.show()
# plt.scatter(x_test['age'], x_test['bmi'], c=test_df['Cluster'], cmap='viridis')
# plt.title('Agglomerative Clustering')
# plt.xlabel('Age')
# plt.ylabel('BMI')
# plt.show()


# ---------------------------------------Hierarchical-Divisive----------------------------
# linkage_matrix = linkage(x_test, method='ward', metric='euclidean')
# agg_cluster = AgglomerativeClustering(n_clusters=2, linkage='ward', metric='euclidean')
# labels = agg_cluster.fit_predict(x_test)
# silhouette_avg = silhouette_score(x_test, labels)
# print(f"Silhouette Score: {silhouette_avg}")
# dendrogram(linkage_matrix, truncate_mode='level', p=2)
# plt.title('Divisive Hierarchical Clustering Dendrogram')
# plt.xlabel('Data points')
# plt.ylabel('Distance')
# plt.show()