import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
matplotlib.use('TkAgg')
import seaborn as sns



from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score



from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report




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
# accTree = accuracy_score(y_test, y_predTree)
# print("Accuracy : " ,  accTree)
#
#
# plt.figure(figsize=(15, 10))
# plot_tree(Tree1, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], filled=True, rounded=True)
# plt.savefig('plot100', dpi=1000)
# tree_rules = export_text(Tree1, feature_names=list(X.columns))
# print("Decision Tree Rules:")
# print(tree_rules)



# ---------------------------------------C4.5 Decision Tree----------------------------
# X = df[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
# y = df["diabetes"]
#
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# Tree1 = DecisionTreeClassifier(criterion="entropy").fit(x_train, y_train)
# y_predTree = Tree1.predict(x_test)
# accTree = accuracy_score(y_test, y_predTree)
# print("Accuracy C4.5 : ", accTree)
#
# plt.figure(figsize=(15, 10))
# plot_tree(Tree1, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], filled=True, rounded=True)
# tree_rules = export_text(Tree1, feature_names=list(X.columns))
# plt.savefig('plot101', dpi=1000)
# print("C4.5 Decision Tree Rules:")
# print(tree_rules)



# ---------------------------------------CART Decision Tree----------------------------
# X = df[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
#        'bmi', 'HbA1c_level', 'blood_glucose_level']]
# y = df["diabetes"]
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# Tree1 = DecisionTreeClassifier(criterion='gini').fit(x_train, y_train)
# y_predTree = Tree1.predict(x_test)
# accTree = accuracy_score(y_test, y_predTree)
# print("Accuracy CART: " ,  accTree)
#
#
# plt.figure(figsize=(15, 10))
# plot_tree(Tree1, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], filled=True, rounded=True)
# plt.savefig('plot102', dpi=1000)
# tree_rules = export_text(Tree1, feature_names=list(X.columns))
# print("Decision Tree Rules:")
# print(tree_rules)





# ---------------------------------------SVM----------------------------
scaler = StandardScaler()
scaled = scaler.fit(df.drop('diabetes',axis=1)).transform(df.drop('diabetes',axis=1))
df_scaled = pd.DataFrame(scaled, columns=df.columns[:-1])
df_scaled.head()
x = df_scaled
y = df['diabetes']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# svc=SVC()
# svc.fit(x_train, y_train)
# y_pred=svc.predict(x_test)
# print('Model accuracy : {0:0.7f}'. format(accuracy_score(y_test, y_pred)))
# cm = confusion_matrix(y_test, y_pred)
# cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
#                                  index=['Predict Positive:1', 'Predict Negative:0'])
#
# sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='mako')
# plt.title('Model')
# plt.show()
# print(classification_report(y_test,y_pred))




# ---------------------------------------linear kernel SVM----------------------------
# linear_classifier= SVC(kernel='linear').fit(x_train,y_train)
# y_pred = linear_classifier.predict(x_test)
# print('linear kernel : {0:0.7f}'. format(accuracy_score(y_test, y_pred)))
# cm = confusion_matrix(y_test, y_pred)
# cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
#                                  index=['Predict Positive:1', 'Predict Negative:0'])
#
# sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='mako')
# plt.title('linear kernel with SVM')
# plt.show()
# print(classification_report(y_test,y_pred))



# ---------------------------------------RBF kernel Nonlinear SVM----------------------------
# rbf_classifier = SVC(kernel='rbf')
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


# ---------------------------------------Polynomial kernel Nonlinear SVM----------------------------
# poly_classifier = SVC(kernel='poly', degree=3)
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




# ---------------------------------------Sigmoid kernel SVM----------------------------
# sigmoid_classifier = SVC(kernel='sigmoid')
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