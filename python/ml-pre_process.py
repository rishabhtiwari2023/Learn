import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Load data
data = pd.read_csv('data.csv')

# 1. Data Cleaning
# Remove duplicate rows from the dataset
data.drop_duplicates(inplace=True)

# 2. Handling Missing Values
# Impute missing values using the mean strategy
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# 3. Data Normalization
# Normalize the data to have a mean of 0 and standard deviation of 1
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data_imputed)

# 4. Data Standardization
# Standardize the data to have a mean of 0 and standard deviation of 1
data_standardized = scaler.fit_transform(data_imputed)

# 5. Feature Scaling
# Scale the features to a standard range
data_scaled = scaler.fit_transform(data_imputed)

# 6. Encoding Categorical Variables
# Encode categorical variables using one-hot encoding
encoder = OneHotEncoder()
data_encoded = encoder.fit_transform(data)

# 7. Handling Outliers
# Remove outliers based on the 5th and 95th percentiles of the 'feature' column
data = data[(data['feature'] >= data['feature'].quantile(0.05)) & (data['feature'] <= data['feature'].quantile(0.95))]

# 8. Feature Selection
# Select features with a correlation greater than 0.5 with the target variable
correlation = data.corr()
features = correlation.index[abs(correlation['target']) > 0.5]

# 9. Feature Extraction
# Extract features using Principal Component Analysis (PCA)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# 10. Dimensionality Reduction
# Reduce the dimensionality of the data using PCA
data_reduced = pca.fit_transform(data)

# 11. Principal Component Analysis (PCA)
# Apply PCA to reduce the data to 2 components
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# 12. Linear Discriminant Analysis (LDA)
# Apply LDA to reduce the data to 1 component
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)
data_lda = lda.fit_transform(data, data['target'])

# 13. Singular Value Decomposition (SVD)
# Apply SVD to reduce the data to 2 components
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=2)
data_svd = svd.fit_transform(data)

# 14. Data Augmentation
# Example for image data augmentation using Keras
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

# 15. Data Splitting (Train/Test/Validation)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, data['target'], test_size=0.2, random_state=42)

# 16. Cross-Validation
# Perform 5-fold cross-validation
scores = cross_val_score(lda, data, data['target'], cv=5)

# 17. Hyperparameter Tuning
# Perform grid search for hyperparameter tuning
from sklearn.model_selection import GridSearchCV
param_grid = {'n_components': [1, 2, 3]}
grid_search = GridSearchCV(lda, param_grid, cv=5)
grid_search.fit(data, data['target'])

# 18. Model Selection
# Select the best model from the grid search
best_model = grid_search.best_estimator_

# 19. Model Evaluation Metrics
# Evaluate the model using accuracy score
from sklearn.metrics import accuracy_score
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 20. Confusion Matrix
# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# 21. ROC Curve and AUC
# Compute the ROC curve and AUC
y_score = best_model.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# 22. Precision-Recall Curve
# Compute the precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_score)

# 23. K-Fold Cross-Validation
# Perform K-Fold cross-validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
kf_scores = cross_val_score(best_model, data, data['target'], cv=kf)

# 24. Stratified Sampling
# Perform stratified sampling for cross-validation
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
skf_scores = cross_val_score(best_model, data, data['target'], cv=skf)

# 25. Bootstrapping
# Perform bootstrapping to create a new sample
from sklearn.utils import resample
data_bootstrap = resample(data, replace=True, n_samples=len(data), random_state=42)

# 26. Bagging and Boosting
# Apply Bagging and Boosting techniques
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
bagging = BaggingClassifier(base_estimator=best_model, n_estimators=10, random_state=42)
boosting = GradientBoostingClassifier(n_estimators=100, random_state=42)

# 27. Ensemble Methods
# Apply ensemble methods using VotingClassifier
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(estimators=[('lda', lda), ('pca', pca)], voting='hard')

# 28. Random Forest
# Apply Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# 29. Gradient Boosting Machines
# Apply Gradient Boosting Machines classifier
from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier(n_estimators=100, random_state=42)

# 30. XGBoost
# Apply XGBoost classifier
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=100, random_state=42)

# 31. LightGBM
# Apply LightGBM classifier
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier(n_estimators=100, random_state=42)

# 32. CatBoost
# Apply CatBoost classifier
from catboost import CatBoostClassifier
catboost = CatBoostClassifier(n_estimators=100, random_state=42, verbose=0)

# 33. Support Vector Machines (SVM)
# Apply Support Vector Machines classifier
from sklearn.svm import SVC
svm = SVC(kernel='linear', probability=True, random_state=42)

# 34. K-Nearest Neighbors (KNN)
# Apply K-Nearest Neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

# 35. Decision Trees
# Apply Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(random_state=42)

# 36. Logistic Regression
# Apply Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression(random_state=42)

# 37. Linear Regression
# Apply Linear Regression model
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()

# 38. Ridge and Lasso Regression
# Apply Ridge and Lasso Regression models
from sklearn.linear_model import Ridge, Lasso
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

# 39. Polynomial Regression
# Apply Polynomial Regression model
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(data.drop('target', axis=1))
poly_regression = LinearRegression().fit(X_poly, data['target'])

# 40. Neural Networks
# Apply Neural Networks using Keras
from keras.models import Sequential
from keras.layers import Dense
neural_network = Sequential()
neural_network.add(Dense(12, input_dim=data.shape[1]-1, activation='relu'))
neural_network.add(Dense(8, activation='relu'))
neural_network.add(Dense(1, activation='sigmoid'))
neural_network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 41. Convolutional Neural Networks (CNN)
# Apply Convolutional Neural Networks using Keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
cnn = Sequential()
cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())
cnn.add(Dense(128, activation='relu'))
cnn.add(Dense(1, activation='sigmoid'))
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 42. Recurrent Neural Networks (RNN)
# Apply Recurrent Neural Networks using Keras
from keras.layers import SimpleRNN
rnn = Sequential()
rnn.add(SimpleRNN(50, input_shape=(10, 1), activation='relu'))
rnn.add(Dense(1, activation='sigmoid'))
rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 43. Long Short-Term Memory (LSTM)
# Apply Long Short-Term Memory networks using Keras
from keras.layers import LSTM
lstm = Sequential()
lstm.add(LSTM(50, input_shape=(10, 1), activation='relu'))
lstm.add(Dense(1, activation='sigmoid'))
lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 44. Generative Adversarial Networks (GANs)
# Apply Generative Adversarial Networks using Keras
from keras.models import Model
from keras.layers import Input, Dense
generator_input = Input(shape=(100,))
generator = Dense(128, activation='relu')(generator_input)
generator = Dense(784, activation='sigmoid')(generator)
generator_model = Model(generator_input, generator)

discriminator_input = Input(shape=(784,))
discriminator = Dense(128, activation='relu')(discriminator_input)
discriminator = Dense(1, activation='sigmoid')(discriminator)
discriminator_model = Model(discriminator_input, discriminator)
discriminator_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 45. Transfer Learning
# Apply Transfer Learning using pre-trained VGG16 model
from keras.applications import VGG16
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# 46. Clustering Algorithms
# Apply K-Means clustering algorithm
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)

# 47. K-Means Clustering
# Fit K-Means clustering algorithm to the data
kmeans.fit(data)

# 48. Hierarchical Clustering
# Apply Hierarchical Clustering and plot dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage
linked = linkage(data, 'single')
dendrogram(linked)

# 49. DBSCAN
# Apply DBSCAN clustering algorithm
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(data)

# 50. Association Rule Learning
# Apply Association Rule Learning using Apriori algorithm
from mlxtend.frequent_patterns import apriori, association_rules
frequent_itemsets = apriori(data, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)