import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize



#PARMS
norm = 'stad'
#norm = 'None'
#norm = 'Norm2'
#norm = 'Norm'
scki = True
initialize = 'random'

#hyper parameters
learning_rates = [ 0.001, 0.01,1e-3,1e-4,1e-5]
regularization_strengths = [1e-3,1e-4,10,1e-2,1e-1,1]
regularization_strengths = [0]
num_iterations = [50]

#normalizing the data
def norm_data(x,y,form='stad'):
	
    if form == "None":
       return x,y

    if form == "stad":
        train_data_sum = np.max(x,axis=0)
        train_data_mean = np.min(x, axis=0)
        norm = (x[:,:]-train_data_mean[None,:]) / (train_data_sum[None,:] - train_data_mean[None,:])
        norm_test = (y[:,:]-train_data_mean[None,:]) / (train_data_sum[None,:] - train_data_mean[None,:])
        #m,n = norm.shape
        #ones=np.ones((m,1))
        #norm  = np.hstack(( ones,norm))
        #norm  = np.hstack(( norm,ones))
        return norm,norm_test

    if form == "Norm":
        train_data_sum = np.std(x,axis=0)
        train_data_mean = np.mean(x, axis=0)
        norm = (x[:,:]-train_data_mean[None,:]) / train_data_sum[None,:] 
        norm_test = (y[:,:]-train_data_mean[None,:]) / train_data_sum[None,:] 
        #m,n = norm.shape
        #ones=np.ones((m,1))
        #norm  = np.hstack(( ones,norm))
        #norm  = np.hstack(( norm,ones))
        return norm,norm_test

    if form == "Norm2":
    	train_data_sum = np.var(x,axis=0)
    	train_data_mean = np.mean(x, axis=0)
    	norm = (x[:,:]-train_data_mean[None,:]) / train_data_sum[None,:] 
    	return norm
      

class LogisticRegressionFromScratch:

    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization_strength=0.01, dimensions=21,initialize='ones'):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization_strength = regularization_strength
        self.dimensions = dimensions        
        self.theta = np.random.rand(dimensions+1)
        self.theta = np.zeros(dimensions+1)
        if initialize == 'zeros':
           self.theta = np.zeros(dimensions+1)
        if initialize == 'ones':
           self.theta = np.ones(dimensions+1)
        if initialize == 'rand':
           self.theta = np.random.rand(dimensions+1)

    def sigmoid(self, z):
        #return (1 / (1 + np.exp(-np.clip(z, -20, 20))))
        return (1 / (1 + np.exp(-z)))
        #return (1 / (1 + np.exp(-np.clip(z, -20, 20))))
    def fit(self, X, y):
        m, n = X.shape
        epsilon = 1e-7
        ones=np.ones((m,1))
        X  = np.hstack(( X,ones))


        for iteration in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)

            ## Calculate the cost (log loss) with smoothing
            #cost = (-y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon)).mean()
            ##l1_regularization = self.regularization_strength * np.sum(np.abs(self.theta))
            #l2_regularization = self.regularization_strength * np.sum(self.theta**2)
            #total_cost = cost  + l2_regularization
            # Calculate the gradient
            gradient_o = (1 / m) * np.dot(X.T, (h - y))
            # Calculate L1 and L2 gradients
            l1_gradient = self.regularization_strength * np.sign(self.theta)
            l2_gradient = self.regularization_strength * self.theta
            # Add the gradient of the L1 and L2 regularization terms
            gradient = (self.learning_rate * gradient_o)  + l2_gradient 
            # Update the model parameters (theta)
            self.theta -= gradient
            # Print the cost for every 100 iterations
            #print(f"Theta {self.theta}")
            #if iteration % 10 == 0:
            #     print(f"Iteration {iteration}: Cost = {cost:.2f}, Total_cost = {total_cost:.2f} \n regression Gradient: {gradient_o}, l2_gradient: {l2_gradient}")

    def predict(self, X):
        m,n=X.shape
        ones=np.ones((m,1))
        X  = np.hstack(( X,ones))
        z = np.dot(X, self.theta)
        probabilities = self.sigmoid(z)
        #Convert probabilities to binary predictions (0 or 1)
        predictions = (probabilities >= 0.5).astype(int)
        return predictions

# Step 1: Load the data
train_data = pd.read_csv("train.csv").values
test_data = pd.read_csv("test.csv").values

test_info = pd.read_csv("test.csv")
train_data_clean = train_data[:,1:-1]

test_data_clean = test_data[:,1:]

train_label_clean = train_data[:,-1]
test_sno_clean = test_data[:,-1]


#train_data_norm = normalize(train_data_clean,axis=0 )
train_data_norm,test_data_norm = norm_data(train_data_clean,test_data_clean,norm)
#print(np.sum(train_data_norm,axis=0))
# Step 2: Data cleaning 

X_train, X_validation, y_train, y_validation = train_test_split(train_data_norm, train_label_clean, test_size=0.002, random_state=42)

# Step 4: Hyperparameter tuning
# Define a list of hyperparameters to search



if scki == True:
	learning_rates = [ 0.001 ]
	regularization_strengths = [0.001]
	num_iterations = [10]

best_accuracy = 0
best_hyperparameters = {}

performance = { 'iterations' : [],
                'lr' : [],
                'reg': [],
                'acc': [],
                'train_acc': [],
}
# Perform hyperparameter tuning using nested loops
for lr in learning_rates:
    for reg_strength in regularization_strengths:
        for iterations in num_iterations:
            # Create and train your logistic regression model with the current hyperparameters
                print(f" learning rate {lr}, reg {reg_strength} ")
                #logistic_model = LogisticRegressionFromScratch(learning_rate=lr, regularization_strength=reg_strength, num_iterations=iterations,initialize=initialize)
                #logistic_model.fit(X_train, y_train)
                #validation_predictions = logistic_model.predict(X_validation)  # Replace X_validation with your validation data

                if scki == True:
                   #clf = LogisticRegression(C=100,max_iter=1000,solver='saga',penalty='elasticnet',l1_ratio=0.5,class_weight='balanced',random_state=0).fit(X_train, y_train)
                   #clf = tree.DecisionTreeClassifier(class_weight='balanced').fit(X_train,y_train)
                   #clf = RandomForestClassifier(bootstrap=True,class_weight='balanced_subsample').fit(X_train,y_train)
                   clf = RandomForestClassifier(random_state=0,bootstrap=False,min_samples_leaf=1).fit(X_train,y_train)
                   #clf = HistGradientBoostingClassifier(max_depth=4).fit(X_train,y_train)
                   #clf = GradientBoostingClassifier().fit(X_train,y_train)
                   validation_predictions = clf.predict(X_validation)
                   train_predictions = clf.predict(X_train)
                   
                # Make predictions on the validation set
                # Calculate accuracy on the validation set
                accuracy = accuracy_score(y_validation, validation_predictions)  # Replace y_validation with your validation labels
                train_accuracy = accuracy_score(y_train, train_predictions)  # Replace y_validation with your validation labels
                performance['iterations'].append( iterations)
                performance['lr'].append( lr)
                performance['reg'].append(  reg_strength)
                performance['acc'].append( accuracy)
                performance['train_acc'].append( train_accuracy)
                # Check if the current hyperparameters produce better accuracy
                # Step 5: Make predictions on the test data using the best hyperparameters

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_hyperparameters = {
                        'learning_rate': lr,
                        'regularization_strength': reg_strength,
                        'num_iterations': iterations
                    }

                    if scki == True:
                        test_predictions = clf.predict(test_data_norm)
                    else:
                        test_predictions = logistic_model.predict(test_data_norm)
#step 6: submission
# Use the best-performing model (Logistic Regression in this case) to make predictions on the test data
#Create a DataFrame with 'SNo' and 'Label' columns
# Save the predictions to a CSV file

                    submission = pd.DataFrame({'SNo': range(1, len(test_predictions) + 1), 'Label': test_predictions})
                    submission.to_csv("submission.csv", index=False)

values2, counts2=np.unique(test_predictions, return_counts=True)
print("test predictions coutns {},{}".format(values2, counts2))

values, counts=np.unique(validation_predictions, return_counts=True)  
print("ttrain predictions coutns {},{}".format(values, counts))

values, counts=np.unique(y_validation, return_counts=True)  
print("train data coutns {},{}".format(values, counts))

print("Best Hyperparameters:", best_hyperparameters)
print("Best Accuracy:", best_accuracy)


per = pd.DataFrame.from_dict(performance)

print("best 5 models ")
print(per.sort_values(by=['acc']).tail(5))

print(per.groupby('lr')['acc'].mean())
print(per.groupby('iterations')['acc'].mean())
print(per.groupby('reg')['acc'].mean())


