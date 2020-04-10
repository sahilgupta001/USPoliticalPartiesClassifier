import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

path="E:\\Development work\\USPoliticalParties\\house-votes-84.data.txt"

features = ['party', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-free-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satelite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile', 'immigaration', 'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa']
data = pd.read_csv(path, na_values = ['?'], names = features)
data.dropna(inplace = True)
data.replace(('y', 'n'), (1, 0), inplace = True)
data.replace(('democrat', 'republican'), (1, 0), inplace = True)


all_features = data[features].drop('party', axis = 1).values
labels = data['party'].values

def create_model():

    model = Sequential()
    model.add(Dense(64, activation = 'relu', kernel_initializer = 'normal', input_shape = (16,)))
    model.add(Dense(64, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(1, kernel_initializer= 'normal', activation = 'sigmoid'))
    model.compile(loss = "binary_crossentropy", optimizer = RMSprop(), metrics = ['accuracy'])
    return model

estimator = KerasClassifier(build_fn = create_model, epochs = 10, verbose = 0)
cv_scores = cross_val_score(estimator , all_features, labels, cv = 10)

