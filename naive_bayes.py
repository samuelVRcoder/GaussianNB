from sklearn.naive_bayes import GaussianNB as nb

import sklearn.datasets as dt

from sklearn.model_selection import train_test_split as tts

from sklearn.metrics import accuracy_score as score

model = nb()

X, y = dt.make_blobs(n_samples=100000)

xtr, xts, ytr, yts = tts(X, y, test_size=0.3)

model.fit(xtr, ytr)

print(score(yts, model.predict(xts)))
