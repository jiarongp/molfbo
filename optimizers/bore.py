from sklearn.ensemble import RandomForestClassifier

from .utils import split_good_bad

class BORE:
    def __init__(
        self,
        top_n_percent=30,
    ):
        self.X = None
        self.y = None

        self.gamma = top_n_percent / 100
 
 
    def fit(self, X, y, z=None):
        self.clf = RandomForestClassifier()

        if z is None:
            _, _, z = split_good_bad(X, y, gamma=self.gamma)

        self.clf.fit(X, z)

    def predict(self, X):
        return self.clf.predict_proba(X)[:, 1]
