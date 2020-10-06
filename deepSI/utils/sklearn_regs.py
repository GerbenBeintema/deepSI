import sklearn.neighbors
import numpy as np

class linear_knn(sklearn.neighbors.KNeighborsRegressor):
    def predict(self, X):
        """Predict the target for the provided data
        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.
        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs), dtype=int
            Target values.
        """
        X = np.array(X)


        neigh_dist, neigh_ind = self.kneighbors(X)
        
        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        Yout = []
        X = np.insert(X,X.shape[1],1,axis=1)
        for neigh_ind_th,Xi in zip(neigh_ind,X):
            Y = _y[neigh_ind_th]
            Xnow = self._fit_X[neigh_ind_th]
            Xnow = np.insert(Xnow,Xnow.shape[1],1,axis=1)
            try:
                th = np.linalg.solve(Xnow.T@Xnow,Xnow.T@Y)
                Yout.append(Xi@th)
            except np.linalg.LinAlgError:
                # print('lin_alg error',Xnow)
                Yout.append([0])
            
        return np.array(Yout)[:,0] if self._y.ndim==1 else np.array(Yout)