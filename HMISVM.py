# -*- coding: utf-8 -*-
import numpy as np


class HMISVM:
    """
    Hard Margin Incremental Support Vector Machine (HMISVM)
    """
    def __init__(self):
        self.data = [[], []]
        self.sv = [[], []]
        self.W, self.b = None, None
        self.d = 0
        self.place_changer = 1

    def fit(self, X, y):
        """
        Fit HMISVM by given X and y
        :param X: (n, d) numpy array
        :param y: (n,) numpy array
        """
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] >= X.shape[1]

        n, self.d = X.shape
        X = [X[y == -1, :], X[y == 1, :]]

        for y_idx in range(2):
            for i in range(self.d):
                self.data[y_idx].append(X[y_idx][i])
                self.sv[y_idx].append(X[y_idx][i])
            X[y_idx] = X[y_idx][self.d:]

        # Set place changer
        self._update_place_changer()

        # Permute X, y randomly
        X, y = np.concatenate(X, axis=0), np.array([-1]*len(X[0]) + [1]*len(X[1]))
        shuffle = np.random.permutation(len(X))
        X, y = X[shuffle], y[shuffle]

        # Add data one-by-one
        for i in range(len(X)):
            X_i, y_i = X[i], y[i]
            self.add_data(X_i, y_i, update=False)

        self._update()

    def add_data(self, X_i, y_i, update=True):
        """
        Fit new data (X_i, y_i)
        :param X_i: (d,) numpy array
        :param y_i: (1,) numpy array
        :param update: Boolean
        """
        y_idx = (int(y_i) + 1) // 2
        self.data[y_idx].append(X_i)

        X_sv = np.array(self.sv[y_idx])
        W = self._find_hyperplane(X_sv)

        # If new data is not in the support vector half-plane
        if self.place_changer * y_i * self._is_above_hyperplane(W, X_i) < 0:
            for i in range(self.d):
                X_sv = np.array(self.sv[y_idx][:i] + self.sv[y_idx][i+1:] + [X_i])
                W = self._find_hyperplane(X_sv)
                if self.place_changer * y_i * self._is_above_hyperplane(W, self.sv[y_idx][i]) > 0:
                    b = True
                    for j in range(len(self.data[1-y_idx])):
                        if self.place_changer * y_i * self._is_above_hyperplane(W, self.data[1-y_idx][j]) > 0:
                            b = False
                    if b:
                        self.sv[y_idx][i] = X_i
                        self._update_place_changer()
                        break

        if update:
            self._update()

    def _update(self):
        """
        Update W and b using found support vectors.
        """
        W = [self._find_hyperplane(self.sv[0]), self._find_hyperplane(self.sv[1])]

        class_valid = [True, True]
        for i in range(2):
            y = i * 2 - 1
            for j in range(self.d):
                if self.place_changer * y * self._is_above_hyperplane(W[i], self.sv[1-i][j]) > 0:
                    class_valid[i] = False
                    break

        if not class_valid[0] and not class_valid[1]:
            class_valid = [True, True]

        min_dist = [np.inf, np.inf]
        min_idx = [(0, 0), (0, 0)]
        for i in range(2):
            if not class_valid[i]:
                continue

            for j in range(self.d):
                dist = np.abs(self.sv[1 - i][j] @ W[i] - 1) / np.linalg.norm(W[i])
                if min_dist[i] > dist:
                    min_dist[i] = dist
                    min_idx[i] = (1 - i, j)

        max_margin = -np.inf
        max_class, max_sv_idx = 0, 0
        for i in range(2):
            if class_valid[i] and min_dist[i] > max_margin:
                max_margin = min_dist[i]
                max_class, max_sv_idx = min_idx[i]

        max_y = max_class * 2 - 1
        self.W = W[1 - max_class]
        self.W *= self.place_changer * np.sign(self.W[self.d - 1]) * 2 / (np.linalg.norm(self.W) * max_margin)
        self.b = max_y - self.sv[max_class][max_sv_idx] @ self.W

    def _find_hyperplane(self, Xs):
        """
        Find w such that Xs * W = 1
        :param Xs: (d, d) numpy array
        :return: (d,) numpy array
        """
        W = np.linalg.solve(Xs + 1e-7 * np.eye(self.d), np.ones(self.d))
        return W

    def _is_above_hyperplane(self, W, x):
        """
        Check x is above the hyperplane w * x = 1
        :param W: (d,) numpy array
        :param x: (d,) numpy array
        :return: 1 if above -1 else
        """
        if x[self.d-1] > (-1 * x[:self.d-1] @ W[:self.d-1] + 1) / W[self.d-1]:
            return 1
        return -1

    def _update_place_changer(self):
        """
        If class '-1' is upper than class '1', update place changer as -1
        """
        W = self._find_hyperplane(np.array(self.sv[1]))
        self.place_changer = -1 * self._is_above_hyperplane(W, self.sv[0][0])

    def predict(self, X):
        y = np.sign(X @ self.W + self.b)
        return y
