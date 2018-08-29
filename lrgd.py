#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

class LinearRegressionGD(object):
  def __init__(self, eta = 0.001, n_iter = 20):
    self.eta = eta
    self.n_iter = n_iter

  # トレーニングを実行
  def fit(self, X, y):
    self.w_ = np.zeros(1 + X.shape[1])
    self.cost_ = []
    for i in range(self.n_iter):
      output = self.net_input(X)
      errors = y - output
      self.w_[1:] += self.eta * X.T.dot(errors)
      self.w_[0] += self.eta * errors.sum()
      cost = (errors**2).sum() / 2.0
      self.cost_.append(cost)
    return self
  
  # 総入力を計算
  def net_input(self, X):
    return np.dot(X, self.w_[1:]) + self.w_[0]

  # 予測値を計算
  def predict(self, X):
    return self.net_input(X)
