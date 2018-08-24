#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
  """
  多数決アンサンブル分類器

  パラメータ
  -----
  classifiers : array-like, shape = [n_classifiers]
  
  vote : str, {'classlabel', 'probability'} (default: 'classlabel')
    'classlabel'の場合: クラスラベルの予測はクラスラベルのargmaxに基づく
    'probability'の場合: クラスラベルの予測はクラスの所属確率のargmaxに基づく

  weights : array-like, shape = [n_classifiers] (optional, default = None)
  """

  def __init__(self, classifiers, vote = 'classlabel', weights = None):
    self.classifiers = classifiers
    self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
    self.vote = vote
    self.weights = weights

  def fit(self, X, y):
    """
    分類器を学習させる

    パラメータ
    -----
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]

    y : array-like, shape = [n_samples]


    戻り値
    -----
    self : object
    """

    self.lablenc_ = LabelEncoder()
    self.lablenc_.fit(y)
    self.classes_ = self.lablenc_.classes_
    self.classifiers_ = []
    for clf in self.classifiers:
      fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
      self.classifiers_.append(fitted_clf)
    return self

  def predict(self, X):
    """
    Xのクラスラベルを予測する

    パラメータ
    -----
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]


    戻り値
    -----
    maj_vote : array-like, shape = [n_samples]
    """

    if self.vote == 'probability':
      maj_vote = np.argmax(self.predict_proba(X), axis = 1)

    else:
      predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
      maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights = self.weights)), axis = 1, arr = predictions)

    maj_vote = self.lablenc_.inverse_transform(maj_vote)
    return maj_vote

  
  def predict_proba(self, X):
    """
    Xのクラスラベルを予測する

    パラメータ
    -----
    X : 同じ


    戻り値
    -----
    avg_proba : array-like, shape = [n_samples, n_classes]
      各サンプルに対する各クラスで重み付けた平均確率
    """

    probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
    avg_proba = np.average(probas, axis = 0, weights = self.weights)
    return avg_proba

  
  def get_params(self, deep = True):
    """
    GridSearchの実行時に分類器のパラメータ名を取得
    """

    if not deep:
      return super(MajorityVoteClassifier, self).get_params(deep = False)

    else:
      # キーを”分類器の名前__パラメータ名”
      # バリューをパラメータの値とするディクショナリを生成
      out = self.named_classifiers.copy()
      for name, step in six.iteritems(self.named_classifiers):
        for key, value in six.iteritems(step.get_params(deep = True)):
          out['%s__%s' % (name, key)] = value
      return out
