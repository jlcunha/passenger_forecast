from sklearn.base             import BaseEstimator, TransformerMixin
from dateutil.relativedelta   import relativedelta
import pandas                 as pd
import numpy                  as np

    
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        X = self.new_features(X)
        return X
    
    def fit_transform(self, X, y=None):
        X = self.new_features(X)
        return X

    def transform(self, X, y=None):
        X = self.new_features(X)
        return X
        
    def new_features(self, data):
        """
        Create new features.
        """

        # data['rota'] = data.aeroporto_de_origem_sigla.astype(str) + " - " + data.aeroporto_de_destino_sigla.astype(str)

        # # FEATURES ENGINEERING FOR TIME SERIES
        # data["mes_sin"] = data.mes.apply(lambda x: np.sin(x * (2. * np.pi / 12)))
        # data["mes_cos"] = data.mes.apply(lambda x: np.cos(x * (2. * np.pi / 12)))


        data = data[['ano', 'aeroporto_de_destino_uf', 'aeroporto_de_origem_uf', 'empresa_nome', 
                     'mes', 'aeroporto_de_destino_sigla', 'aeroporto_de_origem_sigla', 'decolagens']]
        
        return data