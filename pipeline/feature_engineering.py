from sklearn.base             import BaseEstimator, TransformerMixin
from dateutil.relativedelta   import relativedelta
import pandas                 as pd
import numpy                  as np

    
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        X = self.lag_features(X)
        X = self.new_features(X)
        return X
    
    def fit_transform(self, X, y=None):
        X = self.lag_features(X)
        X = self.new_features(X)
        return X

    def transform(self, X, y=None):
        X = self.lag_transform(X)
        X = self.new_features(X)
        return X
    
    def lag_features(self, data ):
        """
        Calculates the lag feature of passengers per flight for each date.
        """

        data_copy = data.copy()
        data_copy['passageiros_voo'] = data_copy.passageiros / data_copy.decolagens
        rolagem = data_copy.groupby('date')[['passageiros_voo']].sum().reset_index()

        self.lag_year = rolagem.copy()
        self.lag_year['date'] = self.lag_year.date.apply(lambda x: x + relativedelta(years=1))
        self.lag_year = self.lag_year.rename(columns={'passageiros_voo': 'lag_year'})

        data = data.merge(self.lag_year, how ='left', on='date')
        data = data.sort_values('date')
        
        return data
    
    def lag_transform(self, data ):
        """
        Apply lag features.
        """
        data = data.merge(self.lag_year, how ='left', on='date')
        data = data.sort_values('date')
        
        return data
    
    def new_features(self, data):
        """
        Create new features.
        """

        data['rota'] = data.aeroporto_de_origem_sigla.astype(str) + " - " + data.aeroporto_de_destino_sigla.astype(str)

        data['distancia_voada_km_por_voo'] = data.distancia_voada_km / data.decolagens
        data['assentos_por_voo'] = data.assentos / data.decolagens
        data['horas_voadas_por_voo'] = data.horas_voadas / data.decolagens

        # FEATURES ENGINEERING FOR TIME SERIES
        data["mes_sin"] = data.mes.apply(lambda x: np.sin(x * (2. * np.pi / 12)))
        data["mes_cos"] = data.mes.apply(lambda x: np.cos(x * (2. * np.pi / 12)))

        #LAG features
        data.lag_year = data.lag_year * data.decolagens
        min_value = data.lag_year.min()
        data.lag_year = data.lag_year.fillna(min_value)

        data = data[['ano', 'aeroporto_de_destino_uf', 'aeroporto_de_origem_uf', 'empresa_nome', 
                     'mes', 'aeroporto_de_destino_sigla', 'aeroporto_de_origem_sigla', 'decolagens']]
        
        return data