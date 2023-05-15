from sklearn.base             import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from unidecode import unidecode

class DataCleaning(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = self.split_names(X)
        X = self.unicode_text(X)
        
        return X
    
    def split_names(self, data):

        """
        Split the airport names into airport name and corresponding state abbreviation.

        Args:
            data (DataFrame): Input DataFrame with columns 'aeroporto_de_origem_nome' and 'aeroporto_de_destino_nome'.

        Returns:
            DataFrame: Modified input DataFrame with additional columns 'aeroporto_de_origem_uf' and 'aeroporto_de_destino_uf'.
                       The 'aeroporto_de_origem_nome' and 'aeroporto_de_destino_nome' columns are updated to contain only the airport names.
                       The 'aeroporto_de_origem_uf' and 'aeroporto_de_destino_uf' columns are converted to the 'category' data type.
        """

        # Origem
        aeroporto_de_origem_uf = pd.DataFrame( data.aeroporto_de_origem_nome.str.split(',').str[1] )
        aeroporto_de_origem_uf = aeroporto_de_origem_uf.rename(columns={'aeroporto_de_origem_nome': 'aeroporto_de_origem_uf'})
        uf_completo = pd.DataFrame( data.aeroporto_de_origem_uf )
        uf_completo.update(aeroporto_de_origem_uf)
        data.aeroporto_de_origem_uf = uf_completo.astype('category')
        data.aeroporto_de_origem_nome = data.aeroporto_de_origem_nome.str.split(',').str[0].astype('category')

        # Destino
        aeroporto_de_destino_uf = pd.DataFrame( data.aeroporto_de_destino_nome.str.split(',').str[1] )
        aeroporto_de_destino_uf = aeroporto_de_destino_uf.rename(columns={'aeroporto_de_destino_nome': 'aeroporto_de_destino_uf'})
        uf_completo = pd.DataFrame( data.aeroporto_de_destino_uf )
        uf_completo.update(aeroporto_de_destino_uf)
        data.aeroporto_de_destino_uf = uf_completo.astype('category')
        data.aeroporto_de_destino_nome = data.aeroporto_de_destino_nome.str.split(',').str[0].astype('category')

        return data

    def unicode_text(self, data ):
        """
        Transforme the text - strip and unidecode.
        """

        for col in data.select_dtypes(include=['category']).columns:
            data[col] = data[col].astype(str).str.strip().apply(lambda x: unidecode(x))
            data[col] = data[col].astype('category')

        return data