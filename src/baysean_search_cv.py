import pandas        as pd
import numpy         as np
from skopt           import gp_minimize
from scipy.stats     import mode
from datetime        import datetime
from datetime        import timedelta
from sklearn.metrics import mean_squared_error

class BayesianSearchCV:
    def __init__(self, kfold, window_size, pipe_processing, params_space, model_params, n_calls=10, verbose=0):
        """
        BayesSearchCV class constructor.
        Args:
        - kfold: number of folds
        - window_size: size in month for window
        - pipe_processing: preprocessing pipeline
        - params_space: hyperparameters space
        - model_params: model parameters
        - n_calls: number of calls for gp_minimize
        - verbose: show results of optimization
        Returns:
        - None
        """
        self.pipe_processing = pipe_processing
        self.params_space = params_space
        self.model_params = model_params
        self.n_calls = n_calls  
        self.verbose = verbose
        self.kfold = kfold
        self.window_size = window_size

        
    def fit(self, X, y):
        """
        Function to fit the model.
        Args:
        - X: input features
        - y: target variable
        Returns:
        - info: dictionary containing model performance for each fold
        """
        data = X.copy()
        data['target'] = y.copy()

        data_max = data['date'].max()  # MAX DATE
        formatted_date = data_max.strftime('%Y-%m-%d')  # Formatação da data como string
        max_date = datetime.strptime(formatted_date, '%Y-%m-%d')  # Conversão para objeto datetime

        all_params = pd.DataFrame()
        train_score = []
        test_score = []

        kfold_local = self.kfold
        window_size_local = self.window_size
        
        fold_no = 1
        
        for k in reversed(range(1, kfold_local + 1)):
            print(f'\nFold: {fold_no}')
            validation_start_date = (max_date - timedelta(days=30*k)).replace(day=1)
            validation_end_date   = (max_date - timedelta(days=30*(k-1))).replace(day=1)
            window = (validation_start_date - timedelta(days=30* window_size_local)).replace(day=1)

            train = data[ (data['date'] >= window) & (data['date'] < validation_start_date)]
            test = data[(data['date'] >= validation_start_date) & (data['date'] <= validation_end_date)]

            params, evaluation_train, evaluation_test = self.training( train, test )            
            train_score.append(evaluation_train)
            test_score.append(evaluation_test)
            
            score = pd.DataFrame.from_dict(params, orient='index').T
            all_params = pd.concat([all_params, score], ignore_index=True)
            
            fold_no += 1
            
        all_params = all_params.fillna('None')                
        all_params['train_score'] = train_score
        all_params['test_score'] = test_score
        
        self.best_parameters_ = self.summary_dataframe(all_params.drop(columns=['train_score', 'test_score']))

        self.cv_results_ = all_params
        
        return None

    def training( self, train, test ): 
        """
        Function to train the model for cross-validation.
        Args:
        - train: training data
        - test: testing data
        Returns:
        - rmse: model performance metric
        - params: model parameters
        """
        self.data = test
        
        x_train = train.drop(columns = 'target')
        self.x_train = self.pipe_processing.fit_transform(x_train)
        self.y_train = train.target
        
        x_test = test.drop(columns = 'target')
        self.x_test = self.pipe_processing.transform(x_test) 
        self.y_test = test.target

        # Tunning model
        model = self.model_tunning( )
        params = model.get_params()
        
        # training model
        model.fit(self.x_train, self.y_train)
        y_train_pred = model.predict(self.x_train)
        evaluation_train = self.make_score( self.y_train, y_train_pred)
        
        y_test_pred = model.predict(self.x_test)
        evaluation_test = self.make_score( self.y_test, y_test_pred)
        
        return params, evaluation_train, evaluation_test
    

    def model_tunning( self ):
        """
        Optimize the hyperparameters of a machine learning model using Bayesian optimization.
        Returns:
            A machine learning model with optimized hyperparameters.
        """

        
        resultados_gp = gp_minimize(self.optimization_function, 
                                    self.params_space, 
                                    random_state=0, 
                                    verbose=self.verbose, 
                                    n_calls=self.n_calls, 
                                    n_random_starts=(int(self.n_calls/3)))   
        
        best_params = resultados_gp.x
        
        mdl = self.model_params(best_params)

        return mdl

    def optimization_function(self, params):
        """
        Defines the function to be minimized during hyperparameter optimization.
        Args:
        - params (list): A list of hyperparameter values to be used to build a model.
        Returns:
        - float: The negative value of the performance metric of the model built with the given hyperparameters.
        """

        mdl = self.model_params(params)

        mdl.fit(self.x_train, self.y_train)

        y_pred = mdl.predict(self.x_test)
        rmse = self.make_score( self.y_test, y_pred)
                
        return rmse
    
    
    def summary_dataframe(self, df):
        """
        Function to summarize the data in a DataFrame.
        Args:
        - df: DataFrame to be summarized
        Returns:
        - summary_df: DataFrame with the summary of the data
        """
        summary_df = pd.DataFrame(columns=['column', 'mean_or_mode'])

        for column in df.columns:
            # Check the columns
            if column == 'C':
                C = df[column].median()
                C_df = pd.DataFrame({'column': [column], 'mean_or_mode': [C]})
                summary_df = pd.concat([summary_df, C_df], ignore_index=True)

            # Check the data type of the column
            elif df[column].dtype == 'int64':
                # Calculate the mean for int columns
                mean = str(int(df[column].mean()))
                num_df = pd.DataFrame({'column': [column], 'mean_or_mode': [mean]})
                summary_df = pd.concat([summary_df, num_df], ignore_index=True)  

            elif df[column].dtype == 'float64':
                # Calculate the mean for int columns
                mean = df[column].mean()
                num_df = pd.DataFrame({'column': [column], 'mean_or_mode': [mean]})
                summary_df = pd.concat([summary_df, num_df], ignore_index=True)

            elif df[column].dtype == 'object':
                # Calculate the mode for object columns
                mode_value = df[column].mode()[0]
                cat_df = pd.DataFrame({'column': [column], 'mean_or_mode': [mode_value]})
                summary_df = pd.concat([summary_df, cat_df], ignore_index=True)

        return summary_df
    
    
    
    ##############################################################################
    def make_score( self, y_val, y_pred):
        """
        Calculates the RMSE.
        """        
        return np.sqrt(mean_squared_error(y_val, y_pred))