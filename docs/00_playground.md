```python
from ISLP import load_data
from ISLP.models import ModelSpec as MS
from ISLP.models import poly, summarize
```


```python
import inspect
```


```python
print(inspect.getsource(load_data))
```

    def load_data(dataset):
        if dataset == 'NCI60':
            with as_file(files('ISLP').joinpath('data', 'NCI60data.npy')) as features:
                X = np.load(features)
            with as_file(files('ISLP').joinpath('data', 'NCI60labs.csv')) as labels:
                Y = pd.read_csv(labels)
            return {'data':X, 'labels':Y}
        elif dataset == 'Khan':
            with as_file(files('ISLP').joinpath('data', 'Khan_xtest.csv')) as xtest:
                xtest = pd.read_csv(xtest)
            xtest = xtest.rename(columns=dict([('V%d' % d, 'G%04d' % d) for d in range(1, len(xtest.columns)+0)]))
            with as_file(files('ISLP').joinpath('data', 'Khan_ytest.csv')) as ytest:
                ytest = pd.read_csv(ytest)
            ytest = ytest.rename(columns={'x':'Y'})
            ytest = ytest['Y']
            
            with as_file(files('ISLP').joinpath('data', 'Khan_xtrain.csv')) as xtrain:
                xtrain = pd.read_csv(xtrain)
                xtrain = xtrain.rename(columns=dict([('V%d' % d, 'G%04d' % d) for d in range(1, len(xtest.columns)+0)]))
    
            with as_file(files('ISLP').joinpath('data', 'Khan_ytrain.csv')) as ytrain:
                ytrain = pd.read_csv(ytrain)
            ytrain = ytrain.rename(columns={'x':'Y'})
            ytrain = ytrain['Y']
    
            return {'xtest':xtest,
                    'xtrain':xtrain,
                    'ytest':ytest,
                    'ytrain':ytrain}
    
        elif dataset == 'Hitters':
            with as_file(files('ISLP').joinpath('data', '%s.csv' % dataset)) as filename:
                df = pd.read_csv(filename)
            for col in ['League', 'Division', 'NewLeague']:
                df[col] = pd.Categorical(df[col])
            return df
        elif dataset == 'Carseats':
            with as_file(files('ISLP').joinpath('data', '%s.csv' % dataset)) as filename:
                df = pd.read_csv(filename)
            for col in ['ShelveLoc', 'Urban', 'US']:
                df[col] = pd.Categorical(df[col])
            return df
        elif dataset == 'NYSE':
            with as_file(files('ISLP').joinpath('data', '%s.csv' % dataset)) as filename: 
                df = pd.read_csv(filename).set_index('date')
            return df
        elif dataset == 'Publication':
            with as_file(files('ISLP').joinpath('data', 'Publication.csv')) as f:
                df = pd.read_csv(f)
            for col in ['mech']:
                df[col] = pd.Categorical(df[col])
            return df
        elif dataset == 'BrainCancer':
            with as_file(files('ISLP').joinpath('data', 'BrainCancer.csv')) as f:
                df = pd.read_csv(f)
            for col in ['sex', 'diagnosis', 'loc', 'stereo']:
                df[col] = pd.Categorical(df[col])
            return df
        elif dataset == 'Bikeshare':
            with as_file(files('ISLP').joinpath('data', '%s.csv' % dataset)) as filename:
                df = pd.read_csv(filename)
            df['weathersit'] = pd.Categorical(df['weathersit'], ordered=False)
            # setting order to avoid alphabetical
            df['mnth'] = pd.Categorical(df['mnth'],
                                        ordered=False,
                                        categories=['Jan', 'Feb',
                                                    'March', 'April',
                                                    'May', 'June',
                                                    'July', 'Aug',
                                                    'Sept', 'Oct',
                                                    'Nov', 'Dec'])
            df['hr'] = pd.Categorical(df['hr'],
                                      ordered=False,
                                      categories=range(24))
            return df
        elif dataset == 'Wage':
            with as_file(files('ISLP').joinpath('data', 'Wage.csv')) as f:
                df = pd.read_csv(f)
                df['education'] = pd.Categorical(df['education'], ordered=True)
            return df
        else:
            with as_file(files('ISLP').joinpath('data', '%s.csv' % dataset)) as filename:
                return pd.read_csv(filename)
    



```python
print(inspect.getsource(MS))
```

    class ModelSpec(TransformerMixin, BaseEstimator):
    
        '''
    
        Parameters
        ----------
    
        terms : sequence (optional)
            Sequence of sets whose
            elements are columns of *X* when fit.
            For :py:class:`pd.DataFrame` these can be column
            names.
    
        intercept : bool (optional)
            Include a column for intercept?
    
        categorical_features : array-like of {bool, int} of shape (n_features) 
                or shape (n_categorical_features,), default=None.
            Indicates the categorical features. Will be ignored if *X* is a :py:class:`pd.DataFrame`
            or :py:class:`pd.Series`.
    
            - None : no feature will be considered categorical for :py:class:`np.ndarray`.
            - boolean array-like : boolean mask indicating categorical features.
            - integer array-like : integer indices indicating categorical
              features.
    
        default_encoders : dict
            Dictionary whose keys are elements of *terms* and values
            are transforms to be applied to the associate columns in the model matrix
            by running the *fit_transform* method when *fit* is called and overwriting
            these values in the dictionary.
        '''
    
        def __init__(self,
                     terms=[],
                     intercept=True,
                     categorical_features=None,
                     default_encoders={'categorical': Contrast(method='drop'),
                                       'ordinal': OrdinalEncoder()}
                     ):
           
            self.intercept = intercept
            self.terms = terms
            self.categorical_features = categorical_features
            self.default_encoders = default_encoders
            
        def fit(self, X, y=None):
    
            """
            Construct parameters for orthogonal
            polynomials in the feature X.
    
            Parameters
            ----------
            X : array-like
                X on which model matrix will be evaluated.
                If a :py:class:`pd.DataFrame` or :py:class:`pd.Series`, variables that are of
                categorical dtype will be treated as categorical.
    
            """
            
            if isinstance(X, (pd.DataFrame, pd.Series)):
                (categorical_features,
                 self.is_ordinal_) = _categorical_from_df(X)
                (self.is_categorical_,
                 self.known_categories_) = _check_categories(categorical_features,
                                                             X)
                self.columns_ = X.columns
                if self.is_categorical_ is None:
                    self.is_categorical_ = np.zeros(X.shape[1], bool)
                self.is_ordinal_ = pd.Series(self.is_ordinal_,
                                             index=self.columns_)
                self.is_categorical_ = pd.Series(self.is_categorical_,
                                                 index=self.columns_)
            else:
                categorical_features = self.categorical_features
                (self.is_categorical_,
                 self.known_categories_) = _check_categories(categorical_features,
                                                             X)
                if self.is_categorical_ is None:
                    self.is_categorical_ = np.zeros(X.shape[1], bool)
                self.is_ordinal_ = np.zeros(self.is_categorical_.shape,
                                            bool)
                self.columns_ = np.arange(X.shape[1])
    
            self.features_ = {}
            self.encoders_ = {}
    
            self.column_info_ = _get_column_info(X,
                                                 self.columns_,
                                                 self.is_categorical_,
                                                 self.is_ordinal_,
                                                 default_encoders=self.default_encoders)
            # include each column as a Feature
            # so that their columns are built if needed
    
            for col_ in self.columns_:
                self.features_[col_] = Feature((col_,), str(col_), None, pure_columns=True) 
    
            # find possible interactions and other features
    
            tmp_cache = {}
    
            for term in self.terms:
                if isinstance(term, Feature):
                    self.features_[term] = term
                    build_columns(self.column_info_,
                                  X,
                                  term,
                                  encoders=self.encoders_,
                                  col_cache=tmp_cache,
                                  fit=True) # these encoders won't have been fit yet
                    for var in term.variables:
                        if var not in self.features_ and isinstance(var, Feature):
                                self.features_[var] = var
                elif term not in self.column_info_:
                    # a tuple of features represents an interaction
                    if type(term) == type((1,)): 
                        names = []
                        column_map = {}
                        column_names = {}
                        idx = 0
                        for var in term:
                            if var in self.features_:
                                var = self.features_[var]
                            cols, cur_names = build_columns(self.column_info_,
                                                            X,
                                                            var,
                                                            encoders=self.encoders_,
                                                            col_cache=tmp_cache,
                                                            fit=True) # these encoders won't have been fit yet
                            column_map[var.name] = range(idx, idx + cols.shape[1])
                            column_names[var.name] = cur_names
                            idx += cols.shape[1]                 
                            names.append(var.name)
                        encoder_ = Interaction(names, column_map, column_names)
                        self.features_[term] = Feature(term, ':'.join(n for n in names), encoder_)
                    elif isinstance(term, Column):
                        self.features_[term] = Feature((term,), term.name, None, pure_columns=True)
                    else:
                        raise ValueError('each element in a term should be a Feature, Column or identify a column')
                    
            # build the mapping of terms to columns and column names
    
            self.column_names_ = {}
            self.column_map_ = {}
            self.terms_ = [self.features_[t] for t in self.terms]
            
            idx = 0
            if self.intercept:
                self.column_map_['intercept'] = slice(0, 1)
                idx += 1 # intercept will be first column
            
            for term, term_ in zip(self.terms, self.terms_):
                term_df, term_names = build_columns(self.column_info_,
                                                    X,
                                                    term_,
                                                    encoders=self.encoders_)
                self.column_names_[term] = term_names
                self.column_map_[term] = slice(idx, idx + term_df.shape[1])
                idx += term_df.shape[1]
        
            return self
        
        def transform(self, X, y=None):
            """
            Build design on X after fitting.
    
            Parameters
            ----------
            X : array-like
    
            y : None
                Ignored. This parameter exists only for compatibility with
                :py:class:`sklearn.pipeline.Pipeline`.
            """
            check_is_fitted(self)
            return build_model(self.column_info_,
                               X,
                               self.terms_,
                               intercept=self.intercept,
                               encoders=self.encoders_)
        
        # ModelSpec specific methods
    
        @property
        def names(self, help='Name for each term in model specification.'):
            names = []
            if self.intercept:
                names = ['intercept']
            return names + [t.name for t in self.terms_]
            
    
        def build_submodel(self,
                           X,
                           terms):
            """
            Build design on X after fitting.
    
            Parameters
            ----------
            X : array-like
                X on which columns are evaluated.
    
            terms : [Feature]
                Sequence of features
    
            Returns
            -------
            D : array-like
                Design matrix created with `terms`
            """
    
            return build_model(self.column_info_,
                               X,
                               terms,
                               intercept=self.intercept,
                               encoders=self.encoders_)
    
        def build_sequence(self,
                           X,
                           anova_type='sequential'):
            """
            Build implied sequence of submodels 
            based on successively including more terms.
    
            Parameters
            ----------
            X : array-like
                X on which columns are evaluated.
    
            anova_type: str
                One of "sequential" or "drop".
    
            Returns
            -------
    
            models : generator
                Generator for sequence of models for ANOVA.
    
            """
    
            check_is_fitted(self)
    
            col_cache = {}  # avoid recomputing the same columns
    
            dfs = []
    
            if self.intercept:
                df_int = pd.DataFrame({'intercept':np.ones(X.shape[0])})
                if isinstance(X, (pd.Series, pd.DataFrame)):
                    df_int.index = X.index
                dfs.append(df_int)
            else:
                df_int = pd.DataFrame({'zero':np.zeros(X.shape[0])})
                if isinstance(X, (pd.Series, pd.DataFrame)):
                    df_int.index = X.index
                dfs.append(df_int)
    
            for term_ in self.terms_:
                term_df, _  = build_columns(self.column_info_,
                                            X,
                                            term_,
                                            col_cache=col_cache,
                                            encoders=self.encoders_,
                                            fit=False)
                if isinstance(X, (pd.Series, pd.DataFrame)):
                    term_df.index = X.index
    
                dfs.append(term_df)
    
            if anova_type == 'sequential':
                if isinstance(X, (pd.Series, pd.DataFrame)):
                    return (pd.concat(dfs[:i], axis=1) for i in range(1, len(dfs)+1))
                else:
                    return (np.column_stack(dfs[:i]) for i in range(1, len(dfs)+1))
            elif anova_type == 'drop':
                if isinstance(X, (pd.Series, pd.DataFrame)):
                    return (pd.concat([dfs[j] for j in range(len(dfs)) if j != i], axis=1) for i in range(len(dfs)))
                else:
                    return (np.column_stack([dfs[j] for j in range(len(dfs)) if j != i]) for i in range(len(dfs)))
            else:
                raise ValueError('anova_type must be one of ["sequential", "drop"]')
    



```python
print(inspect.getsource(poly))
```

    def poly(col,
             degree=1,
             intercept=False,
             raw=False,
             name=None):
    
        """
        Create a polynomial Feature
        for a given column.
        
        Additional `args` and `kwargs`
        are passed to `Poly`.
    
        Parameters
        ----------
    
        col : column identifier or Column
            Column to transform.
    
        degree : int, default=1
            Degree of polynomial.
    
        intercept : bool, default=False
            Include a column for intercept?
    
        raw : bool, default=False
            If False, perform a QR decomposition on the resulting
            matrix of powers of centered and / or scaled features.
    
        name : str (optional)
            Defaults to one derived from col.
    
        Returns
        -------
    
        var : Feature
        """
        shortname, klass = 'poly', Poly
        encoder = klass(degree=degree,
                        raw=raw,
                        intercept=intercept) 
        if name is None:
            if isinstance(col, Column):
                name = col.name
            else:
                name = str(col)
    
            kwargs = {}
            if intercept:
                kwargs['intercept'] = True
            if raw:
                kwargs['raw'] = True
    
            _args = _argstring(degree=degree,
                               **kwargs)
            if _args:
                name = ', '.join([name, _args])
    
            name = f'{shortname}({name})'
    
        return derived_feature([col],
                                name=name,
                                encoder=encoder)
    



```python
print(inspect.getsource(summarize))
```

    def summarize(results,
                  conf_int=False):
        """
        Take a fit statsmodels and summarize it
        by returning the usual coefficient estimates,
        their standard errors, the usual test
        statistics and P-values as well as 
        (optionally) 95% confidence intervals.
    
        Based on:
    
        https://stackoverflow.com/questions/51734180/converting-statsmodels-summary-object-to-pandas-dataframe
    
        Parameters
        ----------
    
        results : a results object
    
        conf_int : bool (optional)
            Include 95% confidence intervals?
    
        """
        tab = results.summary().tables[1]
        results_table = pd.read_html(tab.as_html(),
                                     index_col=0,
                                     header=0)[0]
        if not conf_int:
            columns = ['coef',
                       'std err',
                       't',
                       'P>|t|']
            return results_table[results_table.columns[:-2]]
        return results_table
    

