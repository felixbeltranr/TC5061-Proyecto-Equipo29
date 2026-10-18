from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def create_features():
    #Transformadores para las columnas del DataFrame, para poder generar un modelo de predicción con regresión logística.
    # NUMÉRICAS:
    numericas_power_pipeline = Pipeline( steps=[('impMediana', SimpleImputer(strategy='median')),
                                                ('powerNum', PowerTransformer(method='yeo-johnson', standardize=True))] )

    numericas_power_nombres = ['origen_dpcapacidad',
                            'destino_dpcapacidad']

    numericas_pipeline = Pipeline( steps = [('impMediana', SimpleImputer(strategy='median')),
                                            ('escalaNum', MinMaxScaler())] )

    numericas_pipeline_nombres = ['Año_de_nacimiento',
                                'origen_latitud',
                                'origen_longitud',
                                'destino_latitud',
                                'destino_longitud']

    # BINARIAS:
    catBin_pipeline = Pipeline( steps = [('impModa', SimpleImputer(strategy='most_frequent')),
                                        ('codBinaria', OneHotEncoder(drop='if_binary',handle_unknown='ignore'))] )

    catBin_pipeline_nombres = ['Genero']

    columnasTransformer = ColumnTransformer( transformers = [('numPower', numericas_power_pipeline, numericas_power_nombres),
                                                            ('numpipe', numericas_pipeline, numericas_pipeline_nombres),
                                                            ('catBin', catBin_pipeline, catBin_pipeline_nombres)],
                                            remainder='drop' )

    return columnasTransformer