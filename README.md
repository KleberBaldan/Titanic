# Entendimento do Negócio
- *Tema:*
> Análise de Sobreviventes do Titanic: Machine Learning From Disaster - Kaggle.
- *Problema:*
> Avaliar e modelar os dados de modo a ser possível atingir o objetivo.
- *Objetivo:*
> Predizer a sobrevivência de integrantes que não possuem tal informação.

# Entendimento dos Dados
- *Formato*:
> Temos dois arquivos sendo um com os dados de sobreviência, para treino, e outro sem estes dados, para predição.
- *Tamanho*:
> São 1309 registros.
- *Tipo dos Dados*:
> Os dados são formados por 12 atributos onde 5 são categóricos e 7 quantitativos.

# Definição de  Bibliotecas


```python
import numpy as np

import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
```

# Avaliação dos Dados


```python
treino = pd.read_csv('../data/train.csv')
teste = pd.read_csv('../data/test.csv')
```

### Avaliação de Nulos


```python
treino_nulos = treino.isnull().sum()
teste_nulos = teste.isnull().sum()
pd.concat([treino_nulos, teste_nulos], axis=1).rename({0: 'treino', 1: 'teste'}, axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>treino</th>
      <th>teste</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PassengerId</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Survived</th>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Name</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>177</td>
      <td>86.0</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Ticket</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Cabin</th>
      <td>687</td>
      <td>327.0</td>
    </tr>
    <tr>
      <th>Embarked</th>
      <td>2</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



### Concatenação para Engenharia de Dados


```python
dados = pd.concat([treino, teste])
dados.isnull().sum()
```




    PassengerId       0
    Survived        418
    Pclass            0
    Name              0
    Sex               0
    Age             263
    SibSp             0
    Parch             0
    Ticket            0
    Fare              1
    Cabin          1014
    Embarked          2
    dtype: int64




```python
ATRIBUTOS = dados.columns.values
```


```python
del treino, teste
```


```python
dados.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1309.000000</td>
      <td>891.000000</td>
      <td>1309.000000</td>
      <td>1046.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1308.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>655.000000</td>
      <td>0.383838</td>
      <td>2.294882</td>
      <td>29.881138</td>
      <td>0.498854</td>
      <td>0.385027</td>
      <td>33.295479</td>
    </tr>
    <tr>
      <th>std</th>
      <td>378.020061</td>
      <td>0.486592</td>
      <td>0.837836</td>
      <td>14.413493</td>
      <td>1.041658</td>
      <td>0.865560</td>
      <td>51.758668</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.170000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>328.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.895800</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>655.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>982.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>39.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.275000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1309.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>9.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



### Funções Gerais


```python
RANDOM_STATE = 1

M_CLASSIFICACAO = [
    {
        'modelo': AdaBoostClassifier,
        'parametros': {
            'n_estimators':[10, 25, 50],
            'learning_rate': [0.001, 0.01, ],
            'algorithm':['SAMME.R','SAMME'],
            'random_state': [RANDOM_STATE]
        }
    },
    {
        'modelo': BaggingClassifier,
        'parametros': {
            'n_estimators':[5, 10, 50],
            'max_samples': [0.5, 1.0],
            'max_features': [0.1, 0.5],
            'random_state': [RANDOM_STATE]
        }
    },
    {
        'modelo': ExtraTreesClassifier,
        'parametros': {
            'n_estimators':[50, 100],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 1],
            'min_samples_split': [1, 2],
            'max_features': ['auto', 'sqrt', 'log2'],
            'random_state': [RANDOM_STATE]
        }
    },
    {
        'modelo': GradientBoostingClassifier,
        'parametros': {
            'loss':['deviance', 'exponential'],
            'learning_rate': [0.001, 0.01],
            'n_estimators': [100, 150],
            'subsample': [1.0, 1.5],
            'criterion': ['friedman_mse', 'mse', 'mae'],
            'max_depth': [3, 5],
            'random_state': [RANDOM_STATE]
        }
    },
    {
        'modelo': RandomForestClassifier,
        'parametros': {
            'n_estimators':[50, 100],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 1],
            'min_samples_split': [1, 2],
            'max_features': ['auto', 'sqrt', 'log2'],
            'random_state': [RANDOM_STATE]
        }
    },
    {
        'modelo': RidgeClassifier,
        'parametros': {
            'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            'random_state': [RANDOM_STATE]
        }
    },
    {
        'modelo': SGDClassifier,
        'parametros': {
            'loss': ['hinge', 'log', 'perceptron', 'huber', 'squared_loss'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'random_state': [RANDOM_STATE]
        }
    }
]

def avaliacao_hiper_modelos(modelo: list, treino_teste: list, divisoes: int = 10) -> pd.DataFrame:
    resultado = list()
    gscv = GridSearchCV(estimator=modelo['modelo'](), param_grid=modelo['parametros'], n_jobs=-1, scoring='accuracy',
                        cv=KFold(n_splits=divisoes, random_state=RANDOM_STATE), verbose=1)
    gscv.fit(treino_teste[0], treino_teste[2])
    pred = gscv.predict(treino_teste[1])
    resultado.append({
        'modelo': str(modelo['modelo']).split('.')[-1][:-2],
        'parametros': gscv.best_params_,
        'precisao': gscv.best_score_,
        'predicao': pred,
        'instancia': gscv
    })
    return pd.DataFrame(resultado)

def avaliar_modelos(modelos: list, treino_teste: list) -> pd.DataFrame:
    resultado = list()
    for modelo in modelos:
        clf = modelo['modelo']()
        clf.fit(treino_teste[0], treino_teste[2])
        resultado.append({
            'modelo':str(modelo['modelo']).split('.')[-1][:-2],
            'precisao': accuracy_score(treino_teste[3], clf.predict(treino_teste[1])),
        })
    return pd.DataFrame(resultado)

def exibir_relacoes(dados: pd.DataFrame) -> None:
    plt.figure(dpi=96)
    sns.heatmap(dados.corr(), annot=True, cmap='viridis', fmt='.2f')
    plt.title('Relação Entre Atributos');
```

## Sexo (Sex)


```python
dados['SexId'] = dados.Sex.map({'female': 0, 'male': 1})
```

## Tarifa (Fare)
- Nulos? Como solucionar?
> Conforme avaliado abaixo, estarei utilizando a mediana da tarifa por pessoa dentro da classe ao qual se encontra.
- A Classe Social (Pclass) influencia na tarifa?
> Sim.
- A tarifa (Fare) está relacionada ao ingresso (Ticket)? O ingresso é individual?
> Sim. Vemos que a tarifa cobrada foi por ingresso, e não, o mesmo não é individual.
> Como visto no gráfico, temos até 11 pessoas compartilhando um mesmo ingresso.
- A média de preço do ingresso individual é o mesmo para ingressos utilizados por múltiplas indíviduos?
> Não. Nota-se que conforme aumenta o número de indivíduos, a média da tarifa do ingresso também aumenta dentro da classe, o qual confirma a tarifa por ingresso.
> Ou seja, quando analisado sem o devido agrupamento dos ingressos, avaliamos a tarifa do mesmo ingresso por até 11x.
> Deste modo, afim de evitarmos problemas, substituirei o atributo Tarifa (Fare) pelo atributo Tarifa Individual (IndividualRate) dividindo a mesma pelo número de pessoas que utilizaram o ingresso.


```python
dados.loc[dados.Fare.isnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>SexId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>152</th>
      <td>1044</td>
      <td>NaN</td>
      <td>3</td>
      <td>Storey, Mr. Thomas</td>
      <td>male</td>
      <td>60.5</td>
      <td>0</td>
      <td>0</td>
      <td>3701</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(dpi=96)
sns.boxplot(x='Pclass', y='Fare', data=dados, palette='viridis')
plt.title('Tarifa Por Classe Social')
plt.xlabel('Classe Social')
plt.ylabel('Tarifa');
```


    
![png](README_files/README_19_0.png)
    



```python
dados1 = dados[['PassengerId', 'Ticket']].groupby('Ticket').count().rename({'PassengerId': 'AmountOfPeople'}, axis=1)
```


```python
plt.figure(dpi=96)
sns.barplot(x='AmountOfPeople', y='Ticket', data=dados1.reset_index().groupby('AmountOfPeople').count().reset_index(), palette='viridis')
plt.title('Quant. de Ingressos Utilizados pelo Número de Indivíduos')
plt.xlabel('Número de Indivíduos')
plt.ylabel('Ingressos');
```


    
![png](README_files/README_21_0.png)
    



```python
dados1.reset_index(inplace=True)
dados1['Pclass'], dados1['Fare'] = zip(* dados1.Ticket.apply(lambda ingresso: [dados.Pclass.loc[dados.Ticket==ingresso].min(), dados.Fare.loc[dados.Ticket==ingresso].mean()]))
```


```python
plt.figure(dpi=96)
sns.barplot(x='AmountOfPeople', y='Fare', hue='Pclass', data=dados1, palette='viridis')
plt.title('Tarifa Média por Quant. de Indivíduos e Classe Social')
plt.xlabel('Quantidade de Indivíduos')
plt.ylabel('Tarifa Média');
```


    
![png](README_files/README_23_0.png)
    



```python
dados1 = dados[['Pclass', 'Ticket', 'Fare']].groupby(['Ticket', 'Pclass']).mean().reset_index()
```


```python
fig, [ax0, ax1] = plt.subplots(1, 2, dpi=96)
sns.boxplot(x='Pclass', y='Fare', data=dados, palette='viridis', ax=ax0)
ax0.set_title('Tar. Por Classe')
ax0.set_xlabel('Classe Social')
ax0.set_ylabel('Tarifa')

sns.boxplot(x='Pclass', y='Fare', data=dados1, palette='viridis', ax=ax1)
ax1.set_title('Tar. Agr. por Ingr. e Classe')
ax1.set_xlabel('Classe Social')
ax1.set_ylabel('Tarifa')
plt.tight_layout();
```


    
![png](README_files/README_25_0.png)
    


### Correção de Outliers


```python
def definir_tarifa_individual(args):
    return args[1] / dados.PassengerId.loc[dados.Ticket==args[0]].count()

def ajustar_correcao_outliers(args):
    tarifa, tarifas = args[0], dados.IndividualRate.loc[dados.Pclass==args[1]]
    q1, q3 = tarifas.quantile(.25), tarifas.quantile(.75)
    limite_inferior = q1 - 1.5 * (q3 - q1)
    limite_superior = q3 + 1.5 * (q3 - q1)
    return tarifas.median() if tarifa < limite_inferior or tarifa > limite_superior else tarifa

dados['IndividualRate'] = dados[['Ticket', 'Fare']].apply(definir_tarifa_individual, axis=1)
dados['IndividualRate'] = dados[['IndividualRate', 'Pclass']].apply(ajustar_correcao_outliers, axis=1)
```


```python
plt.figure(dpi=96)
sns.boxplot(x='Pclass', y='IndividualRate', data=dados, palette='viridis')
plt.title('Tarifa Indiv. por Classe')
plt.xlabel('Classe Social')
plt.ylabel('Tarifa');
```


    
![png](README_files/README_28_0.png)
    


### Correção de Nulos


```python
dados.IndividualRate.fillna(dados.IndividualRate.loc[dados.Pclass==3].median(), inplace=True)
```

### Exclusões de Itens Desnecessários


```python
dados.drop('Fare', axis=1, inplace=True)
del ax0, ax1, dados1, fig
```

## Embarque (Embarked)
- Nulos? Como solucionar?
> Devido há ausência de melhores informações, estarei utilizando o embarque padrão.
- Qual a relação de Embarque com os demais atributos?
> Conforme gráfico, não há uma relação forte com os demais atributos.
- Ingresso (Ticket) influencia no Embarque?
> Não. Vemos na lista de Ingressos (Ticket) próximos que apenas 2 registros do mesmo padrão de ingresso apresentam embarque diferente de 'S'


```python
dados.loc[dados.Embarked.isnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>SexId</th>
      <th>IndividualRate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>61</th>
      <td>62</td>
      <td>1.0</td>
      <td>1</td>
      <td>Icard, Miss. Amelie</td>
      <td>female</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>113572</td>
      <td>B28</td>
      <td>NaN</td>
      <td>0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>829</th>
      <td>830</td>
      <td>1.0</td>
      <td>1</td>
      <td>Stone, Mrs. George Nelson (Martha Evelyn)</td>
      <td>female</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>113572</td>
      <td>B28</td>
      <td>NaN</td>
      <td>0</td>
      <td>40.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
dados['EmbarkedId'] = dados.Embarked.map({embarque: indx for indx, embarque in enumerate(dados.Embarked.unique().tolist())})
exibir_relacoes(dados)
```


    
![png](README_files/README_35_0.png)
    



```python
ingresso = 113572
dados1 = dados.loc[dados.Ticket.str.isdecimal()].copy()
dados1.Ticket = dados1.Ticket.astype('int64')
dados2 = dados1[['Ticket', 'Embarked']].loc[dados1.Ticket<ingresso].sort_values('Ticket').tail(5)
dados3 = dados1[['Ticket', 'Embarked']].loc[dados1.Ticket>ingresso].sort_values('Ticket').head(5)
dados1 = pd.concat([dados2, dados3])
print('Lista de Ingressos próximos ao', ingresso)
dados1
```

    Lista de Ingressos próximos ao 113572





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticket</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>166</th>
      <td>113505</td>
      <td>S</td>
    </tr>
    <tr>
      <th>26</th>
      <td>113509</td>
      <td>C</td>
    </tr>
    <tr>
      <th>54</th>
      <td>113509</td>
      <td>C</td>
    </tr>
    <tr>
      <th>351</th>
      <td>113510</td>
      <td>S</td>
    </tr>
    <tr>
      <th>252</th>
      <td>113514</td>
      <td>S</td>
    </tr>
    <tr>
      <th>390</th>
      <td>113760</td>
      <td>S</td>
    </tr>
    <tr>
      <th>802</th>
      <td>113760</td>
      <td>S</td>
    </tr>
    <tr>
      <th>435</th>
      <td>113760</td>
      <td>S</td>
    </tr>
    <tr>
      <th>763</th>
      <td>113760</td>
      <td>S</td>
    </tr>
    <tr>
      <th>185</th>
      <td>113767</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



### Correção de Nulos


```python
dados.Embarked.fillna(dados.Embarked.mode().values[0], inplace=True)
dados['EmbarkedId'] = dados.Embarked.map({embarque: indx for indx, embarque in enumerate(dados.Embarked.unique().tolist())})
```

### Exclusão de Itens Desnecessários


```python
del dados1, dados2, dados3, ingresso
```

## Cabine (Cabin)
- Nulos? Como solucionar?
> Devido há ausência de dados superior a 50%, estou excluindo este atributo.


```python
nulos = dados.Cabin.isnull().sum()
print('Cabin:', nulos, nulos / dados.PassengerId.count())
```

    Cabin: 1014 0.774637127578304


### Exclusão de Itens Desnecessários


```python
dados.drop('Cabin', axis=1, inplace=True)
del nulos
```

## Idade (Age)
- Nulos? Como solucionar?
> Conforme identificado, este campo não terá grandes resultados utilizando modelos de regressão linear.
> Deste modo, criei um atributo para classificação entre crianças, jovens, adultos, idosos.
- Há relações fortes com outros atributos?
> Vejo boas relações com Pclass e IndividualRate, e também a possibilidade do desenvolvimento de atributos que podem ser pertinentes a este, tais como:
    > - Married (Casado): Identifiquei que os indivíduos do sexo feminino, quando casadas, traz como primeiro nome o nome do esposo, portanto, criei o atributo Família (FamilyName)
    > composto pelo primeiro nome e sobrenome. Quando encontrado 2 indivíduos com o mesmo nome de família, identificamos o casal.
    > - Widowers (Viúvo): Avaliando o atributo Casado (Married), notei que indivíduos do sexo feminino nesta situação, possuem o Título (Title) de 'Mrs', porém
    > nem toda Senhora (Mrs) está acompanhada de seu marido (Família / FamilyName). Avaliei a média de idade destes indivíduos e notei que estão próximos aos 40 anos,
    > o qual, me permite supor que estes indivíduos possam ser Viúvas.
    > - Responsable (Responsável): Este atributo será o responsável por informar se o indivíduo é responsável por si mesmo ou dependente de alguém (Criança).


```python
dados.loc[dados.Age.isnull()].sample(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Embarked</th>
      <th>SexId</th>
      <th>IndividualRate</th>
      <th>EmbarkedId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>170</th>
      <td>1062</td>
      <td>NaN</td>
      <td>3</td>
      <td>Lithman, Mr. Simon</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>S.O./P.P. 251</td>
      <td>S</td>
      <td>1</td>
      <td>7.55</td>
      <td>0</td>
    </tr>
    <tr>
      <th>380</th>
      <td>1272</td>
      <td>NaN</td>
      <td>3</td>
      <td>O'Connor, Mr. Patrick</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>366713</td>
      <td>Q</td>
      <td>1</td>
      <td>7.75</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
exibir_relacoes(dados)
```


    
![png](README_files/README_47_0.png)
    


### Criação de Atributos:
- LastName
- Title
- FamilyName
- Married
- Widowers
- Responsable


```python
def extracao_atributos(nome_completo):
    avaliacao, nomes, itens = [',', '.', '('], [], [None, nome_completo]

    for indx in range(len(avaliacao)):
        itens = itens[1].split(avaliacao[indx])
        nomes.append(itens[0].strip().split(' ')[0] if indx < 2 else ' '.join([itens[0].strip().split(' ')[0], nomes[0]]))

    return nomes

dados['LastName'], dados['Title'], dados['FamilyName'] = zip(* dados.Name.apply(extracao_atributos))
```


```python
def definir_casados(args):
    return [1, 0, 1] if not args[1] in ['Master', 'Miss'] and dados.PassengerId.loc[(dados.FamilyName==args[0]) & ((dados.Title!='Master') | (dados.Title!='Miss'))].count() > 1 else [0, 0, 0]

dados['Married'], dados['Widowers'], dados['Responsable'] = zip(* dados[['FamilyName', 'Title']].apply(definir_casados, axis=1))
```


```python
def definir_viuvos(args):
    return 1 if args[0] == 'Mrs' and args[1] == 0 else 0

dados['Widowers'] = dados[['Title', 'Married']].apply(definir_viuvos, axis=1)
```


```python
def identificar_responsaveis(args):
    ingresso, responsavel, sobrenome, familia, idade = args[0], args[1], args[2], args[3], args[4]

    if responsavel == 0:
        familia_correta = dados.FamilyName.loc[(dados.LastName==sobrenome) & (dados.Responsable==1) & (dados.Ticket==ingresso)].values
        familia, responsavel = ([familia, 1] if len(familia_correta) == 0 else [familia_correta, responsavel]) if pd.isna(idade) \
            else ([familia, 0] if len(familia_correta) == 0 and idade < 14 else ([familia, 1] if len(familia_correta) == 0 else [familia_correta[0], responsavel]))
    return familia, responsavel

dados['FamilyName'], dados['Responsable'] = zip(* dados[['Ticket', 'Responsable', 'LastName', 'FamilyName', 'Age']].apply(identificar_responsaveis, axis=1))
```

    /home/kleberbaldan/.anaconda3/envs/Titanic-MLD/lib/python3.8/site-packages/numpy/core/_asarray.py:83: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
      return array(a, dtype, copy=False, order=order)



```python
exibir_relacoes(dados)
```


    
![png](README_files/README_53_0.png)
    


### Definição de Atributos e Visualização dos Dados


```python
dados1 = dados.copy()
dados1.set_index('PassengerId', inplace=True)
dados1.drop([atributo for atributo in dados1.columns.values if dados1[atributo].dtype=='O'], axis=1, inplace=True)
dados1.drop(['Survived', 'SibSp', 'Parch', 'SexId', 'EmbarkedId'], axis=1, inplace=True)
dados1.dropna(inplace=True)
```


```python
X = dados1.drop('Age', axis=1)
y = dados1.Age
```


```python
X_pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(StandardScaler().fit_transform(X))
plt.figure(dpi=96)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='viridis', alpha=.6)
plt.title('Avaliação dos Dados');
```


    
![png](README_files/README_57_0.png)
    


### Criação de Atributos:
- AgeGroup


```python
# Avaliando a menor idade de um indivíduo casado.
dados.Age.loc[(dados.Married==1) & (~dados.Age.isnull())].min()
```




    14.0




```python
# FAIXA_ETARIA = ['Criança', 'Adolescente', 'Adulto', 'Idoso']
# dados['AgeGroup'], dados['AgeGroupId'] = zip(* dados.Age.apply(lambda idade: [FAIXA_ETARIA[0], 0] if idade < 14 else (
#     [FAIXA_ETARIA[1], 1] if idade < 21 else (
#         [FAIXA_ETARIA[2], 2] if idade < 60 else (
#             [FAIXA_ETARIA[3], 3] if not pd.isna(idade) else [pd.NA, np.nan])
#     )
# )))
FAIXA_ETARIA = ['Criança', 'Adulto', 'Idoso']
dados['AgeGroup'], dados['AgeGroupId'] = zip(* dados.Age.apply(lambda idade: [FAIXA_ETARIA[0], 0] if idade < 14 else (
        [FAIXA_ETARIA[1], 1] if idade < 60 else (
            [FAIXA_ETARIA[2], 2] if not pd.isna(idade) else [pd.NA, np.nan])
    )
))
```


```python
dados.set_index('PassengerId', inplace=True)
dados1 = dados.copy()
exibir_relacoes(dados1)
```


    
![png](README_files/README_61_0.png)
    



```python
dados1.drop([atributo for atributo in dados1.columns.values if dados1[atributo].dtype=='O'], axis=1, inplace=True)
dados1.drop(['Survived', 'Age', 'SibSp', 'Parch'], axis=1, inplace=True)
exibir_relacoes(dados1)
```


    
![png](README_files/README_62_0.png)
    



```python
treino = dados1.dropna()
teste = dados1.loc[dados1.AgeGroupId.isnull()].drop('AgeGroupId', axis=1)
```


```python
sns.pairplot(treino, hue='AgeGroupId', palette='viridis');
```

    /home/kleberbaldan/.anaconda3/envs/Titanic-MLD/lib/python3.8/site-packages/seaborn/distributions.py:305: UserWarning: Dataset has 0 variance; skipping density estimate.
      warnings.warn(msg, UserWarning)
    /home/kleberbaldan/.anaconda3/envs/Titanic-MLD/lib/python3.8/site-packages/seaborn/distributions.py:305: UserWarning: Dataset has 0 variance; skipping density estimate.
      warnings.warn(msg, UserWarning)
    /home/kleberbaldan/.anaconda3/envs/Titanic-MLD/lib/python3.8/site-packages/seaborn/distributions.py:305: UserWarning: Dataset has 0 variance; skipping density estimate.
      warnings.warn(msg, UserWarning)
    /home/kleberbaldan/.anaconda3/envs/Titanic-MLD/lib/python3.8/site-packages/seaborn/distributions.py:305: UserWarning: Dataset has 0 variance; skipping density estimate.
      warnings.warn(msg, UserWarning)



    
![png](README_files/README_64_1.png)
    



```python
X = treino.drop('AgeGroupId', axis=1)
y = treino.AgeGroupId
```


```python
std = StandardScaler()
X_normalizado = std.fit_transform(X)
```


```python
X_pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_normalizado)
plt.figure(dpi=96)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', alpha=.6)
plt.title('Avaliação dos Dados');
```


    
![png](README_files/README_67_0.png)
    


### Correção de Nulos


```python
treino_teste = train_test_split(X_normalizado, y, test_size=.3, random_state=RANDOM_STATE)
```


```python
avaliacao = avaliar_modelos(M_CLASSIFICACAO, treino_teste).sort_values('precisao', ascending=False)
avaliacao
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>modelo</th>
      <th>precisao</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AdaBoostClassifier</td>
      <td>0.929936</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SGDClassifier</td>
      <td>0.923567</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BaggingClassifier</td>
      <td>0.910828</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GradientBoostingClassifier</td>
      <td>0.910828</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RandomForestClassifier</td>
      <td>0.910828</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ExtraTreesClassifier</td>
      <td>0.907643</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RidgeClassifier</td>
      <td>0.904459</td>
    </tr>
  </tbody>
</table>
</div>



#### Avaliação dos Melhores Modelos com hiper Parâmetros


```python
modelo = avaliacao.loc[avaliacao.precisao==avaliacao.precisao.max()].index.values[0]
df_hiper = avaliacao_hiper_modelos(modelo=M_CLASSIFICACAO[modelo], divisoes=50, treino_teste=treino_teste)
print(df_hiper['parametros'].values[-1])
df_hiper[['modelo', 'precisao']]
```

    /home/kleberbaldan/.anaconda3/envs/Titanic-MLD/lib/python3.8/site-packages/sklearn/model_selection/_split.py:293: FutureWarning: Setting a random_state has no effect since shuffle is False. This will raise an error in 0.24. You should leave random_state to its default (None), or set shuffle=True.
      warnings.warn(
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  76 tasks      | elapsed:    2.5s
    [Parallel(n_jobs=-1)]: Done 376 tasks      | elapsed:   18.2s
    [Parallel(n_jobs=-1)]: Done 600 out of 600 | elapsed:   30.9s finished


    Fitting 50 folds for each of 12 candidates, totalling 600 fits
    {'algorithm': 'SAMME.R', 'learning_rate': 0.001, 'n_estimators': 10, 'random_state': 1}





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>modelo</th>
      <th>precisao</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AdaBoostClassifier</td>
      <td>0.952667</td>
    </tr>
  </tbody>
</table>
</div>



### Matriz de Confusão


```python
plt.figure(dpi=96)
sns.heatmap(confusion_matrix(treino_teste[3].values, df_hiper['predicao'].values[-1]), annot=True, cmap='viridis', fmt='.2f');
```


    
![png](README_files/README_74_0.png)
    



```python
y = std.transform(teste)
predicao = pd.DataFrame(df_hiper['instancia'].values[-1].predict(y), columns=['AgeGroupId'])
```


```python
teste1 = pd.concat([teste.reset_index(), predicao], axis=1).set_index('PassengerId')
dados1 = pd.concat([treino, teste1])
dados1 = dados1.AgeGroupId
dados.drop('AgeGroupId', axis=1, inplace=True)
dados = pd.concat([dados, dados1], axis=1)
dados.isnull().sum()
```




    Survived          418
    Pclass              0
    Name                0
    Sex                 0
    Age               263
    SibSp               0
    Parch               0
    Ticket              0
    Embarked            0
    SexId               0
    IndividualRate      0
    EmbarkedId          0
    LastName            0
    Title               0
    FamilyName          0
    Married             0
    Widowers            0
    Responsable         0
    AgeGroup          263
    AgeGroupId          0
    dtype: int64




```python
dados.AgeGroup.loc[dados.AgeGroup.isnull()] = dados.AgeGroupId.loc[dados.AgeGroup.isnull()].map({indx: FAIXA_ETARIA[indx] for indx in range(len(FAIXA_ETARIA))})
```

    /home/kleberbaldan/.anaconda3/envs/Titanic-MLD/lib/python3.8/site-packages/pandas/core/indexing.py:670: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      iloc._setitem_with_indexer(indexer, value)


### Exclusão de Itens Desnecessários


```python
del df_hiper, avaliacao, y, treino, teste1, teste, predicao, treino_teste, dados1, X_normalizado, X_pca, X
```

## Sobreviveu (Survived)
- Nulos? Como solucionar?
- Quais foram as tendências de sobrevivência quanto ao sexo?
> Homens, apesar de estar em maior quantidade, tendem a não sobreviver quando em relação as mulheres.
- As classes influenciam na sobrevivência?
> Conforme visto, 3ª classe tende a não sobreviver, porém 1ª classe e 2ª classe seguem bem próximas.
- Faixa etária tem influência na sobrevivência? E isto se mantém dentro das classes?
> Visto que apenas crianças (< 14) tenderam a sobreviver. Não, todas as faixas exceto os idosos tendem a sobreviver na 1ª classe.
> A 2ª classe tende a seguir o modelo indiferente de classes e 3ª classe tende a não sobreviver.


```python
dados.sample(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Embarked</th>
      <th>SexId</th>
      <th>IndividualRate</th>
      <th>EmbarkedId</th>
      <th>LastName</th>
      <th>Title</th>
      <th>FamilyName</th>
      <th>Married</th>
      <th>Widowers</th>
      <th>Responsable</th>
      <th>AgeGroup</th>
      <th>AgeGroupId</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>347</th>
      <td>1.0</td>
      <td>2</td>
      <td>Smith, Miss. Marion Elsie</td>
      <td>female</td>
      <td>40.0</td>
      <td>0</td>
      <td>0</td>
      <td>31418</td>
      <td>S</td>
      <td>0</td>
      <td>13.000000</td>
      <td>0</td>
      <td>Smith</td>
      <td>Miss</td>
      <td>Marion Smith</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Adulto</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1286</th>
      <td>NaN</td>
      <td>3</td>
      <td>Kink-Heilmann, Mr. Anton</td>
      <td>male</td>
      <td>29.0</td>
      <td>3</td>
      <td>1</td>
      <td>315153</td>
      <td>S</td>
      <td>1</td>
      <td>7.341667</td>
      <td>0</td>
      <td>Kink-Heilmann</td>
      <td>Mr</td>
      <td>Anton Kink-Heilmann</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Adulto</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, [ax0, ax1] = plt.subplots(1, 2, dpi=96)

sns.countplot(x='Sex', data=dados[['Sex']].loc[~dados.Survived.isnull()], ax=ax0, palette='viridis')
ax0.set_title('Sobreviventes')
ax0.set_ylabel('Quantidade')
ax0.set_xlabel('Sexo')

sns.countplot(x='Sex', hue='Survived', data=dados[['Survived', 'Sex']].loc[~dados.Survived.isnull()], ax=ax1, palette='viridis')
ax1.set_title('Sobreviventes')
ax1.set_xlabel('Sexo')

plt.tight_layout();
```


    
![png](README_files/README_82_0.png)
    



```python
plt.figure(dpi=96)
sns.countplot(x='Pclass', hue='Sex', data=dados[['Pclass', 'Sex']].loc[~dados.Survived.isnull()], palette='viridis')
plt.title('Sobreviventes por Sexo e Classe')
plt.xlabel('Classe')
plt.ylabel('Quantidade');
```


    
![png](README_files/README_83_0.png)
    



```python
plt.figure(dpi=96)
sns.countplot(x='AgeGroup', hue='Survived', data=dados[['AgeGroup', 'Survived']].loc[~dados.Survived.isnull()], palette='viridis')
plt.title('Sobreviventes por Faixa Etária')
plt.ylabel('Quantidade')
plt.xlabel('Faixa Etária');
```


    
![png](README_files/README_84_0.png)
    



```python
plt.figure(dpi=96)
g = sns.catplot(x='AgeGroup', col='Pclass', hue='Survived', data=dados[['AgeGroup', 'Pclass', 'Survived']].loc[~dados.Survived.isnull()], palette='viridis', kind='count')
g.set_axis_labels('Faixa Etária', 'Quantidade')
g.set_titles('Sobrev. Faixa Etária e Classe: {col_name}')
```




    <seaborn.axisgrid.FacetGrid at 0x7f59b498f970>




    <Figure size 576x384 with 0 Axes>



    
![png](README_files/README_85_2.png)
    



```python
dados.isnull().sum()
```




    Survived          418
    Pclass              0
    Name                0
    Sex                 0
    Age               263
    SibSp               0
    Parch               0
    Ticket              0
    Embarked            0
    SexId               0
    IndividualRate      0
    EmbarkedId          0
    LastName            0
    Title               0
    FamilyName          0
    Married             0
    Widowers            0
    Responsable         0
    AgeGroup            0
    AgeGroupId          0
    dtype: int64




```python
exibir_relacoes(dados)
```


    
![png](README_files/README_87_0.png)
    



```python
dados1 = dados.drop([atributo for atributo in dados.columns.values if dados[atributo].dtype=='O'], axis=1)
dados1.drop(['Age','SibSp', 'Parch'], axis=1, inplace=True)
```


```python
treino = dados1.loc[~dados1.Survived.isnull()]
X = treino.drop('Survived', axis=1)
y = treino.Survived
teste = dados1.loc[dados1.Survived.isnull()].drop('Survived', axis=1)
```


```python
sns.pairplot(treino, hue='Survived', palette='viridis');
```


    
![png](README_files/README_90_0.png)
    



```python
X_normalizado = std.fit_transform(X)
treino_teste = train_test_split(X_normalizado, y, test_size=.2, random_state=RANDOM_STATE)
```


```python
avaliacao = avaliar_modelos(M_CLASSIFICACAO, treino_teste=treino_teste)
avaliacao.sort_values('precisao', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>modelo</th>
      <th>precisao</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AdaBoostClassifier</td>
      <td>0.815642</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GradientBoostingClassifier</td>
      <td>0.810056</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BaggingClassifier</td>
      <td>0.804469</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RandomForestClassifier</td>
      <td>0.804469</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ExtraTreesClassifier</td>
      <td>0.787709</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RidgeClassifier</td>
      <td>0.776536</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SGDClassifier</td>
      <td>0.748603</td>
    </tr>
  </tbody>
</table>
</div>




```python
modelo = avaliacao.loc[avaliacao.precisao==avaliacao.precisao.max()].index.values[0]
df_hiper = avaliacao_hiper_modelos(modelo=M_CLASSIFICACAO[modelo], divisoes=50, treino_teste=treino_teste)
print(df_hiper['parametros'].values[-1])
df_hiper[['modelo', 'precisao']]
```

    /home/kleberbaldan/.anaconda3/envs/Titanic-MLD/lib/python3.8/site-packages/sklearn/model_selection/_split.py:293: FutureWarning: Setting a random_state has no effect since shuffle is False. This will raise an error in 0.24. You should leave random_state to its default (None), or set shuffle=True.
      warnings.warn(
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  76 tasks      | elapsed:    2.6s
    [Parallel(n_jobs=-1)]: Done 376 tasks      | elapsed:   17.1s
    [Parallel(n_jobs=-1)]: Done 600 out of 600 | elapsed:   28.6s finished


    Fitting 50 folds for each of 12 candidates, totalling 600 fits
    {'algorithm': 'SAMME.R', 'learning_rate': 0.001, 'n_estimators': 10, 'random_state': 1}





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>modelo</th>
      <th>precisao</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AdaBoostClassifier</td>
      <td>0.790476</td>
    </tr>
  </tbody>
</table>
</div>



### Matriz de Confusão


```python
plt.figure(dpi=96)
sns.heatmap(confusion_matrix(treino_teste[3].values, df_hiper['predicao'].values[-1]), annot=True, cmap='viridis', fmt='.2f');
```


    
![png](README_files/README_95_0.png)
    



```python
y = std.transform(teste)
predicao = pd.DataFrame(df_hiper['instancia'].values[-1].predict(y), columns=['Survived'])
```


```python
teste1 = pd.concat([teste.reset_index(), predicao], axis=1).set_index('PassengerId')
teste1.Survived.to_csv('../data/survived.csv')
```
