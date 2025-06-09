# Modelo de ML para predição de aluguéis de imóveis

## Modelo de Regressão Linear feito em Python com a biblioteca Scikit Learn

### Visão Geral
Para o projeto foi utilizado um dataset com valores de aluguéis e preços de imóveis de São Paulo de 2019 da plataforma kaggle;
<br/>
ele foi tratado e foram normalizadas tanto as variáveis categóricas quanto as numéricas para o treinamento do modelo.
<p>
  De modo simples o usuário insere as características do imóvel, como a metragem, se tem garagem, piscina, número de quartos, preço do condomínio, dentre outras variáveis, e com isso
  é retornado um preço aproximado do aluguel que deve ser pago/cobrado com base no <strong>modelo de Machine Learning RandomForestRegressor</strong>
</p>
<p>
  Apesar do dataset ser de 2019, foi calculada uma taxa de correção do aumento de aluguel acumulada de 2019 a 2025 com dados de sites de busca de imóveis e o Secovi-sp
</p>

### Estrutura do código
<p>
  O código é separado em arquivos python dentro do folder src que são responsáveis pela implementação dos métodos que realizam ações no main.py. Como exemplo temos o arquivo preprocessing.py
  que é responsável por construir os <strong>pipelines categóricos e numéricos</strong>; o features.py que é responsável por capturar as features/colunas do dataset e classificá-las em categóricas e numéricas
  para serem usadas no <strong>column transformer</strong> posteriormente.
  <br>
  <br>
  No main.py estes métodos são apenas executados e retornam valores para uso posterior no main (quando necessário).
</p>

### Acurácia do Modelo
<p>
  <del>
     Atualmente o modelo tem uma <strong>acurácia = 0.67</strong> e <strong>R2 = 0.66</strong>, isto é, acerta em 67% das vezes e explica por meio da regressão 66% dos dados.
  <br> 
   Ainda estão sendo pensadas técnicas estatísticas para aumentar a acurácia e o valor de R2.
  </del>
</p>

<p>
  O modelo foi trocado de regressão linear para <strong>RandomForestRegressor</strong> e com isso foi atingida uma acurácia de <strong>85%</strong> e R2 de <strong>84%</strong>.
  <br>
  Assim o modelo se mostra mais apto para predições. 
</p>