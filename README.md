# API com Algoritmos de Aprendizado de Máquina para Inferência de Diagnóstico de Diabetes em Mulheres

O objetivo deste projeto é utilizar backend + inteligência artificial para fins didáticos com Python.
Todos os materiais utilizados neste projeto estão disponíveis abertamente na internet.

### Materiais e Ferramentas:

- Framework:
	- FastAPI: https://fastapi.tiangolo.com/
- Machine Learning:
	- scikit-learn: https://scikit-learn.org/stable/index.html
- Base de Dados:
	- Kaggle: https://www.kaggle.com/

### Base de Dados

A base de dados é originalmente do Instituto Nacional de Diabetes e Doenças Digestivas e Renais. O objetivo é prever, com base em medidas diagnósticas, se um paciente tem diabetes. Está disponível no Kaggle e é possível ver mais detalhes por lá.

- Diabetes Dataset: https://www.kaggle.com/datasets/mathchi/diabetes-data-set

### Algoritmos de Aprendizado de Máquina

Na API estão disponíveis os algoritmos de aprendizado de máquina:

- Decision Tree: https://scikit-learn.org/stable/modules/tree.html
- KNN: https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification
- Neural network: https://scikit-learn.org/stable/modules/neural_networks_supervised.html

### Executando o projeto

Para instalar as bibliotecas necessárias do projeto, execute:

    pip install -r requirements.txt


Para executar a API:

    uvicorn main:app --reload

A API poderá ser acessada localmente em: http://127.0.0.1:8000/

Para acessar o Swagger: http://127.0.0.1:8000/docs
