# Guia do código `main.ipynb`

## 1. O que este projeto faz

O notebook `Cod/main.ipynb` constrói um detector de imagens sintéticas usando técnicas clássicas de visão computacional e aprendizado de máquina.

O problema é tratado como uma classificação binária:

- **FAKE (0):** imagem sintética, gerada por IA;
- **REAL (1):** imagem real.

Além de verificar se o detector funciona em imagens originais, o código testa o que acontece quando as imagens passam por transformações comuns na internet, como compressão JPEG, recorte, redimensionamento e desfoque.

O fluxo completo é:

```text
Imagens do CIFAKE
       ↓
Carregamento e conversão para RGB
       ↓
Extração de características (FFT, HOG e LBP)
       ↓
Padronização das características
       ↓
Treinamento dos SVMs
       ↓
Avaliação nas imagens originais (baseline)
       ↓
Transformação apenas das imagens de teste
       ↓
Nova extração de características e avaliação de robustez
       ↓
Tabelas, modelos, CSV e gráficos
```

O ponto central do experimento é: **o modelo é treinado com imagens intactas e depois avaliado com imagens alteradas**. Assim, é possível medir quanto cada transformação prejudica o detector.

---

## 2. Conceitos importantes

### Imagem, característica e rótulo

Uma imagem possui muitos pixels. Em vez de entregar todos os pixels diretamente ao classificador, o código transforma cada imagem em um vetor numérico menor, chamado **vetor de características** ou *feature vector*.

Cada vetor recebe um rótulo:

```text
vetor de características + rótulo esperado
[0.12, 0.53, ...]       + FAKE (0)
[0.87, 0.21, ...]       + REAL (1)
```

O classificador aprende diferenças estatísticas entre os vetores das duas classes.

### Treino e teste

- O conjunto de **treino** ensina o modelo.
- O conjunto de **teste** mede o desempenho em exemplos que não foram usados no treinamento.

Misturar os dois conjuntos causa **vazamento de dados** e produz resultados artificialmente otimistas. O notebook evita um tipo importante de vazamento ao ajustar a padronização somente no conjunto de treino.

### Baseline

O **baseline** é a acurácia nas imagens originais. Ele serve como referência para calcular a degradação provocada por cada transformação:

```text
queda de acurácia = acurácia do baseline − acurácia após transformação
```

Por exemplo, se o baseline for 0,80 e a acurácia após JPEG for 0,65, a queda será 0,15, ou 15 pontos percentuais.

---

## 3. Explicação de cada parte do notebook

## 3.1 Setup e importações — células 1 a 3

O primeiro bloco importa as bibliotecas usadas:

| Biblioteca | Utilização |
|---|---|
| `os`, `pathlib` | Manipulação de caminhos e pastas |
| `random`, `numpy` | Aleatoriedade controlada e operações numéricas |
| `OpenCV (cv2)` | Leitura, conversão e transformação das imagens |
| `PIL` | Biblioteca de imagens importada, mas não utilizada no fluxo atual |
| `scikit-image` | Implementação de HOG e LBP |
| `scikit-learn` | Padronização, SVM, busca de hiperparâmetros e métricas |
| `pandas` | Construção e gravação das tabelas de resultados |
| `matplotlib`, `seaborn` | Visualizações e gráficos |
| `joblib` | Gravação e leitura dos modelos treinados |
| `tqdm` | Barras de progresso |

O código usa `SEED = 42` para tornar operações aleatórias reproduzíveis. Isso significa que, nas mesmas condições, a amostragem e os recortes devem produzir os mesmos resultados.

Depois, procura o CIFAKE em três caminhos possíveis, define as pastas de treino e teste e cria as pastas `models/` e `results/`.

Também registra uma função com `atexit` para fechar figuras abertas quando a execução terminar. Isso reduz o consumo de memória em execuções longas.

### Pontos positivos

- Uso de semente para reprodutibilidade.
- Verificação explícita da existência do dataset.
- Criação automática das pastas de saída.
- Liberação das figuras no encerramento.

### Pontos de melhoria

- `ROOT_DIR = Path.cwd()` depende da pasta a partir da qual o Jupyter foi iniciado. Se o notebook for executado dentro de `Cod/`, modelos e resultados serão criados em `Cod/`; se for iniciado na raiz, serão criados na raiz. É melhor localizar a raiz por um arquivo marcador ou definir um caminho de configuração.
- Existem importações não utilizadas, como `os`, `PIL.Image`, `train_test_split`, `LinearSVC`, `classification_report` e `confusion_matrix`. Elas podem ser removidas ou efetivamente usadas.
- As versões das dependências deveriam ser fixadas em `requirements.txt` ou `environment.yml` para facilitar a reprodução do experimento.

## 3.2 Carregamento do CIFAKE — células 4 a 6

A função `listar_imagens` percorre as pastas `FAKE/` e `REAL/`, aceita arquivos JPG, JPEG e PNG e cria duas listas:

- caminhos das imagens;
- rótulos correspondentes, usando 0 para FAKE e 1 para REAL.

A função `criar_subconjunto` seleciona aleatoriamente a mesma quantidade máxima de exemplos de cada classe. Essa técnica é chamada de **amostragem balanceada**. Ela evita que uma classe fique muito mais representada do que a outra e também permite executar protótipos mais rapidamente.

A função `carregar_imagem` usa o OpenCV para ler o arquivo. Como o OpenCV lê cores no formato BGR, o código converte a imagem para RGB, que é o formato esperado na visualização e nas demais funções.

Por fim, `mostrar_exemplos` apresenta imagens reais e sintéticas para uma inspeção visual inicial.

### Pontos positivos

- Classes balanceadas na criação dos subconjuntos.
- Verificação de erro ao abrir uma imagem.
- Inspeção visual antes do treinamento.

### Pontos de melhoria

- A função diz que mantém imagens em 32 × 32, mas não redimensiona nem valida o tamanho. Isso funciona se todos os arquivos do CIFAKE realmente tiverem essa dimensão, porém uma imagem fora do padrão pode quebrar o HOG ou gerar vetores incompatíveis.
- `pasta_classe.iterdir()` deveria ser ordenado com `sorted(...)`. A ordem de arquivos fornecida pelo sistema não é garantida. Isso também ajudaria a tornar o cache e os experimentos mais reproduzíveis.
- Seria útil detectar arquivos corrompidos antes da execução longa e registrar quais foram ignorados.
- O balanceamento deve ser verificado por contagem, mas também é importante investigar duplicatas ou imagens muito semelhantes entre treino e teste.

## 3.3 FFT: características de frequência — célula 8

A **Transformada Rápida de Fourier (FFT)** representa a imagem no domínio da frequência. Em termos simples, ela descreve quanto da imagem é formado por variações lentas ou rápidas:

- baixas frequências: regiões suaves e formas gerais;
- altas frequências: bordas, detalhes finos e ruído.

O extrator executa os seguintes passos:

1. converte a imagem para tons de cinza;
2. calcula a FFT bidimensional;
3. centraliza as baixas frequências com `fftshift`;
4. usa o logaritmo do espectro de magnitude para reduzir a diferença entre picos muito altos e valores pequenos;
5. agrupa os valores por distância ao centro do espectro;
6. calcula a média de cada faixa radial;
7. normaliza o vetor entre 0 e 1.

O resultado padrão possui 64 valores.

### Para que é usada

Geradores de imagem, redimensionamentos e compressores podem deixar padrões no domínio da frequência. A FFT tenta capturar esses sinais.

### Limitações

- A conversão para cinza descarta informações de cor.
- A média radial descarta a direção dos padrões. Frequências horizontais e verticais na mesma distância do centro tornam-se indistinguíveis.
- A normalização individual entre 0 e 1 remove diferenças absolutas de energia que talvez fossem úteis.
- O código utiliza somente a magnitude e descarta a fase da FFT. A fase contém informação estrutural e pode contribuir para robustez.
- Para imagens de apenas 32 × 32, 64 faixas radiais podem ser excessivas: algumas faixas recebem poucos pixels e podem ser instáveis.

## 3.4 HOG: formas e bordas — célula 9

O **Histogram of Oriented Gradients (HOG)** mede a direção das mudanças de intensidade da imagem. Essas mudanças geralmente correspondem a bordas e contornos.

No notebook:

- a imagem é convertida para cinza;
- é dividida em células de 8 × 8 pixels;
- cada célula conta gradientes em 8 orientações;
- grupos de 2 × 2 células são normalizados com `L2-Hys`.

### Para que é usado

O HOG descreve estrutura espacial, formas e contornos. Ele pode encontrar diferenças de geometria ou nitidez entre imagens reais e sintéticas.

### Limitações

- Em imagens de 32 × 32 há pouca resolução para representar formas complexas.
- Blur e redução de tamanho enfraquecem os gradientes e podem prejudicar bastante a característica.
- A conversão para cinza novamente elimina pistas de cor.
- Os parâmetros foram escolhidos de forma razoável, mas não foram validados sistematicamente.

## 3.5 LBP: textura local — célula 10

O **Local Binary Pattern (LBP)** compara cada pixel com seus vizinhos. Cada comparação informa se o vizinho é mais claro ou mais escuro que o centro. O conjunto das comparações forma um padrão binário.

O código utiliza:

- 8 vizinhos (`P=8`);
- raio 1 (`R=1`);
- padrões uniformes;
- histograma normalizado com 10 posições (`P + 2`).

### Para que é usado

O LBP descreve microtexturas. A hipótese é que imagens geradas por IA possam apresentar texturas locais diferentes das imagens reais.

### Limitações

- Um histograma global informa quais padrões existem, mas perde a posição onde eles aparecem.
- Apenas um raio captura uma única escala de textura.
- Compressão JPEG, blur e resize alteram relações entre pixels vizinhos.
- Assim como FFT e HOG, a versão atual ignora cor.

## 3.6 Extração em lote e cache — célula 11

A função `extrair_e_salvar_features` escolhe o extrator solicitado, processa todas as imagens e monta:

- `X`: matriz em que cada linha representa uma imagem e cada coluna uma característica;
- `y`: vetor de rótulos.

Os dados são salvos em arquivos `.npz` comprimidos. Quando o arquivo já existe, o código o carrega em vez de calcular tudo novamente. Isso é um **cache** e economiza bastante tempo.

### Pontos de melhoria

- O nome do cache contém o tipo da característica e o número de imagens, mas não registra quais imagens foram usadas, sua ordem, parâmetros do extrator ou versão do código. Dois subconjuntos diferentes com o mesmo tamanho podem reutilizar um cache incorreto.
- O cache deveria armazenar um identificador dos caminhos, parâmetros e versão do pipeline, além de validar se os rótulos carregados correspondem ao conjunto atual.
- O processamento poderia ser paralelizado com cuidado, pois cada imagem é independente.

## 3.7 Preparação, padronização e fusão — células 12 a 14

O notebook possui quatro modos de execução:

| Modo | Objetivo |
|---|---|
| `rapido` | Testar se o pipeline funciona |
| `medio` | Fazer experimentos intermediários |
| `forte` | Produzir uma avaliação maior |
| `completo` | Utilizar todo o dataset |

Para cada família de características, o código extrai treino e teste separadamente e faz verificações de dimensões e valores inválidos.

Em seguida aplica o **StandardScaler**, que transforma cada característica usando a média e o desvio-padrão observados no treino. Isso é importante para o SVM, pois impede que uma característica com escala numérica maior domine as demais.

O procedimento está correto:

```text
Treino: fit_transform → aprende média/desvio e transforma
Teste:  transform     → reutiliza média/desvio do treino
```

Depois é feita a **fusão de características** por concatenação:

```text
FUSION = FFT padronizada + HOG padronizado + LBP padronizado
```

### Para que a fusão é usada

Cada descritor observa um aspecto diferente:

- FFT: frequência;
- HOG: bordas e formas;
- LBP: textura.

A fusão tenta aproveitar informações complementares. Entretanto, mais características não garantem melhor generalização: também podem acrescentar ruído e redundância.

### Pontos de melhoria

- Salvar a seleção exata de caminhos usada em cada experimento.
- Verificar explicitamente se os rótulos e a ordem das amostras são idênticos entre FFT, HOG e LBP antes da fusão.
- Avaliar seleção de características ou PCA, sempre ajustados somente nos dados de treino.

## 3.8 Treinamento com SVM RBF — células 15 a 17

O **Support Vector Machine (SVM)** procura uma fronteira que separe as classes. O kernel RBF permite uma fronteira curva e, portanto, relações não lineares entre as características.

Os hiperparâmetros mais importantes são:

- `C`: controla o equilíbrio entre uma margem mais simples e a penalização dos erros de treino;
- `gamma`: controla o alcance da influência de cada amostra no kernel RBF.

O notebook treina quatro modelos:

1. SVM com FFT;
2. SVM com HOG;
3. SVM com LBP;
4. SVM com a fusão das três famílias.

Quando há no máximo 20 mil amostras e `AJUSTAR_HIPERPARAMETROS=True`, o código utiliza `GridSearchCV` com validação cruzada de três partes para escolher `C` entre 1, 10 e 50. Em conjuntos maiores, usa `C=10`. O `gamma` permanece em `"scale"`.

Depois do treinamento, a acurácia no teste original é calculada e os modelos e padronizadores são salvos com `joblib`.

### Técnicas usadas

- **SVM com kernel RBF:** classificação não linear.
- **Validação cruzada:** comparação de configurações usando divisões internas do treino.
- **Grid search:** teste sistemático de valores de hiperparâmetros.
- **Persistência do modelo:** armazenamento em `.pkl` para reutilização.

### Pontos de melhoria

- A busca varia apenas `C`. Deve também testar `gamma` e, idealmente, os parâmetros de FFT, HOG e LBP.
- O limite de 20 mil faz com que a execução completa não ajuste hiperparâmetros. Mais dados com parâmetros inadequados pode ser pior do que uma seleção bem feita em uma amostra representativa.
- A seleção final não deve ser baseada repetidamente no conjunto de teste. O ideal é treino + validação para decisões e um teste final usado uma única vez.
- `SVC` RBF tem custo elevado em grandes bases. Convém comparar tempo, memória e qualidade com `LinearSVC`, SVM aproximado ou outro classificador.
- Os arquivos `pickle/joblib` só devem ser carregados de fonte confiável, pois esse formato pode executar código durante a abertura.

## 3.9 Transformações digitais — células 18 a 22

### Compressão JPEG

`jpeg_compress` codifica e decodifica a imagem em memória com qualidades 90, 70 e 50. Quanto menor a qualidade, maior tende a ser a perda de detalhes. O JPEG pode alterar principalmente componentes de alta frequência.

### Crop aleatório

`crop_aleatorio` seleciona uma região e redimensiona o recorte de volta para 32 × 32. Isso simula enquadramentos e recortes de redes sociais.

Há uma imprecisão importante na descrição atual. Com `percentual=0.25`, cada dimensão mantém 75%, portanto a área mantida é:

```text
0,75 × 0,75 = 0,5625, ou 56,25%
```

Assim, aproximadamente 43,75% da área é removida, e não 25%. É necessário decidir se o parâmetro representa redução de cada dimensão ou redução total da área e corrigir nome, documentação e cálculo.

Além disso, o catálogo passa sempre a mesma semente. Isso faz com que todas as imagens usem a mesma posição relativa de recorte, em vez de uma posição diferente e reproduzível por imagem.

### Resize

`resize_down_up` reduz a imagem e depois retorna ao tamanho original. O processo remove detalhes e cria pixels interpolados, simulando reenvio ou redução de resolução.

### Blur

`blur` aplica desfoque gaussiano. Ele suaviza ruído, textura e bordas. Se o tamanho do kernel for par, o código o transforma no próximo ímpar, conforme exigido pelo OpenCV.

### Pipeline combinado

`pipeline_compartilhamento` executa:

```text
crop → redução e ampliação → JPEG 70
```

Esse caso simula várias etapas de circulação da mesma imagem.

### Pontos de melhoria

- Corrigir a semântica do percentual do crop.
- Gerar uma semente determinística diferente para cada imagem.
- Avaliar várias realizações aleatórias de crop e apresentar média e desvio-padrão, não apenas um único recorte.
- Incluir transformações previstas no objetivo, como brilho, contraste, nitidez, screenshot, rotação, alteração de proporção e múltiplas recompressões.
- Testar diferentes intensidades para construir curvas de degradação, em vez de somente pontos isolados.

## 3.10 Avaliação de robustez — células 23 a 25

Nesta fase, o SVM não é treinado novamente. Cada transformação é aplicada somente ao conjunto de teste. As características são extraídas das novas imagens, transformadas pelos mesmos `StandardScaler` do treino e classificadas pelos modelos já treinados.

Para a fusão, as três famílias padronizadas são concatenadas na mesma ordem usada no treinamento.

Para cada combinação de transformação e característica, o código registra:

- acurácia transformada;
- acurácia original do baseline;
- queda em relação ao baseline.

Os resultados são ordenados da maior para a menor queda e salvos em `results/metrics.csv`.

### Por que não retreinar nessa etapa

O objetivo é medir a robustez do modelo original. Se ele fosse retreinado para cada transformação, o experimento passaria a medir adaptação à transformação, e não resistência a uma alteração desconhecida.

### Pontos de melhoria

- O cache transformado possui a mesma fragilidade do cache original: nomeia transformação e quantidade, mas não identifica exatamente arquivos, parâmetros ou versão.
- Quando modelos são recarregados após reiniciar o kernel e as features originais não estão na memória, o baseline vira `NaN`. Nesse caso, a queda também será `NaN`. O baseline e os metadados do experimento deveriam ser salvos junto ao modelo.
- O código mede apenas acurácia. Devem ser adicionados matriz de confusão, precisão, recall, F1, especificidade, ROC-AUC e, se houver probabilidades calibradas, curvas de calibração.
- É importante informar intervalo de confiança. Uma única acurácia não mostra a incerteza da estimativa.
- A robustez pode ser diferente para REAL e FAKE. É necessário medir separadamente falso positivo e falso negativo.

## 3.11 Gráficos e síntese — células 26 a 28

O código cria dois gráficos:

1. **Barras de acurácia:** compara as famílias de características em cada transformação e mostra os baselines como linhas tracejadas.
2. **Mapa de calor:** mostra a queda de acurácia para cada combinação.

Também imprime automaticamente a maior queda e a queda média por família de característica.

### Pontos de melhoria

- As quatro linhas de baseline sobre o mesmo gráfico podem dificultar a leitura. Facetas, pequenos múltiplos ou uma linha por painel seriam mais claros.
- Um mapa de cores sequencial como `Reds` funciona para quedas positivas, mas pode representar mal casos em que a transformação melhora a acurácia. Uma escala divergente centralizada em zero seria mais adequada.
- Barras de erro ou intervalos de confiança ajudariam a distinguir variação aleatória de uma diferença consistente.

---

## 4. Como a acurácia pode ser aumentada

Antes de buscar uma acurácia maior, é importante garantir que ela seja **real e generalizável**. Uma acurácia elevada por vazamento, duplicatas ou ajuste excessivo ao conjunto de teste não representa um detector melhor.

As melhorias abaixo estão em ordem aproximada de prioridade.

## 4.1 Criar treino, validação e teste final bem controlados

Use três conjuntos:

- treino: ajusta o modelo;
- validação: escolhe características e hiperparâmetros;
- teste final: mede o resultado uma única vez.

Procure duplicatas e imagens quase idênticas entre os conjuntos. Se possível, divida também por origem, gerador ou conteúdo. Isso reduz o risco de o modelo memorizar características específicas do dataset em vez de aprender sinais gerais de geração sintética.

## 4.2 Treinar com transformações, mantendo um teste separado

O modelo atual nunca vê transformações durante o treino. Para criar um detector mais robusto, pode-se aplicar **data augmentation** apenas no treino:

- diferentes qualidades JPEG;
- resize com várias escalas;
- blur leve;
- crops variados;
- brilho e contraste;
- pipelines aleatórios.

Uma estratégia útil é manter uma mistura de imagens originais e transformadas. Isso evita que o modelo aprenda somente artefatos das transformações.

O teste de robustez deve continuar separado e conter combinações e intensidades não usadas diretamente na escolha do modelo.

## 4.3 Ajustar mais hiperparâmetros

Fazer uma busca com validação cruzada para:

- SVM: `C` e `gamma`;
- FFT: quantidade de bins, magnitude, fase e faixas de frequência;
- HOG: orientações, tamanho da célula e tamanho do bloco;
- LBP: diferentes valores de `P`, `R` e múltiplas escalas.

Uma busca aleatória (`RandomizedSearchCV`) pode explorar mais combinações com custo menor do que uma grade grande.

## 4.4 Melhorar as características

Algumas extensões compatíveis com visão clássica são:

- FFT por canal de cor e não apenas em cinza;
- estatísticas de fase e resíduos de alta frequência;
- HOG em mais de uma escala;
- LBP espacial, calculado em regiões da imagem;
- LBP com vários raios;
- histogramas RGB, HSV ou YCbCr;
- GLCM, filtros de Gabor e wavelets;
- resíduos após filtros passa-alta;
- estatísticas de blocos JPEG.

Para imagens sintéticas, pistas de cor, textura e frequência podem ser complementares.

## 4.5 Melhorar a fusão

A concatenação é uma boa referência inicial, mas pode ser melhorada:

- remover características redundantes;
- aplicar PCA ajustado somente no treino;
- selecionar características com validação cruzada;
- treinar classificadores separados e combinar suas decisões (*late fusion*);
- atribuir pesos maiores às famílias mais robustas.

## 4.6 Avaliar resolução maior

O CIFAKE usa imagens pequenas. Em 32 × 32, muitos sinais finos já foram perdidos. Caso o objetivo final envolva imagens de redes sociais, usar uma base com resolução maior pode permitir características mais informativas.

Entretanto, não se deve simplesmente ampliar uma imagem 32 × 32: a interpolação não recria detalhes reais. É necessário partir de arquivos originalmente maiores.

## 4.7 Usar mais dados com diversidade

Mais exemplos podem ajudar, mas a diversidade é tão importante quanto a quantidade. O treino deve conter:

- imagens de diferentes geradores de IA;
- diferentes temas e estilos;
- fotografias de diferentes câmeras e condições;
- diferentes formatos e níveis de compressão.

Um teste externo, vindo de outra base ou de geradores não vistos, é a melhor forma de verificar generalização.

## 4.8 Comparar com aprendizado profundo

Se estiver dentro do escopo do trabalho, uma CNN ou um modelo pré-treinado pode ser usado como comparação. *Transfer learning* com um modelo leve tende a capturar padrões mais complexos do que descritores manuais.

Essa comparação deve manter exatamente os mesmos conjuntos e métricas. O método clássico continua importante porque é mais interpretável e está alinhado ao conteúdo da disciplina.

---

## 5. Plano prático de melhorias

### Prioridade alta

1. Corrigir o cálculo e a documentação do crop.
2. Tornar os caminhos independentes da pasta de execução.
3. Tornar o cache identificável pelos caminhos, parâmetros e versão do pipeline.
4. Separar validação do teste final.
5. Adicionar matriz de confusão, precisão, recall e F1 por classe.
6. Ajustar `C` e `gamma` com validação cruzada.

### Prioridade média

1. Aplicar transformações aleatórias no treino.
2. Avaliar várias sementes e calcular média, desvio-padrão e intervalo de confiança.
3. Testar características de cor, LBP multiescala e informações de fase/frequência.
4. Investigar duplicatas e possíveis atalhos do dataset.
5. Comparar fusão por concatenação com fusão de decisões.

### Prioridade futura

1. Testar uma base externa e geradores não vistos.
2. Trabalhar com imagens de maior resolução.
3. Comparar o pipeline clássico com uma CNN ou modelo pré-treinado.
4. Calibrar as probabilidades para produzir um nível de confiança interpretável.

---

## 6. Como interpretar corretamente os resultados

Uma boa análise não deve responder apenas “qual modelo teve maior acurácia?”. Ela deve investigar:

- qual característica funciona melhor em imagens intactas;
- qual sofre menos com cada transformação;
- se a fusão realmente melhora em dados não vistos;
- se o modelo confunde mais REAL com FAKE ou FAKE com REAL;
- quais transformações provocam as maiores quedas;
- se as diferenças permanecem em outras sementes e outros datasets.

Exemplo de interpretação:

> Se o HOG tiver boa acurácia no baseline, mas cair muito após blur, isso sugere que o modelo dependia de bordas e gradientes finos. Se a FFT cair após JPEG, a compressão provavelmente alterou sinais de frequência usados na classificação. Essas hipóteses devem ser confirmadas pelos números e por repetições do experimento.

Não se deve afirmar antecipadamente que uma característica sempre será mais afetada. A interpretação final deve ser baseada no CSV e nos gráficos realmente produzidos.

---

## 7. Resumo final

O notebook implementa um experimento completo de visão computacional clássica:

- organiza um dataset binário e balanceado;
- transforma imagens em características de frequência, forma e textura;
- padroniza os dados sem usar estatísticas do teste;
- treina SVMs com kernel RBF;
- compara características isoladas e combinadas;
- simula alterações comuns de compartilhamento;
- mede a queda de desempenho;
- salva modelos, métricas e gráficos.

A principal força do trabalho é estudar **robustez**, uma situação mais próxima do uso real do que avaliar somente imagens intactas. As principais oportunidades de melhoria são fortalecer o protocolo experimental, corrigir o crop, tornar o cache seguro, ampliar a busca de hiperparâmetros, usar mais métricas e treinar com transformações controladas.

Neste momento, o repositório não contém `results/metrics.csv` nem os modelos treinados. Por isso, este documento explica o método e como interpretar seus resultados, mas não informa uma acurácia numérica ainda não produzida ou não versionada.
