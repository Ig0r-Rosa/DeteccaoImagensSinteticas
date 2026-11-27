# Roteiro

Slides: `DeteccaoSinteticaAposTransformacoes_Apresentacao.pdf`  
---

### 1. Capa — A

Oi, eu sou [A] e comigo está [B]. A gente é da Engenharia de Computação da UFSC e o trabalho se chama Detecção de Imagens Sintéticas após Transformações.

---

### 2. Motivação — A

Hoje imagens geradas por inteligência artificial aparecem o tempo todo: rede social, anúncio, mensagem. Só que elas quase nunca chegam do jeito que foram geradas, porque no caminho alguém comprime em JPEG, corta, redimensiona ou aplica filtro.

É exatamente isso que aparece nesse slide: uma imagem sintética que o detector marca como falsa e, depois das transformações, passa a ser tratada como real. A gente viu o mesmo tipo de falha no ZeroGPT. Por isso o problema não é só detectar IA, e sim continuar detectando depois que a imagem passou pela internet.

---

### 3. Objetivos — B

A partir disso, o objetivo geral do trabalho é avaliar a robustez de métodos clássicos de visão computacional na detecção de imagens sintéticas depois dessas transformações digitais.

Na prática, a gente queria olhar descritores de frequência, geometria e textura, medir o impacto das transformações, entender o quanto a base de treino muda o resultado e, no fim, entregar uma aplicação, o VerifAI.

A pergunta que guia tudo fica bem nessa frase: quão bem detectores baseados em características clássicas continuam funcionando quando a imagem sofre transformações típicas da internet?

---

### 4. Fundamentação — A

Para responder isso, existem basicamente dois caminhos. No clássico, a gente extrai características com significado visual e um classificador separa real de sintético. No deep learning, a rede aprende sozinha, com mais capacidade, mas menos transparência.

A gente ficou no clássico porque queria um pipeline reproduzível e fácil de ligar cada transformação a um tipo de característica. Na literatura, Bird e Lotfi trazem o CIFAKE com CNN perto de noventa e três por cento; Frank e Bammey mostram marcas no espectro de frequência, o que motiva a FFT; Nirob mostra que fundir descritores clássicos ajuda; e Choi mostra que crop e resize realmente derrubam detectores. O nosso diferencial é juntar essa linha com uma base pareada e um app.

---

### 5. Features — B

Dentro dessa linha clássica, a gente usa três descritores que se complementam. A FFT olha frequência: padrões periódicos, artefatos de geração e também o efeito da compressão JPEG, em sessenta e oito dimensões. O HOG olha bordas e forma, então sofre mais com crop, resize e blur, e gera mil quinhentos e sessenta e oito dimensões. Já o LBP olha textura local, a vizinhança de cada pixel, num vetor curto de dez dimensões.

A ideia é simples: cada um enxerga um tipo de dano diferente na imagem.

---

### 6. Fusão e SVM — A

Com esses três vetores em mãos, o fluxo de classificação fica assim. Como FFT, HOG e LBP têm escalas diferentes, cada um passa por um StandardScaler próprio. Depois a gente concatena tudo num único vetor de fusão e manda para um SVM com kernel RBF, com C igual a dez, gamma automático e peso maior na classe real para reduzir falso positivo no aplicativo.

A saída é real ou fake. No app, isso vira uma chance de IA, e entre quarenta e sessenta por cento a gente trata como inconclusivo.

---

### 7. CIFAKE — B

Os primeiros testes foram no CIFAKE, com imagens reais do CIFAR-10 e sintéticas do Stable Diffusion, tudo em trinta e dois por trinta e dois. Treinando só no original, a fusão chegava perto de noventa e dois por cento no teste limpo, mas depois de crop, blur ou do pipeline caía para perto de cinquenta e quatro por cento, no mesmo estilo da falha do ZeroGPT.

Quando a gente colocou transformações leves no treino, a robustez voltou e no pipeline a fusão ficou acima de setenta e um por cento, com uma queda moderada no teste limpo, para cerca de oitenta e seis. Mesmo assim isso não resolvia o app, porque resolução baixa e domínio CIFAR-10 não representam foto real de celular.

---

### 8. Evolução das bases — A

Por isso a gente foi trocando de base, e o problema se repetia de formas diferentes. No CIFAKE a barreira era a resolução. No CelebA apareciam muitos falsos positivos em fotos gerais, porque o modelo associava “rosto de estúdio” com real. No Harvard AIGI tinha só fake de rede, sem real proporcional. E misturando COCO com Harvard os domínios não batiam, com a FFT perto do acaso, uns cinquenta e três por cento.

A conclusão prática é que, sem o par inverso, o classificador aprende o conteúdo da cena, não o sinal de que a imagem foi gerada. A solução final foi COCO com IA pareada.

---

### 9. Base pareada — B

Essa base final tem cinco mil fotos reais do Microsoft COCO e, para cada uma, uma sintética correspondente. O fluxo usa BLIP para descrever a cena, MiDaS para estimar profundidade, e Stable Diffusion 1.5 com ControlNet Depth para gerar a imagem mantendo pose e composição aproximadas.

No fim ficamos com cinco mil pares alinhados, e o modelo passa a comparar a mesma cena real com a mesma cena sintética, em vez de duas coisas visualmente diferentes.

---

### 10. Pipeline — A

Com essa base, o pipeline final ficou assim: entra a imagem inteira, sem o grid de trinta e dois por trinta e dois; fazemos letterbox para duzentos e vinte e quatro mantendo a proporção, com borda refletida, porque borda preta atrapalhava a FFT; extraímos FFT, HOG e LBP; padronizamos; fundimos; e classificamos com o SVM.

O VerifAI mostra o veredito da fusão e os scores individuais, e o ponto importante é que treino e app usam o mesmo pré-processamento.

---

### 11. Metodologia — B

No experimento, o split é por par, com semente fixa: quatro mil pares no treino e mil no teste. A partir disso treinamos quatro SVMs, um para FFT, um para HOG, um para LBP e um para a fusão.

No treino entra aumento leve, com JPEG, resize, blur e um pipeline combinado. Já na robustez a gente aplica as transformações só no teste, sem retreinar o modelo.

---

### 12. Resultados holdout — A

Nesse holdout, a fusão chega a oitenta e oito vírgula quinze por cento, e é esse o detector que vai para o app. O HOG fica em oitenta e quatro vírgula quarenta, o LBP em oitenta e dois vírgula trinta, e a FFT sozinha em setenta e cinco.

Ou seja, juntar os três dá o melhor equilíbrio entre pegar fake e não acusar real à toa.

---

### 13. Robustez — B

Quando a gente olha as transformações, JPEG, resize e blur mantêm a fusão em geral acima de oitenta e sete por cento. Os piores casos são o crop forte, perto de setenta e quatro e meio no crop de quarenta por cento, e o pipeline combinado, perto de setenta e nove e meio, ainda bem acima do acaso.

Isso mostra que o aumento leve protege bem compressão e blur, mas o recorte continua sendo o ponto mais frágil.

---

### 14. VerifAI — B

Agora a parte prática. Esse é o VerifAI, o app em Streamlit.  
*Mostrar o app*

---

### 15. Discussão — A

Olhando o caminho inteiro, o que mais aparece é que a base pesa tanto quanto o classificador. Foi a fusão que deu a melhor acurácia, o aumento leve que sustentou a robustez nas transformações mais comuns, e o pareamento que evitou o modelo aprender conteúdo no lugar de síntese.

Ao mesmo tempo, as limitações ficam bem visíveis: a gente treinou com um único gerador, no mesmo domínio de treino e teste, ainda sente recorte, e não tem um benchmark in-the-wild. Por isso, dentro do domínio a gente cumpre o objetivo; fora dele, o SVM clássico deixa de ser confiável, e deep learning seria o próximo passo natural.

---

### 16. Conclusão — A

Fechando, a fusão chega a oitenta e oito vírgula quinze por cento no teste original, se mantém acima de oitenta e sete por cento sob JPEG, resize e blur, e o VerifAI já funciona com esse mesmo pipeline.

No fim, a robustez de um detector de imagens sintéticas depende tanto dos descritores e do classificador quanto da representatividade da base. Como próximos passos, a gente pensa em diversificar geradores e dados, especializar a detecção por tipo de transformação e avaliar deep learning para imagens fora do domínio.

Obrigado.
