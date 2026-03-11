import tf from '@tensorflow/tfjs-node';
async function trainModel(inputXs, outputYs) {
  const model = tf.sequential();

  /*
  Primeira camada da rede:
  inputShape:
  Entrada de 7 posições (idade normalizada + 3 cores + 3 localizações
  
  units:
  80 neurônios = aqui coloquei tudo isso porque tem pouca base de treino
  quanto mais neurônios, mais complexidade a rede pode aprender e, consequentemente, 
  mais processamenta ela vai usar

  A RelU age como um filtro:
  É como se ela deixasse somente os dados interessantes seguirem viagem na rede
  Se a informação chegou nesse neurônio é positiva, passa para frente
  Se for zeroou negativa, pode jogar fora porque não serve para nada
  */

  model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));

  /*
  Saída: 3 neurônios
  Um para cada categoria: basic, medium e premium
  */
  model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

  /*
  Compilando o modelo

  optmizer: Adam (Adaptive Moment Estimation)
  É um trabalhador pessoal moderno para redes neurais ajustando os pesos de forma eficiente e inteligente
  Aprende com histórico de erros e acertos

  loss: categoricalCrossentropy
  Ele compara o que o modelo "acha" (os scores de cada categoria)
  A categoria premium será sempre [1, 0, 0]

  metrics:
  Quanto mais distante da previsão do modelo de resposta correta, maior o erro (loss)
  Exemplo clássico: classificação de imagem, recomendação, categorização de usuário
  Qualquer coisa em que a resposta certa é "apenas uma entre várias possíveis"
  */
  model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

  /*
  Treinamento do modelo
  verbose: desabilita o log interno (e usa só callback)
  epochs: quantidade de vezes que vai rodar no dataset
  shuffle: embaralha os dados para evitar vieses
  */
  await model.fit(
    inputXs,
    outputYs,
    {
      verbose: 0,
      epochs: 100,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, log) => console.log(
          `Epoch: ${epoch}: loss = ${log.loss}`
        )
      }
    }
  )

  return model;
}

async function predict(model, person) {
  // Transformar o array js para o tensor (tfjs)
  const tfInput = tf.tensor2d(person);

  // Faz a predição (output será um vetor de 3 probabilidades)
  const pred = model.predict(tfInput);
  const predArray = await pred.array();

   return predArray[0].map((prob, index) => ({ prob, index }));
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

/*
Quanto mais dado, melhor!
Assim o algoritmo consegue entender melhor os padrões complexos
*/
const model = await trainModel(inputXs, outputYs);

const person = {
  nome: 'Dulcetti',
  idade: 43,
  cor: 'verde',
  localizacao: 'Rio'
};

/*
Normalizando a idade da nova pessoa usando o mesmo padrão do treino
Exemplo: idade_min: 25, idade_max: 50, então (43 - 25) / (50 - 25) = 0,72
*/

const personTensorNormalized = [
  [0.72, 0, 0, 1, 0, 1, 0]
];
