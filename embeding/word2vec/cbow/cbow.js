class CBOW {
  /**
   * 初始化CBOW模型
   * @param {object} options 模型参数
   * @param {number} options.embeddingSize 词向量维度
   * @param {number} options.windowSize 上下文窗口大小
   * @param {number} options.learningRate 学习率
   */
  constructor({ embeddingSize = 10, windowSize = 2, learningRate = 0.01 }) {
    this.embeddingSize = embeddingSize;
    this.windowSize = windowSize;
    this.learningRate = learningRate;
    this.wordToIndex = {};      // 词到索引的映射
    this.indexToWord = [];      // 索引到词的映射
    this.vocabSize = 0;         // 词汇表大小
    this.weightsInput = null;   // 输入层权重矩阵 (vocabSize × embeddingSize)
    this.weightsOutput = null;  // 输出层权重矩阵 (embeddingSize × vocabSize)
  }

  /**
   * 构建词汇表
   * @param {string[]} corpus 语料库
   */
  buildVocab(corpus) {
    const words = new Set();
    corpus.forEach(sentence => {
      sentence.split(' ').forEach(word => {
        words.add(word);
      });
    });

    // 构建词与索引的映射
    this.indexToWord = Array.from(words);
    this.indexToWord.forEach((word, index) => {
      this.wordToIndex[word] = index;
    });
    this.vocabSize = this.indexToWord.length;

    // 初始化权重矩阵 (随机初始化)
    this.weightsInput = this.randomMatrix(this.vocabSize, this.embeddingSize);
    this.weightsOutput = this.randomMatrix(this.embeddingSize, this.vocabSize);
  }

  /**
   * 生成训练数据 (上下文-中心词对)
   * @param {string[]} corpus 语料库
   * @returns {Array} 训练样本数组
   */
  generateTrainingData(corpus) {
    const trainingData = [];
    
    corpus.forEach(sentence => {
      const words = sentence.split(' ');
      
      for (let i = 0; i < words.length; i++) {
        // 获取上下文词的索引
        const contextIndices = [];
        
        // 向左取windowSize个词
        for (let j = 1; j <= this.windowSize; j++) {
          if (i - j >= 0) {
            contextIndices.push(this.wordToIndex[words[i - j]]);
          }
        }
        
        // 向右取windowSize个词
        for (let j = 1; j <= this.windowSize; j++) {
          if (i + j < words.length) {
            contextIndices.push(this.wordToIndex[words[i + j]]);
          }
        }
        
        // 确保有足够的上下文词
        if (contextIndices.length > 0) {
          trainingData.push({
            context: contextIndices,
            target: this.wordToIndex[words[i]]
          });
        }
      }
    });
    
    return trainingData;
  }

  /**
   * 训练模型
   * @param {string[]} corpus 语料库
   * @param {number} epochs 训练轮数
   */
  train(corpus, epochs = 100) {
    // 构建词汇表
    this.buildVocab(corpus);
    
    // 生成训练数据
    const trainingData = this.generateTrainingData(corpus);
    console.log(`训练数据量: ${trainingData.length}`);
    console.log(`词汇表大小: ${this.vocabSize}`);

    // 训练循环
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;
      
      // 随机打乱训练数据 (优化: 增加随机性，避免过拟合)
      this.shuffleArray(trainingData);
      
      for (const { context, target } of trainingData) {
        // 前向传播
        const { hidden, output, loss } = this.forward(context, target);
        totalLoss += loss;
        
        // 反向传播更新权重
        this.backward(context, target, hidden, output);
      }
      
      // 每10轮输出一次损失
      if (epoch % 10 === 0) {
        console.log(`Epoch ${epoch}, 平均损失: ${totalLoss / trainingData.length}`);
        
        // 优化: 动态调整学习率
        if (epoch > 0 && epoch % 50 === 0) {
          this.learningRate *= 0.5;
          console.log(`调整学习率为: ${this.learningRate}`);
        }
      }
    }
  }

  /**
   * 前向传播
   * @param {number[]} context 上下文词索引数组
   * @param {number} target 目标词索引
   * @returns {object} 包含隐藏层、输出层和损失的对象
   */
  forward(context, target) {
    // 1. 计算上下文词向量的平均值 (隐藏层)
    const hidden = new Array(this.embeddingSize).fill(0);
    context.forEach(idx => {
      for (let i = 0; i < this.embeddingSize; i++) {
        hidden[i] += this.weightsInput[idx][i];
      }
    });
    // 求平均
    const contextSize = context.length;
    for (let i = 0; i < this.embeddingSize; i++) {
      hidden[i] /= contextSize;
    }

    // 2. 计算输出层 (使用softmax激活)
    const output = new Array(this.vocabSize).fill(0);
    let sumExp = 0;
    
    // 先计算所有输出的指数值
    for (let j = 0; j < this.vocabSize; j++) {
      let score = 0;
      for (let i = 0; i < this.embeddingSize; i++) {
        score += hidden[i] * this.weightsOutput[i][j];
      }
      output[j] = Math.exp(score);
      sumExp += output[j];
    }
    
    // 归一化得到概率分布
    for (let j = 0; j < this.vocabSize; j++) {
      output[j] /= sumExp;
    }

    // 3. 计算交叉熵损失
    const loss = -Math.log(output[target] + 1e-10); // 加小值避免log(0)
    
    return { hidden, output, loss };
  }

  /**
   * 反向传播更新权重
   * @param {number[]} context 上下文词索引数组
   * @param {number} target 目标词索引
   * @param {number[]} hidden 隐藏层输出
   * @param {number[]} output 输出层输出
   */
  backward(context, target, hidden, output) {
    // 1. 计算输出层误差 (优化: 直接计算梯度，避免存储完整误差数组)
    const outputError = [...output];
    outputError[target] -= 1; // 对于交叉熵+softmax，误差简化为output - target_one_hot

    // 2. 更新输出层权重
    for (let i = 0; i < this.embeddingSize; i++) {
      for (let j = 0; j < this.vocabSize; j++) {
        this.weightsOutput[i][j] -= this.learningRate * hidden[i] * outputError[j];
      }
    }

    // 3. 计算隐藏层误差
    const hiddenError = new Array(this.embeddingSize).fill(0);
    for (let i = 0; i < this.embeddingSize; i++) {
      for (let j = 0; j < this.vocabSize; j++) {
        hiddenError[i] += this.weightsOutput[i][j] * outputError[j];
      }
    }

    // 4. 更新输入层权重 (上下文词共享误差)
    const contextSize = context.length;
    for (const idx of context) {
      for (let i = 0; i < this.embeddingSize; i++) {
        this.weightsInput[idx][i] -= this.learningRate * (hiddenError[i] / contextSize);
      }
    }
  }

  /**
   * 获取词向量
   * @param {string} word 单词
   * @returns {number[]} 词向量
   */
  getWordVector(word) {
    if (!this.wordToIndex.hasOwnProperty(word)) {
      throw new Error(`单词 "${word}" 不在词汇表中`);
    }
    return [...this.weightsInput[this.wordToIndex[word]]];
  }

  /**
   * 预测上下文对应的中心词
   * @param {string[]} contextWords 上下文单词数组
   * @returns {object} 包含预测词和概率的对象
   */
  predict(contextWords) {
    const contextIndices = contextWords
      .filter(word => this.wordToIndex.hasOwnProperty(word))
      .map(word => this.wordToIndex[word]);
      
    if (contextIndices.length === 0) {
      throw new Error("上下文单词不在词汇表中");
    }
    
    // 计算隐藏层
    const hidden = new Array(this.embeddingSize).fill(0);
    contextIndices.forEach(idx => {
      for (let i = 0; i < this.embeddingSize; i++) {
        hidden[i] += this.weightsInput[idx][i];
      }
    });
    const contextSize = contextIndices.length;
    for (let i = 0; i < this.embeddingSize; i++) {
      hidden[i] /= contextSize;
    }
    
    // 计算输出层
    const output = new Array(this.vocabSize).fill(0);
    let sumExp = 0;
    for (let j = 0; j < this.vocabSize; j++) {
      let score = 0;
      for (let i = 0; i < this.embeddingSize; i++) {
        score += hidden[i] * this.weightsOutput[i][j];
      }
      output[j] = Math.exp(score);
      sumExp += output[j];
    }
    for (let j = 0; j < this.vocabSize; j++) {
      output[j] /= sumExp;
    }
    
    // 找到概率最大的词
    let maxProb = -Infinity;
    let bestIndex = 0;
    for (let j = 0; j < this.vocabSize; j++) {
      if (output[j] > maxProb) {
        maxProb = output[j];
        bestIndex = j;
      }
    }
    
    return {
      word: this.indexToWord[bestIndex],
      probability: maxProb
    };
  }

  /**
   * 生成随机矩阵
   * @param {number} rows 行数
   * @param {number} cols 列数
   * @returns {number[][]} 随机矩阵
   */
  randomMatrix(rows, cols) {
    const matrix = [];
    for (let i = 0; i < rows; i++) {
      matrix.push([]);
      for (let j = 0; j < cols; j++) {
        // 优化: 使用较小的随机值初始化
        matrix[i].push(Math.random() * 0.2 - 0.1); // 范围: [-0.1, 0.1]
      }
    }
    return matrix;
  }

  /**
   * 打乱数组顺序 (Fisher-Yates算法)
   * @param {Array} array 要打乱的数组
   */
  shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  }
}
  
// 示例演示
function demo() {
  // 示例语料库
  const corpus = [
    "the quick brown fox jumps over the lazy dog",
    "never jump over the lazy dog quickly",
    "quick brown foxes leap over lazy dogs in summer",
    "the quick brown fox jumps high",
    "lazy dogs sleep all day",
    "brown foxes are quick",
    "dogs are lazy animals"
  ];

  // 创建并训练CBOW模型
  console.log("初始化CBOW模型...");
  const cbow = new CBOW({
    embeddingSize: 8,
    windowSize: 4,
    learningRate: 0.05
  });

  console.log("开始训练...");
  cbow.train(corpus, 1000); // 训练200轮

  // 演示1: 获取词向量
  console.log("\n--- 词向量示例 ---");
  const wordsToCheck = ["fox", "dog", "quick", "lazy"];
  wordsToCheck.forEach(word => {
    console.log(`${word}:`, cbow.getWordVector(word).map(v => v.toFixed(4)));
  });

  // 演示2: 预测中心词
  console.log("\n--- 预测示例 ---");
  const testContexts = [
    ["the", "quick", "brown", "jumps"],  // 应该预测 "fox"
    ["over", "the", "dog", "quickly"],   // 应该预测 "lazy"
    ["brown", "foxes", "leap", "lazy"],  // 应该预测 "over"
    ["are", "lazy", "animals"]           // 应该预测 "dogs"
  ];

  testContexts.forEach(context => {
    const prediction = cbow.predict(context);
    console.log(`上下文: [${context.join(", ")}] => 预测中心词: ${prediction.word} (概率: ${prediction.probability.toFixed(4)})`);
  });
}
  
// 导出模型和示例
export {CBOW, demo};