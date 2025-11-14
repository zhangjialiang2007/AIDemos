class SkipGram {
  /**
   * 初始化Skip-gram模型
   * @param {object} options 模型参数
   * @param {number} options.embeddingSize 词向量维度
   * @param {number} options.windowSize 上下文窗口大小
   * @param {number} options.learningRate 学习率
   * @param {number} options.negativeSamples 负采样数量
   */
  constructor({ 
    embeddingSize = 10, 
    windowSize = 2, 
    learningRate = 0.01,
    negativeSamples = 5  // 负采样数量，优化计算效率
  }) {
    this.embeddingSize = embeddingSize;
    this.windowSize = windowSize;
    this.learningRate = learningRate;
    this.negativeSamples = negativeSamples;
    
    this.wordToIndex = {};      // 词到索引的映射
    this.indexToWord = [];      // 索引到词的映射
    this.vocabSize = 0;         // 词汇表大小
    this.wordCounts = {};       // 词频统计，用于负采样
    
    // 权重矩阵
    this.weightsInput = null;   // 输入层权重 (vocabSize × embeddingSize)
    this.weightsOutput = null;  // 输出层权重 (embeddingSize × vocabSize)
  }

  /**
   * 构建词汇表和词频统计
   * @param {string[]} corpus 语料库
   */
  buildVocab(corpus) {
    // 统计词频
    this.wordCounts = {};
    corpus.forEach(sentence => {
      sentence.split(' ').forEach(word => {
        this.wordCounts[word] = (this.wordCounts[word] || 0) + 1;
      });
    });

    // 构建词与索引的映射
    this.indexToWord = Object.keys(this.wordCounts);
    this.indexToWord.forEach((word, index) => {
      this.wordToIndex[word] = index;
    });
    this.vocabSize = this.indexToWord.length;

    // 为负采样准备概率分布 (基于词频的0.75次方)
    this.prepareNegativeSamplingDistribution();

    // 初始化权重矩阵
    this.weightsInput = this.randomMatrix(this.vocabSize, this.embeddingSize);
    this.weightsOutput = this.randomMatrix(this.embeddingSize, this.vocabSize);
  }

  /**
   * 准备负采样的概率分布
   */
  prepareNegativeSamplingDistribution() {
    this.negativeProb = new Array(this.vocabSize);
    let total = 0;

    // 计算词频的0.75次方总和
    this.indexToWord.forEach((word, index) => {
      total += Math.pow(this.wordCounts[word], 0.75);
    });

    // 计算每个词的采样概率
    this.indexToWord.forEach((word, index) => {
      this.negativeProb[index] = Math.pow(this.wordCounts[word], 0.75) / total;
    });

    // 构建累积概率分布，用于快速采样
    this.cumulativeProb = new Array(this.vocabSize);
    this.cumulativeProb[0] = this.negativeProb[0];
    for (let i = 1; i < this.vocabSize; i++) {
      this.cumulativeProb[i] = this.cumulativeProb[i - 1] + this.negativeProb[i];
    }
  }

  /**
   * 负采样：从词汇表中随机采样负例
   * @param {number} target 正例索引（避免采到正例）
   * @returns {number[]} 负例索引数组
   */
  sampleNegatives(target) {
    const negatives = [];
    while (negatives.length < this.negativeSamples) {
      // 基于累积概率分布进行采样
      const r = Math.random();
      let idx = 0;
      while (this.cumulativeProb[idx] < r) {
        idx++;
      }
      // 确保不采到正例
      if (idx !== target) {
        negatives.push(idx);
      }
    }
    return negatives;
  }

  /**
   * 生成训练数据 (中心词-上下文词对)
   * @param {string[]} corpus 语料库
   * @returns {Array} 训练样本数组
   */
  generateTrainingData(corpus) {
    const trainingData = [];
    
    corpus.forEach(sentence => {
      const words = sentence.split(' ');
      
      for (let i = 0; i < words.length; i++) {
        const centerWord = words[i];
        const centerIndex = this.wordToIndex[centerWord];
        
        // 收集上下文词
        for (let j = 1; j <= this.windowSize; j++) {
          // 左侧上下文
          if (i - j >= 0) {
            const contextWord = words[i - j];
            trainingData.push({
              center: centerIndex,
              context: this.wordToIndex[contextWord],
              label: 1  // 正例
            });
          }
          
          // 右侧上下文
          if (i + j < words.length) {
            const contextWord = words[i + j];
            trainingData.push({
              center: centerIndex,
              context: this.wordToIndex[contextWord],
              label: 1  // 正例
            });
          }
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
      
      // 随机打乱训练数据
      this.shuffleArray(trainingData);
      
      for (const { center, context } of trainingData) {
        // 采样负例
        const negatives = this.sampleNegatives(context);
        
        // 处理正例
        const { loss: posLoss } = this.trainPair(center, context, 1);
        totalLoss += posLoss;
        
        // 处理负例
        for (const negContext of negatives) {
          const { loss: negLoss } = this.trainPair(center, negContext, 0);
          totalLoss += negLoss;
        }
      }
      
      // 每10轮输出一次损失
      if (epoch % 10 === 0) {
        console.log(`Epoch ${epoch}, 平均损失: ${(totalLoss / trainingData.length).toFixed(6)}`);
        
        // 动态调整学习率
        if (epoch > 0 && epoch % 50 === 0) {
          this.learningRate *= 0.5;
          console.log(`调整学习率为: ${this.learningRate}`);
        }
      }
    }
  }

  /**
   * 训练单个词对（中心词-上下文词）
   * @param {number} center 中心词索引
   * @param {number} context 上下文词索引
   * @param {number} label 标签（1为正例，0为负例）
   * @returns {object} 包含损失的对象
   */
  trainPair(center, context, label) {
    // 前向传播
    const hidden = this.weightsInput[center];  // 输入层直接输出词向量
    
    // 计算输出（点积 + sigmoid）
    let score = 0;
    for (let i = 0; i < this.embeddingSize; i++) {
      score += hidden[i] * this.weightsOutput[i][context];
    }
    const output = 1 / (1 + Math.exp(-score));  // sigmoid激活函数
    
    // 计算损失（二元交叉熵）
    const loss = - (label * Math.log(output + 1e-10) + (1 - label) * Math.log(1 - output + 1e-10));
    
    // 反向传播
    const error = output - label;  // 误差项
    
    // 更新输出层权重
    for (let i = 0; i < this.embeddingSize; i++) {
      this.weightsOutput[i][context] -= this.learningRate * error * hidden[i];
    }
    
    // 更新输入层权重
    for (let i = 0; i < this.embeddingSize; i++) {
      this.weightsInput[center][i] -= this.learningRate * error * this.weightsOutput[i][context];
    }
    
    return { loss };
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
   * 预测与中心词相关的上下文词
   * @param {string} centerWord 中心词
   * @param {number} topN 返回前N个最可能的词
   * @returns {Array} 包含预测词和概率的对象数组
   */
  predictContext(centerWord, topN = 5) {
    if (!this.wordToIndex.hasOwnProperty(centerWord)) {
      throw new Error(`单词 "${centerWord}" 不在词汇表中`);
    }
    
    const centerIndex = this.wordToIndex[centerWord];
    const hidden = this.weightsInput[centerIndex];
    
    // 计算所有词的概率
    const probabilities = [];
    for (let j = 0; j < this.vocabSize; j++) {
      let score = 0;
      for (let i = 0; i < this.embeddingSize; i++) {
        score += hidden[i] * this.weightsOutput[i][j];
      }
      const prob = 1 / (1 + Math.exp(-score));  // sigmoid
      probabilities.push({
        word: this.indexToWord[j],
        probability: prob
      });
    }
    
    // 排序并返回前N个
    return probabilities
      .sort((a, b) => b.probability - a.probability)
      .filter(item => item.word !== centerWord)  // 排除中心词本身
      .slice(0, topN);
  }

  /**
   * 计算两个词的相似度（余弦相似度）
   * @param {string} word1 第一个词
   * @param {string} word2 第二个词
   * @returns {number} 相似度值
   */
  getSimilarity(word1, word2) {
    const vec1 = this.getWordVector(word1);
    const vec2 = this.getWordVector(word2);
    
    // 计算点积
    let dotProduct = 0;
    // 计算模长
    let norm1 = 0, norm2 = 0;
    
    for (let i = 0; i < this.embeddingSize; i++) {
      dotProduct += vec1[i] * vec2[i];
      norm1 += vec1[i] * vec1[i];
      norm2 += vec2[i] * vec2[i];
    }
    
    // 防止除零
    if (norm1 === 0 || norm2 === 0) return 0;
    
    return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
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
        // 使用较小的随机值初始化，有助于训练稳定性
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
  // 示例语料库 - 关于动物和它们的特征/行为
  const corpus = [
    "cats like to chase mice",
    "dogs like to chase balls",
    "cats and dogs are pets",
    "dogs bark at strangers",
    "cats meow when hungry",
    "dogs and cats need food",
    "pets need love and care",
    "cats sleep during the day",
    "dogs sleep during the night",
    "cats are independent animals",
    "dogs are loyal animals"
  ];

  // 创建并训练Skip-gram模型
  console.log("初始化Skip-gram模型...");
  const skipGram = new SkipGram({
    embeddingSize: 10,
    windowSize: 4,
    learningRate: 0.02,
    negativeSamples: 5
  });

  console.log("开始训练...");
  skipGram.train(corpus, 1000); // 训练300轮

  // 演示1: 获取词向量
  console.log("\n--- 词向量示例 ---");
  const wordsToCheck = ["cats", "dogs", "pets", "chase"];
  wordsToCheck.forEach(word => {
    console.log(`${word}:`, skipGram.getWordVector(word).map(v => v.toFixed(4)));
  });

  // 演示2: 预测上下文词
  console.log("\n--- 上下文预测示例 ---");
  const centerWords = ["cats", "dogs", "pets", "need"];
  centerWords.forEach(word => {
    const predictions = skipGram.predictContext(word, 3);
    console.log(`与 "${word}" 相关的上下文词:`, 
      predictions.map(p => `${p.word} (${p.probability.toFixed(4)})`).join(", "));
  });

  // 演示3: 词相似度计算
  console.log("\n--- 词相似度示例 ---");
  const similarityPairs = [
    ["cats", "dogs"],
    ["cats", "pets"],
    ["dogs", "pets"],
    ["chase", "sleep"],
    ["like", "need"]
  ];
  similarityPairs.forEach(([w1, w2]) => {
    const sim = skipGram.getSimilarity(w1, w2);
    console.log(`${w1} 与 ${w2} 的相似度: ${sim.toFixed(4)}`);
  });
}
  
// 导出模型和示例
export {SkipGram, demo};