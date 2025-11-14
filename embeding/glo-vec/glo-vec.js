class GloVec {
/**
 * 初始化GloVec模型
 * @param {object} options 模型参数
 * @param {number} options.embeddingSize 词向量维度
 * @param {number} options.windowSize 上下文窗口大小
 * @param {number} options.learningRate 学习率
 * @param {number} options.xMax 权重函数的阈值
 * @param {number} options.alpha 权重函数的指数
 */
constructor({
  embeddingSize = 10,
  windowSize = 2,
  learningRate = 0.05,
  xMax = 100,    // 权重函数饱和阈值
  alpha = 0.75   // 权重函数指数
}) {
  this.embeddingSize = embeddingSize;
  this.windowSize = windowSize;
  this.learningRate = learningRate;
  this.xMax = xMax;
  this.alpha = alpha;
  
  this.wordToIndex = {};      // 词到索引的映射
  this.indexToWord = [];      // 索引到词的映射
  this.vocabSize = 0;         // 词汇表大小
  this.cooccurrenceMatrix = {}; // 共现矩阵 {i,j: count}
  
  // 词向量和偏置项
  this.w = null;  // 词向量矩阵 (vocabSize × embeddingSize)
  this.wTilde = null;  // 上下文词向量矩阵 (vocabSize × embeddingSize)
  this.b = null;  // 词偏置项 (vocabSize)
  this.bTilde = null;  // 上下文词偏置项 (vocabSize)
}

/**
 * 构建词汇表和共现矩阵
 * @param {string[]} corpus 语料库
 */
buildVocabAndCooccurrence(corpus) {
  // 第一步：构建词汇表
  const wordSet = new Set();
  corpus.forEach(sentence => {
    sentence.split(' ').forEach(word => wordSet.add(word));
  });
  
  this.indexToWord = Array.from(wordSet);
  this.indexToWord.forEach((word, index) => {
    this.wordToIndex[word] = index;
  });
  this.vocabSize = this.indexToWord.length;

  // 第二步：构建共现矩阵（带距离权重）
  this.cooccurrenceMatrix = {};
  
  corpus.forEach(sentence => {
    const words = sentence.split(' ');
    const len = words.length;
    
    for (let i = 0; i < len; i++) {
      const centerWord = words[i];
      const centerIdx = this.wordToIndex[centerWord];
      
      // 上下文窗口：左侧和右侧
      for (let j = 1; j <= this.windowSize; j++) {
        // 左侧上下文
        if (i - j >= 0) {
          const contextWord = words[i - j];
          const contextIdx = this.wordToIndex[contextWord];
          const distanceWeight = 1 / j; // 距离越近权重越高
          
          this.updateCooccurrence(centerIdx, contextIdx, distanceWeight);
        }
        
        // 右侧上下文
        if (i + j < len) {
          const contextWord = words[i + j];
          const contextIdx = this.wordToIndex[contextWord];
          const distanceWeight = 1 / j; // 距离越近权重越高
          
          this.updateCooccurrence(centerIdx, contextIdx, distanceWeight);
        }
      }
    }
  });
  
  // 初始化向量和偏置
  this.initializeVectors();
}

/**
 * 更新共现矩阵
 * @param {number} i 中心词索引
 * @param {number} j 上下文词索引
 * @param {number} weight 权重（基于距离）
 */
updateCooccurrence(i, j, weight) {
  // 使用字符串键确保i<j，避免重复计数
  const key = i < j ? `${i},${j}` : `${j},${i}`;
  this.cooccurrenceMatrix[key] = (this.cooccurrenceMatrix[key] || 0) + weight;
}

/**
 * 初始化词向量和偏置项
 */
initializeVectors() {
  // 词向量初始化（小随机值）
  this.w = this.randomMatrix(this.vocabSize, this.embeddingSize);
  this.wTilde = this.randomMatrix(this.vocabSize, this.embeddingSize);
  
  // 偏置项初始化（零初始化）
  this.b = new Array(this.vocabSize).fill(0);
  this.bTilde = new Array(this.vocabSize).fill(0);
}

/**
 * 权重函数 f(X_ij)
 * @param {number} x 共现次数
 * @returns {number} 权重值
 */
weightFunction(x) {
  if (x > this.xMax) return 1;
  return Math.pow(x / this.xMax, this.alpha);
}

/**
 * 生成训练样本（共现对）
 * @returns {Array} 训练样本数组
 */
generateTrainingSamples() {
  const samples = [];
  for (const key in this.cooccurrenceMatrix) {
    const [i, j] = key.split(',').map(Number);
    const count = this.cooccurrenceMatrix[key];
    samples.push({ i, j, count });
  }
  return samples;
}

/**
 * 训练模型
 * @param {string[]} corpus 语料库
 * @param {number} epochs 训练轮数
 */
train(corpus, epochs = 100) {
  // 构建词汇表和共现矩阵
  this.buildVocabAndCooccurrence(corpus);
  const trainingSamples = this.generateTrainingSamples();
  
  console.log(`训练样本数: ${trainingSamples.length}`);
  console.log(`词汇表大小: ${this.vocabSize}`);

  // 训练循环
  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;
    
    // 优化：随机打乱样本顺序
    this.shuffleArray(trainingSamples);
    
    for (const { i, j, count } of trainingSamples) {
      // 计算权重
      const w = this.weightFunction(count);
      
      // 前向计算
      let dotProduct = 0;
      for (let k = 0; k < this.embeddingSize; k++) {
        dotProduct += this.w[i][k] * this.wTilde[j][k];
      }
      const logX = Math.log(count);
      const prediction = dotProduct + this.b[i] + this.bTilde[j] - logX;
      const loss = w * prediction * prediction;
      totalLoss += loss;
      
      // 反向传播更新（优化：计算梯度并更新参数）
      const gradient = 2 * w * prediction;
      
      // 更新词向量
      for (let k = 0; k < this.embeddingSize; k++) {
        this.w[i][k] -= this.learningRate * gradient * this.wTilde[j][k];
        this.wTilde[j][k] -= this.learningRate * gradient * this.w[i][k];
      }
      
      // 更新偏置项
      this.b[i] -= this.learningRate * gradient;
      this.bTilde[j] -= this.learningRate * gradient;
    }
    
    // 每10轮输出一次损失
    if (epoch % 10 === 0) {
      console.log(`Epoch ${epoch}, 总损失: ${totalLoss.toFixed(6)}`);
      
      // 优化：动态调整学习率
      if (epoch > 0 && epoch % 50 === 0) {
        this.learningRate *= 0.5;
        console.log(`调整学习率为: ${this.learningRate}`);
      }
    }
  }
}

/**
 * 获取词向量（取词向量和上下文词向量的平均值）
 * @param {string} word 单词
 * @returns {number[]} 词向量
 */
getWordVector(word) {
  if (!this.wordToIndex.hasOwnProperty(word)) {
    throw new Error(`单词 "${word}" 不在词汇表中`);
  }
  
  const idx = this.wordToIndex[word];
  const vector = new Array(this.embeddingSize);
  
  // 优化：使用词向量和上下文向量的平均作为最终向量
  for (let k = 0; k < this.embeddingSize; k++) {
    vector[k] = (this.w[idx][k] + this.wTilde[idx][k]) / 2;
  }
  
  return vector;
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
  
  let dotProduct = 0;
  let norm1 = 0, norm2 = 0;
  
  for (let i = 0; i < this.embeddingSize; i++) {
    dotProduct += vec1[i] * vec2[i];
    norm1 += vec1[i] * vec1[i];
    norm2 += vec2[i] * vec2[i];
  }
  
  if (norm1 === 0 || norm2 === 0) return 0;
  return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
}

/**
 * 查找最相似的词
 * @param {string} word 目标词
 * @param {number} topN 返回前N个最相似的词
 * @returns {Array} 包含相似词和相似度的对象数组
 */
mostSimilar(word, topN = 5) {
  if (!this.wordToIndex.hasOwnProperty(word)) {
    throw new Error(`单词 "${word}" 不在词汇表中`);
  }
  
  const targetVec = this.getWordVector(word);
  const similarities = [];
  
  for (const candidateWord of this.indexToWord) {
    if (candidateWord === word) continue;
    
    const candidateVec = this.getWordVector(candidateWord);
    const sim = this.cosineSimilarity(targetVec, candidateVec);
    similarities.push({ word: candidateWord, similarity: sim });
  }
  
  return similarities
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, topN);
}

/**
 * 计算余弦相似度
 * @param {number[]} vec1 向量1
 * @param {number[]} vec2 向量2
 * @returns {number} 相似度值
 */
cosineSimilarity(vec1, vec2) {
  let dotProduct = 0;
  let norm1 = 0, norm2 = 0;
  
  for (let i = 0; i < vec1.length; i++) {
    dotProduct += vec1[i] * vec2[i];
    norm1 += vec1[i] * vec1[i];
    norm2 += vec2[i] * vec2[i];
  }
  
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
      // 优化：使用较小的随机值初始化，有助于训练稳定性
      matrix[i].push(Math.random() * 0.04 - 0.02); // 范围: [-0.02, 0.02]
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
// 示例语料库 - 关于城市和国家的语料
const corpus = [
  "paris is the capital of france",
  "london is the capital of england",
  "berlin is the capital of germany",
  "rome is the capital of italy",
  "madrid is the capital of spain",
  "tokyo is the capital of japan",
  "beijing is the capital of china",
  "france is a country in europe",
  "germany is a country in europe",
  "japan is a country in asia",
  "china is a country in asia",
  "paris is a city in france",
  "london is a city in england",
  "berlin is a city in germany",
  "tokyo is a city in japan"
];

// 创建并训练GloVec模型
console.log("初始化GloVec模型...");
const glovec = new GloVec({
  embeddingSize: 10,
  windowSize: 3,
  learningRate: 0.05,
  xMax: 100,
  alpha: 0.75
});

console.log("开始训练...");
glovec.train(corpus, 300); // 训练300轮

// 演示1: 获取词向量
console.log("\n--- 词向量示例 ---");
const wordsToCheck = ["paris", "france", "tokyo", "japan"];
wordsToCheck.forEach(word => {
  console.log(`${word}:`, glovec.getWordVector(word).map(v => v.toFixed(4)));
});

// 演示2: 词相似度计算
console.log("\n--- 词相似度示例 ---");
const similarityPairs = [
  ["paris", "france"],
  ["tokyo", "japan"],
  ["berlin", "germany"],
  ["france", "germany"],
  ["asia", "europe"],
  ["paris", "tokyo"],
  ["capital", "city"]
];
similarityPairs.forEach(([w1, w2]) => {
  const sim = glovec.getSimilarity(w1, w2);
  console.log(`${w1} 与 ${w2} 的相似度: ${sim.toFixed(4)}`);
});

// 演示3: 查找最相似的词
console.log("\n--- 最相似词示例 ---");
const targetWords = ["france", "tokyo", "europe"];
targetWords.forEach(word => {
  const similar = glovec.mostSimilar(word, 3);
  console.log(`与 "${word}" 最相似的词:`, 
    similar.map(p => `${p.word} (${p.similarity.toFixed(4)})`).join(", "));
});
}

// 导出模型和示例
export {GloVec, demo};