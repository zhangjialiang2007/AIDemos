class FastText {
    /**
     * 初始化 FastText 模型
     * @param {Object} config 配置参数
     * @param {number} config.dim 词向量维度
     * @param {number} config.minN 最小子词长度
     * @param {number} config.maxN 最大子词长度
     * @param {number} config.learningRate 学习率
     */
    constructor(config = {}) {
      this.dim = config.dim || 100; // 向量维度
      this.minN = config.minN || 3; // 最小子词长度
      this.maxN = config.maxN || 6; // 最大子词长度
      this.lr = config.learningRate || 0.01; // 学习率
  
      this.wordVectors = new Map(); // 词向量表
      this.subwordVectors = new Map(); // 子词向量表
      this.labelWeights = new Map(); // 标签权重（分类头）
      this.vocab = new Map(); // 词表（记录词频）
      this.labels = new Set(); // 标签集合
    }
  
    /**
     * 生成词的所有子词（n-gram）
     * @param {string} word 输入词
     * @returns {Set<string>} 子词集合（包含特殊标记<bos>和<eos>）
     */
    getSubwords(word) {
      const subwords = new Set();
      // 添加特殊标记表示词的开始和结束
      const markedWord = `<bos>${word}<eos>`;
  
      // 生成所有长度在 [minN, maxN] 之间的子词
      for (let n = this.minN; n <= this.maxN; n++) {
        for (let i = 0; i <= markedWord.length - n; i++) {
          subwords.add(markedWord.slice(i, i + n));
        }
      }
      return subwords;
    }
  
    /**
     * 初始化向量（随机生成正态分布向量）
     * @returns {Float32Array} 初始化的向量
     */
    initVector() {
      const vec = new Float32Array(this.dim);
      for (let i = 0; i < this.dim; i++) {
        // 正态分布初始化（均值0，方差0.01）
        vec[i] = (Math.random() - 0.5) * 2 * 0.1;
      }
      return vec;
    }
  
    /**
     * 计算词的总向量（词向量 + 所有子词向量的平均）
     * @param {string} word 输入词
     * @returns {Float32Array} 总向量
     */
    getWordTotalVector(word) {
      // 初始化总向量为0
      const totalVec = new Float32Array(this.dim).fill(0);
      let count = 0;
  
      // 添加词向量（如果存在）
      if (this.wordVectors.has(word)) {
        const wordVec = this.wordVectors.get(word);
        for (let i = 0; i < this.dim; i++) {
          totalVec[i] += wordVec[i];
        }
        count++;
      }
  
      // 添加所有子词向量（如果存在）
      const subwords = this.getSubwords(word);
      subwords.forEach(subword => {
        if (this.subwordVectors.has(subword)) {
          const subVec = this.subwordVectors.get(subword);
          for (let i = 0; i < this.dim; i++) {
            totalVec[i] += subVec[i];
          }
          count++;
        }
      });
  
      // 平均所有向量（防止除以0）
      if (count > 0) {
        for (let i = 0; i < this.dim; i++) {
          totalVec[i] /= count;
        }
      }
      return totalVec;
    }
  
    /**
     * 计算句子的总向量（所有词的总向量的平均）
     * @param {string[]} sentence 分词后的句子
     * @returns {Float32Array} 句子向量
     */
    getSentenceVector(sentence) {
      const sentVec = new Float32Array(this.dim).fill(0);
      let wordCount = 0;
  
      for (const word of sentence) {
        const wordVec = this.getWordTotalVector(word);
        for (let i = 0; i < this.dim; i++) {
          sentVec[i] += wordVec[i];
        }
        wordCount++;
      }
  
      // 平均所有词向量
      if (wordCount > 0) {
        for (let i = 0; i < this.dim; i++) {
          sentVec[i] /= wordCount;
        }
      }
      return sentVec;
    }
  
    /**
     * 训练模型（单样本更新）
     * @param {string[]} sentence 分词后的句子
     * @param {string} label 句子标签
     */
    train(sentence, label) {
      // 1. 计算句子向量
      const sentVec = this.getSentenceVector(sentence);
  
      // 2. 初始化未见过的标签权重
      if (!this.labelWeights.has(label)) {
        this.labelWeights.set(label, this.initVector());
      }
      this.labels.add(label);
  
      // 3. 计算所有标签的得分（点积）
      const scores = new Map();
      let maxScore = -Infinity;
      this.labels.forEach(l => {
        const weight = this.labelWeights.get(l);
        let score = 0;
        for (let i = 0; i < this.dim; i++) {
          score += sentVec[i] * weight[i];
        }
        scores.set(l, score);
        if (score > maxScore) maxScore = score;
      });
  
      // 4. 计算softmax归一化（防止溢出，减去maxScore）
      let expSum = 0;
      this.labels.forEach(l => {
        expSum += Math.exp(scores.get(l) - maxScore);
      });
  
      // 5. 计算梯度并更新（交叉熵损失）
      this.labels.forEach(l => {
        const prob = Math.exp(scores.get(l) - maxScore) / expSum;
        const gradient = (l === label ? 1 - prob : -prob) * this.lr;
  
        // 更新标签权重
        const weight = this.labelWeights.get(l);
        for (let i = 0; i < this.dim; i++) {
          weight[i] += gradient * sentVec[i];
        }
      });
  
      // 6. 更新词向量和子词向量
      for (const word of sentence) {
        // 初始化未见过的词向量
        if (!this.wordVectors.has(word)) {
          this.wordVectors.set(word, this.initVector());
          this.vocab.set(word, (this.vocab.get(word) || 0) + 1);
        }
  
        // 收集当前词的所有向量（词 + 子词）
        const vectors = [];
        vectors.push(this.wordVectors.get(word));
        this.getSubwords(word).forEach(subword => {
          if (!this.subwordVectors.has(subword)) {
            this.subwordVectors.set(subword, this.initVector());
          }
          vectors.push(this.subwordVectors.get(subword));
        });
  
        // 计算词级梯度（所有标签梯度的加权和）
        const wordGradient = new Float32Array(this.dim).fill(0);
        this.labels.forEach(l => {
          const prob = Math.exp(scores.get(l) - maxScore) / expSum;
          const grad = (l === label ? 1 - prob : -prob) * this.lr;
          const weight = this.labelWeights.get(l);
          for (let i = 0; i < this.dim; i++) {
            wordGradient[i] += grad * weight[i];
          }
        });
  
        // 平均梯度并更新所有相关向量
        const avgGrad = wordGradient.map(g => g / vectors.length);
        vectors.forEach(vec => {
          for (let i = 0; i < this.dim; i++) {
            vec[i] += avgGrad[i];
          }
        });
      }
    }
  
    /**
     * 预测句子标签
     * @param {string[]} sentence 分词后的句子
     * @returns {string} 预测的标签
     */
    predict(sentence) {
      const sentVec = this.getSentenceVector(sentence);
      let maxScore = -Infinity;
      let bestLabel = null;
  
      this.labels.forEach(label => {
        const weight = this.labelWeights.get(label);
        let score = 0;
        for (let i = 0; i < this.dim; i++) {
          score += sentVec[i] * weight[i];
        }
        if (score > maxScore) {
          maxScore = score;
          bestLabel = label;
        }
      });
  
      return bestLabel;
    }
}
  
  
// ------------------------------
// 使用示例：文本分类任务
// ------------------------------
async function demo() {
    // 1. 初始化模型
    const model = new FastText({
        dim: 50,      // 向量维度
        minN: 3,      // 最小子词长度
        maxN: 5,      // 最大子词长度
        learningRate: 0.05  // 学习率
    });

    // 2. 训练数据（简单情感分类：正面/负面）
    const trainData = [
        { sentence: ["我", "喜欢", "这个", "电影"], label: "positive" },
        { sentence: ["这部", "电影", "非常", "精彩"], label: "positive" },
        { sentence: ["剧情", "很棒", "演员", "优秀"], label: "positive" },
        { sentence: ["我", "讨厌", "这个", "电影"], label: "negative" },
        { sentence: ["剧情", "无聊", "浪费", "时间"], label: "negative" },
        { sentence: ["演员", "表现", "差", "极了"], label: "negative" }
    ];

    // 3. 训练模型（迭代100次）
    for (let epoch = 0; epoch < 100; epoch++) {
        trainData.forEach(item => {
        model.train(item.sentence, item.label);
        });
    }

    // 4. 预测测试
    const testSentences = [
        ["电影", "很", "好看"],    // 预期：positive
        ["太", "难看", "了"],      // 预期：negative
        ["演员", "很", "棒"]       // 预期：positive
    ];

    testSentences.forEach(sent => {
        const pred = model.predict(sent);
        console.log(`句子：${sent.join(' ')}，预测标签：${pred}`);
    });
}
  
// 运行示例
export {FastText, demo}