# Skip-gram模型详解与实现文档

## 一、原理说明

### 1.1 模型概述
Skip-gram是Word2Vec家族的核心模型之一（另一为CBOW），由Mikolov等人在2013年提出。与CBOW相反，Skip-gram模型通过**中心词预测上下文词**，能够有效学习词的分布式表示，捕捉词之间的语义和语法关系。

### 1.2 核心思想
- 给定一个中心词，模型学习预测其周围一定窗口内的上下文词
- 词的含义由其上下文决定（分布假说）
- 通过神经网络学习到的词向量能够编码词的语义信息，使语义相似的词具有相似的向量表示

### 1.3 数学原理
- 设中心词为$w$，上下文词集合为$Context(w)$
- 模型目标是最大化条件概率：$P(Context(w)|w)$
- 对于每个上下文词$u \in Context(w)$，条件概率通过softmax计算：
  $$P(u|w) = \frac{\exp(v_u^T \cdot v_w)}{\sum_{u' \in V} \exp(v_{u'}^T \cdot v_w)}$$
  其中$v_w$是中心词向量，$v_u$是上下文词向量，$V$是词汇表
- 损失函数采用交叉熵损失：$L = -\sum_{u \in Context(w)} \log P(u|w)$

### 1.4 与CBOW的对比
| 特性 | Skip-gram | CBOW |
|------|-----------|------|
| 预测方向 | 中心词→上下文 | 上下文→中心词 |
| 对低频词 | 表现更好 | 表现较差 |
| 计算复杂度 | 较高（需预测多个上下文词） | 较低（仅预测一个中心词） |
| 训练速度 | 较慢 | 较快 |


## 二、处理流程

Skip-gram模型的完整处理流程分为以下步骤：

### 2.1 数据准备
- 收集语料库（由多个句子组成的文本集合）
- 文本预处理（分词、去噪等基础操作）

### 2.2 词汇表构建
- 从语料库中提取所有唯一词，建立词与索引的映射
- 统计词频，用于后续负采样

### 2.3 训练数据生成
- 滑动窗口遍历每个句子
- 对每个中心词，提取其左右各$windowSize$个词作为上下文
- 生成（中心词，上下文词）训练样本对（每个上下文词都是一个正例）

### 2.4 模型初始化
- 初始化输入权重矩阵（词向量矩阵）
- 初始化输出权重矩阵（上下文向量矩阵）
- 准备负采样的概率分布

### 2.5 模型训练
- 采用负采样（Negative Sampling）优化训练：
  - 对每个正例（中心词，上下文词），采样若干负例（与中心词无关联的词）
  - 将多分类问题转化为多个二分类问题（判断词对是否为真实上下文关系）
- 前向传播：计算预测概率
- 反向传播：使用梯度下降更新权重
- 多轮迭代训练，动态调整学习率

### 2.6 词向量应用
- 训练完成后，输入权重矩阵的每行作为对应词的词向量
- 可用于计算词相似度、上下文预测等任务


## 三、代码解释

### 3.1 类结构与初始化

```javascript
class SkipGram {
  constructor({ 
    embeddingSize = 10, 
    windowSize = 2, 
    learningRate = 0.01,
    negativeSamples = 5  // 负采样数量
  }) {
    this.embeddingSize = embeddingSize;  // 词向量维度
    this.windowSize = windowSize;        // 上下文窗口大小
    this.learningRate = learningRate;    // 学习率
    this.negativeSamples = negativeSamples;  // 负采样数量
    
    // 词汇表相关
    this.wordToIndex = {};      // 词到索引的映射
    this.indexToWord = [];      // 索引到词的映射
    this.vocabSize = 0;         // 词汇表大小
    this.wordCounts = {};       // 词频统计
    
    // 权重矩阵
    this.weightsInput = null;   // 输入层权重 (vocabSize × embeddingSize)
    this.weightsOutput = null;  // 输出层权重 (embeddingSize × vocabSize)
  }
  // ...
}
```

构造函数初始化模型核心参数，包括词向量维度、窗口大小、学习率和负采样数量，并初始化词汇表相关结构。

### 3.2 词汇表与负采样准备

```javascript
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

  // 为负采样准备概率分布
  this.prepareNegativeSamplingDistribution();

  // 初始化权重矩阵
  this.weightsInput = this.randomMatrix(this.vocabSize, this.embeddingSize);
  this.weightsOutput = this.randomMatrix(this.embeddingSize, this.vocabSize);
}
```

`buildVocab`方法完成：
- 词频统计：记录每个词在语料库中出现的次数
- 词汇表构建：建立词与索引的双向映射
- 负采样准备：基于词频构建采样分布（见3.3）
- 权重初始化：使用[-0.1, 0.1]的随机值初始化权重矩阵

### 3.3 负采样实现

```javascript
prepareNegativeSamplingDistribution() {
  this.negativeProb = new Array(this.vocabSize);
  let total = 0;

  // 计算词频的0.75次方总和（平衡高频词和低频词）
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
```

负采样是Skip-gram的关键优化：
- 解决了softmax计算复杂度高的问题（从$O(V)$降至$O(1)$）
- 采样概率基于词频的0.75次方，避免高频词被过度采样
- 累积概率分布用于加速采样过程

### 3.4 训练数据生成

```javascript
generateTrainingData(corpus) {
  const trainingData = [];
  
  corpus.forEach(sentence => {
    const words = sentence.split(' ');
    
    for (let i = 0; i < words.length; i++) {
      const centerWord = words[i];
      const centerIndex = this.wordToIndex[centerWord];
      
      // 收集上下文词（左右各windowSize个）
      for (let j = 1; j <= this.windowSize; j++) {
        // 左侧上下文
        if (i - j >= 0) {
          const contextWord = words[i - j];
          trainingData.push({
            center: centerIndex,
            context: this.wordToIndex[contextWord]
          });
        }
        
        // 右侧上下文
        if (i + j < words.length) {
          const contextWord = words[i + j];
          trainingData.push({
            center: centerIndex,
            context: this.wordToIndex[contextWord]
          });
        }
      }
    }
  });
  
  return trainingData;
}
```

`generateTrainingData`方法生成（中心词，上下文词）训练对：
- 对每个词，将其作为中心词
- 提取其左右窗口内的词作为上下文词
- 每个（中心词，上下文词）对都是一个正例样本

### 3.5 训练过程

```javascript
train(corpus, epochs = 100) {
  this.buildVocab(corpus);                 // 构建词汇表
  const trainingData = this.generateTrainingData(corpus);  // 生成训练数据
  
  console.log(`训练数据量: ${trainingData.length}`);
  console.log(`词汇表大小: ${this.vocabSize}`);

  // 训练循环
  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;
    
    this.shuffleArray(trainingData);  // 随机打乱训练数据
    
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
    
    // 输出损失并调整学习率
    if (epoch % 10 === 0) {
      console.log(`Epoch ${epoch}, 平均损失: ${(totalLoss / trainingData.length).toFixed(6)}`);
      
      if (epoch > 0 && epoch % 50 === 0) {
        this.learningRate *= 0.5;  // 学习率衰减
        console.log(`调整学习率为: ${this.learningRate}`);
      }
    }
  }
}
```

`train`方法是训练主函数：
- 每轮训练前打乱数据顺序，增加随机性
- 对每个正例样本，采样多个负例
- 通过`trainPair`方法处理每个（中心词，上下文词）对（正例标签为1，负例为0）
- 动态调整学习率（每50轮减半），平衡收敛速度和稳定性

### 3.6 单样本训练（前向+反向传播）

```javascript
trainPair(center, context, label) {
  // 前向传播：输入层直接输出词向量
  const hidden = this.weightsInput[center];
  
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
```

`trainPair`方法处理单个样本的训练：
- 前向传播：计算中心词与上下文词的相似度得分，通过sigmoid得到概率
- 损失计算：使用二元交叉熵损失，判断词对是否为真实上下文关系
- 反向传播：计算误差并更新输入层和输出层权重

### 3.7 辅助功能

```javascript
// 获取词向量
getWordVector(word) {
  if (!this.wordToIndex.hasOwnProperty(word)) {
    throw new Error(`单词 "${word}" 不在词汇表中`);
  }
  return [...this.weightsInput[this.wordToIndex[word]]];
}

// 预测上下文词
predictContext(centerWord, topN = 5) {
  // 计算中心词与所有词的相似度，返回前N个最可能的上下文词
  // 实现见完整代码...
}

// 计算词相似度（余弦相似度）
getSimilarity(word1, word2) {
  // 实现见完整代码...
}
```

模型提供了获取词向量、预测上下文词和计算词相似度的功能，方便训练后使用模型进行语义分析。


## 四、优化说明

本实现包含多项关键优化：

1. **负采样优化**：
   - 使用词频的0.75次方构建采样分布，平衡高频词和低频词
   - 用累积概率分布加速采样过程

2. **训练效率优化**：
   - 用sigmoid替代softmax，将多分类转为二分类，降低计算复杂度
   - 每轮训练前打乱数据顺序，避免模型学习顺序偏差

3. **数值稳定性优化**：
   - 权重初始化使用[-0.1, 0.1]的小随机值
   - 计算对数时添加极小值（1e-10），避免log(0)错误

4. **学习率调度**：
   - 动态调整学习率（每50轮减半），兼顾收敛速度和收敛精度


## 五、示例演示说明

示例使用宠物相关语料库（猫、狗及其行为特征），通过训练Skip-gram模型可以观察到：

1. 训练过程中损失逐步下降并趋于稳定，表明模型在学习词之间的关联
2. 语义相关的词（如"cats"和"dogs"）具有较高的相似度
3. 模型能预测合理的上下文词（如"cats"常与"like"、"chase"、"mice"同时出现）
4. 功能词（如"are"、"and"）与内容词的相似度较低

这些结果验证了Skip-gram模型捕捉词间语义关系的能力，生成的词向量可用于文本分类、信息检索等多种自然语言处理任务。