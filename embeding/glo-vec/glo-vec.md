# GloVe模型详解与实现文档

## 一、原理说明

### 1.1 模型概述
GloVe（Global Vectors for Word Representation）是由斯坦福大学研究团队在2014年提出的词向量学习模型，全称为"全局词向量表示"。它结合了Word2Vec的局部上下文信息和潜在语义分析（LSA）的全局统计信息，通过优化加权最小二乘目标函数来学习高质量的词向量。

### 1.2 核心思想
- **全局统计与局部上下文结合**：既利用词共现矩阵的全局统计信息，又保留局部上下文的模式
- **共现概率比**：模型核心是学习词共现概率的比率而非概率本身，这种比率能更好地捕捉词之间的语义关系
- **加权最小二乘优化**：通过最小化加权平方误差来学习词向量，权重函数平衡高频和低频共现对的影响

### 1.3 数学原理
1. **共现矩阵**：定义$X$为共现矩阵，$X_{ij}$表示词$j$出现在词$i$上下文窗口中的次数
2. **共现概率**：$P_{ij} = P(j|i) = X_{ij}/X_i$，其中$X_i = \sum_k X_{ik}$是词$i$的边缘总计数
3. **核心比率**：词向量需要捕捉比率$P_{ij}/P_{ik}$的信息，该比率能反映词$j$和$k$相对于词$i$的语义关系
4. **目标函数**：
   $$J = \sum_{i,j=1}^V f(X_{ij})(w_i \cdot \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$
   其中：
   - $w_i$和$\tilde{w}_j$分别是词$i$的词向量和上下文词向量
   - $b_i$和$\tilde{b}_j$是对应的偏置项
   - $f(X_{ij})$是权重函数

5. **权重函数**：
   $$f(x) = \begin{cases} 
   (x/x_{\text{max}})^\alpha & \text{if } x < x_{\text{max}} \\
   1 & \text{if } x \geq x_{\text{max}}
   \end{cases}$$
   通常取$x_{\text{max}}=100$，$\alpha=0.75$，用于降低高频共现对的权重，避免过度拟合。

### 1.4 与其他模型对比
| 模型 | 核心特点 | 优势 | 劣势 |
|------|----------|------|------|
| GloVe | 全局统计+局部信息 | 语义捕捉能力强，训练稳定 | 需预先计算共现矩阵，内存占用大 |
| Word2Vec(CBOW) | 局部上下文预测 | 训练速度快，内存占用小 | 全局信息利用不足 |
| Word2Vec(Skip-gram) | 中心词预测上下文 | 低频词表现好 | 训练速度慢 |
| LSA | 矩阵分解 | 利用全局统计 | 词向量质量较低，计算复杂 |


## 二、处理流程

GloVe模型的完整处理流程分为以下步骤：

### 2.1 数据准备
- 收集语料库（由多个句子组成的文本集合）
- 文本预处理（分词、基本清洗等）

### 2.2 共现矩阵构建
- 滑动窗口遍历语料库中的每个句子
- 对每个中心词，统计其上下文窗口内所有词的共现次数
- 对共现次数进行距离加权（距离越近权重越高，通常取$1/d$，$d$为距离）
- 存储共现矩阵（通常采用稀疏存储方式）

### 2.3 词汇表构建
- 从语料库中提取所有唯一词，建立词与索引的映射关系
- 确定词汇表大小$V$

### 2.4 模型初始化
- 初始化词向量矩阵$w$（$V \times d$，$d$为词向量维度）
- 初始化上下文词向量矩阵$\tilde{w}$（$V \times d$）
- 初始化偏置项$b$和$\tilde{b}$（长度为$V$的向量）

### 2.5 模型训练
- 生成共现对训练样本（所有$X_{ij} > 0$的词对）
- 迭代优化目标函数：
  1. 计算权重$f(X_{ij})$
  2. 前向计算预测值与真实值（$\log X_{ij}$）的误差
  3. 计算梯度并更新词向量和偏置项
- 动态调整学习率，直至收敛

### 2.6 词向量提取
- 训练完成后，通常取词向量和上下文词向量的平均值（$w_i + \tilde{w}_i)/2$作为最终词向量
- 应用于下游任务（相似度计算、文本分类等）


## 三、代码解释

### 3.1 类结构与初始化

```javascript
class GloVe {
  constructor({
    embeddingSize = 10,
    windowSize = 2,
    learningRate = 0.05,
    xMax = 100,    // 权重函数的阈值
    alpha = 0.75   // 权重函数的指数
  }) {
    this.embeddingSize = embeddingSize;  // 词向量维度
    this.windowSize = windowSize;        // 上下文窗口大小
    this.learningRate = learningRate;    // 学习率
    this.xMax = xMax;                    // 权重函数饱和阈值
    this.alpha = alpha;                  // 权重函数指数
    
    // 词汇表相关
    this.wordToIndex = {};      // 词到索引的映射
    this.indexToWord = [];      // 索引到词的映射
    this.vocabSize = 0;         // 词汇表大小
    this.cooccurrenceMatrix = {}; // 共现矩阵 {i,j: count}
    
    // 模型参数
    this.w = null;  // 词向量矩阵 (vocabSize × embeddingSize)
    this.wTilde = null;  // 上下文词向量矩阵 (vocabSize × embeddingSize)
    this.b = null;  // 词偏置项 (vocabSize)
    this.bTilde = null;  // 上下文词偏置项 (vocabSize)
  }
  // ...
}
```

构造函数初始化模型核心参数，包括词向量维度、窗口大小、学习率以及权重函数的参数，并初始化词汇表和共现矩阵的占位符。

### 3.2 词汇表与共现矩阵构建

```javascript
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
```

`buildVocabAndCooccurrence`方法完成两项核心工作：
1. 词汇表构建：从语料库中提取所有唯一词，建立词与索引的映射
2. 共现矩阵构建：通过滑动窗口统计词对共现次数，并根据距离赋予权重（距离越近权重越高）

### 3.3 共现矩阵更新

```javascript
updateCooccurrence(i, j, weight) {
  // 使用字符串键确保i<j，避免重复计数
  const key = i < j ? `${i},${j}` : `${j},${i}`;
  this.cooccurrenceMatrix[key] = (this.cooccurrenceMatrix[key] || 0) + weight;
}
```

`updateCooccurrence`方法用于更新共现矩阵：
- 使用`i,j`的有序组合作为键（确保`i<j`），避免重复存储（如`(i,j)`和`(j,i)`视为同一对）
- 累加共现权重（基础共现次数乘以距离权重）

### 3.4 权重函数实现

```javascript
weightFunction(x) {
  if (x > this.xMax) return 1;
  return Math.pow(x / this.xMax, this.alpha);
}
```

`weightFunction`实现GloVe的核心权重函数：
- 当共现次数$x$超过$x_{\text{max}}$时，权重饱和为1
- 当$x$小于$x_{\text{max}}$时，权重为$(x/x_{\text{max}})^\alpha$，降低高频共现对的影响

### 3.5 训练数据生成

```javascript
generateTrainingSamples() {
  const samples = [];
  for (const key in this.cooccurrenceMatrix) {
    const [i, j] = key.split(',').map(Number);
    const count = this.cooccurrenceMatrix[key];
    samples.push({ i, j, count });
  }
  return samples;
}
```

`generateTrainingSamples`方法将共现矩阵转换为训练样本：每个样本包含词对索引`(i,j)`和对应的共现次数`count`。

### 3.6 训练过程

```javascript
train(corpus, epochs = 100) {
  // 构建词汇表和共现矩阵
  this.buildVocabAndCooccurrence(corpus);
  const trainingSamples = this.generateTrainingSamples();
  
  console.log(`训练样本数: ${trainingSamples.length}`);
  console.log(`词汇表大小: ${this.vocabSize}`);

  // 训练循环
  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;
    
    // 随机打乱样本顺序
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
      
      // 反向传播更新
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
    
    // 输出损失并调整学习率
    if (epoch % 10 === 0) {
      console.log(`Epoch ${epoch}, 总损失: ${totalLoss.toFixed(6)}`);
      
      if (epoch > 0 && epoch % 50 === 0) {
        this.learningRate *= 0.5;
        console.log(`调整学习率为: ${this.learningRate}`);
      }
    }
  }
}
```

`train`方法是模型训练的主函数：
- 每轮训练前打乱样本顺序，避免模型学习顺序偏差
- 对每个共现对样本：
  1. 计算权重函数值$w = f(X_{ij})$
  2. 前向计算预测误差：$e = w_i \cdot \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}$
  3. 计算损失：$w \cdot e^2$
  4. 反向传播计算梯度，并更新词向量和偏置项
- 动态调整学习率（每50轮减半），优化训练过程

### 3.7 词向量提取与应用

```javascript
// 获取词向量（取词向量和上下文词向量的平均值）
getWordVector(word) {
  if (!this.wordToIndex.hasOwnProperty(word)) {
    throw new Error(`单词 "${word}" 不在词汇表中`);
  }
  
  const idx = this.wordToIndex[word];
  const vector = new Array(this.embeddingSize);
  
  // 取词向量和上下文向量的平均作为最终向量
  for (let k = 0; k < this.embeddingSize; k++) {
    vector[k] = (this.w[idx][k] + this.wTilde[idx][k]) / 2;
  }
  
  return vector;
}

// 计算两个词的相似度（余弦相似度）
getSimilarity(word1, word2) {
  // 实现余弦相似度计算...
}

// 查找最相似的词
mostSimilar(word, topN = 5) {
  // 查找并返回最相似的词...
}
```

GloVe通常将词向量$w_i$和上下文词向量$\tilde{w}_i$的平均值作为最终词向量，这种融合方式能提升词表示的质量。模型还提供了相似度计算和相似词查找功能，方便应用于实际任务。


## 四、优化说明

本实现包含多项关键优化：

1. **共现矩阵优化**：
   - 采用稀疏存储方式（仅存储非零共现对），大幅减少内存占用
   - 对共现对进行距离加权，增强近邻词的影响

2. **数值计算优化**：
   - 权重初始化使用[-0.02, 0.02]的小随机值，提高训练稳定性
   - 样本随机打乱，避免模型学习顺序依赖

3. **训练过程优化**：
   - 动态学习率调整（每50轮减半），平衡收敛速度和精度
   - 权重函数平衡高频和低频共现对的影响，避免过度拟合高频词

4. **词向量质量优化**：
   - 采用词向量与上下文向量的平均值作为最终表示，提升语义捕捉能力


## 五、示例演示说明

示例使用"首都-国家-大洲"关系的语料库，通过训练GloVe模型可以观察到：

1. 训练过程中损失逐步下降并趋于稳定，表明模型在有效学习词间关系
2. 城市与所属国家的相似度高（如"paris"与"france"，"tokyo"与"japan"）
3. 同类国家的相似度高（如欧洲国家之间、亚洲国家之间）
4. 功能词（如"is"、"the"）与实体词的相似度低

这些结果验证了GloVe模型捕捉语义关系的能力，尤其是对"首都-国家"这类特定关系的学习效果。生成的词向量不仅能反映词的相似度，还能捕捉词之间的类比关系（如"paris之于france正如tokyo之于japan"）。

GloVe模型特别适合需要高质量词向量的任务，如文本分类、命名实体识别、机器翻译等，在多数NLP任务中表现优于Word2Vec等模型。