# CBOW模型详解与实现文档

## 一、CBOW模型原理说明

CBOW（Continuous Bag-of-Words）是Word2Vec中的一种核心模型，由Mikolov等人在2013年提出，用于学习高质量的词向量表示。

### 核心思想
CBOW模型通过上下文词来预测中心词，即给定一个中心词周围的上下文词，模型学习输出该中心词的概率分布。与传统的词袋模型不同，CBOW能够捕捉词之间的语义和语法关系，生成的词向量具有良好的线性性质（如"国王-男人+女人≈女王"）。

### 模型结构
CBOW模型由三层组成：
1. **输入层**：将上下文词转换为独热编码（One-Hot Encoding）
2. **隐藏层**：计算所有上下文词向量的平均值
3. **输出层**：通过Softmax函数计算中心词的概率分布

### 数学原理
- 设上下文词集合为$Context(w)$，中心词为$w$
- 隐藏层向量$h$是上下文词向量的平均值：$h = \frac{1}{|Context(w)|} \sum_{u \in Context(w)} v_u$，其中$v_u$是词$u$的输入向量
- 输出层计算：$p(w|Context(w)) = \frac{\exp(v'_w \cdot h)}{\sum_{w' \in V} \exp(v'_{w'} \cdot h)}$，其中$v'_w$是词$w$的输出向量，$V$是词汇表
- 损失函数采用交叉熵损失：$L = -\log p(w|Context(w))$

## 二、处理流程

CBOW模型的完整处理流程分为以下几个步骤：

1. **数据准备**
   - 收集语料库（由多个句子组成的文本集合）
   - 对文本进行预处理（分词等基础操作）

2. **词汇表构建**
   - 从语料库中提取所有唯一词，建立词与索引的映射关系
   - 确定词汇表大小$V$

3. **训练数据生成**
   - 滑动窗口遍历每个句子
   - 对每个中心词，提取其左右各$windowSize$个词作为上下文
   - 生成（上下文，中心词）训练样本对

4. **模型初始化**
   - 初始化输入权重矩阵$W$（$V \times d$，$d$为词向量维度）
   - 初始化输出权重矩阵$W'$（$d \times V$）

5. **模型训练**
   - 前向传播：计算隐藏层向量和输出层概率分布
   - 计算损失：基于预测结果与真实标签的交叉熵
   - 反向传播：使用梯度下降更新权重矩阵
   - 迭代训练：多次遍历训练数据，逐步优化模型参数

6. **词向量提取**
   - 训练完成后，输入权重矩阵$W$的每行即为对应词的词向量
   - 可用于下游任务（如文本分类、相似度计算等）

## 三、代码解释

下面对CBOW模型的JavaScript实现进行详细解释：

### 1. 类结构与初始化

```javascript
class CBOW {
  constructor({ embeddingSize = 10, windowSize = 2, learningRate = 0.01 }) {
    this.embeddingSize = embeddingSize;  // 词向量维度
    this.windowSize = windowSize;        // 上下文窗口大小
    this.learningRate = learningRate;    // 学习率
    
    // 词汇表相关
    this.wordToIndex = {};      // 词到索引的映射
    this.indexToWord = [];      // 索引到词的映射
    this.vocabSize = 0;         // 词汇表大小
    
    // 权重矩阵
    this.weightsInput = null;   // 输入层权重矩阵 (vocabSize × embeddingSize)
    this.weightsOutput = null;  // 输出层权重矩阵 (embeddingSize × vocabSize)
  }
  // ...
}
```

构造函数初始化模型的核心参数，包括词向量维度、窗口大小和学习率，并初始化词汇表相关结构和权重矩阵的占位符。

### 2. 词汇表构建

```javascript
buildVocab(corpus) {
  const words = new Set();
  corpus.forEach(sentence => {
    sentence.split(' ').forEach(word => {
      words.add(word);  // 收集所有唯一词
    });
  });

  // 构建词与索引的双向映射
  this.indexToWord = Array.from(words);
  this.indexToWord.forEach((word, index) => {
    this.wordToIndex[word] = index;
  });
  this.vocabSize = this.indexToWord.length;

  // 初始化权重矩阵 (随机初始化)
  this.weightsInput = this.randomMatrix(this.vocabSize, this.embeddingSize);
  this.weightsOutput = this.randomMatrix(this.embeddingSize, this.vocabSize);
}
```

`buildVocab`方法从语料库中提取所有唯一词，建立词与索引的映射关系，并初始化输入和输出权重矩阵。权重初始化使用[-0.1, 0.1]范围内的随机值，有助于训练稳定性。

### 3. 训练数据生成

```javascript
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
```

`generateTrainingData`方法通过滑动窗口生成训练样本：对于每个中心词，收集其左右各`windowSize`个词作为上下文，形成（上下文，中心词）的训练对。

### 4. 训练过程

```javascript
train(corpus, epochs = 100) {
  this.buildVocab(corpus);                 // 构建词汇表
  const trainingData = this.generateTrainingData(corpus);  // 生成训练数据
  
  // 训练循环
  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;
    
    this.shuffleArray(trainingData);  // 随机打乱训练数据，增加随机性
    
    for (const { context, target } of trainingData) {
      // 前向传播
      const { hidden, output, loss } = this.forward(context, target);
      totalLoss += loss;
      
      // 反向传播更新权重
      this.backward(context, target, hidden, output);
    }
    
    // 输出损失并动态调整学习率
    if (epoch % 10 === 0) {
      console.log(`Epoch ${epoch}, 平均损失: ${totalLoss / trainingData.length}`);
      
      if (epoch > 0 && epoch % 50 === 0) {
        this.learningRate *= 0.5;  // 学习率衰减
        console.log(`调整学习率为: ${this.learningRate}`);
      }
    }
  }
}
```

`train`方法是模型训练的主函数，包含多轮迭代：
- 每轮迭代前打乱训练数据，避免模型学习顺序偏差
- 对每个样本执行前向传播计算损失
- 通过反向传播更新权重参数
- 定期输出损失并动态调整学习率（每50轮减半）

### 5. 前向传播

```javascript
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
  const loss = -Math.log(output[target] + 1e-10);  // 加小值避免log(0)
  
  return { hidden, output, loss };
}
```

`forward`方法实现前向传播：
1. 计算隐藏层向量：上下文词向量的平均值
2. 计算输出层：通过隐藏层与输出权重的点积，再经Softmax函数得到概率分布
3. 计算交叉熵损失：衡量预测分布与真实标签的差距

### 6. 反向传播

```javascript
backward(context, target, hidden, output) {
  // 1. 计算输出层误差
  const outputError = [...output];
  outputError[target] -= 1;  // 交叉熵+softmax的误差简化为output - target_one_hot

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
```

`backward`方法实现反向传播算法：
1. 计算输出层误差：对于Softmax+交叉熵组合，误差简化为预测概率减1（仅目标位置）
2. 更新输出层权重：根据隐藏层输出和输出层误差计算梯度
3. 计算隐藏层误差：通过输出层权重反向传播输出误差
4. 更新输入层权重：上下文词共享隐藏层误差（平均分配）

### 7. 辅助功能

```javascript
// 获取词向量
getWordVector(word) {
  if (!this.wordToIndex.hasOwnProperty(word)) {
    throw new Error(`单词 "${word}" 不在词汇表中`);
  }
  return [...this.weightsInput[this.wordToIndex[word]]];
}

// 预测上下文对应的中心词
predict(contextWords) {
  // 实现见完整代码...
}
```

模型还提供了获取词向量和预测中心词的功能，方便训练后使用模型进行相关任务。

## 四、优化说明

本实现包含多项优化措施：

1. **学习率动态调整**：每50轮将学习率减半，平衡前期快速收敛和后期精细调整
2. **训练数据随机打乱**：每轮训练前打乱样本顺序，避免模型学习到数据的顺序模式
3. **权重初始化优化**：使用[-0.1, 0.1]的小随机值初始化权重，提高训练稳定性
4. **数值稳定性处理**：计算对数时添加极小值（1e-10），避免出现log(0)的情况
5. **上下文窗口处理**：对句子边界进行判断，避免索引越界

这些优化措施有助于提高模型的收敛速度和最终性能，使模型在有限的训练轮数内能够学习到有意义的词向量表示。

## 五、示例演示说明

示例使用关于狐狸和狗的简单语料库，通过训练CBOW模型，能够：
1. 学习到"fox"与"dog"、"quick"与"lazy"等词的语义关系
2. 生成合理的词向量表示，语义相近的词向量具有较高相似度
3. 对给定上下文能够预测出合理的中心词

通过观察训练过程中的损失下降趋势、词向量分布以及预测结果，可以直观了解CBOW模型的工作原理和效果。