import * as tf from '@tensorflow/tfjs';

// 1. 位置编码（与 BERT 一致，为文本添加位置信息）
function positionalEncoding(maxPosition, dModel) {
  return tf.tidy(() => {
    const angleRates = tf.div(1, tf.pow(10000, tf.div(tf.range(0, dModel, 2), dModel)));
    const angleRad = tf.matMul(
      tf.range(0, maxPosition).reshape([-1, 1]),
      angleRates.reshape([1, -1])
    );
    const sin = tf.sin(angleRad);
    const cos = tf.cos(angleRad);
    return tf.concat([sin, cos], 1).reshape([1, maxPosition, dModel]);
  });
}

// 2. 单向自注意力（掩码屏蔽后文）
class CausalSelfAttention {
  constructor(dModel, numHeads) {
    this.dModel = dModel;
    this.numHeads = numHeads;
    this.depth = dModel / numHeads;

    // 权重初始化
    this.wq = tf.variable(tf.randomNormal([dModel, dModel]));
    this.wk = tf.variable(tf.randomNormal([dModel, dModel]));
    this.wv = tf.variable(tf.randomNormal([dModel, dModel]));
    this.dense = tf.variable(tf.randomNormal([dModel, dModel]));
  }

  // 分割注意力头
  splitHeads(x, batchSize) {
    return tf.tidy(() => {
      const reshaped = x.reshape([batchSize, -1, this.numHeads, this.depth]);
      return tf.transpose(reshaped, [0, 2, 1, 3]); // [batch, heads, seqLen, depth]
    });
  }

  // 带因果掩码的点积注意力
  scaledDotProductAttention(q, k, v) {
    return tf.tidy(() => {
      const matmulQk = tf.matMul(q, k, false, true); // [batch, heads, seqLen, seqLen]
      const dk = tf.cast(tf.shape(k)[-1], 'float32');
      const scaled = tf.div(matmulQk, tf.sqrt(dk));

      // 生成因果掩码：上三角矩阵设为 -1e9（屏蔽后文）
      const seqLen = tf.shape(scaled)[2];
      const mask = tf.linalg.bandPart(tf.ones([seqLen, seqLen]), -1, 0); // 下三角为 1，上三角为 0
      const causalMask = tf.where(mask.equal(0), tf.scalar(-1e9), tf.scalar(0));
      const maskedScaled = tf.add(scaled, causalMask);

      const attentionWeights = tf.softmax(maskedScaled);
      const output = tf.matMul(attentionWeights, v); // [batch, heads, seqLen, depth]
      return { output, attentionWeights };
    });
  }

  apply(x) {
    return tf.tidy(() => {
      const batchSize = tf.shape(x)[0];

      // 线性投影 + 分割头
      const q = tf.matMul(x, this.wq);
      const k = tf.matMul(x, this.wk);
      const v = tf.matMul(x, this.wv);
      const qSplit = this.splitHeads(q, batchSize);
      const kSplit = this.splitHeads(k, batchSize);
      const vSplit = this.splitHeads(v, batchSize);

      // 单向注意力计算
      const { output: attnOutput } = this.scaledDotProductAttention(qSplit, kSplit, vSplit);

      // 拼接注意力头 + 最终线性层
      const attnTranspose = tf.transpose(attnOutput, [0, 2, 1, 3]); // [batch, seqLen, heads, depth]
      const concatAttn = attnTranspose.reshape([batchSize, -1, this.dModel]);
      const finalOutput = tf.matMul(concatAttn, this.dense);
      return finalOutput;
    });
  }
}

// 3. 解码器层（GPT 核心模块）
class DecoderLayer {
  constructor(dModel, numHeads, dff, rate = 0.1) {
    this.causalSelfAttn = new CausalSelfAttention(dModel, numHeads);
    this.ffn = this.pointwiseFeedForwardNetwork(dModel, dff);

    this.layernorm1 = tf.layers.layerNormalization({ epsilon: 1e-6 });
    this.layernorm2 = tf.layers.layerNormalization({ epsilon: 1e-6 });

    this.dropout1 = tf.layers.dropout({ rate });
    this.dropout2 = tf.layers.dropout({ rate });
  }

  // 前馈网络（与 BERT 一致）
  pointwiseFeedForwardNetwork(dModel, dff) {
    return tf.sequential({
      layers: [
        tf.layers.dense({ units: dff, activation: 'relu' }),
        tf.layers.dense({ units: dModel })
      ]
    });
  }

  apply(x, training) {
    return tf.tidy(() => {
      // 单向自注意力 + 残差连接 + 归一化
      const attn1 = this.causalSelfAttn.apply(x);
      const attn1Dropout = this.dropout1.apply(attn1, { training });
      const out1 = this.layernorm1.apply(tf.add(x, attn1Dropout));

      // 前馈网络 + 残差连接 + 归一化
      const ffnOutput = this.ffn.predict(out1);
      const ffnDropout = this.dropout2.apply(ffnOutput, { training });
      const out2 = this.layernorm2.apply(tf.add(out1, ffnDropout));

      return out2;
    });
  }
}

// 4. 简化版 GPT 模型（模拟 GPT-3 核心结构）
class SimpleGPT {
  constructor(vocabSize, numLayers, dModel, numHeads, dff, maxSeqLen) {
    this.vocabSize = vocabSize;
    this.dModel = dModel;
    this.numLayers = numLayers;

    // 词嵌入 + 位置编码
    this.embedding = tf.layers.embedding({ inputDim: vocabSize, outputDim: dModel });
    this.posEncoding = positionalEncoding(maxSeqLen, dModel);
    this.dropout = tf.layers.dropout({ rate: 0.1 });

    // 堆叠多个解码器层
    this.decoderLayers = Array.from({ length: numLayers },
      () => new DecoderLayer(dModel, numHeads, dff));

    // 输出层：预测下一个词的概率分布
    this.finalLayer = tf.layers.dense({ units: vocabSize, activation: 'softmax' });
  }

  // 前向传播（生成单步输出）
  apply(x, training) {
    return tf.tidy(() => {
      const seqLen = tf.shape(x)[1];

      // 词嵌入 + 位置编码
      let xEmbed = this.embedding.apply(x);
      xEmbed = tf.multiply(xEmbed, tf.sqrt(tf.scalar(this.dModel)));
      xEmbed = tf.add(xEmbed, this.posEncoding.slice([0, 0, 0], [-1, seqLen, -1]));
      let xOut = this.dropout.apply(xEmbed, { training });

      // 经过所有解码器层
      for (let i = 0; i < this.numLayers; i++) {
        xOut = this.decoderLayers[i].apply(xOut, training);
      }

      // 预测下一个词（取最后一个位置的输出）
      const finalOutput = this.finalLayer.apply(xOut);
      return finalOutput.slice([0, seqLen - 1, 0], [-1, 1, -1]); // [batch, 1, vocabSize]
    });
  }

  // 文本生成逻辑（自回归：逐词生成）
  async generate(inputSeq, maxGenLen) {
    let currentSeq = inputSeq; // 初始输入序列（词ID数组）
    for (let i = 0; i < maxGenLen; i++) {
      // 转换为 TF 张量 [1, seqLen]
      const inputTensor = tf.tensor2d([currentSeq], 'int32');
      // 预测下一个词的概率分布
      const pred = this.apply(inputTensor, false);
      // 取概率最大的词ID
      const predIdx = await pred.argMax(-1).dataSync()[0];
      // 将新生成的词ID添加到序列末尾
      currentSeq.push(predIdx);
      // 清理张量，避免内存泄漏
      tf.dispose([inputTensor, pred]);
    }
    return currentSeq; // 返回完整生成的序列（词ID数组）
  }
}

// 5. 使用示例
async function runSimpleGPT() {
  // 模型参数（简化版：远小于 GPT-3 的 1750 亿参数）
  const vocabSize = 10000; // 简化词汇表大小
  const numLayers = 6;     // 解码器层数
  const dModel = 512;      // 模型维度
  const numHeads = 8;      // 注意力头数
  const dff = 2048;        // 前馈网络维度
  const maxSeqLen = 128;   // 最大序列长度

  // 创建简化版 GPT 模型
  const gpt = new SimpleGPT(vocabSize, numLayers, dModel, numHeads, dff, maxSeqLen);

  // 示例输入：假设 [101, 2023, 2003] 对应词ID（如“我 喜欢 编程”）
  const inputSeq = [101, 2023, 2003];
  // 生成最多 10 个词
  const generatedSeq = await gpt.generate(inputSeq, 10);

  console.log('输入序列（词ID）:', inputSeq);
  console.log('生成序列（词ID）:', generatedSeq);
  // 实际应用中需将词ID转换为对应词汇（需配合词汇表）

  // 清理模型变量
  tf.disposeVariables();
}

// 运行示例
runSimpleGPT();