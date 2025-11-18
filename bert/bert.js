import * as tf from '@tensorflow/tfjs';

// 位置编码实现
function positionalEncoding(maxPosition, dModel) {
  return tf.tidy(() => {
    const angleRates = tf.div(1, tf.pow(10000, tf.div(tf.range(0, dModel, 2), dModel)));
    const angleRad = tf.matMul(
      tf.range(0, maxPosition).reshape([-1, 1]),
      angleRates.reshape([1, -1])
    );
    
    const sin = tf.sin(angleRad);
    const cos = tf.cos(angleRad);
    
    // 交错排列sin和cos
    const posEncoding = tf.concat(
      [sin, cos], 1
    ).reshape([1, maxPosition, dModel]);
    
    return posEncoding;
  });
}

// 多头注意力机制
class MultiHeadAttention {
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
  
  splitHeads(x, batchSize) {
    return tf.tidy(() => {
      const xShape = x.reshape([batchSize, -1, this.numHeads, this.depth]);
      return tf.transpose(xShape, [0, 2, 1, 3]);
    });
  }
  
  scaledDotProductAttention(q, k, v, mask) {
    return tf.tidy(() => {
      const matmulQk = tf.matMul(q, k, false, true);
      const dk = tf.cast(tf.shape(k)[-1], 'float32');
      const scaledAttentionLogits = tf.div(matmulQk, tf.sqrt(dk));
      
      // 应用掩码
      if (mask != null) {
        scaledAttentionLogits.assign(
          tf.where(mask, tf.scalar(-1e9), scaledAttentionLogits)
        );
      }
      
      const attentionWeights = tf.softmax(scaledAttentionLogits);
      const output = tf.matMul(attentionWeights, v);
      
      return { output, attentionWeights };
    });
  }
  
  apply(x, mask) {
    return tf.tidy(() => {
      const batchSize = tf.shape(x)[0];
      
      const q = tf.matMul(x, this.wq);
      const k = tf.matMul(x, this.wk);
      const v = tf.matMul(x, this.wv);
      
      const qSplit = this.splitHeads(q, batchSize);
      const kSplit = this.splitHeads(k, batchSize);
      const vSplit = this.splitHeads(v, batchSize);
      
      const { output: scaledAttention, attentionWeights } = 
        this.scaledDotProductAttention(qSplit, kSplit, vSplit, mask);
      
      const scaledAttentionTranspose = tf.transpose(scaledAttention, [0, 2, 1, 3]);
      const concatAttention = scaledAttentionTranspose.reshape([batchSize, -1, this.dModel]);
      
      const output = tf.matMul(concatAttention, this.dense);
      return { output, attentionWeights };
    });
  }
}

// 前馈网络
function pointwiseFeedForwardNetwork(dModel, dff) {
  return tf.sequential({
    layers: [
      tf.layers.dense({ units: dff, activation: 'relu' }),
      tf.layers.dense({ units: dModel })
    ]
  });
}

// 编码器层
class EncoderLayer {
  constructor(dModel, numHeads, dff, rate = 0.1) {
    this.mha = new MultiHeadAttention(dModel, numHeads);
    this.ffn = pointwiseFeedForwardNetwork(dModel, dff);
    
    this.layernorm1 = tf.layers.layerNormalization({ epsilon: 1e-6 });
    this.layernorm2 = tf.layers.layerNormalization({ epsilon: 1e-6 });
    
    this.dropout1 = tf.layers.dropout({ rate });
    this.dropout2 = tf.layers.dropout({ rate });
  }
  
  apply(x, training, mask) {
    return tf.tidy(() => {
      const { output: attnOutput } = this.mha.apply(x, mask);
      const attnOutputWithDropout = this.dropout1.apply(attnOutput, { training });
      const out1 = this.layernorm1.apply(tf.add(x, attnOutputWithDropout));
      
      const ffnOutput = this.ffn.predict(out1);
      const ffnOutputWithDropout = this.dropout2.apply(ffnOutput, { training });
      const out2 = this.layernorm2.apply(tf.add(out1, ffnOutputWithDropout));
      
      return out2;
    });
  }
}

// BERT模型
class BERT {
  constructor(vocabSize, numLayers, dModel, numHeads, dff, maximumPositionEncoding) {
    this.numLayers = numLayers;
    this.encLayers = Array.from({ length: numLayers }, 
      () => new EncoderLayer(dModel, numHeads, dff));
    
    this.embedding = tf.layers.embedding({
      inputDim: vocabSize,
      outputDim: dModel
    });
    
    this.posEncoding = positionalEncoding(maximumPositionEncoding, dModel);
    this.dropout = tf.layers.dropout({ rate: 0.1 });
    
    // 用于分类任务的输出层
    this.finalLayer = tf.layers.dense({ units: 2 }); // 假设是二分类任务
  }
  
  apply(inputs, training) {
    return tf.tidy(() => {
      const [x, mask] = inputs;
      const seqLen = tf.shape(x)[1];
      
      // 词嵌入 + 位置编码
      let xEmbed = this.embedding.apply(x);
      xEmbed = tf.multiply(xEmbed, tf.sqrt(tf.scalar(this.dModel)));
      xEmbed = tf.add(xEmbed, this.posEncoding.slice([0, 0, 0], [-1, seqLen, -1]));
      
      let xOut = this.dropout.apply(xEmbed, { training });
      
      // 通过多个编码器层
      for (let i = 0; i < this.numLayers; i++) {
        xOut = this.encLayers[i].apply(xOut, training, mask);
      }
      
      // 对于分类任务，使用[CLS]标记的输出
      const clsOutput = xOut.slice([0, 0, 0], [-1, 1, -1]).squeeze([1]);
      const finalOutput = this.finalLayer.apply(clsOutput);
      
      return finalOutput;
    });
  }
}

// 使用示例
async function run() {
  // 模型参数
  const vocabSize = 30522; // BERT基础模型的词汇表大小
  const numLayers = 12;    // BERT基础模型的层数
  const dModel = 768;      // 模型维度
  const numHeads = 12;     // 注意力头数
  const dff = 3072;        // 前馈网络维度
  const maxPosition = 512; // 最大序列长度
  
  // 创建BERT模型
  const bert = new BERT(vocabSize, numLayers, dModel, numHeads, dff, maxPosition);
  
  // 创建示例输入
  const input = tf.tensor2d([[101, 2023, 2003, 1037, 1996, 3052, 102]], 'int32'); // 示例输入ID
  const mask = tf.tensor2d([[1, 1, 1, 1, 1, 1, 1]], 'float32'); // 注意力掩码
  
  // 前向传播
  const output = bert.apply([input, mask], false);
  console.log('模型输出形状:', output.shape);
  
  // 清理内存
  tf.disposeVariables();
}

// 运行示例
run();