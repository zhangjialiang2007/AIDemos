import * as tf from '@tensorflow/tfjs';
import { Utils } from './utils.js';
import { EncoderLayer } from './encoder_layer.js';
import { DecoderLayer } from './decoder_layer.js';

class Transformer {
  constructor(params) {
    const { numLayers, dModel, numHeads, dff, inputVocabSize, targetVocabSize } = params;
    
    this.dModel = dModel;
    this.numLayers = numLayers;
    
    // 编码器堆叠
    this.encLayers = Array.from({ length: numLayers }, () => 
      new EncoderLayer(dModel, numHeads, dff));
    
    // 解码器堆叠
    this.decLayers = Array.from({ length: numLayers }, () => 
      new DecoderLayer(dModel, numHeads, dff));
    
    // 嵌入层
    this.embedding = tf.variable(tf.randomNormal([inputVocabSize, dModel]));
    this.targetEmbedding = tf.variable(tf.randomNormal([targetVocabSize, dModel]));
    
    // 输出层
    this.finalLayer = tf.variable(tf.randomNormal([dModel, targetVocabSize]));
  }

  call(inputs, targets, encPaddingMask, lookAheadMask, decPaddingMask) {
    return tf.tidy(() => {
      // 编码器部分
      let encOutput = tf.matMul(inputs, this.embedding); // 输入嵌入
      encOutput = encOutput.add(Utils.positionalEncoding(encOutput.shape[1], this.dModel)); // 加位置编码
      
      // 编码器层堆叠
      for (let i = 0; i < this.numLayers; i++) {
        encOutput = this.encLayers[i].call(encOutput, encPaddingMask);
      }
      
      // 解码器部分
      let decOutput = tf.matMul(targets, this.targetEmbedding); // 目标嵌入
      decOutput = decOutput.add(Utils.positionalEncoding(decOutput.shape[1], this.dModel)); // 加位置编码
      
      // 解码器层堆叠
      let attentionWeights = {};
      for (let i = 0; i < this.numLayers; i++) {
        const { output, attentionWeights: layerAttnWeights } = 
          this.decLayers[i].call(decOutput, encOutput, lookAheadMask, decPaddingMask);
        
        decOutput = output;
        attentionWeights = { ...attentionWeights, ...layerAttnWeights };
      }
      
      // 最终输出
      const finalOutput = tf.matMul(decOutput, this.finalLayer);
      return { finalOutput, attentionWeights };
    });
  }

  // 计算损失函数
  computeLoss(predictions, labels, labelMask) {
    return tf.tidy(() => {
      // 1. 对预测结果应用softmax，得到概率分布
      const predProbs = tf.softmax(predictions);
      
      // 2. 计算交叉熵损失（分类任务常用，适合词预测）
      // labels需为整数索引（非独热编码），形状为[batch, seq_len]
      const crossEntropy = tf.losses.sparseCategoricalCrossentropy(
        labels, 
        predProbs, 
        { fromLogits: false } // 已应用softmax，故设为false
      );
      
      // 3. 应用掩码（忽略填充符的损失，避免无效值影响训练）
      // labelMask为0/1矩阵，1表示有效位置，0表示填充位置
      if (labelMask != null) {
        const maskedLoss = crossEntropy.mul(labelMask);
        // 求平均损失（除以有效元素数量）
        return maskedLoss.sum().div(labelMask.sum());
      } else {
        // 无掩码时直接求平均
        return crossEntropy.mean();
      }
    });
  }

  // 单步训练（前向传播+反向传播+参数更新）
  async trainStep(inputs, targets, labels, encPaddingMask, lookAheadMask, decPaddingMask, labelMask) {
    // 使用tf.tidy和gradient计算梯度
    return tf.tidy(() => {
      // 1. 前向传播：得到预测结果
      const { finalOutput } = this.call(inputs, targets, encPaddingMask, lookAheadMask, decPaddingMask);
      
      // 2. 计算损失
      const loss = this.computeLoss(finalOutput, labels, labelMask);
      
      // 3. 反向传播：计算损失对所有可训练参数的梯度
      const grads = tf.grad(() => this.computeLoss(finalOutput, labels, labelMask))();
      
      // 4. 优化器更新参数（应用梯度下降）
      this.optimizer.applyGradients(grads.map((g, i) => ({ grad: g, variable: this.trainableVariables[i] })));
      
      return loss;
    });
  }

// 训练---todo
train() {
  const params = {
    numLayers: 2,
    dModel: 64,
    numHeads: 2,
    dff: 128,
    inputVocabSize: 8500,
    targetVocabSize: 8000,
    learningRate: 0.001
  };
  
  const transformer = new Transformer(params);
  
  // 模拟训练10个批次
  for (let epoch = 0; epoch < 10; epoch++) {
    // 这里用随机数据模拟，实际应替换为真实数据
    const inputs = tf.randomUniform([2, 10, 8500]); // 独热编码输入
    const targets = tf.randomUniform([2, 8, 8000]); // 独热编码目标（解码器输入）
    const labels = tf.randomUniform([2, 8], 0, 8000, 'int32'); // 整数标签（用于损失计算）
    const labelMask = tf.ones([2, 8]); // 假设所有位置都是有效词（无填充）
    
    // 单步训练
    const loss = await transformer.trainStep(
      inputs, targets, labels, 
      null, null, null, // 简化示例，忽略掩码
      labelMask
    );
    
    console.log(`Epoch ${epoch}, Loss: ${await loss.data()}`);
    loss.dispose(); // 手动释放内存
  }

}

export {Transformer}
