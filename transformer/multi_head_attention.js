
import * as tf from '@tensorflow/tfjs';
import { Utils } from './utils.js';

class MultiHeadAttention {
  constructor(dModel, numHeads) {
    this.dModel = dModel;
    this.numHeads = numHeads;
    this.depth = dModel / numHeads;

    // 权重矩阵
    this.wq = tf.variable(tf.randomNormal([dModel, dModel]));
    this.wk = tf.variable(tf.randomNormal([dModel, dModel]));
    this.wv = tf.variable(tf.randomNormal([dModel, dModel]));
    this.dense = tf.variable(tf.randomNormal([dModel, dModel]));
  }

  splitHeads(x, batchSize) {
    return tf.tidy(() => {
      const xReshaped = x.reshape([batchSize, -1, this.numHeads, this.depth]);
      return xReshaped.transpose([0, 2, 1, 3]); // [batch, heads, seq_len, depth]
    });
  }

  call(q, k, v, mask) {
    return tf.tidy(() => {
      const batchSize = q.shape[0];

      // 线性投影并分多头
      const q1 = tf.matMul(q, this.wq);
      const k1 = tf.matMul(k, this.wk);
      const v1 = tf.matMul(v, this.wv);

      const qSplit = this.splitHeads(q1, batchSize);
      const kSplit = this.splitHeads(k1, batchSize);
      const vSplit = this.splitHeads(v1, batchSize);

      // 计算注意力
      const { output: scaledAttention, attentionWeights } = 
        Utils.scaledDotProductAttention(qSplit, kSplit, vSplit, mask);

      // 合并多头
      const scaledAttentionTranspose = scaledAttention.transpose([0, 2, 1, 3]);
      const concatAttention = scaledAttentionTranspose.reshape([batchSize, -1, this.dModel]);
      
      // 最终线性投影
      const output = tf.matMul(concatAttention, this.dense);
      return { output, attentionWeights };
    });
  }
}
export {MultiHeadAttention}