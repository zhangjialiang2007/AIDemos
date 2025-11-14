import * as tf from '@tensorflow/tfjs';
import { Utils } from './utils.js';
import { MultiHeadAttention } from './multi_head_attention.js'

class EncoderLayer {
  constructor(dModel, numHeads, dff) {
    this.mha = new MultiHeadAttention(dModel, numHeads);
    this.ffn = Utils.pointwiseFeedForwardNetwork(dModel, dff);
    
    this.layernorm1 = tf.layers.layerNormalization({ epsilon: 1e-6 });
    this.layernorm2 = tf.layers.layerNormalization({ epsilon: 1e-6 });
  }

  call(x, mask) {
    return tf.tidy(() => {
      // 多头注意力 + 残差连接 + 层归一化
      const { output: attnOutput } = this.mha.call(x, x, x, mask);
      const out1 = this.layernorm1.apply(tf.add(x, attnOutput));
      
      // 前馈网络 + 残差连接 + 层归一化
      const ffnOutput = this.ffn.call(out1);
      const out2 = this.layernorm2.apply(tf.add(out1, ffnOutput));
      
      return out2;
    });
  }
}

export {EncoderLayer}