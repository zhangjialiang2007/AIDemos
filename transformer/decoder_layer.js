import * as tf from '@tensorflow/tfjs';
import { Utils } from './utils.js';
import { MultiHeadAttention } from './multi_head_attention.js'

class DecoderLayer {
  constructor(dModel, numHeads, dff) {
    this.mha1 = new MultiHeadAttention(dModel, numHeads); // 掩蔽自注意力
    this.mha2 = new MultiHeadAttention(dModel, numHeads); // 编码器-解码器注意力
    this.ffn = Utils.pointwiseFeedForwardNetwork(dModel, dff);
    
    this.layernorm1 = tf.layers.layerNormalization({ epsilon: 1e-6 });
    this.layernorm2 = tf.layers.layerNormalization({ epsilon: 1e-6 });
    this.layernorm3 = tf.layers.layerNormalization({ epsilon: 1e-6 });
  }

  call(x, encOutput, lookAheadMask, paddingMask) {
    return tf.tidy(() => {
      // 掩蔽自注意力
      const { output: attn1, attentionWeights: attnWeightsBlock1 } = 
        this.mha1.call(x, x, x, lookAheadMask);
      const out1 = this.layernorm1.apply(tf.add(x, attn1));
      
      // 编码器-解码器注意力
      const { output: attn2, attentionWeights: attnWeightsBlock2 } = 
        this.mha2.call(out1, encOutput, encOutput, paddingMask);
      const out2 = this.layernorm2.apply(tf.add(out1, attn2));
      
      // 前馈网络
      const ffnOutput = this.ffn.call(out2);
      const out3 = this.layernorm3.apply(tf.add(out2, ffnOutput));
      
      return { 
        output: out3,
        attentionWeights: { attnWeightsBlock1, attnWeightsBlock2 }
      };
    });
  }
}

export {DecoderLayer}