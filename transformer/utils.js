import * as tf from '@tensorflow/tfjs';

class Utils {
// 位置编码：注入序列位置信息
  static positionalEncoding(length, dModel) {
    return tf.tidy(() => {
      const angleRates = tf.div(1, tf.pow(10000, tf.div(tf.range(0, dModel, 2), dModel)));
      const angleRad = tf.matMul(tf.range(0, length).reshape([-1, 1]), angleRates.reshape([1, -1]));
      
      const sin = tf.sin(angleRad);
      const cos = tf.cos(angleRad);
      
      // 交错拼接sin和cos
      const posEmbedding = tf.stack([sin, cos], 2).reshape([length, dModel]);
      return posEmbedding;
    });
  }

  // 缩放点积注意力
  static scaledDotProductAttention(q, k, v, mask = null) {
    return tf.tidy(() => {
      const matMulQK = tf.matMul(q, k, false, true); // Q·K^T
      const dk = tf.sqrt(tf.scalar(k.shape[k.shape.length - 1]));
      const scaledAttentionLogits = tf.div(matMulQK, dk);

      // 应用掩码（用于解码器的掩蔽注意力）
      if (mask != null) {
        scaledAttentionLogits.assign(scaledAttentionLogits.add(mask.mul(-1e9)));
      }

      const attentionWeights = tf.softmax(scaledAttentionLogits);
      const output = tf.matMul(attentionWeights, v);
      return { output, attentionWeights };
    });
  }

  // 前馈网络
  static pointwiseFeedForwardNetwork(dModel, dff) {
    return {
      dense1: tf.variable(tf.randomNormal([dModel, dff])),
      dense2: tf.variable(tf.randomNormal([dff, dModel])),
      call: (x) => tf.tidy(() => {
        x = tf.matMul(x, this.dense1);
        x = tf.relu(x);
        x = tf.matMul(x, this.dense2);
        return x;
      })
    };
  }
}

export { Utils }

