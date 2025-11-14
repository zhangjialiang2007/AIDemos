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
}

export {Transformer}
