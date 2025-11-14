import * as tf from '@tensorflow/tfjs';
import {Transformer} from './transformer.js'
// 使用示例
async function demo() {
  // 超参数
  const params = {
    numLayers: 2,    // 编码器/解码器层数
    dModel: 64,      // 模型维度
    numHeads: 2,     // 注意力头数
    dff: 128,        // 前馈网络隐藏层维度
    inputVocabSize: 8500,  // 输入词汇表大小
    targetVocabSize: 8000  // 目标词汇表大小
  };

  // 创建Transformer实例
  const transformer = new Transformer(params);
  
  // 随机生成输入数据（batchSize=2, seqLength=10）
  const input = tf.randomUniform([2, 10], 0, params.inputVocabSize, 'int32');
  const target = tf.randomUniform([2, 8], 0, params.targetVocabSize, 'int32');
  
  // 转换为独热编码（简化处理）
  const inputOneHot = tf.oneHot(input, params.inputVocabSize);
  const targetOneHot = tf.oneHot(target, params.targetVocabSize);
  
  // 前向传播
  const { finalOutput } = transformer.call(
    inputOneHot, 
    targetOneHot, 
    null, null, null  // 简化示例，忽略掩码
  );
  
  console.log('输出形状:', finalOutput.shape); // [2, 8, 8000]
}
export {demo}