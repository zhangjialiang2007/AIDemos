/**
 * 计算余弦相似度损失函数 (1 - 余弦相似度)
 * @param {Array<number>} vec1 第一个向量（如预测向量）
 * @param {Array<number>} vec2 第二个向量（如目标向量）
 * @param {number} epsilon 防止除数为0的极小值
 * @returns {number} 损失值（范围 [0, 2]）
 * @example 
 * const pred = [1, 2, 3, 4]; // 预测向量
 * const target = [2, 4, 6, 8]; // 目标向量（与预测向量方向完全一致）
 * console.log(cosineSimilarityLoss(pred, target)); // 接近0（因为方向相同）
 * const opposite = [-1, -2, -3, -4]; // 与预测向量方向完全相反
 * console.log(cosineSimilarityLoss(pred, opposite)); // 接近2（因为方向相反）
 */
function cosineSimilarityLoss(vec1, vec2, epsilon = 1e-8) {
    // 检查向量长度是否一致
    if (vec1.length !== vec2.length) {
      throw new Error('两个向量的长度必须一致');
    }
  
    // 计算点积
    let dotProduct = 0;
    // 计算vec1的模长平方
    let norm1Squared = 0;
    // 计算vec2的模长平方
    let norm2Squared = 0;
  
    for (let i = 0; i < vec1.length; i++) {
      dotProduct += vec1[i] * vec2[i];
      norm1Squared += vec1[i]**2;
      norm2Squared += vec2[i]** 2;
    }
  
    // 计算模长（开平方），加epsilon防止除以0
    const norm1 = Math.sqrt(norm1Squared) + epsilon;
    const norm2 = Math.sqrt(norm2Squared) + epsilon;
  
    // 计算余弦相似度
    const cosineSimilarity = dotProduct / (norm1 * norm2);
  
    // 损失函数：1 - 余弦相似度（范围 [0, 2]）
    return 1 - cosineSimilarity;
  }
  
  export { cosineSimilarityLoss}