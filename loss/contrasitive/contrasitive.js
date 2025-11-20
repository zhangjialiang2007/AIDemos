/**
 * 计算对比损失（Contrastive Loss）
 * @param {Array<number>} x1 第一个样本向量
 * @param {Array<number>} x2 第二个样本向量
 * @param {0 | 1} y 标签（0：相似样本，1：不相似样本）
 * @param {number} margin 阈值（不相似样本的最小距离阈值，通常设为1.0）
 * @returns {number} 损失值
 * @example
 * 相似样本（y=0）：距离越小，损失越小
 * const similar1 = [1, 2, 3];
 * const similar2 = [1.1, 2.1, 3.1];
 * console.log(contrastiveLoss(similar1, similar2, 0)); // 输出较小的损失值
 * 不相似样本（y=1）：距离小于margin时，损失随距离增大而减小
 * const dissimilar1 = [1, 2, 3];
 * const dissimilar2 = [4, 5, 6];
 * console.log(contrastiveLoss(dissimilar1, dissimilar2, 1)); // 若距离>margin，损失为0
 * 不相似样本但距离较近（损失较大）
 * const closeDissimilar1 = [1, 2, 3];
 * const closeDissimilar2 = [1.5, 2.5, 3.5];
 * console.log(contrastiveLoss(closeDissimilar1, closeDissimilar2, 1)); // 输出较大的损失值
 */
function contrastiveLoss(x1, x2, y, margin = 1.0) {
    // 检查向量长度是否一致
    if (x1.length !== x2.length) {
      throw new Error('两个向量的长度必须相同');
    }
  
    // 检查标签合法性
    if (y !== 0 && y !== 1) {
      throw new Error('标签y必须为0或1（0表示相似，1表示不相似）');
    }
  
    // 计算欧氏距离的平方（避免开方运算，提高效率）
    let squaredDistance = 0;
    for (let i = 0; i < x1.length; i++) {
      const diff = x1[i] - x2[i];
      squaredDistance += diff * diff;
    }
  
    // 计算对比损失
    if (y === 0) {
      // 相似样本：损失与距离平方成正比
      return 0.5 * squaredDistance;
    } else {
      // 不相似样本：距离小于margin时才有损失，大于等于margin时损失为0
      const diff = margin - Math.sqrt(squaredDistance);
      const positiveDiff = Math.max(diff, 0); // 确保非负
      return 0.5 * positiveDiff * positiveDiff;
    }
  }

  export { contrastiveLoss }