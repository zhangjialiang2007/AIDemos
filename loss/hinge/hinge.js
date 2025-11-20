/**
 * Hinge 损失函数（用于二分类）
 * @param {number[]} yTrue - 真实标签数组，元素应为 +1 或 -1
 * @param {number[]} yPred - 模型预测得分数组（原始输出，非概率）
 * @returns {number} 平均 hinge 损失
 * @example
 * const yTrue = [1, -1, 1, -1]; // 真实标签（+1 或 -1）
 * const yPredGood = [0.8, -0.9, 0.7, -0.6]; // 预测得分与真实标签一致（符号相同）
 * const yPredBad = [-0.3, 0.2, -0.1, 0.4];  // 预测得分与真实标签相反（符号不同）
 * console.log(hingeLoss(yTrue, yPredGood)); // 较小的损失（约 0.15）
 * console.log(hingeLoss(yTrue, yPredBad));  // 较大的损失（约 1.15）
 */
function hingeLoss(yTrue, yPred) {
    // 检查输入长度是否一致
    if (yTrue.length !== yPred.length) {
        throw new Error('真实标签与预测得分的长度必须一致');
    }

    let totalLoss = 0;
    for (let i = 0; i < yTrue.length; i++) {
        const y = yTrue[i];
        const yHat = yPred[i];

        // 检查真实标签是否为 ±1（二分类约束）
        if (y !== 1 && y !== -1) {
            throw new Error('真实标签必须为 +1 或 -1');
        }

        // 计算单个样本的 hinge 损失：max(0, 1 - y*yHat)
        const loss = Math.max(0, 1 - y * yHat);
        totalLoss += loss;
    }

    // 返回平均损失
    return totalLoss / yTrue.length;
}

/**
 * 平方 Hinge 损失函数（用于二分类）
 * @param {number[]} yTrue - 真实标签数组，元素应为 +1 或 -1
 * @param {number[]} yPred - 模型预测得分数组（原始输出，非概率）
 * @returns {number} 平均平方 hinge 损失
 * @example
 * const yTrue = [1, -1, 1, -1]; // 真实标签（+1 或 -1）
 * const yPredGood = [0.8, -0.9, 0.7, -0.6]; // 预测得分与真实标签一致（符号相同）
 * const yPredBad = [-0.3, 0.2, -0.1, 0.4];  // 预测得分与真实标签相反（符号不同）
 * console.log(squaredHingeLoss(yTrue, yPredGood)); // 更小的损失（约 0.02）
 * console.log(squaredHingeLoss(yTrue, yPredBad));  // 更大的损失（约 1.32）
 */
function squaredHingeLoss(yTrue, yPred) {
    if (yTrue.length !== yPred.length) {
        throw new Error('真实标签与预测得分的长度必须一致');
    }

    let totalLoss = 0;
    for (let i = 0; i < yTrue.length; i++) {
        const y = yTrue[i];
        const yHat = yPred[i];

        if (y !== 1 && y !== -1) {
            throw new Error('真实标签必须为 +1 或 -1');
        }

        // 计算单个样本的平方 hinge 损失：max(0, 1 - y*yHat)^2
        const hinge = Math.max(0, 1 - y * yHat);
        const loss = hinge * hinge;
        totalLoss += loss;
    }

    return totalLoss / yTrue.length;
}

export { hingeLoss, squaredHingeLoss}