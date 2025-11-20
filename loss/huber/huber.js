
/**
 * 计算Huber损失（单个样本）
 * @param {number} yTrue - 真实值
 * @param {number} yPred - 预测值
 * @param {number} delta - 阈值（误差超过此值时用线性损失，默认1.0）
 * @returns {number} 单个样本的Huber损失
 * @example
 * 单个样本Huber损失
 * console.log("真实值=5，预测值=4.5（误差0.5 ≤ 1）→", huberLossSingle(5, 4.5).toFixed(4)); 
 * 输出：0.1250
 * console.log("真实值=5，预测值=3（误差2 > 1）→", huberLossSingle(5, 3).toFixed(4)); 
 * 输出：1.5000
 */
function huberLossSingle(yTrue, yPred, delta = 1.0) {
    const error = yTrue - yPred;
    const absError = Math.abs(error);
    
    if (absError <= delta) {
        // 误差较小时：使用MSE的一半（保持梯度连续性）
        return 0.5 * Math.pow(error, 2);
    } else {
        // 误差较大时：使用线性损失（delta*(|error| - 0.5*delta)）
        return delta * (absError - 0.5 * delta);
    }
}

/**
 * 计算批量样本的平均Huber损失
 * @param {Array<number>} yTrue - 真实值数组
 * @param {Array<number>} yPred - 预测值数组
 * @param {number} delta - 阈值（默认1.0）
 * @returns {number} 平均Huber损失
 * @example
 * 批量样本平均Huber损失
 * const yTrue = [2, 4, 6, 8, 10];
 * const yPred = [1.8, 3.9, 6.2, 7.7, 15]; // 最后一个样本误差较大（5）
 * console.log(`真实值：${yTrue}`);
 * console.log(`预测值：${yPred}`);
 * console.log("平均Huber损失：", huberLossArray(yTrue, yPred).toFixed(4)); 
 * 输出：0.9180
 */
function huberLossArray(yTrue, yPred, delta = 1.0) {
    if (!Array.isArray(yTrue) || !Array.isArray(yPred)) {
        throw new Error("输入必须为数组");
    }
    if (yTrue.length !== yPred.length) {
        throw new Error("真实值与预测值数组长度必须一致");
    }
    
    let sum = 0;
    for (let i = 0; i < yTrue.length; i++) {
        sum += huberLossSingle(yTrue[i], yPred[i], delta);
    }
    return sum / yTrue.length; // 求平均
}

// huber损失
function huberLoss(yTrue, yPred, delta = 1.0){
    if(Array.isArray(yTrue)){
        return huberLossArray(yTrue, yPred, delta)
    }
    return huberLossSingle(yTrue, yPred, delta)
}

export { huberLoss }