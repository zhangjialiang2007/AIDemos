/**
 * 计算单个样本的0-1损失
 * @param {number | string} yTrue - 真实标签（可以是数值或类别字符串）
 * @param {number | string} yPred - 预测标签（需与真实标签类型一致）
 * @returns {number} 0（预测正确）或1（预测错误）
 * @example
 * 单个样本0-1损失：
 * console.log("真实标签=1，预测标签=1 →", zeroOneLossSingle(1, 1)); // 输出：0（正确）
 * console.log("真实标签=0，预测标签=1 →", zeroOneLossSingle(0, 1)); // 输出：1（错误）
 * console.log("真实标签='猫'，预测标签='狗' →", zeroOneLossSingle('猫', '狗')); // 输出：1（错误）
 */
function zeroOneLossSingle(yTrue, yPred) {
    // 直接比较真实标签与预测标签是否一致
    return yTrue === yPred ? 0 : 1;
}

/**
 * 计算批量样本的平均0-1损失（即错误率）
 * @param {Array} yTrue - 真实标签数组
 * @param {Array} yPred - 预测标签数组
 * @returns {number} 平均0-1损失（错误样本数/总样本数）
 * 批量样本平均0-1损失
 * const yTrue = [1, 0, 1, 0, 1]; // 二分类真实标签
 * const yPred = [1, 0, 0, 0, 1]; // 二分类预测标签
 * console.log(`真实标签：${yTrue}`);
 * console.log(`预测标签：${yPred}`);
 * console.log("平均0-1损失：", zeroOneLossArray(yTrue, yPred)); 
 * 输出：0.2
 */
function zeroOneLossArray(yTrue, yPred) {
    if (!Array.isArray(yTrue) || !Array.isArray(yPred)) {
        throw new Error("输入必须均为数组");
    }
    if (yTrue.length !== yPred.length) {
        throw new Error("真实标签与预测标签数组长度必须一致");
    }

    let errorCount = 0;
    for (let i = 0; i < yTrue.length; i++) {
        errorCount += zeroOneLossSingle(yTrue[i], yPred[i]);
    }
    // 平均损失 = 错误样本数 / 总样本数
    return errorCount / yTrue.length;
}

// 01损失函数
function zeroOneLoss(yTrue, yPred){
    if(Array.isArray(yTrue)){
        return zeroOneLossArray(yTrue, yPred)
    }
    return zeroOneLossSingle(yTrue, yPred)
}

export {zeroOneLoss}


