/**
 * 计算均方误差（MSE）
 * @param {Array<number> | number} yTrue - 真实值（单个数值或数组）
 * @param {Array<number> | number} yPred - 预测值（单个数值或数组，需与真实值结构一致）
 * @returns {number} MSE值
 * @example
 * 单个样本MSE：
 * const yTrueSingle = 5;    // 真实值
 * const yPredSingle = 4.5;  // 预测值
 * console.log(`真实值=${yTrueSingle}, 预测值=${yPredSingle} → MSE=${meanSquaredError(yTrueSingle, yPredSingle)}`); 
 * 输出：0.25
 * 
 * 批量样本MSE：
 * const yTrueBatch = [2, 4, 6, 8];  // 真实值数组
 * const yPredBatch = [1.8, 3.9, 6.2, 7.7];  // 预测值数组
 * console.log(`真实值=${yTrueBatch}, 预测值=${yPredBatch} → MSE=${meanSquaredError(yTrueBatch, yPredBatch).toFixed(4)}`); 
 * 输出：0.0450
 */
function meanSquaredError(yTrue, yPred) {
    // 处理单个样本的情况（非数组）
    if (typeof yTrue === 'number' && typeof yPred === 'number') {
        return Math.pow(yTrue - yPred, 2);
    }

    // 处理批量样本（数组）
    if (!Array.isArray(yTrue) || !Array.isArray(yPred)) {
        throw new Error("输入必须同为数组或单个数值");
    }
    if (yTrue.length !== yPred.length) {
        throw new Error("真实值与预测值数组长度必须一致");
    }

    let sum = 0;
    for (let i = 0; i < yTrue.length; i++) {
        const error = yTrue[i] - yPred[i];
        sum += Math.pow(error, 2); // 累加平方误差
    }
    return sum / yTrue.length; // 除以样本数得均值
}

/**
 * 计算均方根误差（RMSE）
 * @param {Array<number> | number} yTrue - 真实值（单个数值或数组）
 * @param {Array<number> | number} yPred - 预测值（单个数值或数组，需与真实值结构一致）
 * @returns {number} RMSE值
 * @example
 * 单个样本RMSE：
 * const yTrueSingle = 5;    // 真实值
 * const yPredSingle = 4.5;  // 预测值
 * console.log(`真实值=${yTrueSingle}, 预测值=${yPredSingle} → RMSE=${rootMeanSquaredError(yTrueSingle, yPredSingle)}`); 
 * 输出：0.5
 *
 * 批量样本RMSE：
 * const yTrueBatch = [2, 4, 6, 8];  // 真实值数组
 * const yPredBatch = [1.8, 3.9, 6.2, 7.7];  // 预测值数组
 * console.log(`真实值=${yTrueBatch}, 预测值=${yPredBatch} → RMSE=${rootMeanSquaredError(yTrueBatch, yPredBatch).toFixed(4)}`); 
 * 输出：0.2121
 */
function rootMeanSquaredError(yTrue, yPred) {
    // 先计算MSE，再取平方根
    const mse = meanSquaredError(yTrue, yPred);
    return Math.sqrt(mse);
}

/**
 * 计算平均绝对误差（MAE）
 * @param {Array<number> | number} yTrue - 真实值（单个数值或数组）
 * @param {Array<number> | number} yPred - 预测值（单个数值或数组，需与真实值结构一致）
 * @returns {number} MAE值
 * @example
 * "单个样本MAE：
 * const yTrueSingle = 5;    // 真实值
 * const yPredSingle = 4.5;  // 预测值
 * console.log(`真实值=${yTrueSingle}, 预测值=${yPredSingle} → MAE=${meanAbsoluteError(yTrueSingle, yPredSingle)}`); 
 * 输出：0.5
 * 
 * 批量样本MAE：
 * const yTrueBatch = [2, 4, 6, 8];  // 真实值数组
 * const yPredBatch = [1.8, 3.9, 6.2, 7.7];  // 预测值数组
 * console.log(`真实值=${yTrueBatch}, 预测值=${yPredBatch} → MAE=${meanAbsoluteError(yTrueBatch, yPredBatch).toFixed(4)}`); 
 * 输出：0.2000
 */
function meanAbsoluteError(yTrue, yPred) {
    // 处理单个样本的情况（非数组）
    if (typeof yTrue === 'number' && typeof yPred === 'number') {
        return Math.abs(yTrue - yPred);
    }

    // 处理批量样本（数组）
    if (!Array.isArray(yTrue) || !Array.isArray(yPred)) {
        throw new Error("输入必须同为数组或单个数值");
    }
    if (yTrue.length !== yPred.length) {
        throw new Error("真实值与预测值数组长度必须一致");
    }

    let sum = 0;
    for (let i = 0; i < yTrue.length; i++) {
        sum += Math.abs(yTrue[i] - yPred[i]); // 累加绝对误差
    }
    return sum / yTrue.length; // 除以样本数得均值
}

export {meanSquaredError as MSE, rootMeanSquaredError as RMSE, meanAbsoluteError as MAE}