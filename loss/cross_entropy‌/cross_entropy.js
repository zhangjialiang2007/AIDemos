
/**
 * 二分类交叉熵计算（自然对数）
 * @param {number} yTrue - 真实标签（0或1）
 * @param {number} yPred - 预测概率（0~1之间）
 * @returns {number} 交叉熵值
 * @example
 * console.log("二分类场景：");
 * console.log("y=1, p=0.8 →", binaryCrossEntropy(1, 0.8).toFixed(3)); // 输出：0.223
 * console.log("y=0, p=0.3 →", binaryCrossEntropy(0, 0.3).toFixed(3)); // 输出：0.357
 */
function binaryCrossEntropy(yTrue, yPred) {
    const epsilon = 1e-10;
    // 防止log(0)或log(1)导致的无穷大
    yPred = Math.max(epsilon, Math.min(1 - epsilon, yPred));
    return - (yTrue * Math.log(yPred) + (1 - yTrue) * Math.log(1 - yPred));
}

/**
 * 多分类交叉熵计算（自然对数）
 * @param {Array<number>} yTrue - 真实标签（one-hot向量，如[1,0,0]）
 * @param {Array<number>} yPred - 预测概率分布（如[0.7,0.2,0.1]）
 * @returns {number} 交叉熵值
 * @example
 * // 示例计算
 * console.log("\n多分类场景：");
 * const yTrue1 = [1, 0, 0]; // 真实为第一类
 * const yPred1 = [0.7, 0.2, 0.1];
 * console.log("P=[1,0,0], Q=[0.7,0.2,0.1] →", categoricalCrossEntropy(yTrue1, yPred1).toFixed(3)); // 输出：0.357
 *
 * const yTrue2 = [0, 1, 0]; // 真实为第二类
 * const yPred2 = [0.1, 0.8, 0.1];
 * console.log("P=[0,1,0], Q=[0.1,0.8,0.1] →", categoricalCrossEntropy(yTrue2, yPred2).toFixed(3)); // 输出：0.223
 */
function categoricalCrossEntropy(yTrue, yPred, epsilon = 1e-10) {
    if (yTrue.length !== yPred.length) {
        throw new Error('样本数量不一致');
    }

    let entropy = 0;
    for (let i = 0; i < yTrue.length; i++) {
        // 防止log(0)
        const pred = Math.max(epsilon, yPred[i]);
        entropy += yTrue[i] * Math.log(pred);
    }
    return -entropy;
}

/**
 * 稀疏多分类交叉熵损失（整数标签）
 * @param {number[]} yTrue - 整数真实标签，如[0, 1, 2]（表示类别0、1、2）
 * @param {number[][]} yPred - 模型预测概率（经过softmax），如[[0.8,0.1,0.1], ...]
 * @param {number} [epsilon=1e-10] - 防止log(0)的极小值
 * @returns {number} 平均稀疏交叉熵损失
 * @example
 * const sparseYTrue = [0, 1, 2];
 * const sparseYPred = [[0.8,0.1,0.1], [0.2,0.7,0.1], [0.1,0.2,0.7]];
 * console.log("稀疏交叉熵：", sparseCategoricalCrossEntropy(sparseYTrue, sparseYPred)); // 与多分类结果一致
 */
function sparseCategoricalCrossEntropy(yTrue, yPred, epsilon = 1e-10) {
    if (yTrue.length !== yPred.length) {
        throw new Error('样本数量不一致');
    }
    let totalLoss = 0;
    for (let i = 0; i < yTrue.length; i++) {
        const c = yTrue[i]; // 整数类别
        const p = yPred[i]; // 该样本的类别概率分布
        if (c < 0 || c >= p.length) {
            throw new Error('标签超出类别范围');
        }
        const prob = Math.max(epsilon, Math.min(1 - epsilon, p[c]));
        totalLoss += Math.log(prob);
    }
    return -totalLoss / yTrue.length;
}

/**
 * 连续分布交叉熵（采样近似，自然对数）
 * @param {number} pMean - 真实分布均值
 * @param {number} pStd - 真实分布标准差
 * @param {number} qMean - 预测分布均值
 * @param {number} qStd - 预测分布标准差
 * @param {number} nSamples - 采样数量（默认10000）
 * @returns {number} 交叉熵近似值
 * @example
 * // 示例计算：P~N(0,1)，Q~N(1,1)
 * console.log("\n连续型场景（近似）：");
 * console.log("P~N(0,1), Q~N(1,1) →", continuousCrossEntropyApprox(0, 1, 1, 1).toFixed(3)); // 输出≈1.418
 */
function continuousCrossEntropyApprox(pMean, pStd, qMean, qStd, nSamples = 10000) {
    // Box-Muller变换：已严格证明从两个独立的均匀分布随机数出发，通过特定变换可以得到两个独立的标准正态分布随机数。
    // 变换公式：sqr(-2*log(u)) * cos(2*PI *v)
    // 方法含义：根据平均数和标准差求随机数
    const normalRandom = (mean = 0, std = 1) => {
        let u = 0, v = 0;
        while (u === 0) u = Math.random(); // 避免log(0)
        while (v === 0) v = Math.random();
        const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
        return z * std + mean;
    }

    const epsilon = 1e-10;
    let sum = 0;
    for (let i = 0; i < nSamples; i++) {
        // 从真实分布P中采样
        const x = normalRandom(pMean, pStd);
        // 计算P在x处的概率密度
        const pDensity = (1 / (Math.sqrt(2 * Math.PI) * pStd)) 
            * Math.exp(-0.5 * Math.pow((x - pMean) / pStd, 2));
        // 计算Q在x处的概率密度
        const qDensity = (1 / (Math.sqrt(2 * Math.PI) * qStd)) 
            * Math.exp(-0.5 * Math.pow((x - qMean) / qStd, 2));
        // 防止log(0)
        const safeQDensity = Math.max(epsilon, qDensity);
        sum += pDensity * Math.log(safeQDensity);
    }
    return -sum / nSamples; // 平均近似积分
}

export {binaryCrossEntropy, categoricalCrossEntropy, sparseCategoricalCrossEntropy, continuousCrossEntropyApprox}