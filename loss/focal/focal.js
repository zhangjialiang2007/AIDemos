/**
 * Focal Loss 损失函数
 * @param {number[]|number[][]} yTrue - 真实标签，二分类为[0,1,...]，多分类为独热编码[[1,0],...]
 * @param {number[]|number[][]} yPred - 模型预测概率，需经过sigmoid(二分类)或softmax(多分类)
 * @param {number} [alpha=0.5] - 类别平衡因子，二分类时建议0.25-0.75，多分类可不传
 * @param {number} [gamma=2] - 聚焦参数，gamma越大对易分类样本的惩罚越弱
 * @param {number} [epsilon=1e-10] - 防止log(0)的极小值
 * @returns {number} 平均Focal Loss
 * @example
 *  二分类使用示例
 * const binaryYTrue = [1, 0, 1, 0, 1];
 * const binaryYPred = [0.9, 0.3, 0.6, 0.8, 0.2]; // 经过sigmoid的概率
 * console.log("二分类Focal Loss:", focalLoss(binaryYTrue, binaryYPred, 0.25, 2));
 * 
 * 多分类使用示例（独热编码）
 * const multiYTrue = [
 *   [1, 0, 0],
 *   [0, 1, 0],
 *   [0, 0, 1],
 *   [1, 0, 0],
 *   [0, 1, 0]
 * ];
 * const multiYPred = [
 *   [0.8, 0.1, 0.1],
 *   [0.2, 0.7, 0.1],
 *   [0.1, 0.3, 0.6],
 *   [0.6, 0.3, 0.1],
 *   [0.3, 0.4, 0.3]
 * ];
 * console.log("多分类Focal Loss:", focalLoss(multiYTrue, multiYPred, 1, 2));
 */
function focalLoss(yTrue, yPred, alpha = 0.5, gamma = 2, epsilon = 1e-10) {
    // 检查输入是否为二分类或多分类格式
    const isBinary = Array.isArray(yTrue[0]) ? false : true;
    
    // 验证输入维度
    if (yTrue.length !== yPred.length) {
        throw new Error('真实标签与预测概率的数量必须一致');
    }
    
    if (!isBinary) {
        for (let i = 0; i < yTrue.length; i++) {
            if (yTrue[i].length !== yPred[i].length) {
                throw new Error('多分类时真实标签与预测概率的维度必须一致');
            }
        }
        // 多分类时不需要alpha参数
        alpha = 1;
    }
    
    let totalLoss = 0;
    
    if (isBinary) {
        // 二分类Focal Loss: -alpha*y*((1-p)^gamma)*log(p) - (1-alpha)*(1-y)*(p^gamma)*log(1-p)
        for (let i = 0; i < yTrue.length; i++) {
            const y = yTrue[i];
            let p = yPred[i];
            
            // 概率值截断，确保在[epsilon, 1-epsilon]范围内
            p = Math.max(epsilon, Math.min(1 - epsilon, p));
            
            const term1 = y * Math.pow(1 - p, gamma) * Math.log(p);
            const term2 = (1 - y) * Math.pow(p, gamma) * Math.log(1 - p);
            totalLoss += -alpha * term1 - (1 - alpha) * term2;
        }
    } else {
        // 多分类Focal Loss: -sum(y*((1-p)^gamma)*log(p))
        for (let i = 0; i < yTrue.length; i++) {
            let sampleLoss = 0;
            for (let j = 0; j < yTrue[i].length; j++) {
                const y = yTrue[i][j];
                let p = yPred[i][j];
                
                p = Math.max(epsilon, Math.min(1 - epsilon, p));
                
                sampleLoss += y * Math.pow(1 - p, gamma) * Math.log(p);
            }
            totalLoss += -sampleLoss;
        }
    }
    
    // 返回平均损失
    return totalLoss / yTrue.length;
}

export {focalLoss}