/**
 * 计算两个概率分布之间的 KL 散度 (Kullback-Leibler Divergence)
 * @param {number[]} p - 第一个概率分布数组（真实分布），所有元素和为1，且均为非负数
 * @param {number[]} q - 第二个概率分布数组（预测分布），所有元素和为1，且均为非负数
 * @param {number} [epsilon=1e-10] - 防止 log(0) 的极小值
 * @returns {number} KL 散度值，值越大表示两个分布差异越大
 * @example
 * const p = [0.2, 0.3, 0.5];  // 真实分布
 * const q1 = [0.2, 0.3, 0.5]; // 与 p 完全相同的分布
 * const q2 = [0.4, 0.3, 0.3]; // 与 p 有差异的分布
 * console.log(klDivergence(p, q1)); // 接近 0（由于数值精度可能不为严格 0）
 * console.log(klDivergence(p, q2)); // 大于 0 的值，表示两个分布有差异
 */
function klDivergence(p, q, epsilon = 1e-10) {
    // 检查输入数组长度是否一致
    if (p.length !== q.length) {
        throw new Error('两个概率分布的长度必须一致');
    }

    let divergence = 0;
    for (let i = 0; i < p.length; i++) {
        // 确保概率值为非负数
        if (p[i] < 0 || q[i] < 0) {
            throw new Error('概率分布中的元素不能为负数');
        }

        // 对 q 中的 0 或极小值进行平滑处理，防止 log(0) 出现
        const qi = Math.max(q[i], epsilon);
        // 计算 p[i] * log(p[i]/q[i])，当 p[i] 为 0 时该项为 0
        divergence += p[i] * Math.log(p[i] / qi);
    }

    return divergence;
}

export {klDivergence}