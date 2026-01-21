// 1. 环境参数定义
const GRID_SIZE = 4;
const STATE_NUM = GRID_SIZE * GRID_SIZE;
const ACTION_NUM = 4; // 上0 下1 左2 右3
const START_STATE = 0;
const END_STATE = 15;
const OBSTACLES = [5, 9, 10];
const ALPHA = 0.1; // 学习率
const GAMMA = 0.9; // 折扣因子
const EPSILON = 0.1; // 探索率
const EPISODES = 1000; // 训练回合数

// 2. 初始化Q表 (替代numpy.zeros)
// 创建一个 STATE_NUM x ACTION_NUM 的二维数组，初始值为0
const qTable = Array.from({ length: STATE_NUM }, () => 
  Array.from({ length: ACTION_NUM }, () => 0)
);

// 3. 状态转移函数 (对应Python的step函数)
function step(state, action) {
  // 将一维状态转换为二维坐标 (row, col)
  const row = Math.floor(state / GRID_SIZE);
  const col = state % GRID_SIZE;
  
  // 执行动作
  let newRow = row;
  let newCol = col;
  switch(action) {
    case 0: newRow -= 1; break; // 上
    case 1: newRow += 1; break; // 下
    case 2: newCol -= 1; break; // 左
    case 3: newCol += 1; break; // 右
  }
  
  // 边界判断：超出边界则停留在原状态，奖励-1
  if (newRow < 0 || newRow >= GRID_SIZE || newCol < 0 || newCol >= GRID_SIZE) {
    return [state, -1];
  }
  
  // 计算新状态
  const nextState = newRow * GRID_SIZE + newCol;
  
  // 障碍判断：碰到障碍停留在原状态，奖励-5
  if (OBSTACLES.includes(nextState)) {
    return [state, -5];
  }
  
  // 终点判断：到达终点奖励10
  if (nextState === END_STATE) {
    return [nextState, 10];
  }
  
  // 普通状态：奖励0
  return [nextState, 0];
}

// 辅助函数：生成0-1之间的随机数 (替代np.random.uniform)
function randomUniform() {
  return Math.random();
}

// 辅助函数：随机选择动作 (替代np.random.choice)
function randomChoice(num) {
  return Math.floor(Math.random() * num);
}

// 辅助函数：获取数组最大值 (替代np.max)
function arrayMax(arr) {
  return Math.max(...arr);
}

// 辅助函数：获取数组最大值的索引 (替代np.argmax)
function arrayArgMax(arr) {
  return arr.indexOf(Math.max(...arr));
}

// 4. Q-Learning训练
for (let episode = 0; episode < EPISODES; episode++) {
  let state = START_STATE;
  
  while (state !== END_STATE) {
    // ε-贪心动作选择
    let action;
    if (randomUniform() < EPSILON) {
      // 探索：随机选择动作
      action = randomChoice(ACTION_NUM);
    } else {
      // 利用：选择Q值最大的动作
      action = arrayArgMax(qTable[state]);
    }
    
    // 执行动作，获取下一个状态和奖励
    const [nextState, reward] = step(state, action);
    
    // Q值更新公式 (Q-Learning核心)
    const target = reward + GAMMA * arrayMax(qTable[nextState]);
    qTable[state][action] += ALPHA * (target - qTable[state][action]);
    
    // 状态转移
    state = nextState;
  }
}

// 5. 打印训练后的Q表与最优策略
console.log("训练后Q表（关键状态）：");
const keyStates = [0, 1, 2, 3, 7, 11, 15];
keyStates.forEach(s => {
  // 保留1位小数，模拟Python的round(1)
  const qValues = qTable[s].map(v => v.toFixed(1));
  console.log(`状态${s}: [${qValues.join(', ')}]`);
});

console.log("\n最优策略（状态→动作）：");
const actionNames = ["上", "下", "左", "右"];
for (let s = 0; s < STATE_NUM; s++) {
  if (s === END_STATE) {
    console.log(`S${s} → 终点`);
    continue;
  }
  const bestAction = arrayArgMax(qTable[s]);
  console.log(`S${s} → ${actionNames[bestAction]}`);
}