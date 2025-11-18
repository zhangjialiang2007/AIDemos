const path = require('path');
const webpack = require('webpack')

module.exports = {
  // 入口文件（项目起点）
  entry: './transformer/index.js',

  // 输出配置
  output: {
    // 打包后的文件目录（绝对路径）
    path: path.resolve(__dirname, '../dist'),
    umdNamedDefine: true,
		libraryTarget: 'umd',
    library:['TRANSFORMER'],
    clean: true, 
  },

  // 开发工具：生成 source-map，方便调试
  devtool: 'inline-source-map',
};