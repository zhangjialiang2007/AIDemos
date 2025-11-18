const webpack = require('webpack');
const { merge } = require('webpack-merge');
const common = require('./webpack.config.js');
const TerserPlugin = require("terser-webpack-plugin");

module.exports = merge(common, {
	target: 'web',
	mode: 'development',
	devtool: false,
	optimization: {
		minimize: true,
		minimizer: [new TerserPlugin()]
	},
	plugins: [
		new webpack.DefinePlugin({
			'_DEBUG': false,
			'_ENABLE_STYLE_DISPOSE_CHECKER': false
		})
	],
	output: {
		filename: 'transformer.umd.min.js'
	}
});