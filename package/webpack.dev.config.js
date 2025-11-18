const webpack = require('webpack');
const { merge } = require('webpack-merge');
const common = require('./webpack.config.js');

module.exports = merge(common, {
	target: 'web',
	mode: 'development',
	devtool: 'source-map',
	plugins: [
		new webpack.DefinePlugin({
			'_DEBUG': true,
			'_ENABLE_STYLE_DISPOSE_CHECKER': true
		})
	],
	output: {
		filename: 'transformer.umd.js'
	},
});