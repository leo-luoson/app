/**
 * chart.js - 图表管理模块
 * 处理单价预测折线图的渲染和数据更新
 */

const ChartManager = (function() {
    let priceChart = null;
    const REAL_DATA_END_YEAR = 2021; // 真实数据的最后一年

    /**
     * 初始化价格折线图
     */
    function initPriceChart() {
        const el = document.getElementById('priceChart');
        if (!el || typeof echarts === 'undefined') {
            console.error('无法初始化图表：找不到容器元素或ECharts库');
            return;
        }

        priceChart = echarts.init(el);

        // 初始化空图表
        const option = {
            backgroundColor: 'transparent',
            title: {
                text: '贸易单价趋势',
                left: 'center',
                textStyle: {
                    color: '#333',
                    fontSize: 16
                }
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'cross'
                },
                formatter: function(params) {
                    let result = `<b>${params[0].axisValue}年</b><br/>`;
                    params.forEach(param => {
                        const value = param.value !== null && param.value !== undefined
                            ? param.value.toFixed(2)
                            : '暂无数据';
                        result += `${param.marker}${param.seriesName}: ${value}<br/>`;
                    });
                    return result;
                }
            },
            legend: {
                data: ['历史真实数据', '预测数据'],
                top: 30,
                textStyle: {
                    color: '#666'
                }
            },
            grid: {
                left: '5%',
                right: '5%',
                bottom: '10%',
                top: '20%',
                containLabel: true
            },
            xAxis: {
                type: 'category',
                data: [],
                axisLine: {
                    lineStyle: {
                        color: '#7f8fa6'
                    }
                },
                axisLabel: {
                    color: '#666'
                }
            },
            yAxis: {
                type: 'value',
                name: '单价',
                axisLine: {
                    lineStyle: {
                        color: '#7f8fa6'
                    }
                },
                axisLabel: {
                    color: '#666'
                },
                splitLine: {
                    lineStyle: {
                        color: 'rgba(127, 143, 166, 0.2)'
                    }
                }
            },
            series: []
        };

        priceChart.setOption(option);

        // 窗口大小变化时重新调整图表
        window.addEventListener('resize', () => {
            if (priceChart) {
                priceChart.resize();
            }
        });
    }

    /**
     * 更新图表数据
     * @param {Object} params - 包含country, province, trade_type, name, unit, start_year, end_year
     */
    async function updateChart(params) {
        if (!priceChart) {
            console.error('图表未初始化');
            return;
        }

        try {
            // 显示加载动画
            priceChart.showLoading({
                text: '数据加载中...',
                color: '#686de0',
                textColor: '#666',
                maskColor: 'rgba(255, 255, 255, 0.8)'
            });

            const startYear = parseInt(params.start_year);
            const endYear = parseInt(params.end_year);

            // 构建年份数组
            const years = [];
            for (let y = startYear; y <= endYear; y++) {
                years.push(y);
            }

            // 分别获取真实数据和预测数据
            const realData = await fetchRealData(params, startYear, Math.min(endYear, REAL_DATA_END_YEAR));
            const predictionData = endYear > REAL_DATA_END_YEAR
                ? await fetchPredictionData(params, REAL_DATA_END_YEAR + 1, endYear)
                : [];

            // 合并数据
            const chartData = mergeChartData(years, realData, predictionData);

            // 更新图表
            updateChartOption(years, chartData);

            // 隐藏加载动画
            priceChart.hideLoading();

        } catch (error) {
            console.error('更新图表失败:', error);
            priceChart.hideLoading();
            alert('获取数据失败: ' + error.message);
        }
    }

    /**
     * 获取真实历史数据
     */
    async function fetchRealData(params, startYear, endYear) {
        if (startYear > endYear) return [];

        try {
            const response = await fetch('/api/get_real_data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    country: params.country,
                    province: params.province || '',
                    trade_type: params.trade_type,
                    name: params.name,
                    unit: params.unit,
                    start_year: startYear,
                    end_year: endYear
                })
            });

            const result = await response.json();

            if (result.success) {
                return result.data;
            } else {
                throw new Error(result.error || '获取真实数据失败');
            }
        } catch (error) {
            console.error('获取真实数据失败:', error);
            throw error;
        }
    }

    /**
     * 获取预测数据
     */
    async function fetchPredictionData(params, startYear, endYear) {
        if (startYear > endYear) return [];

        try {
            const predictions = [];

            // 逐年调用预测接口
            for (let year = startYear; year <= endYear; year++) {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        country: params.country,
                        reg_place: params.province || '',
                        product_code: 'XX', // 暂时使用占位符，后续需要从商品名称映射到编码
                        unit: params.unit,
                        year: year,
                        trade_method: params.trade_type
                    })
                });

                const result = await response.json();

                if (result.success) {
                    predictions.push({
                        year: year,
                        predicted_price: result.predicted_price
                    });
                } else {
                    console.warn(`${year}年预测失败:`, result.error);
                    // 预测失败时使用null
                    predictions.push({
                        year: year,
                        predicted_price: null
                    });
                }
            }

            return predictions;
        } catch (error) {
            console.error('获取预测数据失败:', error);
            throw error;
        }
    }

    /**
     * 合并真实数据和预测数据
     */
    function mergeChartData(years, realData, predictionData) {
        const realDataMap = {};
        const predictionDataMap = {};

        // 构建真实数据映射
        realData.forEach(item => {
            realDataMap[item.year] = item.avg_price;
        });

        // 构建预测数据映射
        predictionData.forEach(item => {
            predictionDataMap[item.year] = item.predicted_price;
        });

        // 为每一年分配数据
        const realValues = [];
        const predictionValues = [];

        years.forEach(year => {
            if (year <= REAL_DATA_END_YEAR) {
                // 2021年及之前使用真实数据
                realValues.push(realDataMap[year] !== undefined ? realDataMap[year] : null);
                predictionValues.push(null);
            } else {
                // 2021年之后使用预测数据
                realValues.push(null);
                predictionValues.push(predictionDataMap[year] !== undefined ? predictionDataMap[year] : null);
            }
        });

        return {
            realValues,
            predictionValues
        };
    }

    /**
     * 更新图表配置
     */
    function updateChartOption(years, chartData) {
        const option = {
            xAxis: {
                data: years
            },
            series: [
                {
                    name: '历史真实数据',
                    type: 'line',
                    data: chartData.realValues,
                    smooth: true,
                    symbol: 'circle',
                    symbolSize: 8,
                    lineStyle: {
                        color: '#686de0',
                        width: 3
                    },
                    itemStyle: {
                        color: '#686de0',
                        borderColor: '#fff',
                        borderWidth: 2
                    },
                    areaStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: 'rgba(104, 109, 224, 0.3)' },
                            { offset: 1, color: 'rgba(104, 109, 224, 0.05)' }
                        ])
                    },
                    connectNulls: false
                },
                {
                    name: '预测数据',
                    type: 'line',
                    data: chartData.predictionValues,
                    smooth: true,
                    symbol: 'diamond',
                    symbolSize: 8,
                    lineStyle: {
                        color: '#f39c12',
                        width: 3,
                        type: 'dashed' // 虚线表示预测数据
                    },
                    itemStyle: {
                        color: '#f39c12',
                        borderColor: '#fff',
                        borderWidth: 2
                    },
                    areaStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: 'rgba(243, 156, 18, 0.3)' },
                            { offset: 1, color: 'rgba(243, 156, 18, 0.05)' }
                        ])
                    },
                    connectNulls: false
                }
            ]
        };

        priceChart.setOption(option);
    }

    /**
     * 清空图表
     */
    function clearChart() {
        if (priceChart) {
            priceChart.setOption({
                xAxis: { data: [] },
                series: []
            });
        }
    }

    /**
     * 获取图表实例（用于其他操作，如截图）
     */
    function getChartInstance() {
        return priceChart;
    }

    // 对外暴露的接口
    return {
        initPriceChart,
        updateChart,
        clearChart,
        getChartInstance
    };
})();