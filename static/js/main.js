/**
 * main.js - 主应用逻辑
 * 保留 Dashboard 页面的图表初始化逻辑
 * Index 页面的逻辑已迁移到 params.js 和 chart.js
 */

const TradeApp = (function () {
    let macroBarChart, mapChart, recognizedProductChart;

    // === Page 2: 特征大屏 ===
    function initDashboardCharts() {
        initMacroBarChart();
        initMapChart();
        initRecognizedProductChart();
    }

    function initMacroBarChart() {
        const el = document.getElementById('macroBarChart');
        if (!el || typeof echarts === 'undefined') return;
        macroBarChart = echarts.init(el);

        const option = {
            backgroundColor: 'transparent',
            tooltip: { trigger: 'axis' },
            grid: { left: '5%', right: '4%', bottom: '8%', top: '12%' },
            xAxis: {
                type: 'category',
                data: ['对象1', '对象2', '对象3', '对象4', '对象5', '对象6', '对象7', '对象8', '对象9', '对象10'],
                axisLine: { lineStyle: { color: '#7f8fa6' } }
            },
            yAxis: {
                type: 'value',
                axisLine: { lineStyle: { color: '#7f8fa6' } },
                splitLine: { lineStyle: { color: 'rgba(127, 143, 166, 0.2)' } }
            },
            series: [{
                type: 'bar',
                data: [30, 24, 18, 15, 14, 13, 10, 9, 8, 7],
                itemStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: '#f39c12' },
                        { offset: 1, color: '#e67e22' }
                    ])
                }
            }]
        };
        macroBarChart.setOption(option);
        window.addEventListener('resize', () => macroBarChart && macroBarChart.resize());
    }

    function initMapChart() {
        const el = document.getElementById('mapChart');
        if (!el || typeof echarts === 'undefined') return;
        mapChart = echarts.init(el);

        // Simplified map-like scatter plot placeholder.
        const option = {
            backgroundColor: 'transparent',
            tooltip: { trigger: 'item' },
            xAxis: { show: false, min: 0, max: 100 },
            yAxis: { show: false, min: 0, max: 100 },
            series: [{
                type: 'scatter',
                symbolSize: val => 8 + val[2] / 10,
                data: [
                    [20, 30, 50],
                    [40, 70, 80],
                    [60, 40, 30],
                    [80, 55, 90],
                    [50, 20, 60]
                ],
                itemStyle: { color: '#22a6b3' }
            }]
        };
        mapChart.setOption(option);
        window.addEventListener('resize', () => mapChart && mapChart.resize());
    }

    function initRecognizedProductChart() {
        const el = document.getElementById('recognizedProductChart');
        if (!el || typeof echarts === 'undefined') return;
        recognizedProductChart = echarts.init(el);

        const years = [];
        for (let y = 2015; y <= 2024; y++) years.push(y);
        const option = {
            backgroundColor: 'transparent',
            tooltip: { trigger: 'axis' },
            grid: { left: '5%', right: '4%', bottom: '8%', top: '12%' },
            xAxis: { type: 'category', data: years },
            yAxis: { type: 'value' },
            series: [{
                name: '识别商品单价',
                type: 'line',
                smooth: true,
                data: [80, 82, 85, 90, 95, 100, 102, 104, 108, 110],
                lineStyle: { color: '#e84393' },
                itemStyle: { color: '#fd79a8' }
            }]
        };
        recognizedProductChart.setOption(option);
        window.addEventListener('resize', () => recognizedProductChart && recognizedProductChart.resize());

        // UI placeholder: enable button only when file chosen
        const fileInput = document.getElementById('productImageInput');
        const btnRecognize = document.getElementById('btnRecognizeProduct');
        if (fileInput && btnRecognize) {
            fileInput.addEventListener('change', () => {
                btnRecognize.disabled = !fileInput.files.length;
            });
            btnRecognize.addEventListener('click', () => {
                // Placeholder: 将来这里调用 /api/recognize_product
                alert('当前为静态示例。未来将在此调用 /api/recognize_product 完成识别并刷新曲线。');
            });
        }
    }

    // 对外暴露的接口（仅保留 Dashboard 相关）
    return {
        initDashboardCharts
    };
})();


