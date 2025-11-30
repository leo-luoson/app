// Simple front-end namespace to avoid globals
const TradeApp = (function () {
    let priceChart, macroBarChart, mapChart, recognizedProductChart;
    let selectedProduct = null;

    // === Page 1: 单价预测 ===
    function initPriceChart() {
        const el = document.getElementById('priceChart');
        if (!el || typeof echarts === 'undefined') return;
        priceChart = echarts.init(el);

        const years = [];
        for (let y = 2015; y <= 2024; y++) years.push(y);

        const option = {
            backgroundColor: 'transparent',
            tooltip: { trigger: 'axis' },
            grid: { left: '5%', right: '4%', bottom: '8%', top: '12%' },
            xAxis: {
                type: 'category',
                data: years,
                axisLine: { lineStyle: { color: '#7f8fa6' } }
            },
            yAxis: {
                type: 'value',
                name: '单价（示意）',
                axisLine: { lineStyle: { color: '#7f8fa6' } },
                splitLine: { lineStyle: { color: 'rgba(127, 143, 166, 0.2)' } }
            },
            series: [{
                name: '预测单价',
                type: 'line',
                smooth: true,
                data: [100, 120, 110, 130, 150, 160, 180, 175, 190, 210],
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: 'rgba(104, 109, 224, 0.5)' },
                        { offset: 1, color: 'rgba(104, 109, 224, 0.0)' }
                    ])
                },
                lineStyle: { color: '#686de0' },
                itemStyle: { color: '#f9ca24' }
            }]
        };
        priceChart.setOption(option);
        window.addEventListener('resize', () => priceChart && priceChart.resize());
    }

    function initProductSelector() {
        const categoryList = document.getElementById('productCategoryList');
        const productList = document.getElementById('productList');
        const confirmBtn = document.getElementById('btnConfirmProduct');

        if (!categoryList || !productList) return;

        // Mock data for demonstration only.
        const mockProducts = {
            electronic: ['手机', '笔记本电脑', '液晶显示器', '路由器', '芯片'],
            agri: ['大豆', '玉米', '小麦', '咖啡豆', '棉花'],
            textile: ['棉布', '毛衣', '牛仔裤', '鞋靴', '箱包']
        };

        function renderProducts(categoryKey) {
            const items = mockProducts[categoryKey] || [];
            productList.innerHTML = items.map(p => `
                <div class="col-6 col-md-4">
                    <button type="button"
                            class="btn btn-outline-light w-100 btn-product-item"
                            data-product="${p}">
                        ${p}
                    </button>
                </div>
            `).join('');

            productList.querySelectorAll('.btn-product-item').forEach(btn => {
                btn.addEventListener('click', () => {
                    productList.querySelectorAll('.btn-product-item').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    selectedProduct = btn.dataset.product;
                });
            });
        }

        // Initial render
        renderProducts('electronic');

        categoryList.querySelectorAll('.list-group-item').forEach(li => {
            li.addEventListener('click', () => {
                categoryList.querySelectorAll('.list-group-item').forEach(el => el.classList.remove('active'));
                li.classList.add('active');
                renderProducts(li.getAttribute('data-category'));
            });
        });

        confirmBtn && confirmBtn.addEventListener('click', () => {
            if (!selectedProduct) {
                alert('请先选择具体商品');
                return;
            }
            // NOTE: Here we only close modal. Later you can store the product to a hidden input
            // and send it to backend API /api/predict when实现预测逻辑.
            const modalEl = document.getElementById('productModal');
            const modal = bootstrap.Modal.getInstance(modalEl);
            modal && modal.hide();
        });
    }

    function bindPredictionEvents() {
        const btnPredict = document.getElementById('btnPredict');
        if (!btnPredict) return;

        btnPredict.addEventListener('click', () => {
            if (!priceChart) return;
            // For now just randomize line data to simulate a "prediction".
            const base = 100;
            const data = [];
            for (let i = 0; i < 10; i++) {
                data.push(Math.round(base + Math.random() * 60));
            }
            priceChart.setOption({
                series: [{ data }]
            });

            // Placeholder: here you could call backend API.
            // Example (to implement later):
            // fetch('/api/predict', {
            //   method: 'POST',
            //   headers: { 'Content-Type': 'application/json' },
            //   body: JSON.stringify({...})
            // }).then(res => res.json()).then(update chart);
        });
    }

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

    return {
        initPriceChart,
        initProductSelector,
        bindPredictionEvents,
        initDashboardCharts
    };
})();


