/**
 * Dashboard 应用程序
 * 负责特征大屏的所有图表和交互逻辑
 */
const DashboardApp = {
    // 图表实例
    charts: {
        lineChart: null,
        pieChart: null,
        macroBarChart: null,
        clusterChart: null
    },

    // 当前状态
    state: {
        currentChapter: null,
        lineParam: '单价',
        pieRelation: '国家',
        pieYear: 2021,
        pieParam: '单价',
        macroStatsYear: 2021
    },

    // AI对话管理器
    llm: {
        currentImage: null,
        chatHistory: []
    },

    /**
     * 初始化应用
     */
    init() {
        this.initCharts();
        this.bindEvents();
        this.loadMacroStats(this.state.macroStatsYear);
    },

    /**
     * 初始化所有图表
     */
    initCharts() {
        this.initLineChart();
        this.initPieChart();
        this.initMacroBarChart();
        this.initClusterChart();
    },

    /**
     * 初始化折线图
     */
    initLineChart() {
        const el = document.getElementById('lineChart');
        if (!el || typeof echarts === 'undefined') return;

        this.charts.lineChart = echarts.init(el);
        const option = {
            backgroundColor: 'transparent',
            tooltip: { trigger: 'axis' },
            grid: { left: '5%', right: '4%', bottom: '8%', top: '12%' },
            xAxis: {
                type: 'category',
                data: [],
                axisLine: { lineStyle: { color: '#7f8fa6' } }
            },
            yAxis: {
                type: 'value',
                name: '单价',
                axisLine: { lineStyle: { color: '#7f8fa6' } },
                splitLine: { lineStyle: { color: 'rgba(127, 143, 166, 0.2)' } }
            },
            series: [{
                name: '单价',
                type: 'line',
                smooth: true,
                data: [],
                lineStyle: { color: '#686de0', width: 3 },
                itemStyle: { color: '#f9ca24' },
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: 'rgba(104, 109, 224, 0.3)' },
                        { offset: 1, color: 'rgba(104, 109, 224, 0.0)' }
                    ])
                }
            }]
        };
        this.charts.lineChart.setOption(option);
        window.addEventListener('resize', () => this.charts.lineChart && this.charts.lineChart.resize());
    },

    /**
     * 初始化饼图
     */
    initPieChart() {
        const el = document.getElementById('pieChart');
        if (!el || typeof echarts === 'undefined') return;

        this.charts.pieChart = echarts.init(el);
        const option = {
            backgroundColor: 'transparent',
            tooltip: {
                trigger: 'item',
                formatter: '{b}: {c} ({d}%)'
            },
            legend: {
                orient: 'vertical',
                right: '5%',
                top: 'center',
                textStyle: { color: '#bdc3ff' }
            },
            series: [{
                name: '分布',
                type: 'pie',
                radius: ['40%', '70%'],
                avoidLabelOverlap: false,
                itemStyle: {
                    borderRadius: 10,
                    borderColor: '#0b1020',
                    borderWidth: 2
                },
                label: {
                    show: false,
                    position: 'center'
                },
                emphasis: {
                    label: {
                        show: true,
                        fontSize: '18',
                        fontWeight: 'bold',
                        color: '#f5f6fa'
                    }
                },
                labelLine: {
                    show: false
                },
                data: []
            }]
        };
        this.charts.pieChart.setOption(option);
        window.addEventListener('resize', () => this.charts.pieChart && this.charts.pieChart.resize());
    },

    /**
     * 初始化条形图
     */
    initMacroBarChart() {
        const el = document.getElementById('macroBarChart');
        if (!el || typeof echarts === 'undefined') return;

        this.charts.macroBarChart = echarts.init(el);
        const option = {
            backgroundColor: 'transparent',
            tooltip: { trigger: 'axis' },
            grid: { left: '5%', right: '4%', bottom: '8%', top: '12%' },
            xAxis: {
                type: 'category',
                data: ['对象1', '对象2', '对象3', '对象4', '对象5', '对象6', '对象7', '对象8', '对象9', '对象10'],
                axisLine: { lineStyle: { color: '#7f8fa6' } },
                axisLabel: { rotate: 30 }
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
        this.charts.macroBarChart.setOption(option);
        window.addEventListener('resize', () => this.charts.macroBarChart && this.charts.macroBarChart.resize());
    },

    /**
     * 初始化聚类图表
     */
    initClusterChart() {
        const el = document.getElementById('clusterChart');
        if (!el || typeof echarts === 'undefined') return;

        this.charts.clusterChart = echarts.init(el);
        const option = {
            backgroundColor: 'transparent',
            tooltip: {
                trigger: 'item',
                formatter: '{b}: {c}'
            },
            xAxis: {
                show: false,
                min: 0,
                max: 100
            },
            yAxis: {
                show: false,
                min: 0,
                max: 100
            },
            series: [{
                type: 'scatter',
                symbolSize: val => 10 + val[2] / 5,
                data: this.generateClusterData(),
                itemStyle: {
                    color: params => {
                        const colors = ['#22a6b3', '#f9ca24', '#eb4d4b', '#6ab04c'];
                        return colors[params.dataIndex % 4];
                    }
                }
            }]
        };
        this.charts.clusterChart.setOption(option);
        window.addEventListener('resize', () => this.charts.clusterChart && this.charts.clusterChart.resize());
    },

    /**
     * 生成聚类数据（示例）
     */
    generateClusterData() {
        const data = [];
        for (let i = 0; i < 50; i++) {
            data.push([
                Math.random() * 100,
                Math.random() * 100,
                Math.random() * 100
            ]);
        }
        return data;
    },

    /**
     * 绑定所有事件
     */
    bindEvents() {
        // 商品识别
        this.bindRecognitionEvents();

        // 折线图参数切换
        this.bindLineChartEvents();

        // 饼图参数切换
        this.bindPieChartEvents();

        // 宏观统计年份切换
        this.bindMacroStatsEvents();

        // AI对话功能
        this.bindLLMEvents();
    },

    /**
     * 绑定商品识别事件
     */
    bindRecognitionEvents() {
        const imageInput = document.getElementById('productImageInput');
        const btnRecognize = document.getElementById('btnRecognizeProduct');
        const manualInput = document.getElementById('manualChapterInput');

        if (btnRecognize && imageInput) {
            btnRecognize.addEventListener('click', async () => {
                if (!imageInput.files || !imageInput.files[0]) {
                    alert('请先选择图片');
                    return;
                }

                const formData = new FormData();
                formData.append('file', imageInput.files[0]);

                this.showLoading('识别中...');

                try {
                    const response = await fetch('/api/recognize_product', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();
                    this.hideLoading();

                    if (result.success) {
                        this.state.currentChapter = result.chapter_name;
                        this.showRecognitionResult(result.chapter_name);
                        this.updateLineChart();
                        this.updatePieChart();
                    } else {
                        alert('识别失败：' + result.error);
                    }
                } catch (error) {
                    this.hideLoading();
                    alert('识别失败：' + error.message);
                }
            });
        }

        if (manualInput) {
            manualInput.addEventListener('change', () => {
                const chapterName = manualInput.value.trim();
                if (chapterName) {
                    this.state.currentChapter = chapterName;
                    this.showRecognitionResult(chapterName);
                    this.updateLineChart();
                    this.updatePieChart();
                }
            });
        }
    },

    /**
     * 显示识别结果
     */
    showRecognitionResult(chapterName) {
        const container = document.getElementById('recognizedChapter');
        const nameDiv = document.getElementById('chapterName');

        if (container && nameDiv) {
            nameDiv.textContent = chapterName;
            container.style.display = 'block';
        }
    },

    /**
     * 绑定折线图事件
     */
    bindLineChartEvents() {
        const radios = document.querySelectorAll('input[name="lineParam"]');
        radios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.state.lineParam = e.target.value;
                this.updateLineChart();
            });
        });
    },

    /**
     * 绑定饼图事件
     */
    bindPieChartEvents() {
        const relationSelect = document.getElementById('pieRelationSelect');
        const yearSelect = document.getElementById('pieYearSelect');
        const paramSelect = document.getElementById('pieParamSelect');

        if (relationSelect) {
            relationSelect.addEventListener('change', (e) => {
                this.state.pieRelation = e.target.value;
                this.updatePieChart();
            });
        }

        if (yearSelect) {
            yearSelect.addEventListener('change', (e) => {
                this.state.pieYear = parseInt(e.target.value);
                this.updatePieChart();
            });
        }

        if (paramSelect) {
            paramSelect.addEventListener('change', (e) => {
                this.state.pieParam = e.target.value;
                this.updatePieChart();
            });
        }
    },

    /**
     * 绑定宏观统计事件
     */
    bindMacroStatsEvents() {
        const yearSelect = document.getElementById('macroStatsYear');
        if (yearSelect) {
            yearSelect.addEventListener('change', (e) => {
                this.state.macroStatsYear = parseInt(e.target.value);
                this.loadMacroStats(this.state.macroStatsYear);
            });
        }
    },

    /**
     * 更新折线图
     */
    async updateLineChart() {
        if (!this.state.currentChapter) {
            return;
        }

        this.showLoading('加载折线图数据...');

        try {
            const response = await fetch('/api/get_line_data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    chapter_name: this.state.currentChapter,
                    param: this.state.lineParam
                })
            });

            const result = await response.json();
            this.hideLoading();

            if (result.success && result.data) {
                const years = result.data.map(d => d.year);
                const values = result.data.map(d => d.value);

                this.charts.lineChart.setOption({
                    xAxis: { data: years },
                    yAxis: { name: this.state.lineParam },
                    series: [{
                        name: this.state.lineParam,
                        data: values
                    }]
                });
            } else {
                console.error('加载折线图失败：', result.error);
            }
        } catch (error) {
            this.hideLoading();
            console.error('加载折线图失败：', error);
        }
    },

    /**
     * 更新饼图
     */
    async updatePieChart() {
        if (!this.state.currentChapter) {
            return;
        }

        this.showLoading('加载饼图数据...');

        try {
            const response = await fetch('/api/get_pie_data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    chapter_name: this.state.currentChapter,
                    relation: this.state.pieRelation,
                    year: this.state.pieYear,
                    param: this.state.pieParam
                })
            });

            const result = await response.json();
            this.hideLoading();

            if (result.success && result.data) {
                const pieData = result.data.map(d => ({
                    name: d.name,
                    value: d.value
                }));

                this.charts.pieChart.setOption({
                    series: [{
                        data: pieData
                    }]
                });
            } else {
                console.error('加载饼图失败：', result.error);
            }
        } catch (error) {
            this.hideLoading();
            console.error('加载饼图失败：', error);
        }
    },

    /**
     * 加载宏观统计数据
     */
    async loadMacroStats(year) {
        try {
            const response = await fetch(`/api/macro_stats/${year}`);
            const result = await response.json();

            if (result.success && result.data) {
                const stats = result.data;
                this.updateStatCards(stats);
            } else {
                console.error('加载宏观统计失败：', result.error);
            }
        } catch (error) {
            console.error('加载宏观统计失败：', error);
        }
    },

    /**
     * 更新统计卡片
     */
    updateStatCards(stats) {
        const formatNumber = (num) => {
            if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
            if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
            if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
            return num;
        };

        document.getElementById('statTotalAmount').textContent =
            formatNumber(stats.total_amount || stats['总金额'] || 0);
        document.getElementById('statTradeCount').textContent =
            formatNumber(stats.trade_count || stats['交易次数'] || 0);
        document.getElementById('statCountries').textContent =
            formatNumber(stats.countries || stats['贸易伙伴总数'] || 0);
        document.getElementById('statProvinces').textContent =
            formatNumber(stats.provinces || stats['省份总数'] || 0);
        document.getElementById('statProducts').textContent =
            formatNumber(stats.products || stats['商品种类总数'] || 0);
    },

    /**
     * 绑定AI对话事件
     */
    bindLLMEvents() {
        const uploadBtn = document.getElementById('dashboardUploadImageBtn');
        const imageUpload = document.getElementById('dashboardImageUpload');
        const captureBtn = document.getElementById('dashboardCaptureChartBtn');
        const removeBtn = document.getElementById('dashboardRemoveImage');
        const sendBtn = document.getElementById('dashboardBtnSendMessage');
        const promptInput = document.getElementById('dashboardLlmPrompt');

        // 上传图片
        if (uploadBtn && imageUpload) {
            uploadBtn.addEventListener('click', () => imageUpload.click());
            imageUpload.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) this.handleImageFile(file);
            });
        }

        // 截取图表
        if (captureBtn) {
            captureBtn.addEventListener('click', () => this.captureChart());
        }

        // 删除图片
        if (removeBtn) {
            removeBtn.addEventListener('click', () => this.clearImage());
        }

        // 发送消息
        if (sendBtn) {
            sendBtn.addEventListener('click', () => this.sendMessage());
        }

        // 回车发送
        if (promptInput) {
            promptInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });

            // 自动调整高度
            promptInput.addEventListener('input', () => {
                promptInput.style.height = 'auto';
                promptInput.style.height = Math.min(promptInput.scrollHeight, 120) + 'px';
            });

            // 粘贴图片
            promptInput.addEventListener('paste', (e) => {
                const items = e.clipboardData?.items;
                if (!items) return;

                for (let i = 0; i < items.length; i++) {
                    if (items[i].type.indexOf('image') !== -1) {
                        e.preventDefault();
                        const blob = items[i].getAsFile();
                        this.handleImageFile(blob);
                        break;
                    }
                }
            });
        }
    },

    /**
     * 处理图片文件
     */
    handleImageFile(file) {
        if (!file || !file.type.startsWith('image/')) {
            alert('请选择有效的图片文件');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            this.llm.currentImage = e.target.result;
            this.showImagePreview(e.target.result);
        };
        reader.readAsDataURL(file);
    },

    /**
     * 显示图片预览
     */
    showImagePreview(dataUrl) {
        const container = document.getElementById('dashboardImagePreviewContainer');
        const preview = document.getElementById('dashboardImagePreview');

        if (container && preview) {
            preview.src = dataUrl;
            container.style.display = 'block';
        }
    },

    /**
     * 清除图片
     */
    clearImage() {
        this.llm.currentImage = null;
        const container = document.getElementById('dashboardImagePreviewContainer');
        const preview = document.getElementById('dashboardImagePreview');

        if (container && preview) {
            preview.src = '';
            container.style.display = 'none';
        }

        const imageUpload = document.getElementById('dashboardImageUpload');
        if (imageUpload) imageUpload.value = '';
    },

    /**
     * 截取图表
     */
    captureChart() {
        // 默认截取折线图，如果没有则截取其他图表
        let chartInstance = null;
        let chartName = '';

        if (this.charts.lineChart && this.state.currentChapter) {
            chartInstance = this.charts.lineChart;
            chartName = '折线图';
        } else if (this.charts.pieChart) {
            chartInstance = this.charts.pieChart;
            chartName = '饼图';
        } else if (this.charts.clusterChart) {
            chartInstance = this.charts.clusterChart;
            chartName = '聚类图';
        }

        if (!chartInstance) {
            alert('未找到可截取的图表');
            return;
        }

        const imageDataUrl = chartInstance.getDataURL({
            type: 'png',
            pixelRatio: 2,
            backgroundColor: '#0b1020'
        });

        this.llm.currentImage = imageDataUrl;
        this.showImagePreview(imageDataUrl);
    },

    /**
     * 发送消息
     */
    async sendMessage() {
        const promptInput = document.getElementById('dashboardLlmPrompt');
        const prompt = promptInput?.value.trim();

        if (!prompt && !this.llm.currentImage) {
            alert('请输入问题或上传图片');
            return;
        }

        // 添加用户消息到历史
        this.addMessageToHistory('user', prompt, this.llm.currentImage);

        // 准备API请求
        const requestData = {
            text_prompt: prompt || '请分析这张图片',
            image_type: 'base64'
        };

        if (this.llm.currentImage) {
            const base64Data = this.llm.currentImage.split(',')[1];
            requestData.image_data = base64Data;
        }

        // 清空输入
        if (promptInput) {
            promptInput.value = '';
            promptInput.style.height = 'auto';
        }
        this.clearImage();

        this.showLoading('AI正在分析中...');

        try {
            const response = await fetch('/api/llm_analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            const result = await response.json();
            this.hideLoading();

            if (result.success) {
                this.addMessageToHistory('assistant', result.result);
            } else {
                this.addMessageToHistory('assistant', `错误：${result.error || '分析失败'}`);
            }
        } catch (error) {
            this.hideLoading();
            this.addMessageToHistory('assistant', `网络错误：${error.message}`);
        }
    },

    /**
     * 添加消息到对话历史
     */
    addMessageToHistory(role, text, imageUrl = null) {
        const chatHistory = document.getElementById('dashboardChatHistory');
        if (!chatHistory) return;

        // 首次消息时，清除欢迎提示
        if (this.llm.chatHistory.length === 0) {
            chatHistory.innerHTML = '';
        }

        // 创建消息容器
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message mb-3 ${role === 'user' ? 'user-message' : 'assistant-message'}`;

        // 消息头
        const headerDiv = document.createElement('div');
        headerDiv.className = 'd-flex align-items-center mb-1';

        const icon = document.createElement('div');
        icon.className = `me-2 rounded-circle d-flex align-items-center justify-content-center ${role === 'user' ? 'bg-primary' : 'bg-success'}`;
        icon.style.width = '24px';
        icon.style.height = '24px';
        icon.style.fontSize = '12px';
        icon.style.color = 'white';
        icon.textContent = role === 'user' ? '我' : 'AI';

        const roleLabel = document.createElement('small');
        roleLabel.className = 'text-muted fw-bold';
        roleLabel.textContent = role === 'user' ? '您' : 'AI助手';

        headerDiv.appendChild(icon);
        headerDiv.appendChild(roleLabel);
        messageDiv.appendChild(headerDiv);

        // 消息内容
        const contentDiv = document.createElement('div');
        contentDiv.className = `message-content p-3 rounded ${role === 'user' ? 'bg-light' : 'bg-white border'}`;

        // 如果有图片
        if (imageUrl) {
            const img = document.createElement('img');
            img.src = imageUrl;
            img.className = 'img-fluid rounded mb-2';
            img.style.maxHeight = '200px';
            contentDiv.appendChild(img);
        }

        // 添加文本
        if (text) {
            const textDiv = document.createElement('div');
            textDiv.className = 'message-text';
            textDiv.innerHTML = text.replace(/\n/g, '<br>');
            contentDiv.appendChild(textDiv);
        }

        messageDiv.appendChild(contentDiv);
        chatHistory.appendChild(messageDiv);

        // 保存到历史
        this.llm.chatHistory.push({ role, text, imageUrl });

        // 滚动到底部
        setTimeout(() => {
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }, 100);
    },

    /**
     * 显示加载状态
     */
    showLoading(text = '处理中...') {
        const loadingText = document.getElementById('loadingText');
        if (loadingText) loadingText.textContent = text;

        const modal = new bootstrap.Modal(document.getElementById('dashboardLoadingModal'));
        modal.show();
    },

    /**
     * 隐藏加载状态
     */
    hideLoading() {
        const modalElement = document.getElementById('dashboardLoadingModal');
        const modal = bootstrap.Modal.getInstance(modalElement);
        if (modal) modal.hide();
    }
};
