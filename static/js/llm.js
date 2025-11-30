/**
 * AI LLM 对话管理器
 * 负责处理图片上传、粘贴、对话历史等功能
 */
const LLMManager = {
    currentImage: null, // 当前选择的图片数据（base64）
    chatHistory: [], // 对话历史记录

    /**
     * 初始化LLM管理器
     */
    init() {
        this.bindEvents();
        this.initPasteHandler();
    },

    /**
     * 绑定事件处理器
     */
    bindEvents() {
        // 上传图片按钮
        const uploadBtn = document.getElementById('uploadImageBtn');
        const imageUpload = document.getElementById('imageUpload');
        if (uploadBtn && imageUpload) {
            uploadBtn.addEventListener('click', () => {
                imageUpload.click();
            });

            imageUpload.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) {
                    this.handleImageFile(file);
                }
            });
        }

        // 截取图表按钮
        const captureBtn = document.getElementById('captureChartBtn');
        if (captureBtn) {
            captureBtn.addEventListener('click', () => {
                this.captureChart();
            });
        }

        // 删除图片按钮
        const removeBtn = document.getElementById('removeImage');
        if (removeBtn) {
            removeBtn.addEventListener('click', () => {
                this.clearImage();
            });
        }

        // 发送消息按钮
        const sendBtn = document.getElementById('btnSendMessage');
        if (sendBtn) {
            sendBtn.addEventListener('click', () => {
                this.sendMessage();
            });
        }

        // 输入框回车发送（Shift+Enter换行）
        const promptInput = document.getElementById('llmPrompt');
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
        }
    },

    /**
     * 初始化粘贴处理器
     */
    initPasteHandler() {
        const promptInput = document.getElementById('llmPrompt');
        if (promptInput) {
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
            this.currentImage = e.target.result;
            this.showImagePreview(e.target.result);
        };
        reader.readAsDataURL(file);
    },

    /**
     * 显示图片预览
     */
    showImagePreview(dataUrl) {
        const container = document.getElementById('imagePreviewContainer');
        const preview = document.getElementById('imagePreview');

        if (container && preview) {
            preview.src = dataUrl;
            container.style.display = 'block';
        }
    },

    /**
     * 清除图片
     */
    clearImage() {
        this.currentImage = null;
        const container = document.getElementById('imagePreviewContainer');
        const preview = document.getElementById('imagePreview');

        if (container && preview) {
            preview.src = '';
            container.style.display = 'none';
        }

        // 重置文件输入
        const imageUpload = document.getElementById('imageUpload');
        if (imageUpload) {
            imageUpload.value = '';
        }
    },

    /**
     * 截取当前图表
     */
    captureChart() {
        const chartDom = document.getElementById('priceChart');
        if (!chartDom) {
            alert('未找到图表');
            return;
        }

        const chartInstance = echarts.getInstanceByDom(chartDom);
        if (!chartInstance) {
            alert('图表未初始化');
            return;
        }

        // 获取图表的base64图片
        const imageDataUrl = chartInstance.getDataURL({
            type: 'png',
            pixelRatio: 2,
            backgroundColor: '#fff'
        });

        this.currentImage = imageDataUrl;
        this.showImagePreview(imageDataUrl);
    },

    /**
     * 发送消息
     */
    async sendMessage() {
        const promptInput = document.getElementById('llmPrompt');
        const prompt = promptInput?.value.trim();

        if (!prompt && !this.currentImage) {
            alert('请输入问题或上传图片');
            return;
        }

        // 添加用户消息到历史
        this.addMessageToHistory('user', prompt, this.currentImage);

        // 准备API请求数据
        const requestData = {
            text_prompt: prompt || '请分析这张图片',
            image_type: 'base64'
        };

        // 如果有图片，提取base64数据（去掉data:image/png;base64,前缀）
        if (this.currentImage) {
            const base64Data = this.currentImage.split(',')[1];
            requestData.image_data = base64Data;
        }

        // 清空输入和图片
        if (promptInput) {
            promptInput.value = '';
            promptInput.style.height = 'auto';
        }
        this.clearImage();

        // 显示加载状态
        this.showLoading();

        try {
            // 调用API
            const response = await fetch('/api/llm_analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });

            const result = await response.json();

            this.hideLoading();

            if (result.success) {
                // 添加AI回复到历史
                this.addMessageToHistory('assistant', result.result);
            } else {
                // 显示错误
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
        const chatHistory = document.getElementById('chatHistory');
        if (!chatHistory) return;

        // 首次消息时，清除欢迎提示
        if (this.chatHistory.length === 0) {
            chatHistory.innerHTML = '';
        }

        // 创建消息容器
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message mb-3 ${role === 'user' ? 'user-message' : 'assistant-message'}`;

        // 消息头（角色标识）
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

        // 如果有图片，先显示图片
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
            // 支持简单的markdown换行
            textDiv.innerHTML = text.replace(/\n/g, '<br>');
            contentDiv.appendChild(textDiv);
        }

        messageDiv.appendChild(contentDiv);
        chatHistory.appendChild(messageDiv);

        // 保存到历史记录
        this.chatHistory.push({ role, text, imageUrl });

        // 滚动到底部
        this.scrollToBottom();
    },

    /**
     * 滚动到底部
     */
    scrollToBottom() {
        const chatHistory = document.getElementById('chatHistory');
        if (chatHistory) {
            setTimeout(() => {
                chatHistory.scrollTop = chatHistory.scrollHeight;
            }, 100);
        }
    },

    /**
     * 显示加载状态
     */
    showLoading() {
        const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
        modal.show();
    },

    /**
     * 隐藏加载状态
     */
    hideLoading() {
        const modalElement = document.getElementById('loadingModal');
        const modal = bootstrap.Modal.getInstance(modalElement);
        if (modal) {
            modal.hide();
        }
    }
};
