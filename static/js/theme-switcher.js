/**
 * 主题切换器
 * 支持在默认深色主题和华为ICT大赛主题之间切换
 */
const ThemeSwitcher = {
    // 主题配置
    themes: {
        'default': '/static/css/style.css',
        'huawei': '/static/css/style-huawei.css'
    },

    // 当前主题
    currentTheme: 'default',

    /**
     * 初始化主题切换器
     */
    init() {
        // 从localStorage读取用户选择的主题
        const savedTheme = localStorage.getItem('selectedTheme');
        if (savedTheme && this.themes[savedTheme]) {
            this.currentTheme = savedTheme;
        }

        // 应用主题
        this.applyTheme(this.currentTheme);

        // 绑定主题切换事件
        this.bindEvents();

        // 更新下拉菜单的激活状态
        this.updateMenuState();
    },

    /**
     * 绑定事件
     */
    bindEvents() {
        const themeOptions = document.querySelectorAll('.theme-option');
        themeOptions.forEach(option => {
            option.addEventListener('click', (e) => {
                e.preventDefault();
                const theme = e.currentTarget.getAttribute('data-theme');
                this.switchTheme(theme);
            });
        });
    },

    /**
     * 切换主题
     */
    switchTheme(themeName) {
        if (!this.themes[themeName]) {
            console.error('未知的主题:', themeName);
            return;
        }

        this.currentTheme = themeName;
        this.applyTheme(themeName);

        // 保存到localStorage
        localStorage.setItem('selectedTheme', themeName);

        // 更新菜单状态
        this.updateMenuState();

        // 显示切换提示
        this.showNotification(`主题已切换为: ${themeName === 'default' ? '默认深色' : '华为ICT大赛'}`);
    },

    /**
     * 应用主题
     */
    applyTheme(themeName) {
        const styleLink = this.getOrCreateStyleLink();
        styleLink.href = this.themes[themeName];
    },

    /**
     * 获取或创建样式链接元素
     */
    getOrCreateStyleLink() {
        let styleLink = document.getElementById('theme-stylesheet');

        if (!styleLink) {
            // 如果不存在，创建新的link元素
            styleLink = document.createElement('link');
            styleLink.id = 'theme-stylesheet';
            styleLink.rel = 'stylesheet';

            // 插入到head中，在Bootstrap CSS之后
            const bootstrapLink = document.querySelector('link[href*="bootstrap"]');
            if (bootstrapLink && bootstrapLink.nextSibling) {
                bootstrapLink.parentNode.insertBefore(styleLink, bootstrapLink.nextSibling);
            } else {
                document.head.appendChild(styleLink);
            }
        }

        return styleLink;
    },

    /**
     * 更新下拉菜单的激活状态
     */
    updateMenuState() {
        const themeOptions = document.querySelectorAll('.theme-option');
        themeOptions.forEach(option => {
            const theme = option.getAttribute('data-theme');
            if (theme === this.currentTheme) {
                option.classList.add('active');
                // 添加勾选图标
                if (!option.querySelector('.check-icon')) {
                    const checkIcon = document.createElement('span');
                    checkIcon.className = 'check-icon float-end';
                    checkIcon.innerHTML = '✓';
                    option.appendChild(checkIcon);
                }
            } else {
                option.classList.remove('active');
                // 移除勾选图标
                const checkIcon = option.querySelector('.check-icon');
                if (checkIcon) {
                    checkIcon.remove();
                }
            }
        });
    },

    /**
     * 显示通知
     */
    showNotification(message) {
        // 创建通知元素
        const notification = document.createElement('div');
        notification.className = 'theme-notification';
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 12px 20px;
            border-radius: 4px;
            z-index: 9999;
            animation: slideIn 0.3s ease-out;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        `;

        // 添加动画样式
        if (!document.querySelector('#notification-style')) {
            const style = document.createElement('style');
            style.id = 'notification-style';
            style.textContent = `
                @keyframes slideIn {
                    from {
                        transform: translateX(100%);
                        opacity: 0;
                    }
                    to {
                        transform: translateX(0);
                        opacity: 1;
                    }
                }
                @keyframes slideOut {
                    from {
                        transform: translateX(0);
                        opacity: 1;
                    }
                    to {
                        transform: translateX(100%);
                        opacity: 0;
                    }
                }
            `;
            document.head.appendChild(style);
        }

        document.body.appendChild(notification);

        // 3秒后自动移除
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-out';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
};

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    ThemeSwitcher.init();
});
