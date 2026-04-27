/**
 * VersTTS 公共 JavaScript 库
 * 包含所有模型页面共享的函数和工具
 */

const VersTTS = {
    // API 基础 URL
    API_BASE: window.location.origin,

    /**
     * 检查服务器健康状态
     */
    async checkHealth() {
        const dot = document.getElementById('healthDot');
        const text = document.getElementById('healthText');

        if (!dot || !text) return;

        try {
            const response = await fetch(`${this.API_BASE}/health`);
            const data = await response.json();
            if (data.status === 'healthy') {
                dot.className = 'health-dot online';
                text.textContent = '在线';
            } else {
                dot.className = 'health-dot offline';
                text.textContent = '离线';
            }
        } catch (error) {
            dot.className = 'health-dot offline';
            text.textContent = '离线';
        }
    },

    /**
     * 显示状态消息
     * @param {string} message - 消息内容
     * @param {boolean} isError - 是否为错误消息
     */
    showStatus(message, isError = false) {
        const statusEl = document.getElementById('statusMessage');
        if (!statusEl) return;

        statusEl.textContent = message;
        statusEl.className = `status-message show ${isError ? 'error' : 'success'}`;
        setTimeout(() => {
            statusEl.classList.remove('show');
        }, 5000);
    },

    /**
     * 检查登录状态，未登录则跳转
     * @returns {boolean} 是否已登录
     */
    checkAuth() {
        const auth = localStorage.getItem('versTTS_auth');
        if (auth !== '1') {
            window.location.href = '../login.html';
            return false;
        }
        return true;
    },

    /**
     * 初始化页面（登录检查 + 健康检查）
     */
    initPage() {
        if (!this.checkAuth()) return false;
        this.checkHealth();
        return true;
    },

    /**
     * 设置按钮加载状态
     * @param {HTMLElement} btn - 按钮元素
     * @param {HTMLElement} btnText - 按钮文本元素
     * @param {boolean} loading - 是否处于加载状态
     * @param {string} originalText - 原始文本（恢复时使用）
     */
    setButtonLoading(btn, btnText, loading, originalText = null) {
        if (loading) {
            btn.disabled = true;
            if (btnText) {
                btnText.dataset.originalText = btnText.innerHTML;
                btnText.innerHTML = '<span class="spinner"></span> 生成中...';
            }
        } else {
            btn.disabled = false;
            if (btnText) {
                const text = originalText || btnText.dataset.originalText;
                if (text) btnText.innerHTML = text;
            }
        }
    },

    /**
     * 模式切换（用于有多个模式的页面）
     * @param {string} mode - 目标模式
     * @param {Object} options - 配置选项
     */
    switchMode(mode, options = {}) {
        const tabSelector = options.tabSelector || '.mode-tab';
        const contentSelector = options.contentSelector || '.mode-content';
        const activeClass = options.activeClass || 'active';

        // 移除所有活动状态
        document.querySelectorAll(tabSelector).forEach(t => t.classList.remove(activeClass));
        document.querySelectorAll(contentSelector).forEach(c => c.classList.remove(activeClass));

        // 添加当前活动状态
        if (event && event.target) {
            event.target.classList.add(activeClass);
        }

        const contentEl = document.getElementById(`mode-${mode}`);
        if (contentEl) {
            contentEl.classList.add(activeClass);
        }

        return mode;
    },

    /**
     * 显示生成结果
     * @param {Object} data - API 返回的数据
     * @param {Object} options - 配置选项
     */
    showResult(data, options = {}) {
        const audioPlayer = document.getElementById(options.audioPlayerId || 'audioPlayer');
        const downloadLink = document.getElementById(options.downloadLinkId || 'downloadLink');
        const resultSection = document.getElementById(options.resultSectionId || 'resultSection');
        const filename = options.filename || `result_${Date.now()}.wav`;

        if (audioPlayer) {
            audioPlayer.src = data.audio_url || `${this.API_BASE}${data.audio_url}`;
        }
        if (downloadLink) {
            downloadLink.href = data.audio_url || `${this.API_BASE}${data.audio_url}`;
            downloadLink.download = filename;
        }
        if (resultSection) {
            resultSection.style.display = 'block';
        }
    },

    /**
     * 通用的 POST 请求封装
     * @param {string} endpoint - API 端点
     * @param {FormData} formData - 表单数据
     * @returns {Promise<Object>} 响应数据
     */
    async postForm(endpoint, formData) {
        const response = await fetch(`${this.API_BASE}${endpoint}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `请求失败: ${response.status}`);
        }

        return await response.json();
    },

    /**
     * 加载说话人列表
     * @param {string} selectId - 下拉框 ID
     * @param {string} defaultText - 默认选项文本
     */
    async loadSpeakers(selectId, defaultText = '-- 请选择说话人 --') {
        const selectEl = document.getElementById(selectId);
        if (!selectEl) return [];

        try {
            const response = await fetch(`${this.API_BASE}/speakers`);
            const data = await response.json();
            if (data.success) {
                const speakers = data.speakers || [];
                const options = speakers.map(s => `<option value="${s.id}">${s.name}</option>`).join('');
                selectEl.innerHTML = `<option value="">${defaultText}</option>` + options;
                return speakers;
            }
        } catch (e) {
            console.error('加载说话人失败:', e);
        }
        return [];
    }
};

// 全局快捷函数（兼容旧代码）
const checkHealth = () => VersTTS.checkHealth();
const showStatus = (msg, err) => VersTTS.showStatus(msg, err);
const initPage = () => VersTTS.initPage();
const switchMode = (mode) => VersTTS.switchMode(mode);
