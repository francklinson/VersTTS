/**
 * 轮询管理器 — 统一管理所有页面的后台轮询
 *
 * 功能:
 * - 页面不可见时自动暂停轮询（节省服务器资源）
 * - 页面关闭时自动清除定时器
 * - 可配置最大轮询时长，防止无限轮询
 * - 支持动态调整轮询间隔
 *
 * 用法:
 *   const poller = new PollingManager({
 *       name: 'voxcpm-task',
 *       interval: 5000,
 *       maxDuration: 900000,  // 15分钟
 *       callback: async () => { ... fetch status ... },
 *       onMaxDuration: () => { ... show timeout ... }
 *   });
 *   poller.start();
 */

class PollingManager {
    constructor(options = {}) {
        this.name = options.name || 'unnamed';
        this.interval = options.interval || 5000;
        this.maxDuration = options.maxDuration || 900000; // 默认15分钟
        this.callback = options.callback || (() => {});
        this.onMaxDuration = options.onMaxDuration || (() => {});
        this.onStop = options.onStop || (() => {});

        this._timerId = null;
        this._startTime = null;
        this._paused = false;
        this._running = false;
    }

    start() {
        if (this._running) return;
        this._running = true;
        this._startTime = Date.now();
        this._paused = false;
        this._scheduleNext();
    }

    stop() {
        this._running = false;
        if (this._timerId) {
            clearTimeout(this._timerId);
            this._timerId = null;
        }
        try { this.onStop(); } catch (e) { console.warn('[PollingManager] onStop error:', e); }
    }

    pause() {
        if (!this._running || this._paused) return;
        this._paused = true;
        if (this._timerId) {
            clearTimeout(this._timerId);
            this._timerId = null;
        }
    }

    resume() {
        if (!this._running || !this._paused) return;
        this._paused = false;
        this._scheduleNext();
    }

    _scheduleNext() {
        if (!this._running || this._paused) return;

        this._timerId = setTimeout(async () => {
            if (!this._running || this._paused) return;

            // 检查最大轮询时长
            if (Date.now() - this._startTime >= this.maxDuration) {
                try { this.onMaxDuration(); } catch (e) { console.warn('[PollingManager] onMaxDuration error:', e); }
                this.stop();
                return;
            }

            // 执行回调
            let shouldContinue = true;
            try {
                shouldContinue = await this.callback();
            } catch (e) {
                console.warn(`[PollingManager:${this.name}] callback error:`, e);
            }

            // 回调返回 false 则停止轮询
            if (shouldContinue === false) {
                this.stop();
                return;
            }

            // 继续下一轮
            this._scheduleNext();
        }, this.interval);
    }
}

// 全局轮询管理器注册表
const _pollers = {};
const _visChangeHandler = null;

/**
 * 注册轮询管理器（自动绑定 visibilitychange 和 beforeunload）
 */
function registerPoller(poller) {
    _pollers[poller.name] = poller;

    // 首次注册时绑定全局事件
    if (Object.keys(_pollers).length === 1) {
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                Object.values(_pollers).forEach(p => p.pause());
            } else {
                Object.values(_pollers).forEach(p => p.resume());
            }
        });

        window.addEventListener('beforeunload', () => {
            Object.values(_pollers).forEach(p => {
                try { p.stop(); } catch (e) {}
            });
        });
    }

    return poller;
}

/**
 * 停止并注销轮询
 */
function unregisterPoller(name) {
    const poller = _pollers[name];
    if (poller) {
        poller.stop();
        delete _pollers[name];
    }
}

/**
 * 停止所有轮询
 */
function stopAllPollers() {
    Object.values(_pollers).forEach(p => {
        try { p.stop(); } catch (e) {}
    });
    for (const key in _pollers) {
        delete _pollers[key];
    }
}
