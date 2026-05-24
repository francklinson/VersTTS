/**
 * 性能测试用例
 * 覆盖页面加载性能、API响应时间、生成耗时等
 *
 * 用例集标签：
 *   @performance - 性能测试
 *   @regression  - 发布前全量回归
 */

const { test, expect } = require('@playwright/test');
const { MockAPI, fillTextInput, setAuthState, navigateWithAuth } = require('./utils');

// 性能测试配置
const PERFORMANCE_THRESHOLDS = {
  pageLoad: 5000,        // 页面加载时间阈值 (ms)
  apiResponse: 3000,     // API响应时间阈值 (ms)
  firstPaint: 2000,      // 首次绘制时间阈值 (ms)
  domContentLoaded: 3000, // DOM内容加载时间阈值 (ms)
};

test.describe('首页性能测试', { tag: ['@performance', '@regression'] }, () => {
  test.beforeEach(async ({ page }) => {
    const mock = new MockAPI(page);
    await mock.mockHealth();
  });

  test('首页加载时间应在合理范围内', { tag: ['@performance'] }, async ({ page }) => {
    const startTime = Date.now();
    await navigateWithAuth(page, './index.html');
    await page.waitForLoadState('networkidle');
    const loadTime = Date.now() - startTime;

    console.log(`首页加载时间: ${loadTime}ms`);
    expect(loadTime).toBeLessThan(PERFORMANCE_THRESHOLDS.pageLoad);
  });

  test('首页首次绘制时间应在合理范围内', { tag: ['@performance'] }, async ({ page }) => {
    await setAuthState(page);
    await page.goto('./index.html');

    // 使用 Performance API 获取首次绘制时间
    const firstPaint = await page.evaluate(() => {
      return new Promise((resolve) => {
        const observer = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const paintEntry = entries.find(entry => entry.name === 'first-contentful-paint');
          if (paintEntry) {
            resolve(paintEntry.startTime);
          }
        });
        observer.observe({ entryTypes: ['paint'] });

        // 超时处理
        setTimeout(() => resolve(0), 5000);
      });
    });

    console.log(`首页首次绘制时间: ${firstPaint}ms`);
    if (firstPaint > 0) {
      expect(firstPaint).toBeLessThan(PERFORMANCE_THRESHOLDS.firstPaint);
    }
  });

  test('首页DOM内容加载时间应在合理范围内', { tag: ['@performance'] }, async ({ page }) => {
    await setAuthState(page);
    await page.goto('./index.html');

    const timing = await page.evaluate(() => {
      const perf = performance.timing;
      return {
        domContentLoaded: perf.domContentLoadedEventEnd - perf.navigationStart,
        loadComplete: perf.loadEventEnd - perf.navigationStart,
      };
    });

    console.log(`DOM内容加载时间: ${timing.domContentLoaded}ms`);
    console.log(`页面完全加载时间: ${timing.loadComplete}ms`);

    expect(timing.domContentLoaded).toBeLessThan(PERFORMANCE_THRESHOLDS.domContentLoaded);
  });
});

test.describe('模型页面性能测试', { tag: ['@performance', '@regression'] }, () => {
  const models = [
    { name: 'qwen3tts', display: 'Qwen3-TTS' },
    { name: 'voxcpm', display: 'VoxCPM' },
    { name: 'omnivoice', display: 'OmniVoice' },
    { name: 'cosyvoice', display: 'CosyVoice' },
  ];

  for (const model of models) {
    test(`${model.display} 页面加载时间应在合理范围内`, { tag: ['@performance'] }, async ({ page }) => {
      const mock = new MockAPI(page);
      await mock.mockAllApis();
      await setAuthState(page);

      const startTime = Date.now();
      await page.goto(`./pages/${model.name}.html`);
      await page.waitForLoadState('networkidle');
      const loadTime = Date.now() - startTime;

      console.log(`${model.display} 页面加载时间: ${loadTime}ms`);
      expect(loadTime).toBeLessThan(PERFORMANCE_THRESHOLDS.pageLoad);
    });

    test(`${model.display} 生成响应时间应在合理范围内`, { tag: ['@performance'] }, async ({ page }) => {
      const mock = new MockAPI(page);
      await mock.mockAllApis();
      await setAuthState(page);
      await page.goto(`./pages/${model.name}.html`);
      await page.waitForLoadState('networkidle');

      await fillTextInput(page, '性能测试文本');

      const startTime = Date.now();
      await page.locator('#generateBtn').click();
      await page.locator('#globalToast, #statusMessage.show').first().waitFor({ timeout: 10000 });
      const responseTime = Date.now() - startTime;

      console.log(`${model.display} 生成响应时间: ${responseTime}ms`);
      expect(responseTime).toBeLessThan(PERFORMANCE_THRESHOLDS.apiResponse);
    });
  }
});

test.describe('任务队列页面性能测试', { tag: ['@performance', '@regression'] }, () => {
  test.beforeEach(async ({ page }) => {
    const mock = new MockAPI(page);
    await mock.mockAllApis();
  });

  test('任务队列页面加载时间应在合理范围内', { tag: ['@performance'] }, async ({ page }) => {
    await setAuthState(page);

    const startTime = Date.now();
    await page.goto('./pages/tasks.html');
    await page.waitForLoadState('networkidle');
    const loadTime = Date.now() - startTime;

    console.log(`任务队列页面加载时间: ${loadTime}ms`);
    expect(loadTime).toBeLessThan(PERFORMANCE_THRESHOLDS.pageLoad);
  });

  test('任务列表刷新响应时间应在合理范围内', { tag: ['@performance'] }, async ({ page }) => {
    await setAuthState(page);
    await page.goto('./pages/tasks.html');
    await page.waitForLoadState('networkidle');

    const refreshBtn = page.locator('button:has-text("刷新")');

    const startTime = Date.now();
    await refreshBtn.click();
    await page.locator('#taskList .task-item').first().waitFor({ timeout: 10000 });
    const responseTime = Date.now() - startTime;

    console.log(`任务列表刷新响应时间: ${responseTime}ms`);
    expect(responseTime).toBeLessThan(PERFORMANCE_THRESHOLDS.apiResponse);
  });

  test('大量任务列表渲染性能应在合理范围内', { tag: ['@performance'] }, async ({ page }) => {
    const mock = new MockAPI(page);

    // 生成大量任务数据
    const largeTaskList = Array.from({ length: 50 }, (_, i) => ({
      task_id: `task_perf_${i.toString().padStart(3, '0')}`,
      status: i % 3 === 0 ? 'completed' : i % 3 === 1 ? 'processing' : 'pending',
      model: ['qwen3tts', 'voxcpm', 'omnivoice', 'cosyvoice'][i % 4],
      mode: 'base',
      text: `性能测试任务 ${i}`,
      audio_url: i % 3 === 0 ? `/outputs/test_${i}.wav` : null,
      created_at: new Date(Date.now() - i * 60000).toISOString(),
      batch_results: null,
    }));

    await mock.mockTaskList(largeTaskList);
    await setAuthState(page);

    const startTime = Date.now();
    await page.goto('./pages/tasks.html');
    await page.waitForLoadState('networkidle');
    const loadTime = Date.now() - startTime;

    // 验证任务列表已渲染
    const taskCount = await page.locator('#taskList .task-item').count();
    console.log(`大量任务列表渲染时间: ${loadTime}ms, 任务数量: ${taskCount}`);

    expect(loadTime).toBeLessThan(PERFORMANCE_THRESHOLDS.pageLoad * 2); // 大量数据允许更长时间
    expect(taskCount).toBeGreaterThan(0);
  });
});

test.describe('API性能测试', { tag: ['@performance', '@regression'] }, () => {
  test.beforeEach(async ({ page }) => {
    await setAuthState(page);
  });

  test('健康检查API响应时间应在合理范围内', { tag: ['@performance'] }, async ({ page }) => {
    const startTime = Date.now();
    const response = await page.request.get('http://localhost:8000/health');
    const responseTime = Date.now() - startTime;

    console.log(`健康检查API响应时间: ${responseTime}ms`);
    expect(responseTime).toBeLessThan(1000); // API响应应在1秒内
    expect(response.ok()).toBe(true);
  });

  test('说话人列表API响应时间应在合理范围内', { tag: ['@performance'] }, async ({ page }) => {
    const startTime = Date.now();
    const response = await page.request.get('http://localhost:8000/speakers');
    const responseTime = Date.now() - startTime;

    console.log(`说话人列表API响应时间: ${responseTime}ms`);
    expect(responseTime).toBeLessThan(PERFORMANCE_THRESHOLDS.apiResponse);
  });

  test('任务列表API响应时间应在合理范围内', { tag: ['@performance'] }, async ({ page }) => {
    const startTime = Date.now();
    const response = await page.request.get('http://localhost:8000/tasks/list');
    const responseTime = Date.now() - startTime;

    console.log(`任务列表API响应时间: ${responseTime}ms`);
    expect(responseTime).toBeLessThan(PERFORMANCE_THRESHOLDS.apiResponse);
  });
});

test.describe('内存和性能监控', { tag: ['@performance', '@regression'] }, () => {
  test('页面内存使用应在合理范围内', { tag: ['@performance'] }, async ({ page }) => {
    await setAuthState(page);
    await page.goto('./index.html');
    await page.waitForLoadState('networkidle');

    // 获取内存使用情况
    const memory = await page.evaluate(() => {
      if (performance.memory) {
        return {
          usedJSHeapSize: performance.memory.usedJSHeapSize,
          totalJSHeapSize: performance.memory.totalJSHeapSize,
          jsHeapSizeLimit: performance.memory.jsHeapSizeLimit,
        };
      }
      return null;
    });

    if (memory) {
      const usedMB = memory.usedJSHeapSize / 1024 / 1024;
      console.log(`页面内存使用: ${usedMB.toFixed(2)}MB`);
      expect(usedMB).toBeLessThan(100); // 内存使用应小于100MB
    }
  });

  test('长时间运行不应有明显内存泄漏', { tag: ['@performance'] }, async ({ page }) => {
    const mock = new MockAPI(page);
    await mock.mockAllApis();
    await setAuthState(page);
    await page.goto('./pages/qwen3tts.html');
    await page.waitForLoadState('networkidle');

    // 记录初始内存
    const initialMemory = await page.evaluate(() => {
      if (performance.memory) {
        return performance.memory.usedJSHeapSize;
      }
      return 0;
    });

    // 执行多次生成操作
    for (let i = 0; i < 5; i++) {
      await fillTextInput(page, `内存测试文本 ${i}`);
      await page.locator('#generateBtn').click();
      await page.waitForTimeout(1000);
    }

    // 强制垃圾回收（如果可用）
    await page.evaluate(() => {
      if (window.gc) {
        window.gc();
      }
    });

    // 记录最终内存
    const finalMemory = await page.evaluate(() => {
      if (performance.memory) {
        return performance.memory.usedJSHeapSize;
      }
      return 0;
    });

    if (initialMemory > 0 && finalMemory > 0) {
      const increaseMB = (finalMemory - initialMemory) / 1024 / 1024;
      console.log(`内存增长: ${increaseMB.toFixed(2)}MB`);
      // 允许一定的内存增长，但不应超过50MB
      expect(increaseMB).toBeLessThan(50);
    }
  });
});
