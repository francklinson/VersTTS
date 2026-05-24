/**
 * VersTTS E2E 测试共享工具函数
 */

const { expect } = require('@playwright/test');

/**
 * 通用的 API Mock 设置
 * 用于在没有真实后端的情况下测试前端 UI 交互
 */
class MockAPI {
  constructor(page) {
    this.page = page;
  }

  /**
   * Mock 健康检查接口
   */
  async mockHealth() {
    await this.page.route('**/health', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'healthy', models: ['qwen3tts', 'voxcpm', 'omnivoice', 'cosyvoice'] }),
      });
    });
  }

  /**
   * Mock 说话人列表接口
   */
  async mockSpeakers(speakers = []) {
    const defaultSpeakers = speakers.length > 0 ? speakers : [
      { id: 'spk_001', name: '测试说话人1', audio_path: '/uploads/test1.wav', reference_text: '测试参考文本一' },
      { id: 'spk_002', name: '测试说话人2', audio_path: '/uploads/test2.wav', reference_text: '测试参考文本二' },
    ];
    await this.page.route('**/speakers', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: true, speakers: defaultSpeakers }),
      });
    });
  }

  /**
   * Mock 任务提交接口
   */
  async mockTaskSubmit(taskId = 'task_20240101_120000_abc12345') {
    await this.page.route('**/tasks/submit', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          task_id: taskId,
          queue_position: 0,
          message: '任务已提交',
        }),
      });
    });
  }

  /**
   * Mock 批量生成接口
   */
  async mockBatchGenerate(count = 5) {
    await this.page.route('**/tts/batch/generate', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          message: `已提交批量生成任务到队列（数量: ${count}）`,
          count: count,
          task_id: 'task_20240101_120000_batch001',
          redirect_url: '/pages/tasks.html',
        }),
      });
    });
  }

  /**
   * Mock 直接生成接口（非队列模式）
   */
  async mockDirectGenerate(model) {
    await this.page.route(`**/tts/${model}`, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          audio_url: '/outputs/test_result.wav',
          message: '生成成功',
        }),
      });
    });
  }

  /**
   * Mock 任务状态查询接口
   */
  async mockTaskStatus(taskId, status = 'completed') {
    await this.page.route(`**/tasks/${taskId}/status`, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          task_id: taskId,
          status: status,
          progress: status === 'processing' ? 50 : 100,
          audio_url: status === 'completed' ? '/outputs/test_result.wav' : null,
          error_message: status === 'failed' ? '生成失败：测试错误' : null,
        }),
      });
    });
  }

  /**
   * Mock 任务列表接口
   */
  async mockTaskList(tasks = []) {
    const defaultTasks = tasks.length > 0 ? tasks : [
      {
        task_id: 'task_20240101_120000_abc001',
        status: 'completed',
        model: 'qwen3tts',
        mode: 'base',
        text: '这是一个测试文本',
        audio_url: '/outputs/test_qwen3tts.wav',
        created_at: new Date().toISOString(),
        batch_results: null,
      },
      {
        task_id: 'task_20240101_120100_abc002',
        status: 'processing',
        model: 'voxcpm',
        mode: 'voice_design',
        text: '音色设计测试',
        audio_url: null,
        created_at: new Date().toISOString(),
        progress: 45,
        batch_results: null,
      },
    ];
    await this.page.route('**/tasks/list**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          tasks: defaultTasks,
        }),
      });
    });
  }

  /**
   * Mock 任务下载接口
   */
  async mockTaskDownload() {
    await this.page.route('**/tasks/*/download', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'audio/wav',
        body: Buffer.from('RIFF....WAVE'), // 最小 wav 文件头
      });
    });
  }

  /**
   * Mock 任务取消接口
   */
  async mockTaskCancel() {
    await this.page.route('**/tasks/*/cancel', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: true, message: '任务已取消' }),
      });
    });
  }

  /**
   * Mock 任务重试接口
   */
  async mockTaskRetry() {
    await this.page.route('**/tasks/*/retry', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: true, new_task_id: 'task_20240101_120300_retry001' }),
      });
    });
  }

  /**
   * Mock 任务删除接口
   */
  async mockTaskDelete() {
    await this.page.route('**/tasks/*', async (route) => {
      if (route.request().method() === 'DELETE') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ success: true, message: '任务已删除' }),
        });
      } else {
        await route.continue();
      }
    });
  }

  /**
   * Mock 全局：拦截所有未匹配的 API 请求并返回 200
   */
  async mockAllApis() {
    await this.mockHealth();
    await this.mockSpeakers();
    await this.mockTaskSubmit();
    await this.mockBatchGenerate();
    await this.mockTaskList();
    await this.mockTaskDownload();
    await this.mockTaskCancel();
    await this.mockTaskRetry();
    await this.mockTaskDelete();
  }
}

/**
 * 等待全局 Toast 提示出现并验证内容
 */
async function waitForToast(page, expectedText, isError = false, timeout = 5000) {
  const toast = page.locator('#globalToast');
  await expect(toast).toBeVisible({ timeout });
  if (expectedText) {
    await expect(toast).toContainText(expectedText, { timeout });
  }
  if (isError) {
    await expect(toast).toHaveCSS('background-color', 'rgb(254, 226, 226)'); // #fee2e2
  }
}

/**
 * 等待状态栏消息出现
 */
async function waitForStatusMessage(page, expectedText, isError = false, timeout = 5000) {
  const statusEl = page.locator('#statusMessage');
  await expect(statusEl).toHaveClass(/show/, { timeout });
  if (expectedText) {
    await expect(statusEl).toContainText(expectedText, { timeout });
  }
  if (isError) {
    await expect(statusEl).toHaveClass(/error/, { timeout });
  }
}

/**
 * 通用页面导航和基础检查
 */
async function navigateToModelPage(page, modelName) {
  await page.goto(`./pages/${modelName}.html`);
  // 等待页面基本元素加载
  await expect(page.locator('#textInput, #genText').first()).toBeVisible({ timeout: 10000 });
  await expect(page.locator('#generateBtn')).toBeVisible();
}

/**
 * 填充文本输入框（兼容 id="textInput" 和 id="genText"）
 */
async function fillTextInput(page, text) {
  const input = page.locator('#textInput, #genText').first();
  await input.fill(text);
}

/**
 * 获取文本输入框的值
 */
async function getTextInputValue(page) {
  const input = page.locator('#textInput, #genText').first();
  return input.inputValue();
}

/**
 * 验证按钮加载状态
 */
async function expectButtonLoading(page, buttonId, textId, loading = true) {
  const btn = page.locator(`#${buttonId}`);
  const btnText = page.locator(`#${textId}`);
  if (loading) {
    await expect(btn).toBeDisabled();
    await expect(btnText).toContainText(/生成中|提交中|加载中|处理中/);
  } else {
    await expect(btn).toBeEnabled();
  }
}

/**
 * 切换模式（通过点击模式标签）
 */
async function switchMode(page, modeName) {
  // 兼容两种模式切换方式：data-mode 属性和 onclick
  const modeTab = page.locator(`.mode-tab[data-mode="${modeName}"], .mode-tab:has-text("${modeName}")`).first();
  if (await modeTab.isVisible().catch(() => false)) {
    await modeTab.click();
    await expect(modeTab).toHaveClass(/active/);
  }
}

/**
 * 执行登录操作
 * 使用默认账号: admin / tp123456
 */
async function doLogin(page, username = 'admin', password = 'tp123456') {
  // 先检查是否已经在登录页面
  const currentUrl = page.url();
  
  // 如果不在登录页面，先跳转到首页，如果需要登录会自动跳转
  if (!currentUrl.includes('login.html')) {
    await page.goto('./index.html');
    await page.waitForLoadState('networkidle');
    
    // 检查是否被重定向到登录页
    if (page.url().includes('login.html')) {
      // 等待登录表单加载
      await page.locator('#loginUsername').waitFor({ timeout: 10000 });
    } else {
      // 已经登录，直接返回
      return;
    }
  }
  
  // 填写登录表单
  await page.locator('#loginUsername').fill(username);
  await page.locator('#loginPassword').fill(password);
  
  // 点击登录按钮
  await page.locator('.btn-login').click();
  
  // 等待跳转到首页
  await page.waitForURL(/index\.html/, { timeout: 10000 });
}

/**
 * 设置登录状态（通过 localStorage）
 * 用于绕过登录页面直接设置认证状态
 */
async function setAuthState(page) {
  await page.addInitScript(() => {
    localStorage.setItem('versTTS_auth', '1');
    localStorage.setItem('versTTS_username', 'admin');
  });
}

/**
 * 带登录状态的页面导航
 */
async function navigateWithAuth(page, url) {
  await setAuthState(page);
  await page.goto(url);
  await page.waitForLoadState('networkidle');
}

module.exports = {
  MockAPI,
  waitForToast,
  waitForStatusMessage,
  navigateToModelPage,
  fillTextInput,
  getTextInputValue,
  expectButtonLoading,
  switchMode,
  doLogin,
  setAuthState,
  navigateWithAuth,
};
