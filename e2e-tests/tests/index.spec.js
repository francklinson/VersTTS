/**
 * 首页（模型列表页）测试
 * 覆盖模型卡片展示、导航、状态显示
 *
 * 用例集标签：
 *   @smoke     - 每次提交后快速验证
 *   @regression - 发布前全量回归
 *   @ui        - 界面元素与样式验证
 */

const { test, expect } = require('@playwright/test');
const { MockAPI, setAuthState, navigateWithAuth } = require('./utils');

test.describe('首页 - 模型列表', { tag: ['@regression', '@ui'] }, () => {
  test.beforeEach(async ({ page }) => {
    const mock = new MockAPI(page);
    await mock.mockHealth();
  });

  test('页面标题应正确', { tag: ['@smoke'] }, async ({ page }) => {
    await navigateWithAuth(page, './index.html');
    await expect(page).toHaveTitle(/VersTTS/);
  });

  test('应展示所有可用模型卡片', { tag: ['@smoke', '@critical'] }, async ({ page }) => {
    await navigateWithAuth(page, './index.html');
    const models = ['cosyvoice', 'qwen3tts', 'voxcpm', 'omnivoice'];
    for (const model of models) {
      const card = page.locator(`.model-card[data-model="${model}"]`);
      await expect(card).toBeVisible();
    }
  });

  test('模型卡片应包含模型名称和描述', { tag: ['@ui'] }, async ({ page }) => {
    await navigateWithAuth(page, './index.html');
    const card = page.locator('.model-card[data-model="qwen3tts"]');
    await expect(card).toContainText('Qwen3-TTS');
    await expect(card.locator('.model-desc')).toBeVisible();
  });

  test('模型卡片应包含进入按钮', { tag: ['@ui'] }, async ({ page }) => {
    await navigateWithAuth(page, './index.html');
    const card = page.locator('.model-card[data-model="voxcpm"]');
    const btn = card.locator('.btn-select');
    await expect(btn).toBeVisible();
    await expect(btn).toHaveText('进入');
  });

  test('任务队列徽标初始应为 0 或隐藏', { tag: ['@ui', '@queue'] }, async ({ page }) => {
    await navigateWithAuth(page, './index.html');
    const badge = page.locator('#taskQueueBadge');
    const text = await badge.textContent().catch(() => '0');
    expect(['0', '']).toContain(text.trim());
  });
});
