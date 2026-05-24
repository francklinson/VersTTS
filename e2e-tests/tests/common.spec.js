/**
 * 通用组件与交互测试
 * 覆盖全局导航、Toast 提示、健康检查等公共功能
 *
 * 用例集标签：
 *   @smoke     - 每次提交后快速验证
 *   @regression - 发布前全量回归
 *   @ui        - 界面元素与样式验证
 *   @api       - API 交互验证
 *   @critical  - P0 级核心用例
 */

const { test, expect } = require('@playwright/test');
const { MockAPI, waitForToast, setAuthState, navigateWithAuth } = require('./utils');

test.describe('全局公共组件', { tag: ['@regression', '@ui'] }, () => {
  test.beforeEach(async ({ page }) => {
    const mock = new MockAPI(page);
    await mock.mockHealth();
    await mock.mockSpeakers();
  });

  test('健康状态检查：页面加载后应显示在线状态', { tag: ['@smoke', '@api', '@critical'] }, async ({ page }) => {
    await navigateWithAuth(page, './index.html');
    const healthDot = page.locator('#healthDot');
    const healthText = page.locator('#healthText');

    // 等待 API 响应后变为在线
    await expect(healthDot).toHaveClass(/online/, { timeout: 10000 });
    await expect(healthText).toHaveText('在线');
  });

  test('用户菜单：点击应展开下拉菜单', { tag: '@ui' }, async ({ page }) => {
    await navigateWithAuth(page, './index.html');
    const userMenu = page.locator('.user-info');
    const dropdown = page.locator('#userDropdown');

    await expect(dropdown).not.toBeVisible();
    await userMenu.click();
    await expect(dropdown).toBeVisible();
    await expect(dropdown).toContainText('退出登录');
  });

  test('说话人管理入口：应能打开说话人管理弹窗', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    await navigateWithAuth(page, './index.html');
    const speakerBtn = page.locator('.speaker-manage-btn').filter({ hasText: '说话人管理' });
    await expect(speakerBtn).toBeVisible();
    await speakerBtn.click();
    await expect(page.locator('.speaker-modal, .modal-overlay').first()).toBeVisible({ timeout: 5000 });
  });

  test('任务队列入口：应能跳转到任务队列页面', { tag: ['@smoke', '@queue'] }, async ({ page }) => {
    await navigateWithAuth(page, './index.html');
    const taskBtn = page.locator('.speaker-manage-btn').filter({ hasText: '任务队列' });
    await expect(taskBtn).toBeVisible();
    await taskBtn.click();
    await expect(page).toHaveURL(/tasks\.html/);
  });

  test('全局 Toast：应显示在页面顶部中央', { tag: ['@smoke', '@ui', '@critical'] }, async ({ page }) => {
    await navigateWithAuth(page, './index.html');

    await page.evaluate(() => {
      if (window.VersTTS) {
        window.VersTTS.showStatus('测试 Toast 消息');
      }
    });

    const toast = page.locator('#globalToast');
    await expect(toast).toBeVisible();
    await expect(toast).toHaveText('测试 Toast 消息');
    await expect(toast).toHaveCSS('position', 'fixed');

    // 验证 5 秒后自动消失
    await expect(toast).not.toBeVisible({ timeout: 7000 });
  });

  test('全局 Toast 错误样式：应显示红色背景', { tag: ['@ui'] }, async ({ page }) => {
    await navigateWithAuth(page, './index.html');
    await page.evaluate(() => {
      if (window.VersTTS) {
        window.VersTTS.showStatus('错误测试', true);
      }
    });

    const toast = page.locator('#globalToast');
    await expect(toast).toBeVisible();
    await expect(toast).toHaveCSS('background-color', 'rgb(254, 226, 226)');
  });
});

test.describe('跨页面导航', { tag: ['@regression', '@ui', '@smoke'] }, () => {
  test.beforeEach(async ({ page }) => {
    const mock = new MockAPI(page);
    await mock.mockHealth();
  });

  const models = [
    { name: 'qwen3tts', display: 'Qwen3-TTS' },
    { name: 'voxcpm', display: 'VoxCPM' },
    { name: 'omnivoice', display: 'OmniVoice' },
    { name: 'cosyvoice', display: 'CosyVoice' },
  ];

  for (const model of models) {
    test(`首页应能导航到 ${model.display} 页面`, { tag: ['@smoke', '@critical'] }, async ({ page }) => {
      await navigateWithAuth(page, './index.html');
      const card = page.locator(`.model-card[data-model="${model.name}"]`);
      await expect(card).toBeVisible();
      await card.click();
      await expect(page).toHaveURL(new RegExp(`${model.name}\\.html`));
    });

    test(`${model.display} 页面应能返回首页`, { tag: ['@smoke'] }, async ({ page }) => {
      await navigateWithAuth(page, `./pages/${model.name}.html`);
      const backBtn = page.locator('.nav-back');
      await expect(backBtn).toBeVisible();
      await backBtn.click();
      await expect(page).toHaveURL(/index\.html/);
    });
  }
});
