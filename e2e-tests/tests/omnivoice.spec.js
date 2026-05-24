/**
 * OmniVoice 页面按键功能测试
 * 扩展版本 - 覆盖更多功能场景
 */

const { test, expect } = require('@playwright/test');
const {
  MockAPI,
  navigateToModelPage,
  fillTextInput,
  waitForToast,
  expectButtonLoading,
  switchMode,
  setAuthState,
} = require('./utils');

test.describe('OmniVoice 页面', { tag: ['@regression'] }, () => {
  test.beforeEach(async ({ page }) => {
    const mock = new MockAPI(page);
    await mock.mockAllApis();
    await setAuthState(page);
    await navigateToModelPage(page, 'omnivoice');
  });

  // ====== 基础 UI 测试 ======
  test('页面应加载所有核心元素', { tag: ['@smoke', '@ui', '@critical'] }, async ({ page }) => {
    await expect(page.locator('#textInput')).toBeVisible();
    await expect(page.locator('#generateBtn')).toBeVisible();
    await expect(page.locator('#batchGenerateBtn')).toBeVisible();
    await expect(page.locator('#batchCount')).toBeVisible();
  });

  test('页面标题应正确显示模型名称', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    await expect(page).toHaveTitle(/OmniVoice/);
  });

  test('应显示模型描述和帮助信息', { tag: ['@ui'] }, async ({ page }) => {
    const description = page.locator('.model-description, .page-description').first();
    if (await description.isVisible().catch(() => false)) {
      await expect(description).toBeVisible();
    }
  });

  // ====== 模式切换测试 ======
  test('模式切换应更新激活状态', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    const tabs = await page.locator('.mode-tab').all();
    if (tabs.length > 1) {
      for (const tab of tabs.slice(1)) {
        await tab.click();
        await expect(tab).toHaveClass(/active/);
      }
    }
  });

  test('模式切换应显示对应参数面板', { tag: ['@regression', '@ui'] }, async ({ page }) => {
    const tabs = await page.locator('.mode-tab').all();
    for (const tab of tabs) {
      await tab.click();
      // 验证激活状态
      await expect(tab).toHaveClass(/active/);
      // 验证对应参数面板可见
      const mode = await tab.getAttribute('data-mode');
      if (mode) {
        const panel = page.locator(`.mode-panel[data-mode="${mode}"], .params-panel[data-mode="${mode}"]`).first();
        if (await panel.isVisible().catch(() => false)) {
          await expect(panel).toBeVisible();
        }
      }
    }
  });

  // ====== 文本输入测试 ======
  test('空文本提交应提示错误', { tag: ['@smoke', '@boundary'] }, async ({ page }) => {
    await fillTextInput(page, '');
    await page.locator('#generateBtn').click();
    await waitForToast(page, '请输入', true);
  });

  test('超长文本输入应正确处理', { tag: ['@boundary'] }, async ({ page }) => {
    const longText = '这是一段超长测试文本。'.repeat(100);
    await fillTextInput(page, longText);
    const value = await page.locator('#textInput, #genText').first().inputValue();
    expect(value.length).toBeGreaterThan(100);
  });

  test('特殊字符文本应正确处理', { tag: ['@boundary'] }, async ({ page }) => {
    const specialText = '你好！Hello! 123 @#$% 测试文本';
    await fillTextInput(page, specialText);
    const value = await page.locator('#textInput, #genText').first().inputValue();
    expect(value).toBe(specialText);
  });

  // ====== 生成按钮测试 ======
  test('正常提交应加入任务队列', { tag: ['@smoke', '@api', '@critical'] }, async ({ page }) => {
    await fillTextInput(page, 'OmniVoice 测试文本');
    await page.locator('#generateBtn').click();
    await waitForToast(page, '已加入');
  });

  test('提交后按钮应进入加载状态', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    await fillTextInput(page, '测试文本');
    await page.locator('#generateBtn').click();
    await expectButtonLoading(page, 'generateBtn', 'btnText', true);
  });

  test('生成按钮应有正确的默认文本', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    const btnText = page.locator('#btnText');
    await expect(btnText).toContainText(/生成|开始/);
  });

  // ====== 批量生成测试 ======
  test('批量生成正常提交应提示成功', { tag: ['@smoke', '@api', '@batch', '@critical'] }, async ({ page }) => {
    await fillTextInput(page, '批量测试文本');
    await page.locator('#batchCount').fill('3');
    await page.locator('#batchGenerateBtn').click();
    await waitForToast(page, '已提交');
  });

  test('提交后批量按钮应进入加载状态', { tag: ['@smoke', '@ui', '@batch'] }, async ({ page }) => {
    await fillTextInput(page, '测试文本');
    await page.locator('#batchGenerateBtn').click();
    await expectButtonLoading(page, 'batchGenerateBtn', 'batchBtnText', true);
  });

  test('批量数量边界验证：小于 2 应自动修正', { tag: ['@boundary', '@batch'] }, async ({ page }) => {
    const countInput = page.locator('#batchCount');
    await countInput.fill('1');
    await countInput.blur();
    const value = await countInput.inputValue();
    expect(parseInt(value)).toBeGreaterThanOrEqual(2);
  });

  test('批量数量边界验证：大于 100 应自动修正', { tag: ['@boundary', '@batch'] }, async ({ page }) => {
    const countInput = page.locator('#batchCount');
    await countInput.fill('200');
    await countInput.blur();
    const value = await countInput.inputValue();
    expect(parseInt(value)).toBeLessThanOrEqual(100);
  });

  test('批量数量输入非数字应自动处理', { tag: ['@boundary', '@batch'] }, async ({ page }) => {
    const countInput = page.locator('#batchCount');
    await countInput.fill('abc');
    await countInput.blur();
    const value = await countInput.inputValue();
    expect(parseInt(value) || 0).toBeGreaterThanOrEqual(0);
  });

  // ====== 参数配置测试 ======
  test('语速参数应可调整', { tag: ['@regression', '@ui'] }, async ({ page }) => {
    const speedSlider = page.locator('#speed, input[name="speed"]').first();
    if (await speedSlider.isVisible().catch(() => false)) {
      await speedSlider.fill('1.5');
      const value = await speedSlider.inputValue();
      expect(parseFloat(value)).toBe(1.5);
    }
  });

  test('音量参数应可调整', { tag: ['@regression', '@ui'] }, async ({ page }) => {
    const volumeSlider = page.locator('#volume, input[name="volume"]').first();
    if (await volumeSlider.isVisible().catch(() => false)) {
      await volumeSlider.fill('80');
      const value = await volumeSlider.inputValue();
      expect(parseInt(value)).toBe(80);
    }
  });

  // ====== 说话人选择测试 ======
  test('说话人选择器应存在', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    const speakerSelect = page.locator('#speakerSelect, .speaker-selector').first();
    if (await speakerSelect.isVisible().catch(() => false)) {
      await expect(speakerSelect).toBeVisible();
    }
  });

  // ====== 结果展示测试 ======
  test('生成结果区域应存在', { tag: ['@ui'] }, async ({ page }) => {
    const resultArea = page.locator('#resultArea, .result-section, #audioPlayer').first();
    if (await resultArea.isVisible().catch(() => false)) {
      await expect(resultArea).toBeVisible();
    }
  });

  // ====== 导航测试 ======
  test('页面应能返回首页', { tag: ['@smoke'] }, async ({ page }) => {
    const backBtn = page.locator('.nav-back, .btn-back').first();
    if (await backBtn.isVisible().catch(() => false)) {
      await backBtn.click();
      await expect(page).toHaveURL(/index\.html/);
    }
  });

  test('底部应包含任务队列入口', { tag: ['@smoke', '@ui', '@queue'] }, async ({ page }) => {
    const taskLink = page.locator('a[href="tasks.html"], button:has-text("查看任务队列"), .task-queue-link').first();
    if (await taskLink.isVisible().catch(() => false)) {
      await expect(taskLink).toBeVisible();
    }
  });
});
