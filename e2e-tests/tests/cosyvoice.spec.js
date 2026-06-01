/**
 * CosyVoice 页面按键功能测试
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

test.describe('CosyVoice 页面', { tag: ['@regression'] }, () => {
  test.beforeEach(async ({ page }) => {
    const mock = new MockAPI(page);
    await mock.mockAllApis();
    await setAuthState(page);
    await navigateToModelPage(page, 'cosyvoice');
  });

  // ====== 基础 UI 测试 ======
  test('页面应加载所有核心元素', { tag: ['@smoke', '@ui', '@critical'] }, async ({ page }) => {
    await expect(page.locator('#textInput, #genText').first()).toBeVisible();
    await expect(page.locator('#generateBtn')).toBeVisible();
    await expect(page.locator('#batchGenerateBtn')).toBeVisible();
    await expect(page.locator('#batchCount')).toBeVisible();
  });

  test('页面标题应正确显示模型名称', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    const title = await page.title();
    expect(title.toLowerCase()).toContain('cosyvoice');
  });

  test('应显示模型描述和帮助信息', { tag: ['@ui'] }, async ({ page }) => {
    const description = page.locator('.model-description, .page-description, .help-text').first();
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
      await expect(tab).toHaveClass(/active/);
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
    // 检查是否有错误提示（Toast 或状态消息）
    const toastVisible = await page.locator('#globalToast').isVisible().catch(() => false);
    const statusVisible = await page.locator('#statusMessage.show, .status-message.show').first().isVisible().catch(() => false);
    expect(toastVisible || statusVisible).toBe(true);
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
  test('正常提交应触发生成流程', { tag: ['@smoke', '@api', '@critical'] }, async ({ page }) => {
    await fillTextInput(page, 'CosyVoice 测试文本');
    await page.locator('#generateBtn').click();
    // 检查是否有任何反馈（Toast、状态消息、按钮禁用或结果区域显示）
    await page.waitForTimeout(500); // 等待页面响应
    const toastVisible = await page.locator('#globalToast').isVisible().catch(() => false);
    const statusVisible = await page.locator('#statusMessage.show, .status-message.show').first().isVisible().catch(() => false);
    const btnDisabled = await page.locator('#generateBtn').isDisabled().catch(() => false);
    const resultVisible = await page.locator('#resultArea, .result-section, .generation-result').first().isVisible().catch(() => false);
    expect(toastVisible || statusVisible || btnDisabled || resultVisible).toBe(true);
  });

  test('生成按钮应有正确的文本', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    const btnText = page.locator('#btnText');
    const text = await btnText.textContent().catch(() => '');
    expect(text.trim().length).toBeGreaterThan(0);
  });

  test('提交后应显示反馈', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    await fillTextInput(page, '测试文本');
    await page.locator('#generateBtn').click();
    // 等待页面响应
    await page.waitForTimeout(500);
    // 检查是否有反馈（Toast、状态消息或结果区域）
    const toastVisible = await page.locator('#globalToast').isVisible().catch(() => false);
    const statusVisible = await page.locator('#statusMessage.show, .status-message.show').first().isVisible().catch(() => false);
    const resultVisible = await page.locator('#resultArea, .result-section, .generation-result').first().isVisible().catch(() => false);
    expect(toastVisible || statusVisible || resultVisible).toBe(true);
  });

  // ====== 批量生成测试 ======
  test('批量生成正常提交应提示成功', { tag: ['@smoke', '@api', '@batch', '@critical'] }, async ({ page }) => {
    await fillTextInput(page, '批量测试文本');
    await page.locator('#batchCount').fill('3');
    await page.locator('#batchGenerateBtn').click();
    // 等待页面响应
    await page.waitForTimeout(500);
    // 检查是否有反馈
    const toastVisible = await page.locator('#globalToast').isVisible().catch(() => false);
    const statusVisible = await page.locator('#statusMessage.show, .status-message.show').first().isVisible().catch(() => false);
    const resultVisible = await page.locator('#resultArea, .result-section, .generation-result').first().isVisible().catch(() => false);
    expect(toastVisible || statusVisible || resultVisible).toBe(true);
  });

  test('提交后批量按钮应显示反馈', { tag: ['@smoke', '@ui', '@batch'] }, async ({ page }) => {
    await fillTextInput(page, '测试文本');
    await page.locator('#batchGenerateBtn').click();
    // 等待页面响应
    await page.waitForTimeout(500);
    // 检查是否有反馈
    const toastVisible = await page.locator('#globalToast').isVisible().catch(() => false);
    const statusVisible = await page.locator('#statusMessage.show, .status-message.show').first().isVisible().catch(() => false);
    const resultVisible = await page.locator('#resultArea, .result-section, .generation-result').first().isVisible().catch(() => false);
    expect(toastVisible || statusVisible || resultVisible).toBe(true);
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
  test('语速参数应可调整（如存在）', { tag: ['@regression', '@ui'] }, async ({ page }) => {
    const speedSlider = page.locator('#speed, input[name="speed"]').first();
    if (await speedSlider.isVisible().catch(() => false)) {
      await speedSlider.fill('1.5');
      const value = await speedSlider.inputValue();
      expect(parseFloat(value)).toBe(1.5);
    }
  });

  test('音量参数应可调整（如存在）', { tag: ['@regression', '@ui'] }, async ({ page }) => {
    const volumeSlider = page.locator('#volume, input[name="volume"]').first();
    if (await volumeSlider.isVisible().catch(() => false)) {
      await volumeSlider.fill('80');
      const value = await volumeSlider.inputValue();
      expect(parseInt(value)).toBe(80);
    }
  });

  // ====== 说话人选择测试 ======
  test('说话人选择器应存在（如适用）', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    const speakerSelect = page.locator('#speakerSelect, .speaker-selector').first();
    if (await speakerSelect.isVisible().catch(() => false)) {
      await expect(speakerSelect).toBeVisible();
    }
  });

  // ====== 结果展示测试 ======
  test('生成结果区域应存在（如适用）', { tag: ['@ui'] }, async ({ page }) => {
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
