/**
 * Qwen3-TTS 页面按键功能测试
 * 覆盖模式切换、生成、批量生成、参数验证
 */

const { test, expect } = require('@playwright/test');
const {
  MockAPI,
  navigateToModelPage,
  fillTextInput,
  waitForToast,
  waitForStatusMessage,
  expectButtonLoading,
  switchMode,
  setAuthState,
  navigateWithAuth,
} = require('./utils');

test.describe('Qwen3-TTS 页面', { tag: ['@regression'] }, () => {
  test.beforeEach(async ({ page }) => {
    const mock = new MockAPI(page);
    await mock.mockAllApis();
    await setAuthState(page);
    await navigateToModelPage(page, 'qwen3tts');
  });

  // ====== 基础 UI 测试 ======
  test('页面应加载所有核心元素', { tag: ['@smoke', '@ui', '@critical'] }, async ({ page }) => {
    await expect(page.locator('#genText')).toBeVisible();
    await expect(page.locator('#generateBtn')).toBeVisible();
    await expect(page.locator('#batchGenerateBtn')).toBeVisible();
    await expect(page.locator('#batchCount')).toBeVisible();
    await expect(page.locator('.mode-tab')).toHaveCount(3);
  });

  test('模式标签应包含所有模式', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    const modes = ['voice_clone', 'custom_voice', 'voice_design'];
    for (const mode of modes) {
      const tab = page.locator(`.mode-tab[data-mode="${mode}"]`);
      await expect(tab).toBeVisible();
    }
  });

  // ====== 模式切换测试 ======
  test('voice_clone 模式应为默认激活状态', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    const voiceCloneTab = page.locator('.mode-tab[data-mode="voice_clone"]');
    await expect(voiceCloneTab).toHaveClass(/active/);
  });

  test('切换到 voice_clone 模式应显示说话人选择', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    await switchMode(page, 'voice_clone');
    await expect(page.locator('#speakerSelect')).toBeVisible();
    await expect(page.locator('.speaker-section, .speaker-select-section').first()).toBeVisible();
  });

  test('切换到 custom_voice 模式应显示预设音色', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    await switchMode(page, 'custom_voice');
    await expect(page.locator('.speaker-card, .preset-speaker').first()).toBeVisible();
  });

  test('切换到 voice_design 模式应显示音色描述输入', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    await switchMode(page, 'voice_design');
    await expect(page.locator('#voiceDesignPrompt')).toBeVisible();
  });

  // ====== 生成按钮测试 ======
  test('空文本提交应提示错误', { tag: ['@smoke', '@boundary'] }, async ({ page }) => {
    await fillTextInput(page, '');
    await page.locator('#generateBtn').click();
    await waitForToast(page, '请输入', true);
    await waitForStatusMessage(page, '请输入', true);
  });

  test('voice_clone 模式下未选择说话人应提示错误', { tag: ['@smoke', '@boundary'] }, async ({ page }) => {
    await switchMode(page, 'voice_clone');
    await fillTextInput(page, '测试文本');
    await page.evaluate(() => {
      const select = document.getElementById('speakerSelect');
      if (select) select.value = '';
    });
    await page.locator('#generateBtn').click();
    await waitForToast(page, '说话人', true);
  });

  test('voice_design 模式下未输入描述应提示错误', { tag: ['@smoke', '@boundary'] }, async ({ page }) => {
    await switchMode(page, 'voice_design');
    await fillTextInput(page, '测试文本');
    await page.locator('#voiceDesignPrompt').fill('');
    await page.locator('#generateBtn').click();
    await waitForToast(page, '音色描述', true);
  });

  test('正常提交应加入任务队列并提示成功', { tag: ['@smoke', '@api', '@critical'] }, async ({ page }) => {
    await fillTextInput(page, '这是一个测试语音合成文本');
    await page.locator('#generateBtn').click();
    await waitForToast(page, '已加入');
  });

  test('提交后按钮应进入加载状态', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    await fillTextInput(page, '测试文本');
    await page.locator('#generateBtn').click();
    await expectButtonLoading(page, 'generateBtn', 'btnText', true);
  });

  test('提交后应开始轮询任务状态', { tag: ['@regression', '@api'] }, async ({ page }) => {
    await fillTextInput(page, '测试文本');
    await page.locator('#generateBtn').click();
    await waitForToast(page, '队列');
  });

  // ====== 批量生成测试 ======
  test('批量生成按钮应存在且可点击', { tag: ['@smoke', '@ui', '@batch'] }, async ({ page }) => {
    await expect(page.locator('#batchGenerateBtn')).toBeVisible();
    await expect(page.locator('#batchBtnText')).toContainText('批量生成');
  });

  test('批量生成空文本应提示错误', { tag: ['@smoke', '@boundary', '@batch'] }, async ({ page }) => {
    await fillTextInput(page, '');
    await page.locator('#batchGenerateBtn').click();
    await waitForToast(page, '请输入', true);
  });

  test('批量生成正常提交应提示成功', { tag: ['@smoke', '@api', '@batch', '@critical'] }, async ({ page }) => {
    await fillTextInput(page, '批量测试文本');
    await page.locator('#batchCount').fill('5');
    await page.locator('#batchGenerateBtn').click();
    await waitForToast(page, '已提交');
  });

  test('批量生成应禁用按钮并显示加载状态', { tag: ['@smoke', '@ui', '@batch'] }, async ({ page }) => {
    await fillTextInput(page, '批量测试文本');
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

  // ====== 模型配置测试 ======
  test('应能切换模型大小', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    const sizes = ['0.6B', '1.7B'];
    for (const size of sizes) {
      const radio = page.locator(`input[name="modelSize"][value="${size}"]`);
      if (await radio.isVisible().catch(() => false)) {
        await radio.check();
        await expect(radio).toBeChecked();
      }
    }
  });

  test('底部应包含任务队列入口', { tag: ['@smoke', '@ui', '@queue'] }, async ({ page }) => {
    const taskLink = page.locator('a[href="tasks.html"], button:has-text("查看任务队列")').first();
    await expect(taskLink).toBeVisible();
  });
});
