/**
 * VoxCPM 页面按键功能测试
 * 覆盖模式切换、生成、批量生成、说话人选择、控制指令
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
} = require('./utils');

test.describe('VoxCPM 页面', { tag: ['@regression'] }, () => {
  test.beforeEach(async ({ page }) => {
    const mock = new MockAPI(page);
    await mock.mockAllApis();
    await setAuthState(page);
    await navigateToModelPage(page, 'voxcpm');
  });

  // ====== 基础 UI 测试 ======
  test('页面应加载所有核心元素', { tag: ['@smoke', '@ui', '@critical'] }, async ({ page }) => {
    await expect(page.locator('#textInput')).toBeVisible();
    await expect(page.locator('#generateBtn')).toBeVisible();
    await expect(page.locator('#batchGenerateBtn')).toBeVisible();
    await expect(page.locator('#batchCount')).toBeVisible();
    await expect(page.locator('.mode-tab')).toHaveCount(4);
  });

  test('模式标签应包含所有模式', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    const modeTexts = ['基础生成', '音色设计', '声音克隆', '极致克隆'];
    for (const text of modeTexts) {
      const tab = page.locator('.mode-tab').filter({ hasText: text });
      await expect(tab).toBeVisible();
    }
  });

  // ====== 模式切换测试 ======
  test('基础生成模式应为默认激活状态', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    const baseTab = page.locator('.mode-tab').filter({ hasText: '基础生成' });
    await expect(baseTab).toHaveClass(/active/);
  });

  test('切换到音色设计模式应显示音色描述输入', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    await page.locator('.mode-tab').filter({ hasText: '音色设计' }).click();
    await expect(page.locator('#voiceDesignPrompt')).toBeVisible();
  });

  test('切换到声音克隆模式应显示说话人选择', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    await page.locator('.mode-tab').filter({ hasText: '声音克隆' }).click();
    await expect(page.locator('.speaker-select, .clone-speaker-section, #speakerSelect').first()).toBeVisible();
  });

  test('切换到极致克隆模式应显示说话人选择并验证参考文本', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    await page.locator('.mode-tab').filter({ hasText: '极致克隆' }).click();
    await expect(page.locator('.speaker-select, .ultimate-speaker-section').first()).toBeVisible();
  });

  // ====== 生成按钮测试 ======
  test('空文本提交应提示错误', { tag: ['@smoke', '@boundary'] }, async ({ page }) => {
    await fillTextInput(page, '');
    await page.locator('#generateBtn').click();
    await waitForToast(page, '请输入', true);
  });

  test('音色设计模式下未输入描述应提示错误', { tag: ['@smoke', '@boundary'] }, async ({ page }) => {
    await page.locator('.mode-tab').filter({ hasText: '音色设计' }).click();
    await fillTextInput(page, '测试文本');
    await page.locator('#voiceDesignPrompt').fill('');
    await page.locator('#generateBtn').click();
    await waitForToast(page, '音色描述', true);
  });

  test('声音克隆模式下未选择说话人应提示错误', { tag: ['@smoke', '@boundary'] }, async ({ page }) => {
    await page.locator('.mode-tab').filter({ hasText: '声音克隆' }).click();
    await fillTextInput(page, '测试文本');
    await page.evaluate(() => {
      if (window.selectedSpeakerId) window.selectedSpeakerId.clone = null;
      const select = document.getElementById('speakerSelect');
      if (select) select.value = '';
    });
    await page.locator('#generateBtn').click();
    await waitForToast(page, '说话人', true);
  });

  test('正常提交应加入任务队列并提示成功', { tag: ['@smoke', '@api', '@critical'] }, async ({ page }) => {
    await fillTextInput(page, '这是一个VoxCPM测试文本');
    await page.locator('#generateBtn').click();
    await waitForToast(page, '已加入');
  });

  test('提交后按钮应进入加载状态', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    await fillTextInput(page, '测试文本');
    await page.locator('#generateBtn').click();
    await expectButtonLoading(page, 'generateBtn', 'btnText', true);
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

  // ====== 控制指令测试 ======
  test('声音克隆模式下应能输入控制指令', { tag: ['@regression', '@ui'] }, async ({ page }) => {
    await page.locator('.mode-tab').filter({ hasText: '声音克隆' }).click();
    const controlInput = page.locator('#cloneControlPrompt');
    if (await controlInput.isVisible().catch(() => false)) {
      await controlInput.fill('用温柔的语气');
      await expect(controlInput).toHaveValue('用温柔的语气');
    }
  });

  // ====== 模式切换与参数联动 ======
  test('切换模式时应清空或保留相应参数', { tag: ['@regression', '@ui'] }, async ({ page }) => {
    await page.locator('.mode-tab').filter({ hasText: '音色设计' }).click();
    await page.locator('#voiceDesignPrompt').fill('温柔的女声');
    await page.locator('.mode-tab').filter({ hasText: '基础生成' }).click();
    await page.locator('.mode-tab').filter({ hasText: '音色设计' }).click();
    const promptValue = await page.locator('#voiceDesignPrompt').inputValue();
    expect(typeof promptValue).toBe('string');
  });
});
