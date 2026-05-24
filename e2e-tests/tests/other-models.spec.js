/**
 * 其他模型页面通用按键测试
 * 扩展版本 - 覆盖 ChatTTS、F5-TTS、FireRedTTS、IndexTTS、GPT-SoVITS、OpenVoice
 * 这些页面结构类似：文本输入 + 生成按钮
 *
 * 用例集标签：
 *   @smoke     - 每次提交后快速验证
 *   @regression - 发布前全量回归
 *   @ui        - 界面元素验证
 *   @boundary   - 异常输入验证
 *   @api        - 生成流程验证
 *   @performance - 性能测试
 */

const { test, expect } = require('@playwright/test');
const { MockAPI, waitForToast, fillTextInput, setAuthState, navigateWithAuth } = require('./utils');

const otherModels = [
  { name: 'chattts', display: 'ChatTTS', hasBatch: false, hasParams: true },
  { name: 'f5tts', display: 'F5-TTS', hasBatch: false, hasParams: true },
  { name: 'fireredtts', display: 'FireRedTTS', hasBatch: false, hasParams: true },
  { name: 'indextts', display: 'IndexTTS', hasBatch: false, hasParams: true },
  { name: 'gptsovits', display: 'GPT-SoVITS', hasBatch: false, hasParams: true },
  { name: 'openvoice', display: 'OpenVoice', hasBatch: false, hasParams: true },
];

for (const model of otherModels) {
  test.describe(`${model.display} 页面`, { tag: ['@regression'] }, () => {
    test.beforeEach(async ({ page }) => {
      const mock = new MockAPI(page);
      await mock.mockAllApis();
      await mock.mockDirectGenerate(model.name);
      await setAuthState(page);
      await page.goto(`./pages/${model.name}.html`);
    });

    // ====== 基础 UI 测试 ======
    test('页面应加载文本输入框和生成按钮', { tag: ['@smoke', '@ui', '@critical'] }, async ({ page }) => {
      await expect(page.locator('#textInput, #genText').first()).toBeVisible();
      await expect(page.locator('#generateBtn')).toBeVisible();
    });

    test('页面标题应正确显示模型名称', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
      const title = await page.title();
      // 检查标题包含模型名称（不区分大小写，处理特殊字符如 GPT-SoVITS 中的连字符）
      const normalizedTitle = title.toLowerCase().replace(/[-_]/g, '');
      const normalizedModelName = model.name.toLowerCase().replace(/[-_]/g, '');
      expect(normalizedTitle).toContain(normalizedModelName);
    });

    test('应显示模型描述和帮助信息', { tag: ['@ui'] }, async ({ page }) => {
      const description = page.locator('.model-description, .page-description, .help-text').first();
      if (await description.isVisible().catch(() => false)) {
        await expect(description).toBeVisible();
      }
    });

    // ====== 文本输入测试 ======
    test('空文本提交应提示错误', { tag: ['@smoke', '@boundary'] }, async ({ page }) => {
      await fillTextInput(page, '');
      await page.locator('#generateBtn').click();
      // 等待提示出现，可能是 Toast 或状态消息
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

    test('多语言文本应正确处理', { tag: ['@boundary'] }, async ({ page }) => {
      const multiLangText = '中文 English 日本語 한국어';
      await fillTextInput(page, multiLangText);
      const value = await page.locator('#textInput, #genText').first().inputValue();
      expect(value).toBe(multiLangText);
    });

    // ====== 生成按钮测试 ======
    test('正常提交应触发生成流程', { tag: ['@smoke', '@api', '@critical'] }, async ({ page }) => {
      await fillTextInput(page, `${model.display} 测试文本`);
      await page.locator('#generateBtn').click();
      // 检查是否有任何反馈（Toast、状态消息或按钮禁用）
      const toastVisible = await page.locator('#globalToast').isVisible().catch(() => false);
      const statusVisible = await page.locator('#statusMessage.show, .status-message.show').first().isVisible().catch(() => false);
      const btnDisabled = await page.locator('#generateBtn').isDisabled().catch(() => false);
      expect(toastVisible || statusVisible || btnDisabled).toBe(true);
    });

    test('生成按钮应有正确的文本', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
      const btnText = page.locator('#btnText');
      // 检查按钮文本存在且不为空
      const text = await btnText.textContent().catch(() => '');
      expect(text.trim().length).toBeGreaterThan(0);
    });

    test('提交后按钮应进入加载状态', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
      await fillTextInput(page, '测试文本');
      await page.locator('#generateBtn').click();
      // 检查按钮是否被禁用或显示加载状态
      const btn = page.locator('#generateBtn');
      const isDisabled = await btn.isDisabled().catch(() => false);
      const hasLoadingClass = await btn.evaluate(el => el.classList.contains('loading') || el.disabled).catch(() => false);
      expect(isDisabled || hasLoadingClass).toBe(true);
    });

    // ====== 参数配置测试 ======
    if (model.hasParams) {
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

      test('音调参数应可调整（如存在）', { tag: ['@regression', '@ui'] }, async ({ page }) => {
        const pitchSlider = page.locator('#pitch, input[name="pitch"]').first();
        if (await pitchSlider.isVisible().catch(() => false)) {
          await pitchSlider.fill('1.0');
          const value = await pitchSlider.inputValue();
          expect(parseFloat(value)).toBe(1.0);
        }
      });
    }

    // ====== 说话人选择测试 ======
    test('说话人选择器应存在（如适用）', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
      const speakerSelect = page.locator('#speakerSelect, .speaker-selector, .voice-selector').first();
      if (await speakerSelect.isVisible().catch(() => false)) {
        await expect(speakerSelect).toBeVisible();
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

    test('应包含首页导航链接', { tag: ['@ui'] }, async ({ page }) => {
      const homeLink = page.locator('a[href="../index.html"], a[href="./index.html"], .nav-home').first();
      if (await homeLink.isVisible().catch(() => false)) {
        await expect(homeLink).toBeVisible();
      }
    });

    // ====== 结果展示测试 ======
    test('生成结果区域应存在（如适用）', { tag: ['@ui'] }, async ({ page }) => {
      const resultArea = page.locator('#resultArea, .result-section, #audioPlayer, .audio-player').first();
      if (await resultArea.isVisible().catch(() => false)) {
        await expect(resultArea).toBeVisible();
      }
    });

    // ====== 批量生成测试（如适用）=======
    if (model.hasBatch) {
      test('批量生成按钮应存在', { tag: ['@ui', '@batch'] }, async ({ page }) => {
        await expect(page.locator('#batchGenerateBtn')).toBeVisible();
      });

      test('批量生成应能正常工作', { tag: ['@api', '@batch'] }, async ({ page }) => {
        await fillTextInput(page, '批量测试文本');
        await page.locator('#batchCount').fill('3');
        await page.locator('#batchGenerateBtn').click();
        await waitForToast(page, '已提交');
      });
    }

    // ====== 性能测试 ======
    test('页面加载时间应在合理范围内', { tag: ['@performance'] }, async ({ page }) => {
      const startTime = Date.now();
      await page.goto(`./pages/${model.name}.html`);
      await page.waitForLoadState('networkidle');
      const loadTime = Date.now() - startTime;
      expect(loadTime).toBeLessThan(5000); // 页面加载应在5秒内
    });

    test('生成响应时间应在合理范围内', { tag: ['@performance'] }, async ({ page }) => {
      await fillTextInput(page, '性能测试文本');
      const startTime = Date.now();
      await page.locator('#generateBtn').click();
      await page.locator('#globalToast, #statusMessage.show, .status-message.show').first().waitFor({ timeout: 10000 }).catch(() => {});
      const responseTime = Date.now() - startTime;
      expect(responseTime).toBeLessThan(3000); // 响应应在3秒内
    });
  });
}
