// @ts-check
const { defineConfig, devices } = require('@playwright/test');

/**
 * VersTTS E2E 测试配置
 *
 * 用例集说明：
 *   @smoke     - 冒烟测试：每次代码提交后运行，验证最核心功能
 *   @regression - 回归测试：发布前运行，覆盖全部功能路径
 *   @boundary   - 边界测试：异常输入、极限值、空值等
 *   @ui         - UI 测试：界面元素、样式、布局、交互反馈
 *   @api        - API 集成测试：前后端交互、Mock 验证
 *   @critical   - 核心用例：必须通过的 P0 级用例
 *   @batch      - 批量生成相关测试
 *   @queue      - 任务队列相关测试
 *   @performance - 性能测试
 *
 * 运行方式：
 *   npx playwright test --grep "@smoke"        仅跑冒烟测试
 *   npx playwright test --grep "@regression"   仅跑回归测试
 *   npx playwright test --grep "@boundary"     仅跑边界测试
 *   npx playwright test --grep "@critical"     仅跑核心用例
 *   npx playwright test --grep "@batch"        仅跑批量生成测试
 *   npx playwright test --grep "@queue"        仅跑任务队列测试
 *   npx playwright test --grep "@performance"  仅跑性能测试
 *
 * 测试留痕功能：
 *   - 每个测试步骤自动截图
 *   - 失败时录制视频
 *   - 完整的 Trace 追踪（网络请求、控制台日志、点击事件等）
 *   - HTML 报告包含所有截图和视频
 *
 * @see https://playwright.dev/docs/test-configuration
 */
module.exports = defineConfig({
  testDir: './tests',

  /* 每个 test 文件完全独立并行运行 */
  fullyParallel: true,

  /* CI 上禁止失败重试，本地允许重试 1 次 */
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 0 : 1,

  /* 本地使用 4 个工作进程，CI 使用 1 个 */
  workers: process.env.CI ? 1 : 4,

  /* 报告器配置 - 增强版 */
  reporter: [
    ['html', { 
      open: 'never',
      outputFolder: 'playwright-report',
    }],
    ['list'],
    /* JSON 报告，方便集成 */
    ['json', { outputFile: 'test-results/test-results.json' }],
    /* JUnit 报告，方便 CI 集成 */
    ['junit', { outputFile: 'test-results/junit-results.xml' }],
  ],

  /* 全局测试超时 */
  timeout: 60000,

  /* 期望超时 */
  expect: {
    timeout: 10000,
  },

  /* 共享配置 */
  use: {
    /* 基础 URL：前端页面服务地址 */
    baseURL: process.env.FRONTEND_URL || 'http://localhost:8080',

    /* API 基础 URL：后端服务地址 */
    apiBaseURL: process.env.API_BASE_URL || 'http://localhost:8000',

    /* 
     * Trace 追踪配置 - 最详细的留痕
     * 'on-all-retries' - 每次重试都记录
     * 包含：网络请求、控制台日志、点击事件、页面状态等
     */
    trace: 'on-all-retries',

    /* 
     * 截图配置
     * 'on' - 每个测试都截图
     * 'only-on-failure' - 仅失败时截图
     * 'off' - 不截图
     */
    screenshot: 'on',

    /* 
     * 视频录制配置
     * 'on' - 每个测试都录制
     * 'retain-on-failure' - 失败时保留
     * 'on-first-retry' - 第一次重试时录制
     */
    video: 'on',

    /* 视口大小 */
    viewport: { width: 1440, height: 900 },

    /* 动作超时 */
    actionTimeout: 10000,

    /* 导航超时 */
    navigationTimeout: 15000,

    /* 测试上下文选项 */
    contextOptions: {
      /* 记录所有控制台日志 */
      recordConsole: true,
      /* 记录所有网络请求 */
      recordNetwork: true,
    },

    /* 测试信息 */
    testIdAttribute: 'data-testid',
  },

  /* 项目配置：按用例集 + 浏览器矩阵运行 */
  projects: [
    /* ====== 全量回归 - 多浏览器 ====== */
    {
      name: 'regression-chromium',
      use: { 
        ...devices['Desktop Chrome'],
        /* 回归测试记录更详细的 trace */
        trace: 'on',
      },
    },
    {
      name: 'regression-firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'regression-webkit',
      use: { ...devices['Desktop Safari'] },
    },

    /* ====== 冒烟测试 - 仅 Chromium（最快）====== */
    {
      name: 'smoke-chromium',
      use: { 
        ...devices['Desktop Chrome'],
        /* 冒烟测试也记录详细 trace */
        trace: 'on',
      },
      grep: /@smoke/,
    },

    /* ====== 边界测试 - 仅 Chromium ====== */
    {
      name: 'boundary-chromium',
      use: { ...devices['Desktop Chrome'] },
      grep: /@boundary/,
    },

    /* ====== 核心用例 - 多浏览器 ====== */
    {
      name: 'critical-chromium',
      use: { 
        ...devices['Desktop Chrome'],
        /* 核心用例记录最详细的 trace */
        trace: 'on',
        screenshot: 'on',
        video: 'on',
      },
      grep: /@critical/,
    },
    {
      name: 'critical-firefox',
      use: { ...devices['Desktop Firefox'] },
      grep: /@critical/,
    },

    /* ====== UI 测试 - 仅 Chromium ====== */
    {
      name: 'ui-chromium',
      use: { 
        ...devices['Desktop Chrome'],
        /* UI 测试需要截图 */
        screenshot: 'on',
      },
      grep: /@ui/,
    },

    /* ====== API 集成测试 - 仅 Chromium ====== */
    {
      name: 'api-chromium',
      use: { ...devices['Desktop Chrome'] },
      grep: /@api/,
    },

    /* ====== 性能测试 - 仅 Chromium ====== */
    {
      name: 'performance-chromium',
      use: { 
        ...devices['Desktop Chrome'],
        /* 性能测试不需要视频 */
        video: 'off',
        screenshot: 'off',
      },
      grep: /@performance/,
    },
  ],

  /* 输出目录配置 */
  outputDir: 'test-results/',

  /* 本地开发服务器配置（可选） */
  webServer: process.env.SKIP_WEB_SERVER
    ? undefined
    : {
        command: 'npx http-server ../frontend -p 8080 --cors',
        url: 'http://localhost:8080',
        reuseExistingServer: !process.env.CI,
        timeout: 120000,
      },
});
