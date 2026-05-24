# VersTTS 前端 E2E 自动化测试

## 用例集设计

测试用例按功能维度划分为 **6 大用例集**，每个用例可携带多个标签，便于按需组合执行。

| 用例集 | 标签 | 说明 | 执行频率 |
|--------|------|------|----------|
| **冒烟测试** | `@smoke` | 每次代码提交后运行，验证最核心功能路径（页面加载、生成提交、Toast提示） | 每次 commit |
| **回归测试** | `@regression` | 发布前全量运行，覆盖所有功能路径 | 每次发布前 |
| **边界测试** | `@boundary` | 异常输入、空值、极限值验证 | 每次修改输入校验后 |
| **UI 测试** | `@ui` | 界面元素可见性、样式、布局、按钮状态变化 | 每次修改样式后 |
| **API 集成测试** | `@api` | 前后端交互、Mock 验证、请求响应 | 每次修改接口后 |
| **核心用例** | `@critical` | P0 级必须通过的用例（生成成功、批量生成、试听） | 每次关键修复后 |

### 业务维度标签

| 标签 | 说明 |
|------|------|
| `@batch` | 批量生成相关测试 |
| `@queue` | 任务队列相关测试 |

---

## 快速开始

```bash
# 1. 进入测试目录
cd e2e-tests

# 2. 安装依赖
npm install

# 3. 安装浏览器（首次需要）
npx playwright install

# 4. 运行测试
npm test
```

---

## 按用例集执行

### 冒烟测试（每次提交后）
```bash
# 快速验证核心功能
npm run smoke

# 带界面运行（便于调试）
npm run smoke:headed

# UI 模式（可逐条执行）
npm run smoke:ui
```

### 回归测试（发布前全量）
```bash
# 全量回归 - 所有浏览器
npm run regression

# 仅 Chromium（最快）
npm run regression:chromium

# 仅 Firefox
npm run regression:firefox

# 仅 WebKit
npm run regression:webkit
```

### 按标签单独执行
```bash
# 边界测试
npm run boundary

# UI 测试
npm run ui

# API 集成测试
npm run api

# 核心用例
npm run critical

# 批量生成相关
npm run batch

# 任务队列相关
npm run queue
```

### 组合标签（命令行）
```bash
# 同时满足多个标签（与关系）
npx playwright test --grep "@smoke @critical"

# 满足任一标签（或关系）
npx playwright test --grep "@smoke|@critical"

# 排除特定标签
npx playwright test --grep-invert "@boundary"
```

---

## 用例标签分配原则

### `@smoke`（冒烟测试）
每次代码提交后必跑的用例，覆盖最核心的用户路径：
- 首页模型卡片可见
- 各模型页面核心元素加载
- 模式切换 UI 更新
- 空文本/必填项校验
- 正常提交生成成功
- 批量生成提交成功
- 任务队列页面加载
- 全局 Toast 显示

### `@regression`（回归测试）
发布前全量运行，覆盖所有功能路径：
- 包含所有 `@smoke` 用例
- 额外的边界场景
- 模式参数联动
- 轮询状态更新
- 批量清理操作

### `@boundary`（边界测试）
验证异常输入和边界条件：
- 空文本提交
- 未选择说话人
- 未输入音色描述
- 批量数量 < 2 或 > 100

### `@ui`（UI 测试）
验证界面元素和交互反馈：
- 元素可见性、数量
- 按钮加载状态变化
- 模式标签激活样式
- 进度条显示
- 试听播放器显示
- Toast 位置和样式

### `@api`（API 集成测试）
验证前后端交互：
- 健康检查 API
- 任务提交 API
- 批量生成 API
- 任务状态查询
- 取消/重试/删除 API

### `@critical`（核心用例）
P0 级必须通过的用例：
- 各模型页面核心元素加载
- 正常生成提交成功
- 批量生成提交成功
- 任务队列试听播放器显示
- 全局 Toast 功能正常

### `@batch`（批量生成）
所有与批量生成功能相关的用例：
- 批量按钮存在性
- 批量空文本校验
- 批量提交成功
- 批量按钮加载状态
- 批量数量边界
- 批量结果试听

### `@queue`（任务队列）
所有与任务队列相关的用例：
- 任务队列页面加载
- 刷新按钮
- 状态筛选
- 自动刷新开关
- 试听/下载/取消/重试/删除
- 批量清理

---

## 配置说明

### playwright.config.js

```javascript
projects: [
  // 全量回归 - 多浏览器
  { name: 'regression-chromium', use: { ...devices['Desktop Chrome'] } },
  { name: 'regression-firefox',  use: { ...devices['Desktop Firefox'] } },
  { name: 'regression-webkit',   use: { ...devices['Desktop Safari'] } },

  // 冒烟测试 - 仅 Chromium（最快）
  { name: 'smoke-chromium', use: { ...devices['Desktop Chrome'] }, grep: /@smoke/ },

  // 边界测试 - 仅 Chromium
  { name: 'boundary-chromium', use: { ...devices['Desktop Chrome'] }, grep: /@boundary/ },

  // 核心用例 - 多浏览器
  { name: 'critical-chromium', use: { ...devices['Desktop Chrome'] }, grep: /@critical/ },
  { name: 'critical-firefox',  use: { ...devices['Desktop Firefox'] }, grep: /@critical/ },

  // UI 测试 - 仅 Chromium
  { name: 'ui-chromium', use: { ...devices['Desktop Chrome'] }, grep: /@ui/ },

  // API 集成测试 - 仅 Chromium
  { name: 'api-chromium', use: { ...devices['Desktop Chrome'] }, grep: /@api/ },
]
```

---

## Mock 设计

所有测试使用 `MockAPI` 工具类拦截后端请求，无需启动真实后端服务即可验证前端交互逻辑：

| Mock 接口 | 用途 |
|-----------|------|
| `mockHealth()` | 健康检查 |
| `mockSpeakers()` | 说话人列表 |
| `mockTaskSubmit()` | 任务提交 |
| `mockBatchGenerate()` | 批量生成 |
| `mockDirectGenerate()` | 直接生成（非队列模型） |
| `mockTaskStatus()` | 任务状态查询 |
| `mockTaskList()` | 任务列表 |
| `mockTaskDownload()` | 任务下载 |
| `mockTaskCancel()` | 取消任务 |
| `mockTaskRetry()` | 重试任务 |
| `mockTaskDelete()` | 删除任务 |
| `mockAllApis()` | 一键启用所有 Mock |

---

## CI/CD 集成建议

```yaml
# .github/workflows/e2e.yml 示例
name: E2E Tests
on: [push, pull_request]
jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: cd e2e-tests && npm install && npx playwright install
      - run: cd e2e-tests && npm run smoke

  regression:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: cd e2e-tests && npm install && npx playwright install
      - run: cd e2e-tests && npm run regression
      - uses: actions/upload-artifact@v4
        with:
          name: playwright-report
          path: e2e-tests/playwright-report/
```
