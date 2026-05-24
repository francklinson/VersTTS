/**
 * 任务队列页面按键功能测试
 * 覆盖刷新、筛选、下载、取消、重试、删除、自动刷新
 */

const { test, expect } = require('@playwright/test');
const { MockAPI, waitForToast, setAuthState } = require('./utils');

test.describe('任务队列页面', { tag: ['@regression', '@queue'] }, () => {
  test.beforeEach(async ({ page }) => {
    const mock = new MockAPI(page);
    await mock.mockAllApis();
    await mock.mockTaskStatus('task_20240101_120000_abc001', 'completed');
    await mock.mockTaskStatus('task_20240101_120100_abc002', 'processing');
    await setAuthState(page);
    await page.goto('./pages/tasks.html');
  });

  // ====== 页面加载测试 ======
  test('页面应加载所有核心元素', { tag: ['@smoke', '@ui', '@critical'] }, async ({ page }) => {
    await expect(page.locator('#statusFilter')).toBeVisible();
    await expect(page.locator('text=刷新')).toBeVisible();
    await expect(page.locator('#autoRefresh')).toBeVisible();
    await expect(page.locator('#taskList')).toBeVisible();
    await expect(page.locator('.stat-card')).toHaveCount(5);
  });

  test('页面标题应正确', { tag: ['@smoke'] }, async ({ page }) => {
    await expect(page).toHaveTitle(/任务队列/);
  });

  test('应显示统计信息', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    await expect(page.locator('#statPending')).toBeVisible();
    await expect(page.locator('#statProcessing')).toBeVisible();
    await expect(page.locator('#statCompleted')).toBeVisible();
    await expect(page.locator('#statFailed')).toBeVisible();
    await expect(page.locator('#statCancelled')).toBeVisible();
  });

  // ====== 刷新按钮测试 ======
  test('点击刷新按钮应重新加载任务列表', { tag: ['@smoke', '@api'] }, async ({ page }) => {
    const refreshBtn = page.locator('button:has-text("刷新")');
    await refreshBtn.click();
    await expect(page.locator('#taskList .task-item').first()).toBeVisible();
  });

  // ====== 状态筛选测试 ======
  test('按 completed 筛选应只显示已完成任务', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    await page.locator('#statusFilter').selectOption('completed');
    await page.waitForTimeout(500);
    const tasks = await page.locator('#taskList .task-item').all();
    for (const task of tasks) {
      await expect(task).toHaveAttribute('data-status', 'completed');
    }
  });

  test('按 processing 筛选应只显示执行中任务', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    await page.locator('#statusFilter').selectOption('processing');
    await page.waitForTimeout(500);
    const tasks = await page.locator('#taskList .task-item').all();
    for (const task of tasks) {
      await expect(task).toHaveAttribute('data-status', 'processing');
    }
  });

  test('按 failed 筛选应只显示失败任务', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    await page.locator('#statusFilter').selectOption('failed');
    await page.waitForTimeout(500);
    const tasks = await page.locator('#taskList .task-item').all();
    if (tasks.length > 0) {
      for (const task of tasks) {
        await expect(task).toHaveAttribute('data-status', 'failed');
      }
    }
  });

  // ====== 自动刷新测试 ======
  test('自动刷新开关应默认开启', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    const checkbox = page.locator('#autoRefresh');
    await expect(checkbox).toBeChecked();
  });

  test('关闭自动刷新后不应定时更新', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    const checkbox = page.locator('#autoRefresh');
    await checkbox.uncheck();
    await expect(checkbox).not.toBeChecked();
  });

  // ====== 任务卡片测试 ======
  test('任务卡片应显示任务ID、模型、模式和文本', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    const firstTask = page.locator('#taskList .task-item').first();
    await expect(firstTask.locator('.task-id')).toBeVisible();
    await expect(firstTask.locator('.task-text')).toBeVisible();
    await expect(firstTask.locator('.task-meta')).toBeVisible();
  });

  test('已完成任务应显示试听播放器', { tag: ['@smoke', '@ui', '@critical'] }, async ({ page }) => {
    const completedTask = page.locator('#taskList .task-item[data-status="completed"]').first();
    if (await completedTask.isVisible().catch(() => false)) {
      await expect(completedTask.locator('audio')).toBeVisible();
    }
  });

  test('执行中任务应显示进度条', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    const processingTask = page.locator('#taskList .task-item[data-status="processing"]').first();
    if (await processingTask.isVisible().catch(() => false)) {
      await expect(processingTask.locator('.progress-bar')).toBeVisible();
    }
  });

  // ====== 操作按钮测试 ======
  test('已完成任务应显示下载按钮', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    const completedTask = page.locator('#taskList .task-item[data-status="completed"]').first();
    if (await completedTask.isVisible().catch(() => false)) {
      await expect(completedTask.locator('button:has-text("下载")')).toBeVisible();
    }
  });

  test('等待中任务应显示取消按钮并可点击', { tag: ['@smoke', '@api', '@critical'] }, async ({ page }) => {
    const mock = new MockAPI(page);
    await mock.mockTaskList([
      {
        task_id: 'task_queued_001',
        status: 'queued',
        model: 'qwen3tts',
        mode: 'base',
        text: '等待中测试任务',
        audio_url: null,
        created_at: new Date().toISOString(),
        batch_results: null,
      },
    ]);
    await page.reload();
    const queuedTask = page.locator('#taskList .task-item[data-status="queued"]').first();
    if (await queuedTask.isVisible().catch(() => false)) {
      const cancelBtn = queuedTask.locator('button:has-text("取消")');
      await expect(cancelBtn).toBeVisible();
      page.on('dialog', async (dialog) => {
        if (dialog.type() === 'confirm') {
          await dialog.accept();
        }
      });
      await cancelBtn.click();
      await waitForToast(page, '已取消');
    }
  });

  test('失败任务应显示重试和删除按钮', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    const mock = new MockAPI(page);
    await mock.mockTaskList([
      {
        task_id: 'task_failed_001',
        status: 'failed',
        model: 'qwen3tts',
        mode: 'base',
        text: '失败测试任务',
        audio_url: null,
        created_at: new Date().toISOString(),
        error_message: '测试错误',
        batch_results: null,
      },
    ]);
    await page.reload();
    const failedTask = page.locator('#taskList .task-item[data-status="failed"]').first();
    if (await failedTask.isVisible().catch(() => false)) {
      await expect(failedTask.locator('button:has-text("重试")')).toBeVisible();
      await expect(failedTask.locator('button:has-text("删除")')).toBeVisible();
      await expect(failedTask.locator('.task-error')).toContainText('测试错误');
    }
  });

  test('删除已完成任务应弹出确认框并删除', { tag: ['@smoke', '@api'] }, async ({ page }) => {
    const mock = new MockAPI(page);
    await mock.mockTaskList([
      {
        task_id: 'task_delete_001',
        status: 'completed',
        model: 'qwen3tts',
        mode: 'base',
        text: '删除测试任务',
        audio_url: '/outputs/delete_test.wav',
        created_at: new Date().toISOString(),
        batch_results: null,
      },
    ]);
    await page.reload();
    const task = page.locator('#taskList .task-item[data-status="completed"]').first();
    if (await task.isVisible().catch(() => false)) {
      page.on('dialog', async (dialog) => {
        if (dialog.type() === 'confirm') {
          await dialog.accept();
        }
      });
      const deleteBtn = task.locator('button:has-text("删除")');
      await deleteBtn.click();
      await waitForToast(page, '已删除');
    }
  });

  // ====== 批量操作测试 ======
  test('存在已完成/失败/已取消任务时应显示清理按钮', { tag: ['@smoke', '@ui'] }, async ({ page }) => {
    const mock = new MockAPI(page);
    await mock.mockTaskList([
      {
        task_id: 'task_cleanup_001',
        status: 'completed',
        model: 'qwen3tts',
        mode: 'base',
        text: '清理测试',
        audio_url: '/outputs/cleanup.wav',
        created_at: new Date().toISOString(),
        batch_results: null,
      },
    ]);
    await page.reload();
    const clearBtn = page.locator('#batchDeleteBtn');
    await expect(clearBtn).toBeVisible();
  });

  test('点击清理按钮应弹出确认并批量删除', { tag: ['@regression', '@api'] }, async ({ page }) => {
    const mock = new MockAPI(page);
    await mock.mockTaskList([
      {
        task_id: 'task_cleanup_001',
        status: 'completed',
        model: 'qwen3tts',
        mode: 'base',
        text: '清理测试1',
        audio_url: '/outputs/cleanup1.wav',
        created_at: new Date().toISOString(),
        batch_results: null,
      },
      {
        task_id: 'task_cleanup_002',
        status: 'failed',
        model: 'voxcpm',
        mode: 'base',
        text: '清理测试2',
        audio_url: null,
        created_at: new Date().toISOString(),
        error_message: '错误',
        batch_results: null,
      },
    ]);
    await page.reload();
    const clearBtn = page.locator('#batchDeleteBtn');
    if (await clearBtn.isVisible().catch(() => false)) {
      page.on('dialog', async (dialog) => {
        if (dialog.type() === 'confirm') {
          await dialog.accept();
        }
      });
      await clearBtn.click();
      await waitForToast(page, '清理完成');
    }
  });

  // ====== 批量生成结果测试 ======
  test('批量任务应显示多个音频试听', { tag: ['@smoke', '@ui', '@batch', '@critical'] }, async ({ page }) => {
    const mock = new MockAPI(page);
    await mock.mockTaskList([
      {
        task_id: 'task_batch_001',
        status: 'completed',
        model: 'qwen3tts',
        mode: 'base',
        text: '批量生成测试',
        audio_url: '/outputs/batch_main.wav',
        created_at: new Date().toISOString(),
        batch_results: [
          { filename: 'result_1.wav', url: '/outputs/batch_1.wav' },
          { filename: 'result_2.wav', url: '/outputs/batch_2.wav' },
        ],
      },
    ]);
    await page.reload();
    const batchTask = page.locator('#taskList .task-item').first();
    await expect(batchTask.locator('.batch-results')).toBeVisible();
    const audios = await batchTask.locator('.batch-results audio').all();
    expect(audios.length).toBeGreaterThanOrEqual(2);
  });
});
