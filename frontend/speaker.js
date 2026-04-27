// ==================== 说话人管理功能 ====================

        let selectedAudioFile = null;
        let savedSpeakers = [];
        let currentScriptText = '';  // 当前选中的朗读文本
        let currentScripts = [];     // 缓存的文本列表

        // 加载朗读文本
        async function loadRecordingScripts() {
            const length = document.getElementById('scriptLength').value;
            try {
                const response = await fetch(`${API_BASE}/recording_scripts?length=${length}`);
                if (response.ok) {
                    const data = await response.json();
                    if (data.success) {
                        currentScripts = data.scripts;
                        // 随机选择一条
                        selectRandomScript();
                    }
                }
            } catch (e) {
                console.error('加载朗读文本失败:', e);
                document.getElementById('scriptDisplay').textContent = '加载失败，请刷新重试';
            }
        }

        // 随机选择一条文本
        function selectRandomScript() {
            if (currentScripts.length > 0) {
                const script = currentScripts[Math.floor(Math.random() * currentScripts.length)];
                currentScriptText = script.text;
                document.getElementById('scriptDisplay').textContent = script.text;
                
                // 显示类型标签
                const typeBadge = document.getElementById('scriptTypeBadge');
                typeBadge.textContent = script.type || '通用';
                
                // 根据类型设置不同颜色
                const typeColors = {
                    '唐诗': { bg: '#fee2e2', color: '#dc2626' },
                    '宋词': { bg: '#fce7f3', color: '#db2777' },
                    '散文': { bg: '#e0f2fe', color: '#0369a1' },
                    '名言': { bg: '#fef3c7', color: '#d97706' },
                    '小说': { bg: '#f3e8ff', color: '#7c3aed' },
                    '古文': { bg: '#ecfdf5', color: '#059669' },
                    '现代诗': { bg: '#f0fdf4', color: '#16a34a' },
                    '诗词赏析': { bg: '#eff6ff', color: '#2563eb' },
                    '演讲': { bg: '#fff7ed', color: '#ea580c' },
                    '对话': { bg: '#f1f5f9', color: '#475569' },
                    '新闻': { bg: '#f8fafc', color: '#64748b' }
                };
                const colors = typeColors[script.type] || { bg: '#e0f2fe', color: '#0369a1' };
                typeBadge.style.background = colors.bg;
                typeBadge.style.color = colors.color;
                
                // 显示出处
                document.getElementById('scriptSource').textContent = script.source || '';
                document.getElementById('scriptDuration').textContent = script.duration || '5-8秒';
            }
        }

        // 刷新文本
        function refreshScript() {
            selectRandomScript();
        }

        // 当前激活的子页面
        let currentSpeakerTab = 'record';

        // 切换子页面
        function switchSpeakerTab(tab) {
            currentSpeakerTab = tab;

            // 更新选项卡样式
            const tabRecord = document.getElementById('tabRecord');
            const tabUpload = document.getElementById('tabUpload');

            if (tab === 'record') {
                tabRecord.classList.add('active');
                tabRecord.style.color = '#0ea5e9';
                tabRecord.style.borderBottomColor = '#0ea5e9';
                tabRecord.style.fontWeight = '600';

                tabUpload.classList.remove('active');
                tabUpload.style.color = '#64748b';
                tabUpload.style.borderBottomColor = 'transparent';
                tabUpload.style.fontWeight = 'normal';

                document.getElementById('pageRecord').style.display = 'block';
                document.getElementById('pageUpload').style.display = 'none';
            } else {
                tabUpload.classList.add('active');
                tabUpload.style.color = '#0ea5e9';
                tabUpload.style.borderBottomColor = '#0ea5e9';
                tabUpload.style.fontWeight = '600';

                tabRecord.classList.remove('active');
                tabRecord.style.color = '#64748b';
                tabRecord.style.borderBottomColor = 'transparent';
                tabRecord.style.fontWeight = 'normal';

                document.getElementById('pageRecord').style.display = 'none';
                document.getElementById('pageUpload').style.display = 'block';
            }

            // 重置表单状态
            updateSaveButton();
        }

        // 获取当前页面的名称输入元素ID
        function getSpeakerNameInputId() {
            return currentSpeakerTab === 'record' ? 'speakerNameRecord' : 'speakerNameUpload';
        }

        // 获取当前页面的名称状态元素ID
        function getNameStatusId() {
            return currentSpeakerTab === 'record' ? 'nameStatusRecord' : 'nameStatusUpload';
        }

        // 获取当前页面的保存按钮元素ID
        function getSaveButtonId() {
            return currentSpeakerTab === 'record' ? 'saveSpeakerBtnRecord' : 'saveSpeakerBtnUpload';
        }

        // 获取当前页面的进度条元素ID
        function getProgressElementIds() {
            if (currentSpeakerTab === 'record') {
                return { fill: 'progressFillRecord', text: 'progressTextRecord', container: 'uploadProgressRecord' };
            } else {
                return { fill: 'progressFillUpload', text: 'progressTextUpload', container: 'uploadProgressUpload' };
            }
        }

        // 显示说话人管理模态框
        function showSpeakerManager() {
            document.getElementById('speakerModal').classList.add('active');
            loadSpeakersList();
            loadRecordingScripts(); // 加载朗读文本
            // 默认显示录音页面
            switchSpeakerTab('record');
        }

        // 关闭说话人管理模态框
        function closeSpeakerManager() {
            document.getElementById('speakerModal').classList.remove('active');
            resetSpeakerForm();
        }

        // 处理文件选择
        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                validateAndSetFile(file);
            }
        }

        // 验证并设置文件
        function validateAndSetFile(file) {
            const allowedTypes = ['audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/flac', 'audio/ogg', 'audio/mp4', 'audio/x-m4a'];
            const allowedExtensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a'];
            const fileExt = '.' + file.name.split('.').pop().toLowerCase();

            if (!allowedExtensions.includes(fileExt)) {
                showSpeakerStatus('不支持的音频格式，请上传 MP3、WAV、FLAC、OGG 或 M4A 格式', 'error');
                return;
            }

            selectedAudioFile = file;
            document.getElementById('fileName').textContent = file.name;
            document.getElementById('selectedFile').classList.add('show');
            document.getElementById('uploadArea').classList.add('has-file');
            showSpeakerStatus('');
            updateSaveButton();
        }

        // 移除选中的文件
        function removeSelectedFile() {
            selectedAudioFile = null;
            document.getElementById('audioFileInput').value = '';
            document.getElementById('selectedFile').classList.remove('show');
            document.getElementById('uploadArea').classList.remove('has-file');
            updateSaveButton();
        }

        // 检查说话人名称
        async function checkSpeakerName() {
            const nameInputId = getSpeakerNameInputId();
            const statusId = getNameStatusId();
            const nameInput = document.getElementById(nameInputId);
            const name = nameInput.value.trim();
            const statusEl = document.getElementById(statusId);

            if (!name) {
                statusEl.textContent = '';
                nameInput.classList.remove('error', 'success');
                updateSaveButton();
                return;
            }

            // 检查名称格式
            if (name.length < 2 || name.length > 20) {
                statusEl.textContent = '名称长度应在 2-20 个字符之间';
                statusEl.className = 'name-status error';
                nameInput.classList.add('error');
                nameInput.classList.remove('success');
                updateSaveButton();
                return;
            }

            try {
                const response = await fetch(`${API_BASE}/speakers/check-name`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: `name=${encodeURIComponent(name)}`
                });
                const data = await response.json();

                if (data.exists) {
                    statusEl.textContent = '该名称已被使用';
                    statusEl.className = 'name-status error';
                    nameInput.classList.add('error');
                    nameInput.classList.remove('success');
                } else {
                    statusEl.textContent = '名称可用';
                    statusEl.className = 'name-status success';
                    nameInput.classList.remove('error');
                    nameInput.classList.add('success');
                }
            } catch (error) {
                console.error('检查名称失败:', error);
            }
            updateSaveButton();
        }

        // 更新保存按钮状态
        function updateSaveButton() {
            const nameInputId = getSpeakerNameInputId();
            const statusId = getNameStatusId();
            const btnId = getSaveButtonId();

            const name = document.getElementById(nameInputId).value.trim();
            const btn = document.getElementById(btnId);
            const isNameValid = name.length >= 2 && name.length <= 20 && !document.getElementById(statusId).classList.contains('error');

            // 检查是否有音频文件
            const hasContent = selectedAudioFile !== null;

            btn.disabled = !(hasContent && isNameValid);
        }

        // 显示状态消息
        function showSpeakerStatus(message, type = '') {
            const statusEl = document.getElementById('speakerStatusMessage');
            if (!message) {
                statusEl.classList.remove('show');
                return;
            }
            statusEl.textContent = message;
            statusEl.className = `status-message-speaker show ${type}`;
        }

        // 更新进度条
        function updateProgress(percent, text) {
            const ids = getProgressElementIds();
            document.getElementById(ids.fill).style.width = percent + '%';
            document.getElementById(ids.text).textContent = text;
            if (percent > 0) {
                document.getElementById(ids.container).classList.add('show');
            }
        }

        // ==================== 录音功能 ====================

        let mediaRecorder = null;
        let recordedChunks = [];
        let recordStartTime = null;
        let recordTimerInterval = null;
        let isRecording = false;
        let recordedAudioBlob = null;

        // 切换录音状态
        async function toggleRecording() {
            if (isRecording) {
                stopRecording();
            } else {
                await startRecording();
            }
        }

        // 开始录音
        async function startRecording() {
            try {
                // 请求麦克风权限 - 使用基本配置确保兼容性
                console.log('请求麦克风权限...');
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true
                        // 不指定 sampleRate，使用浏览器默认值
                    }
                });
                console.log('麦克风权限获取成功');

                // 检测浏览器支持的录音格式
                let mimeType = 'audio/webm';
                if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
                    mimeType = 'audio/webm;codecs=opus';
                    console.log('使用音频格式: audio/webm;codecs=opus');
                } else if (MediaRecorder.isTypeSupported('audio/webm')) {
                    mimeType = 'audio/webm';
                    console.log('使用音频格式: audio/webm');
                } else if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
                    mimeType = 'audio/ogg;codecs=opus';
                    console.log('使用音频格式: audio/ogg;codecs=opus');
                } else {
                    console.log('使用默认音频格式: audio/webm');
                }

                // 创建 MediaRecorder
                mediaRecorder = new MediaRecorder(stream, { mimeType });
                recordedChunks = [];
                console.log('MediaRecorder 创建成功, 状态:', mediaRecorder.state);

                mediaRecorder.ondataavailable = (event) => {
                    console.log('收到音频数据, 大小:', event.data.size, '类型:', event.data.type);
                    if (event.data.size > 0) {
                        recordedChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = () => {
                    console.log('录音停止, 收集到的 chunks:', recordedChunks.length);
                    recordedAudioBlob = new Blob(recordedChunks, { type: mimeType });
                    console.log('Blob 创建成功, 大小:', recordedAudioBlob.size, '类型:', recordedAudioBlob.type);
                    showAudioPreview();
                    // 停止所有音频轨道
                    stream.getTracks().forEach(track => track.stop());
                };

                mediaRecorder.onerror = (e) => {
                    console.error('MediaRecorder 错误:', e);
                    showSpeakerStatus('录音过程出错', 'error');
                };

                // 开始录音
                mediaRecorder.start(100); // 每100ms收集一次数据
                console.log('录音开始, 状态:', mediaRecorder.state);
                isRecording = true;
                recordStartTime = Date.now();

                // 更新UI
                updateRecordUI();
                startRecordTimer();

                showSpeakerStatus('正在录音...', 'info');

            } catch (error) {
                console.error('录音启动失败:', error);
                console.error('错误名称:', error.name);
                console.error('错误消息:', error.message);
                console.error('错误堆栈:', error.stack);
                let errorMsg = '录音启动失败';
                if (error.name === 'NotAllowedError') {
                    errorMsg = '请允许使用麦克风权限';
                } else if (error.name === 'NotFoundError') {
                    errorMsg = '未找到麦克风设备';
                } else if (error.name === 'NotReadableError') {
                    errorMsg = '麦克风被其他应用占用';
                } else if (error.message) {
                    errorMsg = '录音启动失败: ' + error.message;
                }
                showSpeakerStatus(errorMsg, 'error');
            }
        }

        // 停止录音
        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                stopRecordTimer();
                updateRecordUI();
                showSpeakerStatus('录音完成', 'success');
            }
        }

        // 更新录音UI
        function updateRecordUI() {
            const btn = document.getElementById('recordBtn');
            const status = document.getElementById('recordStatus');
            const waves = document.getElementById('recordWaves');
            const timer = document.getElementById('recordTimer');

            if (isRecording) {
                btn.classList.add('recording');
                btn.innerHTML = '⏹️';
                status.textContent = '正在录音，点击停止';
                waves.classList.add('show');
                timer.style.display = 'block';
            } else {
                btn.classList.remove('recording');
                btn.innerHTML = '🎤';
                status.textContent = recordedAudioBlob ? '录音完成' : '点击按钮开始录音';
                waves.classList.remove('show');
                if (!recordedAudioBlob) {
                    timer.style.display = 'none';
                }
            }
        }

        // 开始计时器
        function startRecordTimer() {
            recordTimerInterval = setInterval(() => {
                const elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
                const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
                const seconds = (elapsed % 60).toString().padStart(2, '0');
                document.getElementById('recordTimer').textContent = `${minutes}:${seconds}`;
            }, 1000);
        }

        // 停止计时器
        function stopRecordTimer() {
            if (recordTimerInterval) {
                clearInterval(recordTimerInterval);
                recordTimerInterval = null;
            }
        }

        // 将 AudioBuffer 转换为 WAV 格式的 ArrayBuffer
        function audioBufferToWav(buffer) {
            const numOfChannels = buffer.numberOfChannels;
            const sampleRate = buffer.sampleRate;
            const format = 1; // PCM
            const bitDepth = 16;
            const bytesPerSample = bitDepth / 8;
            const blockAlign = numOfChannels * bytesPerSample;
            const byteRate = sampleRate * blockAlign;
            const dataLength = buffer.length * blockAlign;
            const headerLength = 44;
            const arrayBuffer = new ArrayBuffer(headerLength + dataLength);
            const view = new DataView(arrayBuffer);
            let offset = 0;

            function writeString(str) {
                for (let i = 0; i < str.length; i++) {
                    view.setUint8(offset++, str.charCodeAt(i));
                }
            }

            writeString('RIFF');
            view.setUint32(offset, 36 + dataLength, true); offset += 4;
            writeString('WAVE');
            writeString('fmt ');
            view.setUint32(offset, 16, true); offset += 4;
            view.setUint16(offset, format, true); offset += 2;
            view.setUint16(offset, numOfChannels, true); offset += 2;
            view.setUint32(offset, sampleRate, true); offset += 4;
            view.setUint32(offset, byteRate, true); offset += 4;
            view.setUint16(offset, blockAlign, true); offset += 2;
            view.setUint16(offset, bitDepth, true); offset += 2;
            writeString('data');
            view.setUint32(offset, dataLength, true); offset += 4;

            // 写入采样数据
            const channels = [];
            for (let i = 0; i < numOfChannels; i++) {
                channels.push(buffer.getChannelData(i));
            }
            for (let i = 0; i < buffer.length; i++) {
                for (let ch = 0; ch < numOfChannels; ch++) {
                    let sample = Math.max(-1, Math.min(1, channels[ch][i]));
                    sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
                    view.setInt16(offset, sample, true);
                    offset += 2;
                }
            }
            return arrayBuffer;
        }

        // 显示录音预览 - 将 webm 转换为 WAV 以确保浏览器兼容性
        async function showAudioPreview() {
            if (!recordedAudioBlob) {
                console.error('没有录音数据');
                return;
            }

            // 调试信息
            console.log('录音 Blob 信息:', {
                size: recordedAudioBlob.size,
                type: recordedAudioBlob.type,
                chunks: recordedChunks.length
            });

            // 计算录制时长
            let recordedDuration = 0;
            if (recordStartTime) {
                recordedDuration = Math.round((Date.now() - recordStartTime) / 1000);
            }
            // 保底：从计时器显示获取
            if (recordedDuration === 0) {
                const timerText = document.getElementById('recordTimer').textContent;
                const parts = timerText.split(':');
                if (parts.length === 2) {
                    recordedDuration = parseInt(parts[0]) * 60 + parseInt(parts[1]);
                }
            }

            const audio = document.getElementById('recordedAudio');
            const durationEl = document.getElementById('recordDuration');
            const errorMsg = document.getElementById('audioErrorMsg');

            // 先显示录制的时长
            if (durationEl && recordedDuration > 0) {
                durationEl.textContent = recordedDuration;
            }

            // 重置状态
            errorMsg.style.display = 'none';

            // 移除旧的 audio 元素，并释放其 Blob URL
            const audioContainer = audio.parentNode;
            if (audio.src && audio.src.startsWith('blob:')) {
                URL.revokeObjectURL(audio.src);
            }
            audio.remove();

            // 创建新的 audio 元素
            const newAudio = document.createElement('audio');
            newAudio.id = 'recordedAudio';
            newAudio.controls = true;
            newAudio.style.width = '100%';
            newAudio.preload = 'auto';
            newAudio.volume = 1.0;
            newAudio.muted = false;

            // 尝试将 webm 转换为 WAV，确保浏览器能正常播放
            try {
                const arrayBuffer = await recordedAudioBlob.arrayBuffer();
                const audioContext = new AudioContext();
                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                console.log('Web Audio API 解码成功，采样率:', audioBuffer.sampleRate, '通道数:', audioBuffer.numberOfChannels, '时长:', audioBuffer.duration);

                // 检查音量，判断麦克风是否有输入
                const channelData = audioBuffer.getChannelData(0);
                let maxAmp = 0;
                let sumAmp = 0;
                for (let i = 0; i < channelData.length; i++) {
                    const abs = Math.abs(channelData[i]);
                    if (abs > maxAmp) maxAmp = abs;
                    sumAmp += abs;
                }
                const avgAmp = sumAmp / channelData.length;
                console.log('录音音量分析 - 最大振幅:', maxAmp.toFixed(4), '平均振幅:', avgAmp.toFixed(4));
                if (maxAmp < 0.01) {
                    showSpeakerStatus('警告：录音音量极低，请检查麦克风是否正常', 'warning');
                }

                // 转换为 WAV
                const wavBuffer = audioBufferToWav(audioBuffer);
                const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
                const wavUrl = URL.createObjectURL(wavBlob);
                newAudio.src = wavUrl;
                console.log('已转换为 WAV 预览，URL:', wavUrl, '大小:', wavBlob.size);

                // 同时保留原始 webm blob，供后端使用
                // recordedAudioBlob 不变
            } catch (decodeErr) {
                console.error('Web Audio API 解码失败，回退到原始 webm 预览:', decodeErr);
                const audioUrl = URL.createObjectURL(recordedAudioBlob);
                newAudio.src = audioUrl;
                showSpeakerStatus('音频预览可能无法播放，但录音数据仍可使用', 'info');
            }

            // 添加事件监听
            newAudio.onloadedmetadata = function() {
                console.log('音频元数据加载成功，时长:', newAudio.duration, '秒');
                if (durationEl && newAudio.duration && !isNaN(newAudio.duration) && newAudio.duration !== Infinity && newAudio.duration > 0) {
                    durationEl.textContent = Math.round(newAudio.duration);
                }
            };

            newAudio.oncanplay = function() {
                console.log('音频可以播放，时长:', newAudio.duration);
                if (durationEl && newAudio.duration && !isNaN(newAudio.duration) && newAudio.duration !== Infinity && newAudio.duration > 0) {
                    durationEl.textContent = Math.round(newAudio.duration);
                }
            };

            newAudio.onerror = function(e) {
                console.error('音频加载失败:', e);
                console.error('音频错误代码:', newAudio.error ? newAudio.error.code : 'unknown');
                errorMsg.style.display = 'block';
                errorMsg.textContent = '音频预览加载失败，但录音数据仍可使用';
            };

            // 插入到 DOM 中（在错误消息之前）
            audioContainer.insertBefore(newAudio, errorMsg);

            // 显示预览区域
            document.getElementById('audioPreview').classList.add('show');

            // 禁用上传区域
            document.getElementById('uploadArea').style.opacity = '0.5';
            document.getElementById('uploadArea').style.pointerEvents = 'none';
        }

        // 使用录音
        function useRecordedAudio() {
            if (!recordedAudioBlob) return;

            // 根据 MIME 类型确定文件扩展名
            let fileExt = '.webm';
            const mimeType = recordedAudioBlob.type || 'audio/webm';
            if (mimeType.includes('ogg')) {
                fileExt = '.ogg';
            } else if (mimeType.includes('webm')) {
                fileExt = '.webm';
            }

            // 转换为 File 对象
            const file = new File([recordedAudioBlob], `recording_${Date.now()}${fileExt}`, {
                type: recordedAudioBlob.type
            });

            selectedAudioFile = file;

            // 更新UI
            document.getElementById('fileName').textContent = file.name;
            document.getElementById('selectedFile').classList.add('show');
            document.getElementById('uploadArea').classList.add('has-file');

            showSpeakerStatus('录音已选中，请填写名称后保存', 'success');
            updateSaveButton();
        }

        // 丢弃录音
        function discardRecordedAudio() {
            recordedAudioBlob = null;
            recordedChunks = [];
            recordStartTime = null; // 重置开始时间

            const audio = document.getElementById('recordedAudio');
            const errorMsg = document.getElementById('audioErrorMsg');

            if (audio) {
                // 暂停音频
                audio.pause();

                // 释放 Blob URL
                if (audio.src && audio.src.startsWith('blob:')) {
                    URL.revokeObjectURL(audio.src);
                }

                // 移除 audio 元素
                audio.remove();
            }

            // 重新创建初始的 audio 结构
            const audioPreview = document.getElementById('audioPreview');

            // 找到错误消息元素
            const existingErrorMsg = document.getElementById('audioErrorMsg');

            // 创建新的 audio 元素 - 直接设置 src，避免 source 子元素引发无效URI错误
            const newAudio = document.createElement('audio');
            newAudio.id = 'recordedAudio';
            newAudio.controls = true;
            newAudio.style.width = '100%';
            newAudio.preload = 'none';
            newAudio.style.display = 'none';
            newAudio.src = '';

            // 插入到错误消息之前（audioPreview 是共同父容器）
            if (existingErrorMsg && existingErrorMsg.parentNode === audioPreview) {
                audioPreview.insertBefore(newAudio, existingErrorMsg);
            } else {
                audioPreview.appendChild(newAudio);
            }

            // 重置显示
            document.getElementById('recordDuration').textContent = '0';
            if (existingErrorMsg) {
                existingErrorMsg.style.display = 'none';
            }
            document.getElementById('audioPreview').classList.remove('show');
            document.getElementById('recordTimer').textContent = '00:00';
            document.getElementById('recordTimer').style.display = 'none';

            // 恢复上传区域
            document.getElementById('uploadArea').style.opacity = '';
            document.getElementById('uploadArea').style.pointerEvents = '';

            updateRecordUI();
            showSpeakerStatus('');
        }

        // 保存说话人
        async function saveSpeaker() {
            if (!selectedAudioFile) {
                showSpeakerStatus('请先选择音频文件', 'error');
                return;
            }

            const nameInputId = getSpeakerNameInputId();
            const btnId = getSaveButtonId();

            const name = document.getElementById(nameInputId).value.trim();
            if (!name) {
                showSpeakerStatus('请输入说话人名称', 'error');
                return;
            }

            const btn = document.getElementById(btnId);
            btn.disabled = true;
            btn.innerHTML = '⏳ 处理中...';

            updateProgress(10, '正在上传音频...');

            try {
                // 第一步：上传音频（与模型解耦，不再提取 embedding）
                const formData = new FormData();
                formData.append('audio', selectedAudioFile);
                formData.append('speaker_name', name);

                // 根据当前页面获取参考文本
                let referenceText = '';
                if (currentSpeakerTab === 'record') {
                    // 录音模式：使用当前朗读文本
                    referenceText = currentScriptText;
                } else {
                    // 上传模式：使用用户输入的参考文本
                    referenceText = document.getElementById('uploadReferenceText').value.trim();
                }

                if (referenceText) {
                    formData.append('reference_text', referenceText);
                }

                updateProgress(40, '正在处理音频...');

                const uploadResponse = await fetch(`${API_BASE}/speakers/upload`, {
                    method: 'POST',
                    body: formData
                });

                const uploadData = await uploadResponse.json();

                if (!uploadData.success) {
                    throw new Error(uploadData.message || '上传音频失败');
                }

                updateProgress(70, '正在保存说话人信息...');

                // 第二步：保存说话人信息（与模型解耦，不再保存 embedding）
                const saveFormData = new FormData();
                saveFormData.append('name', name);
                saveFormData.append('audio_path', uploadData.audio_path || '');
                // 添加参考文本
                if (referenceText) {
                    saveFormData.append('reference_text', referenceText);
                }

                const saveResponse = await fetch(`${API_BASE}/speakers/save`, {
                    method: 'POST',
                    body: saveFormData
                });

                const saveData = await saveResponse.json();

                if (saveData.success) {
                    console.log('说话人保存成功，准备刷新列表...');
                    updateProgress(100, '保存成功！');
                    showSpeakerStatus('说话人保存成功！', 'success');
                    resetSpeakerForm();
                    await loadSpeakersList();

                    // 刷新参考人声列表（用于所有 TTS 模型）
                    await loadReferenceVoices();
                    console.log('列表刷新完成');
                } else {
                    throw new Error(saveData.message || '保存说话人失败');
                }

            } catch (error) {
                console.error('保存说话人失败:', error);
                showSpeakerStatus('保存失败: ' + error.message, 'error');
            } finally {
                btn.disabled = false;
                btn.innerHTML = '💾 保存说话人';
                setTimeout(() => {
                    updateProgress(0, '准备上传...');
                }, 2000);
            }
        }

        // 重置表单
        function resetSpeakerForm() {
            removeSelectedFile();
            discardRecordedAudio(); // 重置录音

            // 重置录音页面
            document.getElementById('speakerNameRecord').value = '';
            document.getElementById('nameStatusRecord').textContent = '';
            document.getElementById('nameStatusRecord').className = 'name-status';
            document.getElementById('speakerNameRecord').classList.remove('error', 'success');

            // 重置上传页面
            document.getElementById('speakerNameUpload').value = '';
            document.getElementById('nameStatusUpload').textContent = '';
            document.getElementById('nameStatusUpload').className = 'name-status';
            document.getElementById('speakerNameUpload').classList.remove('error', 'success');
            document.getElementById('uploadReferenceText').value = '';

            showSpeakerStatus('');
            updateSaveButton();

            // 重置进度条
            updateProgress(0, '准备上传...');
        }

        // 加载说话人列表
        async function loadSpeakersList() {
            try {
                console.log('[DEBUG] 开始加载说话人列表...');
                console.log('[DEBUG] API_BASE:', API_BASE);
                const url = `${API_BASE}/speakers`;
                console.log('[DEBUG] 请求URL:', url);

                const response = await fetch(url);
                console.log('[DEBUG] 响应状态:', response.status, response.statusText);

                const data = await response.json();
                console.log('[DEBUG] API返回数据:', data);

                if (data.success) {
                    savedSpeakers = data.speakers || [];
                    console.log('[DEBUG] 加载到说话人数目:', savedSpeakers.length);
                    renderSpeakersList();
                    const countEl = document.getElementById('speakersCount');
                    if (countEl) {
                        countEl.textContent = savedSpeakers.length;
                    }
                    // 更新 ChatTTS 页面的说话人选择下拉框
                    updateChatTTSSpeakerSelect();
                } else {
                    console.error('[DEBUG] API返回success=false:', data);
                }
            } catch (error) {
                console.error('[DEBUG] 加载说话人列表失败:', error);
                console.error('[DEBUG] 错误详情:', error.message);
            }
        }

        // 渲染说话人列表
        function renderSpeakersList() {
            console.log('[DEBUG] 开始渲染说话人列表');
            const listEl = document.getElementById('speakersList');
            console.log('[DEBUG] speakersList 元素:', listEl);
            console.log('[DEBUG] savedSpeakers 数据:', savedSpeakers);
            console.log('[DEBUG] savedSpeakers 长度:', savedSpeakers.length);

            if (!listEl) {
                console.error('[DEBUG] 找不到 speakersList 元素');
                return;
            }

            if (savedSpeakers.length === 0) {
                console.log('[DEBUG] 说话人列表为空，显示空状态');
                listEl.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">📭</div>
                        <div>暂无已保存的说话人</div>
                    </div>
                `;
                return;
            }

            console.log('[DEBUG] 开始渲染', savedSpeakers.length, '个说话人');

            listEl.innerHTML = savedSpeakers.map(speaker => {
                const hasRefText = speaker.has_reference_text;
                const refTextPreview = hasRefText ? (speaker.reference_text || '').substring(0, 50) + '...' : '';
                
                return `
                <div class="speaker-item" data-id="${speaker.id}">
                    <div class="speaker-icon">🎙️</div>
                    <div class="speaker-info">
                        <div class="speaker-name">${escapeHtml(speaker.name)}</div>
                        <div class="speaker-meta">
                            创建于 ${formatDate(speaker.created_at)}
                            ${hasRefText ? '<span style="color: #0ea5e9; margin-left: 8px;">✓ 有参考文本</span>' : ''}
                        </div>
                        ${hasRefText ? `<div class="speaker-ref-text" style="font-size: 0.85rem; color: #64748b; margin-top: 4px; font-style: italic;">${escapeHtml(refTextPreview)}</div>` : ''}
                    </div>
                    <div class="speaker-actions">
                        <button class="btn-icon btn-use" onclick="useSpeaker('${speaker.id}')" title="使用此说话人">
                            ✓
                        </button>
                        <button class="btn-icon btn-delete" onclick="deleteSpeaker('${speaker.id}')" title="删除">
                            🗑️
                        </button>
                    </div>
                </div>
            `}).join('');
            console.log('说话人列表渲染完成');
        }

        // 使用指定的说话人
        async function useSpeaker(speakerId) {
            // 查找说话人信息
            const speaker = savedSpeakers.find(s => s.id === speakerId);
            if (!speaker) {
                showSpeakerStatus('说话人不存在', 'error');
                return;
            }

            // 关闭模态框
            closeSpeakerManager();

            // 设置为当前选中的说话人
            selectedSpeakerId = speakerId;

            // 更新下拉框的选中状态
            const speakerSelect = document.getElementById('chattts-speaker-select');
            if (speakerSelect) {
                speakerSelect.value = speakerId;
                console.log('[DEBUG] 下拉框已更新为:', speakerId);
            }

            // 显示提示
            showStatus(`已选择说话人: ${speaker.name}`);

            // 更新 ChatTTS 选项中的显示
            updateChatTTSSpeakerDisplay(speaker.name);
        }

        // 删除说话人
        async function deleteSpeaker(speakerId) {
            const speaker = savedSpeakers.find(s => s.id === speakerId);
            if (!confirm(`确定要删除说话人 "${speaker?.name || ''}" 吗？`)) {
                return;
            }

            try {
                const response = await fetch(`${API_BASE}/speakers/${speakerId}`, {
                    method: 'DELETE'
                });

                const data = await response.json();

                if (data.success) {
                    showSpeakerStatus('说话人已删除', 'success');
                    loadSpeakersList();

                    // 如果删除的是当前选中的说话人，清除选择
                    if (selectedSpeakerId === speakerId) {
                        selectedSpeakerId = null;
                        updateChatTTSSpeakerDisplay('');
                    }

                    // 更新下拉框
                    updateChatTTSSpeakerSelect();
                } else {
                    showSpeakerStatus(data.message || '删除失败', 'error');
                }
            } catch (error) {
                console.error('删除说话人失败:', error);
                showSpeakerStatus('删除失败: ' + error.message, 'error');
            }
        }

        // 更新 ChatTTS 的说话人选择下拉框
        function updateChatTTSSpeakerSelect() {
            console.log('[DEBUG] 更新 ChatTTS 说话人选择下拉框');
            const select = document.getElementById('chattts-speaker-select');
            if (!select) {
                console.log('[DEBUG] 找不到 chattts-speaker-select 元素');
                return;
            }

            // 保留第一个选项（随机说话人）
            const firstOption = select.options[0];
            console.log('[DEBUG] 保留第一个选项:', firstOption ? firstOption.textContent : '无');
            select.innerHTML = '';
            if (firstOption) {
                select.appendChild(firstOption);
            }

            // 添加已保存的说话人
            console.log('[DEBUG] 添加', savedSpeakers.length, '个说话人到下拉框');
            savedSpeakers.forEach(speaker => {
                const option = document.createElement('option');
                option.value = speaker.id;
                option.textContent = speaker.name;
                select.appendChild(option);
                console.log('[DEBUG] 添加选项:', speaker.name, '(' + speaker.id + ')');
            });
            console.log('[DEBUG] 下拉框更新完成，共', select.options.length, '个选项');
        }

        // 更新 ChatTTS 选项中的说话人显示
        function updateChatTTSSpeakerDisplay(speakerName) {
            const displayEl = document.getElementById('chattts-current-speaker');
            if (displayEl) {
                displayEl.textContent = speakerName || '随机说话人';
            }
        }

        // 辅助函数：HTML 转义
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // 辅助函数：格式化日期
        function formatDate(isoString) {
            if (!isoString) return '';
            const date = new Date(isoString);
            return date.toLocaleDateString('zh-CN') + ' ' + date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
        }

        // 拖拽上传支持
        document.addEventListener('DOMContentLoaded', () => {
            const uploadArea = document.getElementById('uploadArea');
            if (uploadArea) {
                uploadArea.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    uploadArea.classList.add('dragover');
                });

                uploadArea.addEventListener('dragleave', () => {
                    uploadArea.classList.remove('dragover');
                });

                uploadArea.addEventListener('drop', (e) => {
                    e.preventDefault();
                    uploadArea.classList.remove('dragover');
                    const files = e.dataTransfer.files;
                    if (files.length > 0) {
                        validateAndSetFile(files[0]);
                    }
                });
            }

            // ESC 键关闭模态框
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    const modal = document.getElementById('speakerModal');
                    if (modal && modal.classList.contains('active')) {
                        closeSpeakerManager();
                    }
                }
            });

            // 点击模态框外部关闭
            document.getElementById('speakerModal').addEventListener('click', (e) => {
                if (e.target.id === 'speakerModal') {
                    closeSpeakerManager();
                }
            });

            // 初始化用户信息
            const username = localStorage.getItem('versTTS_username') || 'admin';
            const currentUserEl = document.getElementById('currentUser');
            if (currentUserEl) {
                currentUserEl.textContent = username;
            }

            // 初始化应用
            checkHealth();
            setInterval(checkHealth, 30000); // 每30秒检查一次
            selectModel('chattts'); // 默认选择ChatTTS
            loadReferenceVoices(); // 加载参考人声列表
            loadSpeakersList(); // 加载说话人列表
            checkQwen3TTSStatus(); // 检查 Qwen3-TTS 模型状态
        });

        // 当前选中的说话人ID（用于TTS合成）
        let selectedSpeakerId = null;

        // 处理说话人选择变化
        function onChatTTSSpeakerChange() {
            const select = document.getElementById('chattts-speaker-select');
            const speakerId = select.value;
            selectedSpeakerId = speakerId || null;

            // 更新显示
            if (speakerId) {
                const speaker = savedSpeakers.find(s => s.id === speakerId);
                updateChatTTSSpeakerDisplay(speaker ? speaker.name : '');
            } else {
                updateChatTTSSpeakerDisplay('');
            }
        }

        // 登录相关函数
        const DEFAULT_PASSWORD = 'tp123456';
        // AUTH_KEY 已在文件顶部声明
        const DEFAULT_USERNAME = 'admin';

        function doLogout() {
            localStorage.removeItem(AUTH_KEY);
            localStorage.removeItem('versTTS_username');
            window.location.href = 'login.html';
        }
    </script>

    <!-- ==================== 说话人管理模态框 ==================== -->
    <div id="speakerModal" class="modal-overlay">
        <div class="modal-content" style="max-width: 700px; max-height: 90vh; overflow-y: auto;">
            <div class="modal-header">
                <h2>👤 说话人管理</h2>
                <button class="modal-close" onclick="closeSpeakerManager()">&times;</button>
            </div>
            <div class="modal-body">
                <!-- 状态消息 -->
                <div id="speakerStatusMessage" class="status-message-speaker"></div>

                <!-- 子页面选项卡 -->
                <div class="speaker-tabs" style="display: flex; border-bottom: 2px solid #e2e8f0; margin-bottom: 20px;">
                    <button id="tabRecord" class="speaker-tab active" onclick="switchSpeakerTab('record')" style="flex: 1; padding: 12px 20px; background: none; border: none; font-size: 1rem; cursor: pointer; color: #0ea5e9; border-bottom: 2px solid #0ea5e9; font-weight: 600;">
                        🎙️ 朗读录音
                    </button>
                    <button id="tabUpload" class="speaker-tab" onclick="switchSpeakerTab('upload')" style="flex: 1; padding: 12px 20px; background: none; border: none; font-size: 1rem; cursor: pointer; color: #64748b; border-bottom: 2px solid transparent;">
                        📤 上传音频
                    </button>
                </div>

                <!-- ========== 录音模式子页面 ========== -->
                <div id="pageRecord" class="speaker-page">
                    <!-- 朗读文本选择区域 -->
                    <div class="script-section" style="margin-bottom: 20px; padding: 15px; background: #f0f9ff; border-radius: 12px; border: 1px solid #bae6fd;">
                        <div style="font-weight: 600; color: #0369a1; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                            📖 朗读文本
                            <span style="font-size: 0.85em; font-weight: normal; color: #64748b;">（请朗读以下文本进行录音）</span>
                        </div>
                        <div style="margin-bottom: 12px;">
                            <select id="scriptLength" onchange="loadRecordingScripts()" style="padding: 8px 12px; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 0.9rem; margin-right: 10px;">
                                <option value="short">短文本 (5-8秒)</option>
                                <option value="medium">中等 (10-15秒)</option>
                                <option value="long">长文本 (15-20秒)</option>
                            </select>
                            <button onclick="refreshScript()" style="padding: 8px 16px; background: #0ea5e9; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 0.9rem;">
                                🔄 换一段
                            </button>
                        </div>
                        <div id="scriptDisplay" style="padding: 15px; background: white; border-radius: 8px; border: 1px solid #e2e8f0; font-size: 1.1rem; line-height: 1.8; color: #1e293b; min-height: 60px;">
                            正在加载朗读文本...
                        </div>
                        <div id="scriptInfo" style="margin-top: 8px; font-size: 0.85rem; color: #64748b;">
                            <span id="scriptTypeBadge" style="display: inline-block; padding: 2px 8px; background: #e0f2fe; color: #0369a1; border-radius: 4px; margin-right: 8px;">-</span>
                            <span id="scriptSource" style="color: #059669; font-style: italic;"></span>
                            <span style="margin-left: 8px;">|</span>
                            <span style="margin-left: 8px;">预计时长: <span id="scriptDuration">-</span></span>
                        </div>
                    </div>

                    <!-- 录音区域 -->
                    <div class="record-section">
                        <div class="record-title">
                            🎙️ 录音
                        </div>
                        <div class="record-controls">
                            <button id="recordBtn" class="btn-record" onclick="toggleRecording()" title="开始/停止录音">
                                🎤
                            </button>
                            <div class="record-status" id="recordStatus">点击按钮开始录音</div>
                            <div class="record-timer" id="recordTimer" style="display: none;">00:00</div>
                            <div class="record-waves" id="recordWaves">
                                <div class="record-wave-bar"></div>
                                <div class="record-wave-bar"></div>
                                <div class="record-wave-bar"></div>
                                <div class="record-wave-bar"></div>
                                <div class="record-wave-bar"></div>
                                <div class="record-wave-bar"></div>
                                <div class="record-wave-bar"></div>
                            </div>
                            <div class="record-hint">
                                建议录制 5-20 秒的清晰语音
                            </div>
                        </div>
                        <!-- 录音预览 -->
                        <div id="audioPreview" class="audio-preview">
                            <div style="margin-bottom: 10px; color: #374151; font-weight: 500;">
                                🔊 录音预览（时长: <span id="recordDuration">0</span> 秒）
                            </div>
                            <audio id="recordedAudio" controls preload="none" src="" style="width: 100%; display: none;">
                                您的浏览器不支持音频播放
                            </audio>
                            <div id="audioErrorMsg" style="display: none; color: #dc2626; margin-top: 10px; padding: 10px; background: #fee2e2; border-radius: 8px;">
                                音频格式不受支持，但数据仍可正常使用
                            </div>
                            <div class="audio-preview-actions">
                                <button class="btn-use-record" onclick="useRecordedAudio()">
                                    ✓ 使用此录音
                                </button>
                                <button class="btn-discard-record" onclick="discardRecordedAudio()">
                                    ✕ 重新录制
                                </button>
                            </div>
                        </div>
                    </div>

                    <!-- 命名区域 -->
                    <div class="name-section" style="margin-top: 20px;">
                        <label for="speakerNameRecord">说话人名称</label>
                        <div class="name-input-wrapper">
                            <input type="text" id="speakerNameRecord" class="name-input" placeholder="请输入说话人名称（如：小明、女声1等）" onblur="checkSpeakerName()">
                        </div>
                        <div id="nameStatusRecord" class="name-status"></div>
                    </div>

                    <!-- 进度条 -->
                    <div id="uploadProgressRecord" class="progress-container" style="margin-top: 15px;">
                        <div class="progress-bar">
                            <div class="progress-fill" id="progressFillRecord"></div>
                        </div>
                        <div class="progress-text" id="progressTextRecord">准备上传...</div>
                    </div>

                    <!-- 操作按钮 -->
                    <div class="action-buttons" style="margin-top: 20px;">
                        <button id="saveSpeakerBtnRecord" class="btn-primary" onclick="saveSpeaker()" disabled>
                            💾 保存说话人
                        </button>
                        <button class="btn-secondary" onclick="resetSpeakerForm()">
                            重置
                        </button>
                    </div>
                </div>

                <!-- ========== 上传模式子页面 ========== -->
                <div id="pageUpload" class="speaker-page" style="display: none;">
                    <!-- 上传区域 -->
                    <div class="upload-section" style="margin-bottom: 20px;">
                        <div id="uploadArea" class="upload-area" onclick="document.getElementById('audioFileInput').click()">
                            <div class="upload-icon">🎵</div>
                            <div class="upload-text">点击或拖拽上传音频文件</div>
                            <div class="upload-hint">支持 MP3、WAV、FLAC、OGG、M4A、WEBM 格式</div>
                            <input type="file" id="audioFileInput" class="file-input" accept="audio/*" onchange="handleFileSelect(event)">
                        </div>
                        <div id="selectedFile" class="selected-file">
                            <span class="file-name" id="fileName"></span>
                            <button class="file-remove" onclick="removeSelectedFile()">移除</button>
                        </div>
                    </div>

                    <!-- 参考文本输入区域 -->
                    <div class="script-section" style="margin-bottom: 20px; padding: 15px; background: #f0fdf4; border-radius: 12px; border: 1px solid #86efac;">
                        <div style="font-weight: 600; color: #15803d; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                            📝 参考文本
                            <span style="font-size: 0.85em; font-weight: normal; color: #64748b;">（输入音频对应的文本内容）</span>
                        </div>
                        <textarea id="uploadReferenceText" rows="4" placeholder="请输入音频文件中朗读的文本内容..." oninput="updateSaveButton()" style="width: 100%; padding: 12px; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 1rem; line-height: 1.6; resize: vertical;"></textarea>
                        <div style="margin-top: 8px; font-size: 0.85rem; color: #64748b;">
                            💡 提示：准确的参考文本有助于提高声音克隆质量
                        </div>
                    </div>

                    <!-- 命名区域 -->
                    <div class="name-section">
                        <label for="speakerNameUpload">说话人名称</label>
                        <div class="name-input-wrapper">
                            <input type="text" id="speakerNameUpload" class="name-input" placeholder="请输入说话人名称（如：小明、女声1等）" onblur="checkSpeakerName()">
                        </div>
                        <div id="nameStatusUpload" class="name-status"></div>
                    </div>

                    <!-- 进度条 -->
                    <div id="uploadProgressUpload" class="progress-container" style="margin-top: 15px;">
                        <div class="progress-bar">
                            <div class="progress-fill" id="progressFillUpload"></div>
                        </div>
                        <div class="progress-text" id="progressTextUpload">准备上传...</div>
                    </div>

                    <!-- 操作按钮 -->
                    <div class="action-buttons" style="margin-top: 20px;">
                        <button id="saveSpeakerBtnUpload" class="btn-primary" onclick="saveSpeaker()" disabled>
                            💾 保存说话人
                        </button>
                        <button class="btn-secondary" onclick="resetSpeakerForm()">
                            重置
                        </button>
                    </div>
                </div>

                <!-- 已保存说话人列表 -->
                <div class="speakers-list-section" style="margin-top: 30px; border-top: 1px solid #e2e8f0; padding-top: 20px;">
                    <h3>已保存的说话人 <span id="speakersCount" class="speakers-count">0</span></h3>
                    <div id="speakersList" class="speakers-list">
                        <!-- 动态填充 -->
                        <div class="empty-state">
                            <div class="empty-state-icon">📭</div>
                            <div>暂无已保存的说话人</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>