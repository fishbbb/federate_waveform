// 联邦学习可视化前端应用

// 全局变量
let socket;
let lossChart, f1Chart, accuracyChart;
let startTime = null;
let elapsedInterval = null;

// 通知函数
function showNotification(message, type = 'info') {
    // 创建通知元素
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#667eea'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        z-index: 10000;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    // 3秒后自动移除
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    connectWebSocket();
    startElapsedTime();
    
    // 初始化时加载节点状态
    refreshNodes();
    loadLogs();
    
    // 定期请求状态更新（每3秒）
    setInterval(() => {
        if (socket && socket.connected) {
            socket.emit('request_state');
        }
    }, 3000);
    
    // 定期刷新节点状态（每5秒）
    setInterval(refreshNodes, 5000);
});

// 初始化图表
function initializeCharts() {
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: {
                display: true,
                position: 'top',
            },
            tooltip: {
                mode: 'index',
                intersect: false,
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(0, 0, 0, 0.1)'
                }
            },
            x: {
                grid: {
                    color: 'rgba(0, 0, 0, 0.1)'
                }
            }
        }
    };

    // Loss 图表
    const lossCtx = document.getElementById('loss-chart').getContext('2d');
    lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: '全局损失',
                data: [],
                borderColor: 'rgb(239, 68, 68)',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: chartOptions
    });

    // F1 图表
    const f1Ctx = document.getElementById('f1-chart').getContext('2d');
    f1Chart = new Chart(f1Ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'F1 分数',
                data: [],
                borderColor: 'rgb(59, 130, 246)',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: chartOptions
    });

    // Accuracy 图表
    const accuracyCtx = document.getElementById('accuracy-chart').getContext('2d');
    accuracyChart = new Chart(accuracyCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: '准确率',
                data: [],
                borderColor: 'rgb(16, 185, 129)',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: chartOptions
    });
}

// 连接WebSocket
function connectWebSocket() {
    socket = io();
    
    socket.on('connect', function() {
        console.log('Connected to server');
        updateConnectionStatus(true);
        socket.emit('request_state');
    });
    
    socket.on('disconnect', function() {
        console.log('Disconnected from server');
        updateConnectionStatus(false);
    });
    
    socket.on('connected', function(data) {
        console.log('Server message:', data.message);
    });
    
    socket.on('state_update', function(state) {
        console.log('State update received');
        updateUI(state);
    });
    
    socket.on('update', function(data) {
        console.log('Update event:', data.event_type);
        handleUpdate(data.event_type, data.data);
    });
    
    socket.on('new_log', function(logEntry) {
        addLogToUI(logEntry);
    });
}

// ==================== 控制功能 ====================

// 标签页切换
function showTab(tabName) {
    // 隐藏所有标签页
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // 显示选中的标签页
    document.getElementById(`tab-${tabName}`).classList.add('active');
    event.target.classList.add('active');
}

// 节点管理
async function startNode() {
    const nodeId = document.getElementById('node-select').value;
    
    // 显示加载状态
    const btn = event.target;
    const originalText = btn.textContent;
    btn.disabled = true;
    btn.textContent = '启动中...';
    
    try {
        const response = await fetch('/api/nodes/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                node_id: nodeId,
                auto_add_dataset: true
            })
        });
        
        const data = await response.json();
        if (data.success) {
            // 使用更友好的提示
            showNotification(`节点 ${nodeId} 启动成功 (PID: ${data.pid})`, 'success');
            refreshNodes();
            // 自动刷新节点状态
            setTimeout(refreshNodes, 2000);
        } else {
            showNotification(`启动失败: ${data.error}`, 'error');
        }
    } catch (error) {
        showNotification(`启动节点失败: ${error.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = originalText;
    }
}

async function stopNode() {
    const nodeId = document.getElementById('node-select').value;
    
    if (!confirm(`确定要停止节点 ${nodeId} 吗？`)) {
        return;
    }
    
    // 显示加载状态
    const btn = event.target;
    const originalText = btn.textContent;
    btn.disabled = true;
    btn.textContent = '停止中...';
    
    try {
        const response = await fetch('/api/nodes/stop', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({node_id: nodeId})
        });
        
        const data = await response.json();
        if (data.success) {
            showNotification(`节点 ${nodeId} 已停止`, 'success');
            refreshNodes();
        } else {
            showNotification(`停止失败: ${data.error}`, 'error');
        }
    } catch (error) {
        showNotification(`停止节点失败: ${error.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = originalText;
    }
}

async function refreshNodes() {
    try {
        const response = await fetch('/api/nodes/list');
        const data = await response.json();
        
        const statusContainer = document.getElementById('nodes-status');
        if (!statusContainer) return;
        
        if (data.nodes.length === 0) {
            statusContainer.innerHTML = '<p class="empty-message">没有配置的节点。请先准备数据并配置节点。</p>';
            return;
        }
        
        statusContainer.innerHTML = data.nodes.map(node => {
            const statusClass = node.status === 'running' ? 'running' : 
                               node.status === 'stopped' ? 'stopped' : 'idle';
            const statusText = node.status === 'running' ? '运行中' : 
                              node.status === 'stopped' ? '已停止' : '空闲';
            
            return `
            <div class="node-status-item">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <strong>${node.node_id}</strong>
                    <span class="status-tag ${statusClass}">${statusText}</span>
                    ${node.pid ? `<span style="color: #6b7280; font-size: 12px;">PID: ${node.pid}</span>` : ''}
                </div>
                <div style="margin-top: 8px; font-size: 12px; color: #6b7280;">
                    ${node.device_type ? `<span>设备类型: ${node.device_type}</span>` : ''}
                    ${node.compute_power ? `<span> | 算力: ${node.compute_power}</span>` : ''}
                </div>
                <div class="node-metrics">
                    <span>数据量: ${node.data_size || 0}</span>
                    ${node.metrics && node.metrics.f1 ? `<span>F1: ${node.metrics.f1.toFixed(3)}</span>` : ''}
                    ${node.metrics && node.metrics.loss ? `<span>损失: ${node.metrics.loss.toFixed(4)}</span>` : ''}
                </div>
            </div>
        `;
        }).join('');
    } catch (error) {
        console.error('刷新节点状态失败:', error);
        const statusContainer = document.getElementById('nodes-status');
        if (statusContainer) {
            statusContainer.innerHTML = `<p class="error">刷新失败: ${error.message}</p>`;
        }
    }
}

// 训练控制
async function startTraining() {
    const rounds = document.getElementById('rounds-input').value;
    const batchSize = document.getElementById('batch-size-input').value;
    const lr = document.getElementById('lr-input').value;
    
    try {
        const response = await fetch('/api/training/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                rounds: parseInt(rounds),
                batch_size: parseInt(batchSize),
                learning_rate: parseFloat(lr)
            })
        });
        
        const data = await response.json();
        if (data.success) {
            alert(`训练已启动 (PID: ${data.pid})`);
        } else {
            alert(`启动训练失败: ${data.error}`);
        }
    } catch (error) {
        alert(`启动训练失败: ${error.message}`);
    }
}

async function stopTraining() {
    if (!confirm('确定要停止训练吗？')) return;
    
    try {
        const response = await fetch('/api/training/stop', {
            method: 'POST'
        });
        
        const data = await response.json();
        if (data.success) {
            alert('训练已停止');
        } else {
            alert(`停止训练失败: ${data.error}`);
        }
    } catch (error) {
        alert(`停止训练失败: ${error.message}`);
    }
}

// 日志管理
async function loadLogs() {
    try {
        const level = document.getElementById('log-level-filter').value;
        const url = level === 'all' ? '/api/logs' : `/api/logs?level=${level}`;
        const response = await fetch(url);
        const data = await response.json();
        
        const logsContainer = document.getElementById('logs-container');
        logsContainer.innerHTML = data.logs.map(log => `
            <div class="log-entry log-${log.level}">
                <span class="log-time">${new Date(log.timestamp).toLocaleTimeString()}</span>
                <span class="log-level">[${log.level.toUpperCase()}]</span>
                <span class="log-message">${log.message}</span>
            </div>
        `).join('');
        
        // 滚动到底部
        logsContainer.scrollTop = logsContainer.scrollHeight;
    } catch (error) {
        console.error('加载日志失败:', error);
    }
}

function addLogToUI(logEntry) {
    const logsContainer = document.getElementById('logs-container');
    const logEl = document.createElement('div');
    logEl.className = `log-entry log-${logEntry.level}`;
    logEl.innerHTML = `
        <span class="log-time">${new Date(logEntry.timestamp).toLocaleTimeString()}</span>
        <span class="log-level">[${logEntry.level.toUpperCase()}]</span>
        <span class="log-message">${logEntry.message}</span>
    `;
    logsContainer.appendChild(logEl);
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

function filterLogs() {
    loadLogs();
}

async function clearLogs() {
    if (!confirm('确定要清空日志吗？')) return;
    
    try {
        const response = await fetch('/api/logs/clear', {method: 'POST'});
        const data = await response.json();
        if (data.success) {
            document.getElementById('logs-container').innerHTML = '';
        }
    } catch (error) {
        console.error('清空日志失败:', error);
    }
}

// 数据分析
async function analyzeConvergence() {
    try {
        const response = await fetch('/api/analysis/convergence');
        const data = await response.json();
        
        const analysisEl = document.getElementById('convergence-analysis');
        analysisEl.innerHTML = `
            <div class="analysis-card">
                <h5>收敛状态</h5>
                <p><strong>是否收敛:</strong> ${data.is_converged ? '是' : '否'}</p>
                <p><strong>损失变化:</strong> ${data.loss_change.toFixed(6)}</p>
                <p><strong>损失标准差:</strong> ${data.loss_std.toFixed(6)}</p>
                <p><strong>收敛率:</strong> ${data.convergence_rate.toFixed(6)}</p>
                <p><strong>总轮数:</strong> ${data.total_rounds}</p>
            </div>
        `;
    } catch (error) {
        console.error('分析收敛性失败:', error);
        document.getElementById('convergence-analysis').innerHTML = 
            '<p class="error">分析失败: ' + error.message + '</p>';
    }
}

async function analyzeNodePerformance() {
    try {
        const response = await fetch('/api/analysis/data');
        const data = await response.json();
        
        const performanceEl = document.getElementById('node-performance');
        const nodes = data.node_performance;
        
        if (Object.keys(nodes).length === 0) {
            performanceEl.innerHTML = '<p class="empty-message">暂无节点性能数据</p>';
            return;
        }
        
        performanceEl.innerHTML = Object.entries(nodes).map(([nodeId, perf]) => `
            <div class="analysis-card">
                <h5>${nodeId}</h5>
                <p><strong>平均损失:</strong> ${perf.avg_loss.toFixed(4)}</p>
                <p><strong>平均F1:</strong> ${perf.avg_f1.toFixed(4)}</p>
                <p><strong>平均准确率:</strong> ${perf.avg_accuracy.toFixed(4)}</p>
                <p><strong>数据量:</strong> ${perf.data_size}</p>
            </div>
        `).join('');
    } catch (error) {
        console.error('分析节点性能失败:', error);
        document.getElementById('node-performance').innerHTML = 
            '<p class="error">分析失败: ' + error.message + '</p>';
    }
}

// 启动所有节点
async function startAllNodes() {
    if (!confirm('确定要启动所有节点吗？')) return;
    
    const nodes = ['node_1', 'node_2', 'node_3'];
    let successCount = 0;
    let failCount = 0;
    
    for (const nodeId of nodes) {
        try {
            const response = await fetch('/api/nodes/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({node_id: nodeId, auto_add_dataset: true})
            });
            
            const data = await response.json();
            if (data.success) {
                successCount++;
                showNotification(`节点 ${nodeId} 启动成功`, 'success');
            } else {
                failCount++;
                showNotification(`节点 ${nodeId} 启动失败: ${data.error}`, 'error');
            }
            // 每个节点之间等待1秒
            await new Promise(resolve => setTimeout(resolve, 1000));
        } catch (error) {
            failCount++;
            showNotification(`节点 ${nodeId} 启动失败: ${error.message}`, 'error');
        }
    }
    
    showNotification(`启动完成: ${successCount} 成功, ${failCount} 失败`, 
                     failCount === 0 ? 'success' : 'warning');
    refreshNodes();
}

// 初始化时加载日志
setInterval(loadLogs, 2000);  // 每2秒刷新一次日志
setInterval(refreshNodes, 5000);  // 每5秒刷新一次节点状态

// 更新连接状态
function updateConnectionStatus(connected) {
    const statusEl = document.getElementById('connection-status');
    if (connected) {
        statusEl.textContent = '已连接';
        statusEl.className = 'status-indicator online';
    } else {
        statusEl.textContent = '未连接';
        statusEl.className = 'status-indicator offline';
    }
}

// 处理更新事件
function handleUpdate(eventType, data) {
    console.log('Handling update:', eventType, data);
    
    switch(eventType) {
        case 'experiment_started':
            startTime = new Date();
            updateExperimentStatus('running');
            // 显示详细训练状态面板
            const statusPanel = document.getElementById('detailed-training-status');
            if (statusPanel) {
                statusPanel.style.display = 'block';
            }
            // 如果数据中包含配置信息，立即更新
            if (data.config) {
                updateExperimentConfig(data.config);
            }
            // 如果数据中包含轮次信息，立即更新
            if (data.rounds !== undefined || data.total_rounds !== undefined) {
                const totalRounds = data.total_rounds || data.rounds || 0;
                const currentRound = data.current_round || 0;
                updateProgress(currentRound, totalRounds);
            }
            // 请求完整状态以更新详细状态
            socket.emit('request_state');
            break;
        case 'experiment_ended':
            updateExperimentStatus('completed');
            // 隐藏详细训练状态面板（可选，也可以保留显示最终状态）
            // const statusPanel = document.getElementById('detailed-training-status');
            // if (statusPanel) {
            //     statusPanel.style.display = 'none';
            // }
            break;
        case 'experiment_error':
            updateExperimentStatus('error');
            alert('训练出错: ' + (data.error || '未知错误'));
            break;
        case 'round_started':
            if (data.round !== undefined || data.round_index !== undefined) {
                // 使用 round（1-based）或 round_index + 1
                const roundNum = data.round || (data.round_index !== undefined ? data.round_index + 1 : 0);
                const totalRounds = data.total_rounds || 0;
                updateProgress(roundNum, totalRounds);
            }
            break;
        case 'round_metrics_updated':
            updateRoundMetrics(data);
            break;
        case 'metrics_updated':
            // 单个指标更新
            if (data.round !== undefined && data.metric && data.value !== undefined) {
                updateSingleMetric(data.round, data.metric, data.value);
            }
            break;
        case 'node_status_updated':
            updateNodeStatus(data.node_id, data.data);
            break;
        case 'progress_update':
            // 更新进度
            if (data.round !== undefined && data.total_rounds !== undefined) {
                // data.round 可能是0-based或1-based，这里统一处理
                const roundNum = data.round >= 0 ? (data.round > 0 ? data.round : data.round + 1) : data.round + 1;
                updateProgress(roundNum, data.total_rounds);
            }
            break;
    }
    
    // 请求最新状态
    socket.emit('request_state');
}

// 更新单个指标
function updateSingleMetric(round, metric, value) {
    // 更新图表
    const charts = {
        'loss': lossChart,
        'f1': f1Chart,
        'accuracy': accuracyChart
    };
    
    const chart = charts[metric];
    if (chart) {
        // 确保数据数组足够大
        while (chart.data.datasets[0].data.length <= round) {
            chart.data.datasets[0].data.push(0);
            if (chart.data.labels.length <= round) {
                chart.data.labels.push(`轮次 ${chart.data.labels.length}`);
            }
        }
        
        chart.data.datasets[0].data[round] = value;
        chart.update('none'); // 不显示动画，更快更新
    }
}

// 更新UI
function updateUI(state) {
    // 更新实验状态
    if (state.experiment_running) {
        updateExperimentStatus('running');
        if (state.start_time && !startTime) {
            startTime = new Date(state.start_time);
        }
    } else if (state.end_time) {
        updateExperimentStatus('completed');
    }
    
    // 更新实验配置
    updateExperimentConfig(state.experiment_config);
    
    // 更新进度 - 确保使用正确的值
    const currentRound = state.current_round || 0;
    const totalRounds = state.total_rounds || (state.experiment_config?.round_limit || 0);
    updateProgress(currentRound, totalRounds);
    
    // 更新节点拓扑
    updateNodeTopology(state.nodes);
    
    // 更新图表
    updateCharts(state.global_metrics);
    
    // 更新详细训练状态
    updateDetailedTrainingStatus(state);
    
    // 更新节点详情
    updateNodesDetail(state.nodes);
    
    // 更新训练历史
    updateTrainingHistory(state.round_history);
}

// 更新实验状态
function updateExperimentStatus(status) {
    const statusEl = document.getElementById('experiment-status');
    statusEl.className = 'status-badge';
    
    switch(status) {
        case 'running':
            statusEl.textContent = '运行中';
            statusEl.classList.add('running');
            break;
        case 'completed':
            statusEl.textContent = '已完成';
            statusEl.classList.add('completed');
            break;
        default:
            statusEl.textContent = '未运行';
    }
}

// 更新实验配置
function updateExperimentConfig(config) {
    const configEl = document.getElementById('experiment-config');
    if (!config || Object.keys(config).length === 0) {
        configEl.innerHTML = '<p>等待实验开始...</p>';
        return;
    }
    
    configEl.innerHTML = `
        <p><strong>训练轮数:</strong> ${config.round_limit || 'N/A'}</p>
        <p><strong>数据集标签:</strong> ${(config.tags || []).join(', ')}</p>
        <p><strong>每轮Epochs:</strong> ${config.training_args?.epochs || 'N/A'}</p>
        <p><strong>学习率:</strong> ${config.training_args?.optimizer_args?.lr || 'N/A'}</p>
    `;
}

// 更新进度
function updateProgress(currentRound, totalRounds) {
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    
    // 确保数值有效
    currentRound = currentRound || 0;
    totalRounds = totalRounds || 0;
    
    if (totalRounds > 0) {
        const percentage = Math.min(100, Math.max(0, (currentRound / totalRounds) * 100));
        progressBar.style.width = `${percentage}%`;
        progressText.textContent = `轮次: ${currentRound} / ${totalRounds}`;
    } else {
        progressBar.style.width = '0%';
        progressText.textContent = '轮次: 0 / 0';
    }
}

// 更新轮次进度（单独函数）
function updateRoundProgress(roundNum) {
    // 直接使用socket请求状态
    socket.emit('request_state');
}

// 更新详细训练状态
function updateDetailedTrainingStatus(state) {
    const statusPanel = document.getElementById('detailed-training-status');
    if (!statusPanel) return;
    
    // 如果实验未运行，隐藏面板
    if (!state.experiment_running) {
        statusPanel.style.display = 'none';
        return;
    }
    
    // 显示面板
    statusPanel.style.display = 'block';
    
    const detailedStatus = state.detailed_status || {};
    const currentRound = state.current_round || 0;
    const totalRounds = state.total_rounds || 0;
    
    // 更新当前轮次
    document.getElementById('current-round-detail').textContent = 
        currentRound > 0 ? `${currentRound} / ${totalRounds}` : '-';
    
    // 更新轮次进度
    const roundProgress = totalRounds > 0 ? 
        Math.round((currentRound / totalRounds) * 100) : 0;
    document.getElementById('round-progress').textContent = `${roundProgress}%`;
    
    // 计算本轮已用时间
    let roundElapsed = '00:00';
    if (detailedStatus.round_start_time) {
        const startTime = new Date(detailedStatus.round_start_time);
        const elapsed = Math.floor((Date.now() - startTime.getTime()) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        roundElapsed = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }
    document.getElementById('round-elapsed-time').textContent = roundElapsed;
    
    // 计算预计剩余时间
    let estimatedTime = '计算中...';
    const roundTimes = detailedStatus.round_times || [];
    if (roundTimes.length > 0 && currentRound > 0 && totalRounds > 0) {
        const avgRoundTime = roundTimes.reduce((a, b) => a + b, 0) / roundTimes.length;
        const remainingRounds = totalRounds - currentRound;
        const estimatedSeconds = Math.ceil(avgRoundTime * remainingRounds);
        const hours = Math.floor(estimatedSeconds / 3600);
        const minutes = Math.floor((estimatedSeconds % 3600) / 60);
        const seconds = estimatedSeconds % 60;
        
        if (hours > 0) {
            estimatedTime = `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        } else {
            estimatedTime = `${minutes}:${String(seconds).padStart(2, '0')}`;
        }
    }
    document.getElementById('estimated-remaining-time').textContent = estimatedTime;
    
    // 计算平均每轮时间
    let avgRoundTime = '-';
    if (roundTimes.length > 0) {
        const avg = roundTimes.reduce((a, b) => a + b, 0) / roundTimes.length;
        const minutes = Math.floor(avg / 60);
        const seconds = Math.floor(avg % 60);
        avgRoundTime = `${minutes}:${String(seconds).padStart(2, '0')}`;
    }
    document.getElementById('avg-round-time').textContent = avgRoundTime;
    
    // 更新当前指标
    const currentMetrics = detailedStatus.current_metrics || {};
    document.getElementById('current-loss').textContent = 
        currentMetrics.loss !== null && currentMetrics.loss !== undefined ? 
        currentMetrics.loss.toFixed(4) : '-';
    document.getElementById('current-f1').textContent = 
        currentMetrics.f1 !== null && currentMetrics.f1 !== undefined ? 
        currentMetrics.f1.toFixed(4) : '-';
    document.getElementById('current-accuracy').textContent = 
        currentMetrics.accuracy !== null && currentMetrics.accuracy !== undefined ? 
        (currentMetrics.accuracy * 100).toFixed(2) + '%' : '-';
    
    // 更新节点训练状态
    updateNodesTrainingDetail(state.nodes, detailedStatus.nodes_training || {});
}

// 更新节点详细训练状态
function updateNodesTrainingDetail(nodes, nodesTraining) {
    const container = document.getElementById('nodes-training-detail');
    if (!container) return;
    
    if (!nodes || Object.keys(nodes).length === 0) {
        container.innerHTML = '<p class="empty-message">等待节点连接...</p>';
        return;
    }
    
    container.innerHTML = '';
    
    // 只显示有效的节点（node_1, node_2, node_3）
    const validNodeIds = ['node_1', 'node_2', 'node_3'];
    
    Object.entries(nodes)
        .filter(([nodeId]) => validNodeIds.includes(nodeId))
        .forEach(([nodeId, nodeInfo]) => {
            const trainingInfo = nodesTraining[nodeId] || {};
            const status = nodeInfo.status || trainingInfo.status || 'idle';
            
            const nodeEl = document.createElement('div');
            nodeEl.className = `node-detail-item ${status}`;
            
            const statusText = {
                'training': '训练中',
                'completed': '已完成',
                'idle': '空闲',
                'running': '运行中'
            }[status] || '未知';
            
            nodeEl.innerHTML = `
                <div class="node-detail-info">
                    <div class="node-detail-name">${nodeId}</div>
                    <div class="node-detail-status">状态: ${statusText}</div>
                </div>
                <div class="node-detail-progress">
                    ${nodeInfo.data_size ? `数据量: ${nodeInfo.data_size.toLocaleString()}` : ''}
                </div>
            `;
            
            container.appendChild(nodeEl);
        });
}

// 更新轮次指标
function updateRoundMetrics(data) {
    if (!data || !data.global_metrics) return;
    
    const round = data.round || 0;
    const metrics = data.global_metrics;
    
    // 更新图表
    if (metrics.loss !== undefined) {
        updateSingleMetric(round, 'loss', metrics.loss);
    }
    if (metrics.f1 !== undefined) {
        updateSingleMetric(round, 'f1', metrics.f1);
    }
    if (metrics.accuracy !== undefined) {
        updateSingleMetric(round, 'accuracy', metrics.accuracy);
    }
    
    // 请求完整状态更新
    socket.emit('request_state');
}

// 更新节点拓扑
function updateNodeTopology(nodes) {
    const container = document.getElementById('nodes-container');
    
    if (!nodes || Object.keys(nodes).length === 0) {
        container.innerHTML = '<p class="empty-message">等待节点连接...</p>';
        return;
    }
    
    container.innerHTML = '';
    
    Object.values(nodes).forEach(node => {
        const nodeEl = document.createElement('div');
        nodeEl.className = 'node-item';
        
        const statusClass = getNodeStatusClass(node.status);
        const icon = getNodeIcon(node.status);
        
        nodeEl.innerHTML = `
            <div class="node-connection"></div>
            <div class="node-icon node ${statusClass}">${icon}</div>
            <div class="node-label">节点 ${node.id}</div>
        `;
        
        container.appendChild(nodeEl);
    });
}

// 获取节点状态类
function getNodeStatusClass(status) {
    switch(status) {
        case 'training':
            return 'training';
        case 'completed':
            return 'completed';
        case 'error':
            return 'error';
        default:
            return '';
    }
}

// 获取节点图标
function getNodeIcon(status) {
    switch(status) {
        case 'training':
            return '⚙️';
        case 'completed':
            return '✅';
        case 'error':
            return '❌';
        default:
            return '🖥️';
    }
}

// 更新图表
function updateCharts(metrics) {
    if (!metrics || !metrics.rounds) return;
    
    const rounds = metrics.rounds.map(r => `轮次 ${r}`);
    
    // 更新Loss图表
    if (metrics.loss && metrics.loss.length > 0) {
        lossChart.data.labels = rounds;
        lossChart.data.datasets[0].data = metrics.loss;
        lossChart.update();
    }
    
    // 更新F1图表
    if (metrics.f1 && metrics.f1.length > 0) {
        f1Chart.data.labels = rounds;
        f1Chart.data.datasets[0].data = metrics.f1;
        f1Chart.update();
    }
    
    // 更新Accuracy图表
    if (metrics.accuracy && metrics.accuracy.length > 0) {
        accuracyChart.data.labels = rounds;
        accuracyChart.data.datasets[0].data = metrics.accuracy;
        accuracyChart.update();
    }
}

// 更新节点详情
function updateNodesDetail(nodes) {
    const container = document.getElementById('nodes-detail');
    
    if (!nodes || Object.keys(nodes).length === 0) {
        container.innerHTML = '<p class="empty-message">等待节点数据...</p>';
        return;
    }
    
    container.innerHTML = '';
    
    Object.values(nodes).forEach(node => {
        const card = document.createElement('div');
        card.className = 'node-detail-card';
        
        const metrics = node.metrics || {};
        const statusTag = getStatusTag(node.status);
        
        card.innerHTML = `
            <h4>节点 ${node.id}</h4>
            ${statusTag}
            <div class="metric">
                <span class="metric-label">数据量:</span>
                <span class="metric-value">${node.data_size || 0}</span>
            </div>
            <div class="metric">
                <span class="metric-label">损失:</span>
                <span class="metric-value">${metrics.loss ? metrics.loss.toFixed(4) : 'N/A'}</span>
            </div>
            <div class="metric">
                <span class="metric-label">F1分数:</span>
                <span class="metric-value">${metrics.f1 ? metrics.f1.toFixed(4) : 'N/A'}</span>
            </div>
            <div class="metric">
                <span class="metric-label">准确率:</span>
                <span class="metric-value">${metrics.accuracy ? (metrics.accuracy * 100).toFixed(2) + '%' : 'N/A'}</span>
            </div>
            <div class="metric">
                <span class="metric-label">训练时间:</span>
                <span class="metric-value">${metrics.training_time ? metrics.training_time.toFixed(2) + 's' : 'N/A'}</span>
            </div>
            <div class="metric">
                <span class="metric-label">最后更新:</span>
                <span class="metric-value">${formatTime(node.last_update)}</span>
            </div>
        `;
        
        container.appendChild(card);
    });
}

// 获取状态标签
function getStatusTag(status) {
    const statusMap = {
        'idle': { text: '空闲', class: 'idle' },
        'training': { text: '训练中', class: 'training' },
        'uploading': { text: '上传中', class: 'training' },
        'completed': { text: '已完成', class: 'completed' },
        'error': { text: '错误', class: 'error' }
    };
    
    const statusInfo = statusMap[status] || statusMap['idle'];
    return `<span class="status-tag ${statusInfo.class}">${statusInfo.text}</span>`;
}

// 更新训练历史
function updateTrainingHistory(history) {
    const container = document.getElementById('training-history');
    
    if (!history || history.length === 0) {
        container.innerHTML = '<p class="empty-message">暂无训练历史</p>';
        return;
    }
    
    container.innerHTML = '';
    
    // 按轮次倒序显示
    [...history].reverse().forEach(roundData => {
        const item = document.createElement('div');
        item.className = 'history-item';
        
        const globalMetrics = roundData.global || {};
        const nodeCount = Object.keys(roundData.nodes || {}).length;
        
        item.innerHTML = `
            <div class="history-item-header">
                <h4>第 ${roundData.round + 1} 轮训练</h4>
                <span class="timestamp">${formatTime(roundData.timestamp)}</span>
            </div>
            <div class="history-metrics">
                <div class="history-metric">
                    <div class="history-metric-label">参与节点</div>
                    <div class="history-metric-value">${nodeCount}</div>
                </div>
                <div class="history-metric">
                    <div class="history-metric-label">全局损失</div>
                    <div class="history-metric-value">${globalMetrics.loss ? globalMetrics.loss.toFixed(4) : 'N/A'}</div>
                </div>
                <div class="history-metric">
                    <div class="history-metric-label">F1分数</div>
                    <div class="history-metric-value">${globalMetrics.f1 ? globalMetrics.f1.toFixed(4) : 'N/A'}</div>
                </div>
                <div class="history-metric">
                    <div class="history-metric-label">准确率</div>
                    <div class="history-metric-value">${globalMetrics.accuracy ? (globalMetrics.accuracy * 100).toFixed(2) + '%' : 'N/A'}</div>
                </div>
            </div>
        `;
        
        container.appendChild(item);
    });
}

// 这些函数已经在上面定义了，删除重复定义

// 格式化时间
function formatTime(timestamp) {
    if (!timestamp) return 'N/A';
    try {
        const date = new Date(timestamp);
        return date.toLocaleString('zh-CN');
    } catch (e) {
        return timestamp;
    }
}

// 开始计时
function startElapsedTime() {
    elapsedInterval = setInterval(() => {
        if (startTime) {
            const elapsed = Math.floor((new Date() - startTime) / 1000);
            const hours = Math.floor(elapsed / 3600);
            const minutes = Math.floor((elapsed % 3600) / 60);
            const seconds = elapsed % 60;
            
            document.getElementById('elapsed-time').textContent = 
                `运行时间: ${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        }
    }, 1000);
}
