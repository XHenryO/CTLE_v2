

%% 1. 独立设置三种方案的参数
% 方案一：1-stage CTLE
ctle_1stage = struct('fz_GHz', 5, 'fp_GHz', [15, 35], 'A0', 1);

% 方案二：2-stage CTLE A，括号中的 1 / 2 代表第一级 / 第二级
ctle_2stage_A = [ ...
    struct('fz_GHz', 0.1, 'fp_GHz', [0.4, 0.6], 'A0', 1), ... % 第一级
    struct('fz_GHz', 5, 'fp_GHz', [15, 35], 'A0', 1)];    % 第二级

% 方案三：2-stage CTLE B，每一级均可独立修改
ctle_2stage_B = [ ...
    struct('fz_GHz', 5, 'fp_GHz', [15, 35], 'A0', 1), ... % 第一级
    struct('fz_GHz', 5, 'fp_GHz', [15, 35], 'A0', 1)];    % 第二级

%% 2. 公共绘图设置
f_min_GHz = 0.1;
f_max_GHz = 100;
num_points = 3000;
save_plot = false; % true：保存 PNG、FIG 到项目 result/custom_ctle

validateattributes([f_min_GHz, f_max_GHz], {'numeric'}, ...
    {'real', 'finite', 'positive', 'vector', 'numel', 2});
assert(f_max_GHz > f_min_GHz, '结束频率必须大于起始频率。');
validateattributes(num_points, {'numeric'}, {'scalar', 'integer', '>=', 2});
validate_stages(ctle_1stage, 1);
validate_stages(ctle_2stage_A, 2);
validate_stages(ctle_2stage_B, 2);

%% 3. 三种方案的传输函数，可在工作区直接调用
% 例如 H_2stage_A(0) 或 H_2stage_A(1i*2*pi*10e9)。
H_1stage   = @(s) cascade_response(s, ctle_1stage);
H_2stage_A = @(s) cascade_response(s, ctle_2stage_A);
H_2stage_B = @(s) cascade_response(s, ctle_2stage_B);

frequency_GHz = logspace(log10(f_min_GHz), log10(f_max_GHz), num_points).';
frequency_Hz = frequency_GHz*1e9;
s = 1i*2*pi*frequency_Hz;
responses = [H_1stage(s), H_2stage_A(s), H_2stage_B(s)];
magnitude_dB = 20*log10(abs(responses));
dc_gains = [H_1stage(0), H_2stage_A(0), H_2stage_B(0)];

%% 4. 只绘制一张幅度图，三条曲线叠加
fig = figure('Color', 'w', 'Name', 'CTLE stage comparison', ...
    'NumberTitle', 'off', 'Position', [150, 100, 1100, 640]);
ax = axes(fig);
colors = [0.10 0.36 0.68; 0.85 0.32 0.10; 0.18 0.56 0.34];
styles = {'-', '--', '-.'};
labels = {'1-stage CTLE', '2-stage CTLE A', '2-stage CTLE B'};
for k = 1:3
    semilogx(ax, frequency_GHz, magnitude_dB(:,k), styles{k}, ...
        'Color', colors(k,:), 'LineWidth', 2.3, 'DisplayName', labels{k});
    hold(ax, 'on');
end
yline(ax, 0, ':', 'Color', [0.55 0.55 0.55], 'HandleVisibility', 'off');
grid(ax, 'on');
set(ax, 'XScale', 'log', 'XMinorGrid', 'on', 'FontName', 'Arial', ...
    'FontSize', 12, 'LineWidth', 1, 'Box', 'off');
xlim(ax, [f_min_GHz, f_max_GHz]);
xlabel(ax, 'Frequency (GHz, log scale)');
ylabel(ax, 'Magnitude (dB)');
title(ax, 'CTLE magnitude: one stage vs. two cascaded stages', 'FontSize', 17);
subtitle(ax, 'Independent stage parameters | Absolute gain, no normalization', 'FontSize', 12);
legend(ax, 'Location', 'northwest');

% 输出实际参数和所画频率范围内的峰值，方便核对。
settings = {ctle_1stage, ctle_2stage_A, ctle_2stage_B};
for k = 1:3
    fprintf('\n%s\n', labels{k});
    for j = 1:numel(settings{k})
        p = settings{k}(j);
        fprintf('  Stage %d: A0=%g, zero=%g GHz, poles=%s GHz\n', ...
            j, p.A0, p.fz_GHz, mat2str(p.fp_GHz));
    end
    [peak_dB, index] = max(magnitude_dB(:,k));
    fprintf('  DC gain = %g (%+.3f dB)\n', dc_gains(k), 20*log10(abs(dc_gains(k))));
    fprintf('  Peak within plotted range = %.3f dB at %.3f GHz\n', ...
        peak_dB, frequency_GHz(index));
end

if save_plot
    project_dir = fileparts(fileparts(mfilename('fullpath')));
    output_dir = fullfile(project_dir, 'result', 'custom_ctle');
    if ~isfolder(output_dir), mkdir(output_dir); end
    exportgraphics(fig, fullfile(output_dir, 'custom_ctle_stages.png'), 'Resolution', 180);
    savefig(fig, fullfile(output_dir, 'custom_ctle_stages.fig'));
    fprintf('\nSaved to: %s\n', output_dir);
end

%% 本地函数：每一级独立计算，再将各级传输函数相乘
function validate_stages(stages, expected_count)
assert(numel(stages)==expected_count, 'CTLE 级数与方案定义不符。');
for k = 1:numel(stages)
    validateattributes(stages(k).fz_GHz, {'numeric'}, {'scalar','real','finite','positive'});
    validateattributes(stages(k).fp_GHz, {'numeric'}, {'vector','nonempty','real','finite','positive'});
    validateattributes(stages(k).A0, {'numeric'}, {'scalar','real','finite','nonzero'});
end
end

function H = cascade_response(s, stages)
H = ones(size(s));
for k = 1:numel(stages)
    p = stages(k);
    stage_H = p.A0 .* (1 + s./(2*pi*p.fz_GHz*1e9));
    for pole_GHz = p.fp_GHz(:).'
        stage_H = stage_H ./ (1 + s./(2*pi*pole_GHz*1e9));
    end
    H = H .* stage_H;
end
end
