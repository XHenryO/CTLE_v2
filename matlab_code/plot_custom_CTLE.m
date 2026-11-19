%% 可调 CTLE 对比：一零点两极点 / 一零点三极点
% 修改下面的参数，然后直接运行本脚本。
% 连续时间传输函数：
%                  (1 + s / wz)
% H(s) = A0 * ---------------------------
%             (1 + s / wp1)(1 + s / wp2)
% wz = 2*pi*fz，wp1 = 2*pi*fp1，wp2 = 2*pi*fp2。
% H(0) = A0。这里画模拟传输函数，不做采样或双线性变换。
% 第二种 CTLE：H3(s) = A0_3pole*(1+s/wz3) /
%                         ((1+s/wp13)*(1+s/wp23)*(1+s/wp33))。
% 两种 CTLE 独立设置，在同一幅频图、相频图中叠加比较。
% 不需要 Control System Toolbox 或 Signal Processing Toolbox。

%% 1. 在这里设置参数：第一种 CTLE（一零点、两极点）
fz_GHz  = 5;       % 零点频率，单位 GHz
fp1_GHz = 15;      % 第一个极点频率，单位 GHz
fp2_GHz = 35;      % 第二个极点频率，单位 GHz
A0 = 1;           % 直流（零频）幅度增益，线性倍数；1 对应 0 dB
% 如果希望用 dB 设置，例如 6 dB，可把上一行换成：A0 = 10^(6/20);

%% 第二种 CTLE（一零点、三极点），所有参数均可独立修改
fz_3pole_GHz  = 5;    % 零点频率，GHz
fp1_3pole_GHz = 15;   % 第一个极点，GHz
fp2_3pole_GHz = 35;   % 第二个极点，GHz
fp3_3pole_GHz = 100;   % 新增的第三个极点，GHz
A0_3pole = 1;        % 直流增益，线性倍数

%% 公共绘图设置
f_min_GHz = 0.1;   % 绘图起始频率；对数横轴必须大于 0
f_max_GHz = 100;   % 绘图结束频率
num_points = 3000;
save_plot = false; % 改成 true，会将 PNG 和 FIG 保存到 result/custom_ctle
show_corner_lines = false; % 改成 true，显示两种 CTLE 的零极点位置线

%% 2. 定义传输函数并计算频率响应
validateattributes([fz_GHz, fp1_GHz, fp2_GHz], {'numeric'}, ...
    {'real', 'finite', 'positive', 'vector', 'numel', 3});
validateattributes(A0, {'numeric'}, {'real', 'finite', 'scalar', 'nonzero'});
validateattributes([fz_3pole_GHz, fp1_3pole_GHz, fp2_3pole_GHz, fp3_3pole_GHz], ...
    {'numeric'}, {'real', 'finite', 'positive', 'vector', 'numel', 4});
validateattributes(A0_3pole, {'numeric'}, {'real', 'finite', 'scalar', 'nonzero'});
validateattributes([f_min_GHz, f_max_GHz], {'numeric'}, ...
    {'real', 'finite', 'positive', 'vector', 'numel', 2});
assert(f_max_GHz > f_min_GHz, '结束频率必须大于起始频率。');
validateattributes(num_points, {'numeric'}, {'scalar', 'integer', '>=', 2});

wz  = 2*pi*fz_GHz*1e9;
wp1 = 2*pi*fp1_GHz*1e9;
wp2 = 2*pi*fp2_GHz*1e9;

% H 是可以直接调用的传输函数，例如 H(0)、H(1i*2*pi*10e9)。
H = @(s) A0 .* (1 + s./wz) ./ ((1 + s./wp1) .* (1 + s./wp2));
wz3  = 2*pi*fz_3pole_GHz*1e9;
wp13 = 2*pi*fp1_3pole_GHz*1e9;
wp23 = 2*pi*fp2_3pole_GHz*1e9;
wp33 = 2*pi*fp3_3pole_GHz*1e9;
H3 = @(s) A0_3pole .* (1 + s./wz3) ./ ...
    ((1 + s./wp13) .* (1 + s./wp23) .* (1 + s./wp33));

% 分子、分母按 s 的降幂排列；如安装了控制工具箱，可用 tf(num, den)。
num = A0 * [1/wz, 1];
den = conv([1/wp1, 1], [1/wp2, 1]);
num3 = A0_3pole * [1/wz3, 1];
den3 = conv(conv([1/wp13, 1], [1/wp23, 1]), [1/wp33, 1]);

frequency_GHz = logspace(log10(f_min_GHz), log10(f_max_GHz), num_points);
frequency_Hz = frequency_GHz * 1e9;
response = H(1i*2*pi*frequency_Hz);
magnitude_dB = 20*log10(abs(response));
phase_deg = unwrap(angle(response))*180/pi;
dc_gain_dB = 20*log10(abs(A0));
response3 = H3(1i*2*pi*frequency_Hz);
magnitude3_dB = 20*log10(abs(response3));
phase3_deg = unwrap(angle(response3))*180/pi;
dc_gain3_dB = 20*log10(abs(A0_3pole));

%% 3. 绘图：对数频率横轴
fig = figure('Color', 'w', 'Name', 'Adjustable CTLE response', ...
    'NumberTitle', 'off', 'Position', [150, 80, 1150, 820]);
layout = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
title(layout, 'CTLE comparison: two poles vs. three poles', 'FontSize', 18);
subtitle(layout, {sprintf('2 poles: A_0 = %g (%+.2f dB), f_z = %g, f_{p1} = %g, f_{p2} = %g GHz', ...
    A0, dc_gain_dB, fz_GHz, fp1_GHz, fp2_GHz), ...
    sprintf('3 poles: A_0 = %g (%+.2f dB), f_z = %g, f_{p1} = %g, f_{p2} = %g, f_{p3} = %g GHz', ...
    A0_3pole, dc_gain3_dB, fz_3pole_GHz, fp1_3pole_GHz, fp2_3pole_GHz, fp3_3pole_GHz)}, ...
    'FontSize', 12);
color_two = [0.10, 0.36, 0.68];
color_three = [0.85, 0.32, 0.10];

ax_magnitude = nexttile(layout);
semilogx(ax_magnitude, frequency_GHz, magnitude_dB, 'LineWidth', 2.2, ...
    'Color', color_two, 'DisplayName', '1 zero, 2 poles');
hold(ax_magnitude, 'on');
semilogx(ax_magnitude, frequency_GHz, magnitude3_dB, '--', 'LineWidth', 2.2, ...
    'Color', color_three, 'DisplayName', '1 zero, 3 poles');
ylabel(ax_magnitude, 'Magnitude (dB)');
title(ax_magnitude, 'Magnitude response');

ax_phase = nexttile(layout);
semilogx(ax_phase, frequency_GHz, phase_deg, 'LineWidth', 2.2, ...
    'Color', color_two, 'DisplayName', '1 zero, 2 poles');
hold(ax_phase, 'on');
semilogx(ax_phase, frequency_GHz, phase3_deg, '--', 'LineWidth', 2.2, ...
    'Color', color_three, 'DisplayName', '1 zero, 3 poles');
ylabel(ax_phase, 'Phase (degrees)');
title(ax_phase, 'Phase response');

corner_groups = {[fz_GHz, fp1_GHz, fp2_GHz], ...
    [fz_3pole_GHz, fp1_3pole_GHz, fp2_3pole_GHz, fp3_3pole_GHz]};
corner_colors = [color_two; color_three];
for ax = [ax_magnitude, ax_phase]
    if show_corner_lines
        for group = 1:2
            for corner = corner_groups{group}
                if corner >= f_min_GHz && corner <= f_max_GHz
                    xline(ax, corner, ':', 'Color', corner_colors(group,:), ...
                        'LineWidth', 0.8, 'HandleVisibility', 'off');
                end
            end
        end
    end
    grid(ax, 'on');
    set(ax, 'XScale', 'log', 'XMinorGrid', 'on', 'FontName', 'Arial', ...
        'FontSize', 12, 'LineWidth', 1, 'Box', 'off');
    xlim(ax, [f_min_GHz, f_max_GHz]);
    xlabel(ax, 'Frequency (GHz, log scale)');
    legend(ax, 'Location', 'southwest');
end
linkaxes([ax_magnitude, ax_phase], 'x');

fprintf('H(0) = %g, DC magnitude = %+.4f dB\n', H(0), dc_gain_dB);
fprintf('Zero: s = -%.6g rad/s\n', wz);
fprintf('Poles: s = -%.6g, -%.6g rad/s\n', wp1, wp2);
fprintf('H3(0) = %g, DC magnitude = %+.4f dB\n', H3(0), dc_gain3_dB);
fprintf('3-pole CTLE zero: s = -%.6g rad/s\n', wz3);
fprintf('3-pole CTLE poles: s = -%.6g, -%.6g, -%.6g rad/s\n', wp13, wp23, wp33);

%% 4. 可选保存
if save_plot
    project_dir = fileparts(fileparts(mfilename('fullpath')));
    output_dir = fullfile(project_dir, 'result', 'custom_ctle');
    if ~isfolder(output_dir), mkdir(output_dir); end
    exportgraphics(fig, fullfile(output_dir, 'custom_ctle_response.png'), 'Resolution', 180);
    savefig(fig, fullfile(output_dir, 'custom_ctle_response.fig'));
    fprintf('Saved to: %s\n', output_dir);
end
