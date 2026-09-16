function summary = analyze_VLC_channel_rolloff(sample_rate_hz, channel_Fs, factor, signal_length)
% Measure the linear roll-off of the current VLC_channel_JY implementation.
% Run with defaults matching B1_gen_WGN.m:
%   analyze_VLC_channel_rolloff
% Or specify sample rate, channel Fs, attenuation factor and record length:
%   analyze_VLC_channel_rolloff(100e9, 100, 30, 204960)
% The nonlinear argument is set to zero for this measurement. The channel's
% internal ISI and AWGN switches must also be off. Original data are not saved
% or replaced. Outputs go to <repository>/result/channel_rolloff/.

if nargin < 1, sample_rate_hz = 100e9; end
if nargin < 2, channel_Fs = 100; end
if nargin < 3, factor = 30; end
if nargin < 4, signal_length = 204960; end
validateattributes(sample_rate_hz, {'numeric'}, {'scalar','positive','finite'});
validateattributes(channel_Fs, {'numeric'}, {'scalar','positive','finite'});
validateattributes(factor, {'numeric'}, {'scalar','positive','finite'});
validateattributes(signal_length, {'numeric'}, {'scalar','integer','>=',2048,'even'});

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
output_dir = fullfile(fileparts(script_dir), 'result', 'channel_rolloff');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
rng_before = rng;
restore_rng = onCleanup(@() rng(rng_before)); %#ok<NASGU>

N = signal_length;
carrierno = 512;
upsampleno = 4;
CP = 0;
impulse = zeros(N,1);
impulse(1) = 1;
h = VLC_channel_JY(impulse, carrierno, upsampleno, CP, channel_Fs, factor, 70, 0);
H = fft(h);

% Independent frequency-grid calculation, including spline interpolation and
% the real(ifft(...)) operation used by the channel for real input signals.
base_length = (carrierno + CP) * upsampleno;
internal_frequency = (1:base_length/2) * (2*channel_Fs/base_length);
positive_side = exp(-internal_frequency/factor);
raw_response = spline(1:base_length, [positive_side fliplr(positive_side)].', ...
    linspace(1,base_length,N)).';
negative_bin = [1, N:-1:2];
expected_response = (raw_response + raw_response(negative_bin))/2;
verification_error = max(abs(H-expected_response));
assert(verification_error < 1e-10, ...
    'Channel differs from linear roll-off. Check ISI/AWGN switches and function version.');

bins = (0:N/2).';
frequency_hz = bins * sample_rate_hz/N;
gain = abs(H(1:N/2+1));
dc_gain = gain(1);
relative_gain = gain/dc_gain;
relative_db = 20*log10(relative_gain);
absolute_db = 20*log10(gain);

% Approximation: internal frequency spans 0..channel_Fs while physical
% frequency spans 0..sample_rate_hz/2. The exact curve also has grid offsets.
approximate_gain = exp(-2*channel_Fs/factor * frequency_hz/sample_rate_hz);
approximate_db = 20*log10(approximate_gain);
slope_db_per_GHz = -(20/log(10)) * 2*channel_Fs/factor / sample_rate_hz * 1e9;
first_below = find(relative_db <= -3,1);
if isempty(first_below)
    bandwidth_3db_hz = NaN;
else
    interval = first_below-1:first_below;
    bandwidth_3db_hz = interp1(relative_db(interval), frequency_hz(interval), -3);
end

requested_GHz = [0;1;2.5;5;10;12.5;20;25;40;50];
requested_GHz = requested_GHz(requested_GHz <= sample_rate_hz/2/1e9);
measured_gain = interp1(frequency_hz, gain, requested_GHz*1e9);
gain_relative_to_dc = measured_gain/dc_gain;
loss_db = -20*log10(gain_relative_to_dc);
power_percent = 100*gain_relative_to_dc.^2;
summary = table(requested_GHz, measured_gain, gain_relative_to_dc, loss_db, power_percent, ...
    'VariableNames', {'frequency_GHz','absolute_amplitude_gain', ...
    'amplitude_relative_to_DC','loss_relative_to_DC_dB','power_relative_to_DC_percent'});
writetable(summary, fullfile(output_dir,'key_frequencies.csv'));
curve = table(frequency_hz/1e9, gain, absolute_db, relative_db, approximate_db, ...
    'VariableNames', {'frequency_GHz','amplitude_gain','absolute_gain_dB', ...
    'gain_relative_to_DC_dB','approximate_relative_gain_dB'});
writetable(curve, fullfile(output_dir,'frequency_response.csv'));
save(fullfile(output_dir,'channel_rolloff.mat'), 'h','H','frequency_hz', ...
    'summary','sample_rate_hz','channel_Fs','factor','N','bandwidth_3db_hz', ...
    'slope_db_per_GHz','verification_error');

fig = figure('Visible','off','Color','w','Position',[100,100,1120,440]);
close_figure = onCleanup(@() close(fig)); %#ok<NASGU>
layout = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
ax1 = nexttile(layout);
plot(ax1, frequency_hz/1e9,relative_db,'Color',[0.12,0.36,0.68],'LineWidth',2);
hold(ax1,'on');
plot(ax1, frequency_hz/1e9,approximate_db,'--','Color',[0.8,0.35,0.1],'LineWidth',1.3);
yline(ax1,-3,':','-3 dB','LabelHorizontalAlignment','left','HandleVisibility','off');
if isfinite(bandwidth_3db_hz)
    xline(ax1,bandwidth_3db_hz/1e9,':',sprintf('%.3f GHz',bandwidth_3db_hz/1e9), ...
        'LabelVerticalAlignment','top','LabelOrientation','horizontal', ...
        'LabelHorizontalAlignment','right','HandleVisibility','off');
end
xlabel(ax1,'Frequency (GHz)'); ylabel(ax1,'Gain relative to DC (dB)');
title(ax1,'Linear channel: high-frequency attenuation');
legend(ax1,{'Impulse measurement','Exponential approximation'},'Location','southwest');
grid(ax1,'on'); xlim(ax1,[0,sample_rate_hz/2/1e9]);
ax2 = nexttile(layout);
plot(ax2,frequency_hz/1e9,relative_gain,'Color',[0.12,0.36,0.68],'LineWidth',2);
hold(ax2,'on');
scatter(ax2,requested_GHz,gain_relative_to_dc,24,[0.8,0.35,0.1],'filled');
xlabel(ax2,'Frequency (GHz)'); ylabel(ax2,'Amplitude / DC amplitude');
title(ax2,'Remaining signal amplitude'); grid(ax2,'on');
xlim(ax2,[0,sample_rate_hz/2/1e9]); ylim(ax2,[0,1.05]);
set([ax1,ax2],'FontName','Arial','FontSize',11,'LineWidth',1,'Box','off');
title(layout,sprintf('Sample rate %.0f GSa/s | channel Fs = %g | factor = %g | nonlinearity / extra FIR / AWGN off', ...
    sample_rate_hz/1e9,channel_Fs,factor),'FontSize',12);
exportgraphics(fig,fullfile(output_dir,'channel_rolloff.png'),'Resolution',160);
plot_VLC_channel_log_response;

fprintf('Source: %s\n',which('VLC_channel_JY'));
fprintf('Sample rate = %.6g GSa/s; N = %d; channel Fs = %g; factor = %g\n', ...
    sample_rate_hz/1e9,N,channel_Fs,factor);
fprintf('DC amplitude gain = %.9g (%+.6f dB)\n',dc_gain,absolute_db(1));
fprintf('Approximate slope = %.6f dB/GHz\n',slope_db_per_GHz);
fprintf('-3 dB bandwidth relative to DC = %.6f GHz\n',bandwidth_3db_hz/1e9);
fprintf('Impulse vs independent calculation max error = %.3g\n',verification_error);
disp(summary);
fprintf('Outputs: %s\n',output_dir);
end
