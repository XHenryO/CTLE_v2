function results = plot_post_stage_sweep_responses(visible)
%PLOT_POST_STAGE_SWEEP_RESPONSES Plot the trained Post sweep at SNR 5/10/15.
%   results = plot_post_stage_sweep_responses('off') reads the best saved
%   1--6-stage CTLE models and the 9-tap FPost model.  It creates one
%   two-panel figure per SNR: equalizer response and channel+equalizer
%   response.  All frequency axes are logarithmic and all three figures use
%   the same y limits so that the SNR cases can be compared directly.
%
%   The Keras 2 .keras files used by this project are HDF5 files.  Their
%   trained weights are read directly, so this function does not need
%   Python or TensorFlow.

if nargin < 1, visible = 'on'; end

root = fileparts(fileparts(mfilename('fullpath')));
model_dir = fullfile(root, 'model', 'model_trained');
output_dir = fullfile(root, 'model', 'result', 'post_stage_sweep', ...
    'frequency_response');
if ~isfolder(output_dir), mkdir(output_dir); end

snr_values = [5 10 15];
model_names = {'1-stage CTLE', '2-stage CTLE', '3-stage CTLE', ...
    '4-stage CTLE', '5-stage CTLE', '6-stage CTLE', '9-tap FPost'};
file_patterns = {'pw_post_ctle_nn_channel_snr%d_model.keras', ...
    'pw_post_2stage_ctle_nn_channel_snr%d_model.keras', ...
    'pw_post_3stage_ctle_nn_channel_snr%d_model.keras', ...
    'pw_post_4stage_ctle_nn_channel_snr%d_model.keras', ...
    'pw_post_5stage_ctle_nn_channel_snr%d_model.keras', ...
    'pw_post_6stage_ctle_nn_channel_snr%d_model.keras', ...
    'fpost_nn_channel_snr%d_tap9_model.keras'};
expected_stages = [1 2 3 4 5 6 0];

results = repmat(struct(), 1, numel(snr_values));
for snr_index = 1:numel(snr_values)
    snr_db = snr_values(snr_index);
    models = cell(1, numel(model_names));
    reference_Fs = NaN;
    for model_index = 1:numel(model_names)
        file = fullfile(model_dir, sprintf(file_patterns{model_index}, snr_db));
        assert(isfile(file), 'Best model does not exist: %s', file);
        models{model_index} = read_saved_model(file, ...
            expected_stages(model_index), reference_Fs);
        if ~isfinite(reference_Fs), reference_Fs = models{model_index}.Fs; end
    end
    assert(all(cellfun(@(model) model.Fs == reference_Fs, models)), ...
        'All models must use the same sample rate.');

    frequency_Hz = logspace(log10(0.1e9), ...
        log10(0.499 * reference_Fs), 2400).';
    equalizer = complex(zeros(numel(frequency_Hz), numel(models)));
    channel = equalizer;
    combined = equalizer;
    for model_index = 1:numel(models)
        model = models{model_index};
        equalizer(:, model_index) = equalizer_response(model, frequency_Hz);
        omega = 2*pi*frequency_Hz / model.Fs;
        channel(:, model_index) = exp(-1i * omega * ...
            model.channel_delays.') * model.channel_taps;
        combined(:, model_index) = equalizer(:, model_index) .* ...
            channel(:, model_index);
    end

    results(snr_index).snr_db = snr_db;
    results(snr_index).frequency_Hz = frequency_Hz;
    results(snr_index).model_names = model_names;
    results(snr_index).models = models;
    results(snr_index).equalizer = equalizer;
    results(snr_index).channel = channel;
    results(snr_index).combined = combined;
end

% Shared limits make the three SNR figures visually comparable.
equalizer_cells = arrayfun(@(x) magnitude_db(x.equalizer), ...
    results, 'UniformOutput', false);
combined_cells = arrayfun(@(x) magnitude_db(x.combined), ...
    results, 'UniformOutput', false);
all_equalizer_db = vertcat(equalizer_cells{:});
all_combined_db = vertcat(combined_cells{:});
equalizer_limits = padded_limits(all_equalizer_db);
combined_limits = padded_limits(all_combined_db);

colors = [0.0000 0.4470 0.7410; 0.8500 0.3250 0.0980; ...
          0.4660 0.6740 0.1880; 0.4940 0.1840 0.5560; ...
          0.3010 0.7450 0.9330; 0.6350 0.0780 0.1840; ...
          0.9290 0.6940 0.1250];
line_styles = {'-', '--', '-.', ':', '-', '--', '-.'};

for snr_index = 1:numel(results)
    result = results(snr_index);
    snr_db = result.snr_db;
    fig = figure('Color', 'w', 'Visible', visible, ...
        'Position', [80 80 1450 620]);
    tiles = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', ...
        'Padding', 'compact');
    title(tiles, sprintf('Post equalizers trained at SNR = %g dB', snr_db), ...
        'FontSize', 19, 'FontWeight', 'bold');
    subtitle(tiles, sprintf(['Absolute magnitude | F_s = %g GSa/s | ' ...
        'Best saved model for each architecture'], ...
        result.models{1}.Fs/1e9), 'FontSize', 12);

    plot_panel(nexttile(tiles, 1), result.frequency_Hz, result.equalizer, ...
        model_names, colors, line_styles, equalizer_limits, ...
        'Equalizer response', false, []);
    plot_panel(nexttile(tiles, 2), result.frequency_Hz, result.combined, ...
        model_names, colors, line_styles, combined_limits, ...
        'Channel + equalizer response', true, result.channel(:,1));
    xlabel(tiles, ['Dotted vertical line: 12.5 GHz (25 GBd Nyquist). ' ...
        'The response excludes AWGN and sampling operations.'], 'FontSize', 11);

    base = fullfile(output_dir, sprintf('post_stage_sweep_snr%d', snr_db));
    save([base '.mat'], 'result');
    savefig(fig, [base '.fig']);
    exportgraphics(fig, [base '.png'], 'Resolution', 180);

    table_out = table(result.frequency_Hz, ...
        'VariableNames', {'frequency_Hz'});
    keys = {'ctle_1stage', 'ctle_2stage', 'ctle_3stage', ...
        'ctle_4stage', 'ctle_5stage', 'ctle_6stage', 'fpost_9tap'};
    for model_index = 1:numel(keys)
        table_out.([keys{model_index} '_equalizer_dB']) = ...
            magnitude_db(result.equalizer(:,model_index));
        table_out.([keys{model_index} '_combined_dB']) = ...
            magnitude_db(result.combined(:,model_index));
    end
    table_out.channel_dB = magnitude_db(result.channel(:,1));
    writetable(table_out, [base '.csv']);
    fprintf('Saved SNR %g dB figure and response data: %s\n', snr_db, base);
    if strcmpi(visible, 'off'), close(fig); end
end
end

function plot_panel(ax, frequency_Hz, response, labels, colors, ...
        line_styles, y_limits, panel_title, show_channel, channel)
hold(ax, 'on');
lines = gobjects(1, size(response,2));
for model_index = 1:size(response,2)
    lines(model_index) = semilogx(ax, frequency_Hz/1e9, ...
        magnitude_db(response(:,model_index)), line_styles{model_index}, ...
        'LineWidth', 2.1, 'Color', colors(model_index,:));
end
if show_channel
    channel_line = semilogx(ax, frequency_Hz/1e9, magnitude_db(channel), ...
        '-', 'LineWidth', 1.6, 'Color', [0.35 0.35 0.35]);
end
yline(ax, 0, ':', 'Color', [0.55 0.55 0.55], ...
    'HandleVisibility', 'off');
xline(ax, 12.5, ':', 'Color', [0.45 0.45 0.45], ...
    'HandleVisibility', 'off');
grid(ax, 'on');
set(ax, 'XScale', 'log', 'XMinorGrid', 'on', 'FontName', 'Arial', ...
    'FontSize', 11, 'Box', 'off', 'LineWidth', 1);
xlim(ax, [0.1 50]);
ylim(ax, y_limits);
xticks(ax, [0.1 1 10 50]);
xticklabels(ax, {'0.1', '1', '10', '50'});
xlabel(ax, 'Frequency (GHz, log scale)');
ylabel(ax, 'Magnitude (dB)');
title(ax, panel_title, 'FontSize', 15);
if show_channel
    legend(ax, [lines channel_line], [labels {'Channel only'}], ...
        'Location', 'southwest', 'FontSize', 9);
else
    legend(ax, lines, labels, 'Location', 'southwest', 'FontSize', 9);
end
end

function H_eq = equalizer_response(model, frequency_Hz)
s = 2i*pi*frequency_Hz;
if ~isempty(model.ffe_taps)
    omega = 2*pi*frequency_Hz/model.Fs;
    H_eq = exp(-1i*omega*model.ffe_delays.') * model.ffe_taps;
    return
end
H_eq = ones(size(frequency_Hz));
for stage_index = 1:numel(model.ctle)
    p = model.ctle(stage_index);
    H_eq = H_eq .* p.A0 .* (1 + s*p.tau_z) ./ ...
        ((1 + s*p.tau_p1) .* (1 + s*p.tau_p2));
end
end

function model = read_saved_model(file, expected_stages, Fs)
if nargin < 3, Fs = NaN; end
config = jsondecode(h5readatt(file, '/', 'model_config'));
layers = config.config.layers;
if ~iscell(layers), layers = num2cell(layers); end
ctle = struct([]);
channel_name = '';
dense_name = '';
ffe_taps = [];
training_snr_db = NaN;
upsample_sps = NaN;
ffe_pad = NaN;
for layer_index = 1:numel(layers)
    layer = layers{layer_index};
    if strcmp(layer.class_name, 'Lambda')
        if strcmp(layer.config.name, 'awgn_noise')
            training_snr_db = layer.config.arguments.snr;
        elseif strcmp(layer.config.name, 'tx_upsampling')
            upsample_sps = layer.config.arguments.sps;
        elseif strcmp(layer.config.name, 'rx_ffe_circular_pad')
            ffe_pad = layer.config.arguments.kernel_size;
        end
    elseif strcmp(layer.class_name, 'Conv1D')
        c = layer.config;
        assert(strcmp(c.name,'rx_ffe') && c.filters==1 && ~c.use_bias ...
            && strcmp(c.activation,'linear') && all(c.strides==1) ...
            && all(c.dilation_rate==1) && strcmp(c.padding,'valid'), ...
            'Unsupported FPost configuration.');
        ffe_taps = double(h5read(file, ...
            ['/model_weights/' c.name '/' c.name '/kernel:0']));
        ffe_taps = ffe_taps(:);
    elseif strcmp(layer.class_name, 'CTLELayer')
        name = layer.config.name;
        Fs = layer.config.Fs;
        p.name = name;
        base = ['/model_weights/' name '/' name '/'];
        tau = zeros(1,3);
        weight_names = {'log_tau_z', 'log_tau_p1', 'log_tau_p2'};
        for weight_index = 1:3
            tau(weight_index) = exp(double(h5read(file, ...
                [base weight_names{weight_index} ':0'])));
        end
        % Match CTLELayer._params(), including its frequency limits.
        fz = min(max(1/(2*pi*tau(1)), 0.01*Fs), 0.5*Fs);
        fp1 = min(max(1/(2*pi*tau(2)), max(0.01*Fs,1.05*fz)), ...
            0.49*Fs);
        fp2 = min(max(1/(2*pi*tau(3)), max(0.01*Fs,1.05*fz)), ...
            0.49*Fs);
        p.tau_z = 1/(2*pi*fz);
        p.tau_p1 = 1/(2*pi*fp1);
        p.tau_p2 = 1/(2*pi*fp2);
        p.A0 = min(max(exp(double(h5read(file, ...
            [base 'log_A0:0']))), 1e-6), 1e3);
        if isempty(ctle), ctle = p; else, ctle(end+1) = p; end %#ok<AGROW>
    elseif startsWith(layer.class_name, 'CTLE')
        error('Unsupported CTLE class %s in %s.', layer.class_name, file);
    elseif strcmp(layer.class_name, 'Functional')
        channel_name = layer.config.name;
        sublayers = layer.config.layers;
        if ~iscell(sublayers), sublayers = num2cell(sublayers); end
        dense = sublayers(cellfun(@(x) strcmp(x.class_name,'Dense'), sublayers));
        assert(isscalar(dense) && strcmp(dense{1}.config.activation,'linear') ...
            && ~dense{1}.config.use_bias, ...
            'Expected one linear, bias-free channel Dense layer.');
        dense_name = dense{1}.config.name;
    end
end
assert(numel(ctle)==expected_stages, ...
    'Unexpected CTLE stage count in %s.', file);
assert(~isempty(channel_name), 'Embedded channel not found in %s.', file);
assert(isfinite(Fs) && isfinite(training_snr_db) && upsample_sps==4, ...
    'Missing or unexpected sample-rate/SNR metadata in %s.', file);
if expected_stages==0
    assert(~isempty(ffe_taps) && ffe_pad==numel(ffe_taps) ...
        && mod(ffe_pad,2)==1, 'Expected centered odd-length FPost.');
else
    assert(isempty(ffe_taps), 'CTLE model unexpectedly includes FPost.');
end

% MATLAB reverses HDF5 dimensions: Python [input,output] -> [output,input].
kernel = double(h5read(file, ['/model_weights/' channel_name '/' ...
    dense_name '/kernel:0'])).';
assert(isequal(size(kernel), [157 85]), ...
    'Unexpected channel kernel shape in %s.', file);
model.source_file = file;
model.Fs = Fs;
model.ctle = ctle;
model.ffe_taps = ffe_taps;
model.ffe_delays = (numel(ffe_taps)-1)/2 - (0:numel(ffe_taps)-1).';
model.training_snr_db = training_snr_db;
model.channel_taps = kernel(:, floor(size(kernel,2)/2)+1);
model.channel_delays = (size(kernel,1)-1)/2 - (0:size(kernel,1)-1).';
end

function limits = padded_limits(values)
values = values(isfinite(values));
lo = floor(min(values, [], 'all')/5)*5;
hi = ceil(max(values, [], 'all')/5)*5;
if lo == hi, lo = lo-5; hi = hi+5; end
limits = [lo hi];
end

function y = magnitude_db(x)
y = 20*log10(max(abs(x), 1e-15));
end
