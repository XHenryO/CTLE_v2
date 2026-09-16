function result = plot_trained_post_responses(snr_db, visible)
% Compare saved best Post / Post_2stage models on logarithmic frequency axes.
% Run: plot_trained_post_responses(10)
% Reads the Keras 2 HDF5 .keras files directly; Python is not required.
% Magnitude is absolute 20*log10(abs(H)), with no DC normalization.
% Channel weights come from EACH saved model, not the current channel file.
% Excludes AWGN, symbol repetition and downsampling; plots the linear link
% at the waveform sample rate. The cascaded equalizer is H1 * H2.
if nargin < 1, snr_db = 10; end
if nargin < 2, visible = 'on'; end
root = fileparts(fileparts(mfilename('fullpath')));
model_dir = fullfile(root, 'model', 'model_trained');
output_dir = fullfile(root, 'model', 'result', 'post_frequency_response');
if ~isfolder(output_dir), mkdir(output_dir); end
prefixes = {'pw_post_ctle_nn_channel_', 'pw_post_2stage_ctle_nn_channel_'};
row_names = {'Single CTLE', 'Cascaded CTLEs'};
for row = 1:2
    file = fullfile(model_dir, sprintf('%ssnr%g_model.keras', prefixes{row}, snr_db));
    assert(isfile(file), 'Best model does not exist: %s', file);
    models(row) = read_saved_model(file, row); %#ok<AGROW>
end
assert(models(1).Fs == models(2).Fs, 'Models must use the same sample rate.');
% Stay below Nyquist: the rFFT implementation has a real-only Nyquist bin.
frequency_Hz = logspace(log10(0.1e9), log10(0.499 * models(1).Fs), 2400).';
result.frequency_Hz = frequency_Hz;
result.snr_db = snr_db;
result.models = models;
result.equalizer = complex(zeros(numel(frequency_Hz), 2));
result.channel = result.equalizer;
result.combined = result.equalizer;
fig = figure('Color', 'w', 'Visible', visible, 'Position', [80 80 1500 550]);
tiles = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tiles, sprintf('Trained post equalizers | SNR = %g dB | F_s = %g GSa/s', ...
    snr_db, models(1).Fs / 1e9), 'FontSize', 19, 'FontWeight', 'bold');
subtitle(tiles, 'Absolute magnitude response; channel extracted from each saved best model', 'FontSize', 12);
colors = [0.10 0.36 0.68; 0.85 0.32 0.10];
for row = 1:2
    model = models(row);
    s = 2i * pi * frequency_Hz;
    stages = complex(zeros(numel(frequency_Hz), numel(model.ctle)));
    for k = 1:numel(model.ctle)
        p = model.ctle(k);
        stages(:, k) = p.A0 .* (1 + s*p.tau_z) ./ ...
            ((1 + s*p.tau_p1) .* (1 + s*p.tau_p2));
    end
    H_eq = prod(stages, 2);
    omega = 2*pi*frequency_Hz / model.Fs;
    H_ch = exp(-1i * omega * model.channel_delays.') * model.channel_taps;
    H_link = H_eq .* H_ch;
    result.equalizer(:,row) = H_eq;
    result.channel(:,row) = H_ch;
    result.combined(:,row) = H_link;
    fprintf('\n%s: %s\n', row_names{row}, model.source_file);
    for k = 1:numel(model.ctle)
        p = model.ctle(k);
        fprintf('  %s: A0=%.7g, fz=%.5g, fp1=%.5g, fp2=%.5g GHz\n', ...
            p.name, p.A0, 1/(2*pi*p.tau_z)/1e9, ...
            1/(2*pi*p.tau_p1)/1e9, 1/(2*pi*p.tau_p2)/1e9);
    end
end
responses = {result.equalizer, result.channel, result.combined};
panel_names = {'Equalizer', 'Channel', 'Channel x equalizer'};
for col = 1:3
    ax = nexttile(tiles, col);
    h1 = semilogx(ax, frequency_Hz/1e9, db(responses{col}(:,1)), ...
        '-', 'LineWidth', 2.3, 'Color', colors(1,:));
    hold(ax, 'on');
    h2 = semilogx(ax, frequency_Hz/1e9, db(responses{col}(:,2)), ...
        '--', 'LineWidth', 2.3, 'Color', colors(2,:));
    yline(ax, 0, ':', 'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');
    xline(ax, 12.5, ':', 'Color', [0.55 0.55 0.55], 'HandleVisibility', 'off');
    grid(ax, 'on');
    set(ax, 'XScale', 'log', 'XMinorGrid', 'on', 'FontName', 'Arial', ...
        'FontSize', 12, 'Box', 'off', 'LineWidth', 1);
    xlim(ax, [0.1 50]);
    xticks(ax, [0.1 1 10 50]);
    xticklabels(ax, {'0.1', '1', '10', '50'});
    xlabel(ax, 'Frequency (GHz, log scale)');
    ylabel(ax, 'Magnitude (dB)');
    title(ax, panel_names{col}, 'FontSize', 15);
    legend(ax, [h1 h2], {'Single CTLE', 'Cascaded CTLEs (H_1 H_2)'}, ...
        'Location', 'southwest', 'FontSize', 11);
    if col == 2 && max(abs(db(result.channel(:,1))-db(result.channel(:,2)))) < 1e-8
        text(ax, 0.04, 0.25, 'Identical channel responses (overlap)', ...
            'Units', 'normalized', 'FontSize', 11, 'Color', [0.35 0.35 0.35]);
    end
end
xlabel(tiles, 'Dotted vertical line: 12.5 GHz (25 GBd Nyquist). Linear response excludes noise and sampling operations.', 'FontSize', 11);
base = fullfile(output_dir, sprintf('post_comparison_snr%g', snr_db));
save([base '.mat'], 'result');
savefig(fig, [base '.fig']);
exportgraphics(fig, [base '.png'], 'Resolution', 180);
table_out = table(frequency_Hz, db(result.equalizer(:,1)), db(result.channel(:,1)), ...
    db(result.combined(:,1)), db(result.equalizer(:,2)), db(result.channel(:,2)), ...
    db(result.combined(:,2)), 'VariableNames', {'frequency_Hz', 'single_equalizer_dB', ...
    'single_channel_dB', 'single_combined_dB', 'cascade_equalizer_dB', ...
    'cascade_channel_dB', 'cascade_combined_dB'});
writetable(table_out, [base '.csv']);
fprintf('\nSaved figure and response data: %s\n', base);
end

function model = read_saved_model(file, expected_stages)
config = jsondecode(h5readatt(file, '/', 'model_config'));
layers = config.config.layers;
if ~iscell(layers), layers = num2cell(layers); end
ctle = struct([]);
channel_name = '';
for k = 1:numel(layers)
    layer = layers{k};
    if strcmp(layer.class_name, 'CTLELayer')
        name = layer.config.name;
        Fs = layer.config.Fs;
        p.name = name;
        % Match CTLELayer._params(), including its frequency bounds.
        base = ['/model_weights/' name '/' name '/'];
        tau = zeros(1,3);
        names = {'log_tau_z', 'log_tau_p1', 'log_tau_p2'};
        for j = 1:3
            tau(j) = exp(double(h5read(file, [base names{j} ':0'])));
        end
        fz = min(max(1/(2*pi*tau(1)), 0.01*Fs), 0.5*Fs);
        fp1 = min(max(1/(2*pi*tau(2)), max(0.01*Fs,1.05*fz)), 0.49*Fs);
        fp2 = min(max(1/(2*pi*tau(3)), max(0.01*Fs,1.05*fz)), 0.49*Fs);
        p.tau_z = 1/(2*pi*fz);
        p.tau_p1 = 1/(2*pi*fp1);
        p.tau_p2 = 1/(2*pi*fp2);
        p.A0 = min(max(exp(double(h5read(file, [base 'log_A0:0']))),1e-6),1e3);
        if isempty(ctle)
            ctle = p;
        else
            ctle(end+1) = p; %#ok<AGROW>
        end
    elseif startsWith(layer.class_name, 'CTLE')
        error('Unsupported CTLE class %s; use its actual transfer function.', layer.class_name);
    elseif strcmp(layer.class_name, 'Functional')
        channel_name = layer.config.name;
        sublayers = layer.config.layers;
        if ~iscell(sublayers), sublayers = num2cell(sublayers); end
        dense = sublayers(cellfun(@(x) strcmp(x.class_name,'Dense'), sublayers));
        assert(numel(dense)==1 && strcmp(dense{1}.config.activation,'linear') ...
            && ~dense{1}.config.use_bias, 'Expected a single linear, bias-free channel Dense layer.');
        dense_name = dense{1}.config.name;
    end
end
assert(numel(ctle)==expected_stages, 'Unexpected CTLE stage count in %s', file);
assert(~isempty(channel_name), 'Embedded channel not found.');
% MATLAB reverses HDF5 dimensions: Python [input,output] -> [output,input].
kernel = double(h5read(file, ['/model_weights/' channel_name '/' dense_name '/kernel:0'])).';
assert(isequal(size(kernel), [157 85]), 'Unexpected channel kernel shape.');
model.source_file = file;
model.Fs = Fs;
model.ctle = ctle;
model.channel_taps = kernel(:, floor(size(kernel,2)/2)+1);
model.channel_delays = (size(kernel,1)-1)/2 - (0:size(kernel,1)-1).';
end

function y = db(x)
y = 20*log10(max(abs(x), 1e-15));
end
