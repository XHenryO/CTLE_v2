function result = plot_trained_pnn2_response(snr_db, operating_levels, visible)
% Plot the trained PNN2 equalizer's local small-signal frequency response.
% Usage: result = plot_trained_pnn2_response(10);
%        result = plot_trained_pnn2_response(10, [-0.5 0 0.5]);
%
% PNN2 contains ReLU activations, so it has no single input-independent
% transfer function. For each constant input level x0, this function
% differentiates the network output with respect to its 63 input samples.
% The resulting 63-tap FIR is the exact local linearization at x0.
% Levels refer to the normalized waveform entering rx_pnn_windows, after
% the channel and AWGN, rather than to the transmitted PAM symbols.
% MATLAB reads the saved Keras 2 HDF5 model directly; Python is not needed.

if nargin < 1 || isempty(snr_db), snr_db = 10; end
if nargin < 2 || isempty(operating_levels), operating_levels = [-0.5 0 0.5]; end
if nargin < 3 || isempty(visible), visible = 'on'; end
operating_levels = double(operating_levels(:).');
assert(all(isfinite(operating_levels)), 'Operating levels must be finite.');

root = fileparts(fileparts(mfilename('fullpath')));
name = sprintf('pw_post_pnn2_mlp63_64_32_nn_channel_snr%g', snr_db);
model_file = fullfile(root, 'model', 'model_trained', [name '_model.keras']);
assert(isfile(model_file), 'Trained PNN2 model not found: %s', model_file);
out_dir = fullfile(root, 'model', 'result', 'pnn2_frequency_response');
if ~isfolder(out_dir), mkdir(out_dir); end

config = jsondecode(h5readatt(model_file, '/', 'model_config'));
layers = config.config.layers;
if ~iscell(layers), layers = num2cell(layers); end
names = cellfun(@(x) x.config.name, layers, 'UniformOutput', false);
expected = {'rx_pnn_windows', 'rx_pnn_hidden_1', 'rx_pnn_hidden_2', ...
    'rx_pnn_correction', 'rx_pnn_residual'};
assert(all(ismember(expected, names)), 'Saved model is not the expected PNN2 architecture.');
for k = 1:2
    layer = layers{strcmp(names, sprintf('rx_pnn_hidden_%d', k))};
    assert(strcmp(layer.class_name, 'Dense') && strcmp(layer.config.activation, 'relu'), ...
        'Expected a ReLU Dense hidden layer.');
end
layer = layers{strcmp(names, 'rx_pnn_correction')};
assert(strcmp(layer.class_name, 'Dense') && strcmp(layer.config.activation, 'linear'), ...
    'Expected a linear correction layer.');

% MATLAB's h5read reverses Python HDF5 matrix dimensions.
W1 = read_kernel(model_file, 'rx_pnn_hidden_1');
W2 = read_kernel(model_file, 'rx_pnn_hidden_2');
W3 = read_kernel(model_file, 'rx_pnn_correction');
b1 = read_bias(model_file, 'rx_pnn_hidden_1');
b2 = read_bias(model_file, 'rx_pnn_hidden_2');
assert(isequal(size(W1), [63 64]) && isequal(size(W2), [64 32]) ...
    && isequal(size(W3), [32 1]) && numel(b1)==64 && numel(b2)==32, ...
    'Unexpected saved PNN2 weight shapes.');

Fs = 100e9;  % Model_ch_PNN2.py: Rs = 25e9, sps = 4.
f_Hz = logspace(log10(0.1e9), log10(Fs/2), 2000).';
omega = 2*pi*f_Hz/Fs;
delays = 31 - (0:62).';  % Window order: x[n-31], ..., x[n+31].
number_of_levels = numel(operating_levels);
H = complex(zeros(numel(f_Hz), number_of_levels));
taps = zeros(63, number_of_levels);
for k = 1:number_of_levels
    x0 = operating_levels(k);
    z1 = x0*sum(W1, 1) + b1;
    z2 = max(z1, 0)*W2 + b2;
    active1 = double(z1 > 0).';
    active2 = double(z2 > 0).';
    h = W1 * (active1 .* (W2 * (active2 .* W3)));
    h(32) = h(32) + 1;  % PNN2 adds its input through rx_pnn_residual.
    taps(:,k) = h;
    H(:,k) = exp(-1i*omega*delays.') * h;
end
magnitude_dB = 20*log10(max(abs(H), 1e-15));

fig = figure('Color', 'w', 'Visible', visible, ...
    'Name', 'Trained PNN2 local frequency response');
ax = axes(fig);
hold(ax, 'on');
colors = lines(number_of_levels);
for k = 1:number_of_levels
    semilogx(ax, f_Hz/1e9, magnitude_dB(:,k), 'LineWidth', 1.8, ...
        'Color', colors(k,:), 'DisplayName', sprintf('Input level %g', operating_levels(k)));
end
grid(ax, 'on');
set(ax, 'XScale', 'log', 'XMinorGrid', 'on');
xlim(ax, [0.1 50]);
xlabel(ax, 'Frequency (GHz, log scale)');
ylabel(ax, 'Local gain (dB)');
title(ax, sprintf('PNN2 equalizer: local response at constant input | SNR %g dB', snr_db));
legend(ax, 'Location', 'best');

result.model_file = model_file;
result.Fs = Fs;
result.operating_levels = operating_levels;
result.frequency_Hz = f_Hz;
result.response = H;
result.magnitude_dB = magnitude_dB;
result.taps = taps;
result.table = table(f_Hz, 'VariableNames', {'frequency_Hz'});
for k = 1:number_of_levels
    result.table.(sprintf('level_%d_dB', k)) = magnitude_dB(:,k);
end

base = fullfile(out_dir, sprintf('pnn2_snr%g_local_response', snr_db));
save([base '.mat'], 'result');
writetable(result.table, [base '.csv']);
savefig(fig, [base '.fig']);
exportgraphics(fig, [base '.png'], 'Resolution', 180);

fprintf('Model: %s\n', model_file);
fprintf('Local linearization of 63-tap nonlinear NN equalizer; Fs = %.0f GSa/s.\n', Fs/1e9);
fprintf('Operating input levels: %s\n', mat2str(operating_levels));
fprintf('Full response: result.table; local FIR taps: result.taps.\n');
sample_GHz = [0.1 1 5 10 12.5 20 30 40 50];
fprintf(' Frequency (GHz)');
for k = 1:number_of_levels, fprintf('    level %g (dB)', operating_levels(k)); end
fprintf('\n');
for target = sample_GHz
    [~, idx] = min(abs(f_Hz/1e9 - target));
    fprintf(' %15.3f', f_Hz(idx)/1e9);
    fprintf(' %16.3f', magnitude_dB(idx,:));
    fprintf('\n');
end
fprintf('Saved: %s.[png,fig,csv,mat]\n', base);
end

function W = read_kernel(model_file, layer_name)
path = ['/model_weights/' layer_name '/' layer_name '/kernel:0'];
W = double(h5read(model_file, path)).';
end

function b = read_bias(model_file, layer_name)
path = ['/model_weights/' layer_name '/' layer_name '/bias:0'];
b = double(h5read(model_file, path));
b = b(:).';
end
