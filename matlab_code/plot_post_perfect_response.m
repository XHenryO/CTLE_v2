function result = plot_post_perfect_response(snr_db, visible)
% Plot and print the trained frequency response of Model_ch_Post_P.
% Usage: result = plot_post_perfect_response(10);
% Reads the saved Keras 2 HDF5 model directly; Python is not required.
% result.table contains all 10001 FFT bins, including DC and Nyquist.
if nargin < 1, snr_db = 10; end
if nargin < 2, visible = 'on'; end

root = fileparts(fileparts(mfilename('fullpath')));
model_file = fullfile(root, 'model', 'model_trained', ...
    sprintf('pw_post_perfect_5000pt_nn_channel_snr%g_repeated_awgn_model.keras', snr_db));
assert(isfile(model_file), 'Trained Post_P model not found: %s', model_file);

config = jsondecode(h5readatt(model_file, '/', 'model_config'));
layers = config.config.layers;
if ~iscell(layers), layers = num2cell(layers); end
eq_layer = [];
channel_layer = [];
for k = 1:numel(layers)
    layer = layers{k};
    if strcmp(layer.config.name, 'rx_ctle')
        assert(contains(layer.class_name, 'Layer_perfect'), ...
            'rx_ctle is not Layer_perfect.');
        eq_layer = layer;
    elseif strcmp(layer.config.name, 'linear_nn_channel')
        channel_layer = layer;
    end
end
assert(~isempty(eq_layer) && ~isempty(channel_layer), ...
    'Equalizer or channel is missing from the saved model.');

Fs = double(eq_layer.config.Fs);
n_points = double(eq_layer.config.n_points);
n_fft = double(eq_layer.config.fft_length);
assert(n_points >= 3 && n_fft == round(n_fft) && mod(n_fft, 2) == 0);
base = '/model_weights/rx_ctle/rx_ctle/';
h_real = double(h5read(model_file, [base 'H_real:0']));
h_imag_inner = double(h5read(model_file, [base 'H_imag_interior:0']));
h_real = h_real(:);
h_imag_inner = h_imag_inner(:);
assert(numel(h_real) == n_points && numel(h_imag_inner) == n_points-2, ...
    'Saved equalizer weight count differs from model configuration.');
h_control = complex(h_real, [0; h_imag_inner; 0]);

% Match Layer_perfect.frequency_response(): linear interpolation of the
% real and imaginary control points onto the RFFT bins, including endpoints.
n_bins = n_fft/2 + 1;
frequency_Hz = (0:n_bins-1).' * Fs/n_fft;
position = (0:n_bins-1).' * (n_points-1) * 2/n_fft;
left = floor(position) + 1;  % MATLAB's 1-based index
right = min(left + 1, n_points);
fraction = position - floor(position);
H_eq = (1-fraction).*h_control(left) + fraction.*h_control(right);

% The embedded channel is a bias-free Dense kernel. Its central output
% column is the convolutional channel tap vector used at every sample.
sub = channel_layer.config.layers;
if ~iscell(sub), sub = num2cell(sub); end
dense = sub(cellfun(@(x) strcmp(x.class_name, 'Dense'), sub));
assert(numel(dense) == 1 && strcmp(dense{1}.config.activation, 'linear') ...
    && ~dense{1}.config.use_bias, 'Unsupported embedded channel.');
dense_name = dense{1}.config.name;
channel_name = channel_layer.config.name;
kernel = double(h5read(model_file, ...
    ['/model_weights/' channel_name '/' dense_name '/kernel:0'])).';
assert(isequal(size(kernel), [157 85]), 'Unexpected channel kernel size.');
channel_taps = kernel(:, floor(size(kernel,2)/2)+1);
channel_delays = (size(kernel,1)-1)/2 - (0:size(kernel,1)-1).';
omega = 2*pi*frequency_Hz/Fs;
H_ch = exp(-1i*omega*channel_delays.') * channel_taps;
H_combined = H_eq .* H_ch;

eq_dB = 20*log10(max(abs(H_eq), 1e-15));
ch_dB = 20*log10(max(abs(H_ch), 1e-15));
combined_dB = 20*log10(max(abs(H_combined), 1e-15));
result.model_file = model_file;
result.Fs = Fs;
result.fft_length = n_fft;
result.control_frequency_Hz = linspace(0, Fs/2, n_points).';
result.control_response = h_control;
result.frequency_Hz = frequency_Hz;
result.equalizer = H_eq;
result.channel = H_ch;
result.combined = H_combined;
result.table = table(frequency_Hz, real(H_eq), imag(H_eq), eq_dB, ...
    ch_dB, combined_dB, 'VariableNames', ...
    {'frequency_Hz','equalizer_real','equalizer_imag', ...
     'equalizer_dB','channel_dB','combined_dB'});

mask = frequency_Hz > 0;
figure('Color', 'w', 'Visible', visible, 'Name', 'Trained Post_P frequency response');
semilogx(frequency_Hz(mask)/1e9, eq_dB(mask), 'LineWidth', 1.6); hold on;
semilogx(frequency_Hz(mask)/1e9, ch_dB(mask), 'LineWidth', 1.6);
semilogx(frequency_Hz(mask)/1e9, combined_dB(mask), 'LineWidth', 1.6);
grid on; xlabel('Frequency (GHz, log scale)'); ylabel('Magnitude (dB)');
xlim([0.1 Fs/2/1e9]);
title(sprintf('Post\\_P trained response | %g control points | SNR %g dB', ...
    n_points, snr_db));
legend('Equalizer', 'Channel', 'Channel x equalizer', 'Location', 'best');

fprintf('Model: %s\n', model_file);
fprintf('Fs = %.6g GSa/s, FFT length = %d, control points = %d\n', ...
    Fs/1e9, n_fft, n_points);
fprintf('Full numeric response: result.table (%d frequency bins).\n', n_bins);
fprintf(' Frequency(GHz)     Re(H_eq)       Im(H_eq)     Eq(dB)   Channel(dB)  Combined(dB)\n');
sample_GHz = [0.1 1 5 10 12.5 20 30 40 Fs/2/1e9];
for target = sample_GHz
    [~, idx] = min(abs(frequency_Hz/1e9-target));
    fprintf(' %12.4f  %13.6g  %13.6g  %9.3f  %11.3f  %12.3f\n', ...
        frequency_Hz(idx)/1e9, real(H_eq(idx)), imag(H_eq(idx)), ...
        eq_dB(idx), ch_dB(idx), combined_dB(idx));
end
end
