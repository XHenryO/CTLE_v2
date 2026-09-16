function plot_VLC_channel_log_response
% Plot the saved channel response with a logarithmic frequency axis.
% Uses exactly the same response data as analyze_VLC_channel_rolloff.
script_dir = fileparts(mfilename('fullpath'));
output_dir = fullfile(fileparts(script_dir), 'result', 'channel_rolloff');
response_file = fullfile(output_dir, 'frequency_response.csv');
if ~isfile(response_file)
    error('Run analyze_VLC_channel_rolloff first to generate the response data.');
end
data = readtable(response_file);
f = data.frequency_GHz;
g = data.gain_relative_to_DC_dB;
positive = f > 0;
upper = max(f);
lower = max(min(f(positive)), upper/500);

fig = figure('Visible','off','Color','w','Position',[100,100,920,540]);
close_figure = onCleanup(@() close(fig)); %#ok<NASGU>
ax = axes(fig);
semilogx(ax,f(positive),g(positive),'Color',[0.12,0.36,0.68],'LineWidth',2.4);
hold(ax,'on');
grid(ax,'on');
set(ax,'XMinorGrid','on','YMinorGrid','off','FontName','Arial', ...
    'FontSize',12,'LineWidth',1,'Box','off');
xlim(ax,[lower,upper]);
ylim(ax,[min(g)-3,1.5]);
xlabel(ax,'Frequency (GHz, logarithmic scale)');
ylabel(ax,'Gain relative to DC (dB)');
title(ax,'Same channel response on a logarithmic frequency axis','FontSize',15);
ticks = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500];
ticks = ticks(ticks >= lower & ticks <= upper);
xticks(ax,ticks);
xticklabels(ax,compose('%g',ticks));
yline(ax,-3,':','-3 dB','LabelHorizontalAlignment','left', ...
    'Color',[0.45,0.45,0.45],'HandleVisibility','off');

first_below = find(g <= -3,1);
if ~isempty(first_below) && first_below > 1
    interval = first_below-1:first_below;
    f3 = interp1(g(interval),f(interval),-3);
    scatter(ax,f3,-3,40,[0.8,0.35,0.1],'filled');
    text(ax,f3*0.9,-4.8,sprintf('%.3f GHz: -3 dB',f3), ...
        'HorizontalAlignment','right','FontSize',11);
end
markers = [12.5,25,50];
markers = markers(markers >= lower & markers <= upper);
values = interp1(f,g,markers);
scatter(ax,markers,values,40,[0.8,0.35,0.1],'filled');
for k = 1:numel(markers)
    text(ax,markers(k)*0.90,values(k)-0.4, ...
        sprintf('%g GHz: %.2f dB',markers(k),values(k)), ...
        'HorizontalAlignment','right','VerticalAlignment','top','FontSize',11);
end
exportgraphics(fig,fullfile(output_dir,'channel_rolloff_log_frequency.png'),'Resolution',180);
savefig(fig,fullfile(output_dir,'channel_rolloff_log_frequency.fig'));
fprintf('Log-frequency plot: %s\n',fullfile(output_dir,'channel_rolloff_log_frequency.png'));
end
