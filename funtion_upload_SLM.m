Blank=zeros(1080,1920);
delay=0.1;
%% SLM-2
monitor1=2;
FigH1=figure('Color', zeros(1, 3), 'Renderer', 'OpenGL');
axes('Visible', 'off', 'Units', 'normalized', 'Position', [0, 0, 1, 1]);
FigPos1=get(FigH1, 'Position');
WindowAPI(FigH1, 'Position', FigPos1, monitor1); % 3/2 is the monitor number for LCoS
WindowAPI(FigH1, 'ToMonitor');
WindowAPI(FigH1, 'Position', 'full');
pause(delay);
% imshow(SLM1);
%imshow(Blank);
imshow(strcat('C:\Users\MP\Desktop\optmization\point_AS50_plane.bmp'));
pause(delay);
%% SLM-3
monitor2=3;
FigH2=figure('Color', zeros(1, 3), 'Renderer', 'OpenGL');
axes('Visible', 'off', 'Units', 'normalized', 'Position', [0, 0, 1, 1]);
FigPos2=get(FigH2, 'Position');
WindowAPI(FigH2, 'Position', FigPos2, monitor2); % 3/2 is the monitor number for LCoS
WindowAPI(FigH2, 'ToMonitor');
WindowAPI(FigH2, 'Position', 'full');
pause(delay);
% imshow(SLM2);
%imshow(Blank);
imshow(strcat('C:\Users\MP\Desktop\optmization\point_AS50_plane.bmp'));
pause(delay);