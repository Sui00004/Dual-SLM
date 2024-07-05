close all;clear;clc;tic
%% parameter
slm.pix = 6.4e-3;
slm.Nx = 1920; slm.Ny = 1080; 
opt.Nx = 2*slm.Nx; opt.Ny = 2*slm.Ny;
dh = 6.4e-3;
opt.lambda = 520e-6;
opt.k=2*pi/opt.lambda;
%% illumination pattern at SLM
slm.window = ones(slm.Ny, slm.Nx);
slm.window = padarray(slm.window,[slm.Ny/2,slm.Nx/2]);
opt.source = slm.window;
%% construct object matrix
F1=imread('9140.jpg');
F1=rgb2gray(F1);
% load('pointarray.mat')
% F1=G;
F1=im2double(F1);
F1 = imresize(F1, [slm.Ny, slm.Nx]);
obj = padarray(F1,[slm.Ny/2,slm.Nx/2]);
Masks = obj;
E=sum(Masks(:));
%% convolution kernel (band-limited angular spectrum)
depth = 50;
[fx,fy]=meshgrid(linspace(-1/(2*slm.pix),1/(2*slm.pix),opt.Nx),linspace(-1/(2*slm.pix),1/(2*slm.pix),opt.Ny));
Sm=opt.Nx*dh;Sn=opt.Ny*dh;
delta_m=(2*Sm).^(-1);delta_n=(2*Sn).^(-1);
lim_m=((2*delta_m*depth).^2+1).^(-1/2)./opt.lambda;
lim_n=((2*delta_n*depth).^2+1).^(-1/2)./opt.lambda;
bandlim_m=(lim_m-abs(fx));
bandlim_n=(lim_n-abs(fy));
bandlim_m=imbinarize(bandlim_m,0);
bandlim_n=imbinarize(bandlim_n,0);
bandlim_AS=bandlim_m.*bandlim_n;
HTrans = bandlim_AS.*exp(1i*opt.k*sqrt(1-(opt.lambda*fy).^2-(opt.lambda*fx).^2)*depth);
figure,imshow(F1);
%% generation of the starting value
phaa=2*pi*ones(opt.Ny,opt.Nx);
pha=exp(1i*phaa);
amp=sqrt(Masks);
hologram=fftshift(fft2(fftshift(amp .* pha))) .* HTrans;
hologram = ifftshift(ifft2(ifftshift(hologram)));
A=abs(hologram);
A=A/max(max(A));
fai=angle(hologram);
fai=mod(fai,2*pi);
Pha1=fai-acos(A);Pha1=mod(Pha1,2*pi);
Pha2=fai+acos(A);Pha2=mod(Pha2,2*pi);
% Pha1=phaa;Pha2=phaa;
%% set camera
video_rec=videoinput('gentl','1');
set(video_rec,'FramesPerTrigger',1);
set(video_rec,'TriggerRepeat',Inf);
set(video_rec,'ReturnedColorSpace','grayscale');
triggerconfig(video_rec,'Manual');
preview(video_rec);
pause(1);
frame=getsnapshot(video_rec);
pause(1);
I=im2double(frame);%E=sum(I(:))/2;
I=E*I/sum(sum(I));
figure,imshow(I);
rec=I(423:1812,40:2460);
[n,m]=size(rec);
FF1=imresize(F1,[n,m]);
EE=sum(FF1(:));
rec=EE*rec/sum(sum(rec));
imwrite(rec,'9140Lcos3.bmp');
%% set screen
monitor1=2;
FigH1=figure('Color', zeros(1, 3), 'Renderer', 'OpenGL');
axes('Visible', 'off', 'Units', 'normalized', 'Position', [0, 0, 1, 1]);
FigPos1=get(FigH1, 'Position');
    WindowAPI(FigH1, 'Position', FigPos1, monitor1); 
    WindowAPI(FigH1, 'ToMonitor');
    WindowAPI(FigH1, 'Position', 'full');
monitor2=3;
FigH2=figure('Color', zeros(1, 3), 'Renderer', 'OpenGL');
axes('Visible', 'off', 'Units', 'normalized', 'Position', [0, 0, 1, 1]);
FigPos2=get(FigH2, 'Position');
    WindowAPI(FigH2, 'Position', FigPos2, monitor2); 
    WindowAPI(FigH2, 'ToMonitor');
    WindowAPI(FigH2, 'Position', 'full');
%% optimization
times=250;
RMSE=zeros(times,1);
PhaHolo1=Pha1(opt.Ny/4+1:opt.Ny*3/4,opt.Nx/4+1:opt.Nx*3/4);
PhaHolo2=Pha2(opt.Ny/4+1:opt.Ny*3/4,opt.Nx/4+1:opt.Nx*3/4);
for i=2:times
%% Display SLM2
    PhaHolo1=mod(PhaHolo1,2*pi);
    figure(FigH1),imshow(PhaHolo1,[0,2*pi]);
    pause(0.1);
%% Take photo
    frame=getsnapshot(video_rec);
    pause(1);
    I=im2double(frame);
    I=I(423:1812,40:2460);
    I=imresize(I,[slm.Ny, slm.Nx]);
    I=E*I/sum(sum(I));
%     objectField=opt.source.*exp(1i.*Pha);
%     imagez = fftshift(fft2(fftshift(objectField))) .* conj(HTrans);
%     imagez = ifftshift(ifft2(ifftshift(imagez))); 
%     amp=abs(imagez);
%     I=amp(opt.Ny/4+1:opt.Ny*3/4,opt.Nx/4+1:opt.Nx*3/4).^2;
%     I=E*I/sum(sum(I));
    figure(1);imshow(I);
    [loss1, df1] = Gradient_L2_AS_Pha1(Pha1,Pha2,I,opt.source, opt.Nx, opt.Ny, Masks, HTrans);
    [updates1, state1] = Optmization_SGD_ADAM(df1, []);
    Pha1=Pha1-updates1;
    PhaHolo1=Pha1(opt.Ny/4+1:opt.Ny*3/4,opt.Nx/4+1:opt.Nx*3/4);
%% Display SLM3
    PhaHolo2=mod(PhaHolo2,2*pi);
    PhaHolo2=fliplr(PhaHolo2);
    figure(FigH2),imshow(PhaHolo2,[0,2*pi]);
    pause(0.1);
%% Take photo
    frame=getsnapshot(video_rec);
    pause(1);
    I=im2double(frame);
    rec=I(423:1812,40:2460);
    I=imresize(rec,[slm.Ny, slm.Nx]);
    I=E*I/sum(sum(I));
%     objectField=opt.source.*exp(1i.*Pha);
%     imagez = fftshift(fft2(fftshift(objectField))) .* conj(HTrans);
%     imagez = ifftshift(ifft2(ifftshift(imagez))); 
%     amp=abs(imagez);
%     I=amp(opt.Ny/4+1:opt.Ny*3/4,opt.Nx/4+1:opt.Nx*3/4).^2;
%     I=E*I/sum(sum(I));
    figure(1);imshow(I);
    rec=EE*rec/sum(sum(rec));
    imwrite(rec,['green' num2str(i) 'rec.bmp']);
    [loss2, df2] = Gradient_L2_AS_Pha2(Pha1,Pha2,I,opt.source, opt.Nx, opt.Ny, Masks, HTrans);
    [updates2, state2] = Optmization_SGD_ADAM(df2, []);
    Pha2=Pha2-updates2;
    PhaHolo2=Pha2(opt.Ny/4+1:opt.Ny*3/4,opt.Nx/4+1:opt.Nx*3/4);    
%% show result
    Diff=double(I)-double(F1);
    MSE=gather(sum(Diff(:).^2)/numel(I));
    RMSE(i,1)=sqrt(MSE);
end
save('holo2','PhaHolo2','PhaHolo1');
% rec.phase = reshape(Pha, [opt.Ny, opt.Nx]);
% rec.phase = exp(1i.*rec.phase);
%% reconstruction
% Rec = fftshift(fft2(fftshift(opt.source.*rec.phase))).* conj(HTrans); 
% Rec = ifftshift(ifft2(ifftshift(Rec))); 
% I = abs(Rec).^2;
% I=I(opt.Ny/4+1:opt.Ny*3/4,opt.Nx/4+1:opt.Nx*3/4);
% I= E*I/sum(sum(I));
% toc
% imwrite(I,'SGD_AS_complex2.bmp');
% Phase = mod(angle(Rec),2*pi);
figure,imshow(I);