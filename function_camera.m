
video_rec=videoinput('gentl','1');
set(video_rec,'FramesPerTrigger',1);
set(video_rec,'TriggerRepeat',Inf);
set(video_rec,'ReturnedColorSpace','grayscale');
triggerconfig(video_rec,'Manual');
preview(video_rec);
frame=getsnapshot(video_rec);
rec=im2double(frame);
I=rec(535:1779,1:2250);

%     ph(1,1:Imin)=2*(1+ph(1,1:Imin));
%     ph(1,Imin+1:Imax)=2*(1-ph(1,Imin+1:Imax));
%     ph(1,Imax+1:end)=2*(ph(1,Imax+1:end));
pmax=max(max(I));
pmin=min(min(I));
    norm=(I-pmin)/(pmax-pmin);
    ph=asin(sqrt(norm))/pi;
figure,imshow(norm);
save('SLM2_pi','norm','ph')
imwrite(norm,'SLM2_pi.bmp');
%{
stop(video_rec);
closepreview(video_rec);
delete(video_rec);
%}
