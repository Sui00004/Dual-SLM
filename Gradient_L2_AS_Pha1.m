function [loss, df ] = Gradient_L2_AS_Pha1( phase1,phase2, rec,source, Nx, Ny,  mask, HTrans)
% This function calculate the loss function and the gradient with repect to
% the phase-only hologram. Angular spectrum theory is utilized for propagation.

df = zeros(Ny, Nx);
loss = 0; 
V = mask;
mass2 = sum(V(:));
% phase = reshape(phase, [Ny, Nx]);
%% compute loss
    objectField = source.*(exp(1i*phase1)+exp(1i*phase2));
    imagez = fftshift(fft2(fftshift(objectField))) .* conj(HTrans);
    imagez = ifftshift(ifft2(ifftshift(imagez))); 
    amp=abs(imagez);
    I = amp.^2;
    mass1 = sum(I(:));
    I = mass2*I/mass1;
    I_rec=I(Ny/4+1:Ny*3/4,Nx/4+1:Nx*3/4);
    mass3 = sum(I_rec(:));
    I_rec=mass3*rec/sum(rec(:));
    I(Ny/4+1:Ny*3/4,Nx/4+1:Nx*3/4)=I_rec;
    amp=sqrt(mass1*I/mass2);
    imagez=amp.*exp(1i.*angle(imagez));
%     I=rec;
    diffh = (I-V).^2;
    L2 = sum(sum(diffh)); 
    loss = loss + L2; 
%% Compute gradient 
    temph = 2*(I-V);
    temph = mass1*temph/mass2;
    temph = 2*imagez.*temph;
%     temph = temph.*field;
    temph = fftshift(fft2(fftshift(temph))).*HTrans;
    temph = ifftshift(ifft2(ifftshift(temph)));
    df = df + temph;
    df=source.*df;
    dfphase = - real(df).*sin(phase1) + imag(df) .* cos(phase1);
    df = real(dfphase);
end
