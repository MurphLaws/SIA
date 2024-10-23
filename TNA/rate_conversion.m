function [rate_converted_signal,output_frequency] = rate_conversion(L,M, Fs, signal)

f = (Fs/2) * min(1/L, 1/M);
attenuation = 96;

d = designfilt('lowpassiir', 'PassbandFrequency', 0.9*f, 'StopbandFrequency', 1.1*f, ...
               'PassbandRipple', 5, 'StopbandAttenuation', attenuation, 'DesignMethod', 'ellip', ...
               'SampleRate', Fs);


upsampled_signal = upsample(signal,L);
upsampled_filtered_signal = filter(d, upsampled_signal);
rate_converted_signal = downsample(upsampled_filtered_signal, M);
output_frequency = round(Fs*L/M);

end