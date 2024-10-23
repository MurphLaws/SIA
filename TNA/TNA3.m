% Load your audio signal
clear;
clc;
close all;
load("input48k.mat")
y = w48;
fs = 48000;






%%


figure;
subplot(3,2,1);
plot(time, w48);
title('Original Signal');
xlabel('Time (seconds)');
ylabel('Amplitude');
subplot(3,2,3);
reconstructed_signal = applyGainAndReconstruct([1,1,1,1], y,fs);
time_r = (0:length(reconstructed_signal)-1) / fs;
plot(time_r, reconstructed_signal);
title('Reconstructed Signal with Gain [1 1 1 1]' );
xlabel('Time (seconds)');
ylabel('Amplitude');
subplot(3,2,5);
reconstructed_signal2 =  applyGainAndReconstruct([0,0,1/3,1],y,fs);
time_r2 = (0:length(reconstructed_signal2)-1) / fs;
plot(time_r2, reconstructed_signal2);
title('Reconstructed Signal with Gain [0 0 1/3 1]');
xlabel('Time (seconds)');
ylabel('Amplitude');
fft_original = abs(fft(w48));
fft_original = fft_original(1:length(fft_original)/2); % Keep only the first half
frequencies = linspace(0, fs/2, length(fft_original));

fft_reconstructed = abs(fft(reconstructed_signal));
fft_reconstructed = fft_reconstructed(1:length(fft_reconstructed)/2); % Keep only the first half
frequenciesr = linspace(0, fs/2, length(fft_reconstructed));

fft_reconstructed2 = abs(fft(reconstructed_signal2));
fft_reconstructed2 = fft_reconstructed2(1:length(fft_reconstructed2)/2); % Keep only the first half
frequenciesr2 = linspace(0, fs/2, length(fft_reconstructed2));
subplot(3,2,2);
plot(frequencies, fft_original);
title('FFT of Original Signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
subplot(3,2,4);
plot(frequenciesr, fft_reconstructed);
title('FFT of Reconstructed Signal \n Gain [1 1 1 1]');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
subplot(3,2,6);
plot(frequenciesr2, fft_reconstructed2);
title('FFT of Reconstructed Signal \n Gain [0 0 1/3 1]');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
%%

