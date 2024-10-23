clear;
clc;
close all;

load("playback_44100.mat")

 
x = w441;
w441_frag = x(1:1000); 


%% FIRST EXPERIMENT


Fin = 441e2;

[first_block,  F1] = rate_conversion(4,3,Fin, x);
[second_block, F2] = rate_conversion(8,7,F1, first_block);
[third_block,  F3] = rate_conversion(5,7,F2, second_block);

t_in  = linspace(0, length(x)/ Fin, length(x));
t_out = 0:1/F3:0.25-1/F3;

plot(linspace(0, length(third_block)/ F3, length(third_block)),third_block)
%stem(t_out,third_block)

