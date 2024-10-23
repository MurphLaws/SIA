clc;
clear;

%% 2 Generation de signaux test
n = 1000;
x = rand(n,1) - 1/2;
plot(x)
b = 1;
a = [1 ,0.3 , -0.1, 0.2].';
d = filter (b, a, x);
%plot(d)

% 4 Mise en oeuvre de l'algorithme LMS
mu = 0.1;
P = 4;

[allw, w,y,e] = algolms(x,d,P,mu);

subplot(2,2,1)
plot(x)
title('Input signal x ')

subplot(2,2,2)
plot(y)
title('Signal y')

subplot(2,2,3)
plot(e)
title('error e ')

subplot(2,2,4)
hold on
for i=1:P
    plot(allw(:,i));
end
hold off
title('Coefficients over time')

%%  Test de l'algorithme LMS
%Test of the effect of different P values on the convergance

n = 10000;
x = rand(n,1) - 1/2;
noise = rand(1,n)*0.5 - 0.5/2;
f = 0.5;
P = 5;
mu = 0.01;

P1 = 5;
b1 = fir1(P1-1, f);
d1 = filter(b,1, x) + noise;

P2 = 10;
b2 = fir1(P2-1, f);
d2 = filter(b2,1, x)+ noise;

P3 = 20;
b3 = fir1(P3-1, f);
d3 = filter(b3,1, x)+ noise;

[~, ~,~,e_p5] = algolms(x,d1,P1,mu);
[~, ~,~,e_p10] = algolms(x,d2,P2,mu);
[~, ~,~,e_p20] = algolms(x,d3,P3,mu);

subplot(2,2,1)
plot(e_p5)
title('P = 5')

subplot(2,2,2)
plot(e_p10)
title('P = 10')

subplot(2,2,3)
plot(e_p20)
title('P = 20')


%% Test of the effect of different mu values on the convergance

n = 10000;
x = rand(n,1) - 1/2;
noise = rand(1,n)*0.5 - 0.5/2;
f = 0.5;
P = 5;
mu = 0.01;

b = fir1(P-1, f);
d = filter(b,1, x) + noise;

[~, ~,~,e_mu001] = algolms(x,d,P,0.01);
[~, ~,~,e_mu01] = algolms(x,d,P,0.1);
[~, ~,~,e_mu05] = algolms(x,d,P,0.5);

subplot(2,2,1)
plot(e_mu001)
title('mu = 0.01')

subplot(2,2,2)
plot(e_mu01)
title('mu = 0.1')

subplot(2,2,3)
plot(e_mu05)
title('mu = 0.5')

%% Application

[y, Fs] = audioread('Voix1.wav');
sound(y,Fs);

room_impulse_response = importdata('Rep.dat');

d4 = filter(room_impulse_response,1,y);
subplot(2,2,1)
plot(y)
title('y')
subplot(2,2,2)
plot(d4)
title('y filtered')
subplot(2,2,3)
white_noise = rand(1,n)*0.5 - 0.5/2;
plot(white_noise)
title('noise')

subplot(2,2,4)
d5 = d4 + white_noise;
plot(d5)
title('y + noise filtered')
