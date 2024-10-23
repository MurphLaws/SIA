
% algolms():

% Input:
%   x = Original Input Signal
%   d = Signal after filtering and adding a Noise
%   P = Spectrum Order of the filter
%   mu = Learning Rate


% Output:
%   allw = A matrix that store the the coefficient arrays at every
%   iteration (n)
%   w = Filter Coefficients
%   y = Output signal after after appliying the LMS Algorithm
%   e = Output Error



function [allw, w, y, e] = algolms(x, d, P, mu)

    N = length(x);

    %Initialization 
    w = zeros(P, 1);
    y = zeros(N, 1);
    e = zeros(N, 1);
    allw = zeros(N,P);
    for n = P:N
        %Definin the window with the last P signal moments
        xn = x(n:-1:n-P+1);

        %Appliying the filter to said last P moments
        y(n) = w' * xn;

        %Error caomputing
        e(n) = d(n) - y(n);
        w = w + mu * e(n) * xn;

        %Storing all the filter coefficients at every given iteration
        allw(n,:) = w;
    end
end

