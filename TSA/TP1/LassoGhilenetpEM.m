img = importdata("img1.dat");
img2 =  imread("img2.png");
img3 =  imread("img3.png");
img4 =  imread("img4.png");

size(img); %Should be square

vect_x = img(:);
vect_x2 = im2double(img2(:));
vect_x3 = im2double(img3(:));
vect_x4 = im2double(img4(:));

size(vect_x); % Should be 50 * 50 
%[pi_1, pi_2, m1, sig1, m2, sig2]
%[N0, N0, 0, N0, 0, N0)
theta_0 = [0.6,0.4,5,10,20,1];

%algoEM(vec_x, theta_param)

[final_theta,b,all_thetas]=algoEM(vect_x, theta_0);

final_theta_cell = num2cell(final_theta);
[p1,p2,m1,s1,m2,s2] = final_theta_cell{:};

%%


%title("img1.data Histogram")


range = -10:0.1:25;


first_normal  = pdf(makedist('Normal','mu',m1,'sigma',s1),range);
second_normal = pdf(makedist('Normal','mu',m2,'sigma',s2),range);
%final_theta

hold on;
plot(range,first_normal)
plot(range,second_normal)

histogram(vect_x, 20,'Normalization', 'probability', 'DisplayName', 'Normalized Histogram');
legend;

hold off;

%%

%P's
all_p1 = cellfun(@(x) x(1), all_thetas);
all_p2 = cellfun(@(x) x(2), all_thetas);
hold on
plot(all_p1)
plot(all_p2)
legend('\pi 1','\pi 2')

xlabel('c')
ylabel('\pis values')
title("\pis converge over c")
hold off

%%
%Means
all_m1 = cellfun(@(x) x(3), all_thetas);
all_m2 = cellfun(@(x) x(5), all_thetas);
hold on
plot(all_m1)
plot(all_m2)
legend('\mu 1','\mu 2')

xlabel('c')
ylabel('\mus values')
title("\mus converge over c")
hold off

%%
%Sigmas
all_s1 = cellfun(@(x) x(4), all_thetas);
all_s2 = cellfun(@(x) x(6), all_thetas);
hold on
plot(all_s1)
plot(all_s2)
legend('\sigma 1','\sigma 2')

xlabel('c')
ylabel('\sigmas values')
title("\sigmas converge over c")
hold off

%%



%% Img 2
theta_0 = [0.5,0.5,0.3,0.2,0.9,0.1];

[final_theta,b,all_thetas]=algoEM(vect_x2, theta_0);

final_theta_cell = num2cell(final_theta);
[p1,p2,m1,s1,m2,s2] = final_theta_cell{:};

range = 0:0.01:1;


first_normal  = pdf(makedist('Normal','mu',m1,'sigma',s1),range);
second_normal = pdf(makedist('Normal','mu',m2,'sigma',s2),range);

first_normal = first_normal/(max(first_normal)*6);
second_normal = second_normal/(max(second_normal)*4);


hold on;
plot(range,first_normal)
plot(range,second_normal)

histogram(vect_x2, 10,'Normalization', 'probability', 'DisplayName', 'Normalized Histogram');
legend;

hold off;




