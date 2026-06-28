% Example of the use of MU vs. the new version, extrapolated, MUe
% on the CBCL data set with beta = 3/2 
% See the paper 
% L.T.K. Hien,  V. Leplat, and N Gillis, "Block Majorization Minimization 
% with Extrapolation and Application to ?-NMF", January 2024. 
% See https://arxiv.org/abs/2401.06646 

clear all; close all; clc; 
%% Load data set and options 
load CBCL; 
r = 49; 
options.beta = 3/2; 
[m,n] = size(X); 
options.accuracy = 0;
options.maxiter = 100; 
options.timemax = Inf;
%% Initialization 
W0 = max(eps,rand(m,r));
H0 = max(eps,rand(r,n)); 
% To avoid an abrupt decrease of the objective at the first iteration,
% improve the intialization by 1-step MU 
options.H = MUbeta(X,W0,H0,options.beta); 
options.W = MUbeta(X',options.H',W0',options.beta); 
options.W = options.W'; 
%% Running MU without extrapolation 
options.extrapol = 'noextrap'; 
disp('***Running MU for beta-NMF without extrapolation***'); 
[W,H,e,t] = betaNMF(X,r,options);
%% Running MU without Nesterov extrapolation 
options.extrapol = 'nesterov'; 
disp('***Running MU for beta-NMF with extrapolation (that is, MUe)***'); 
[We,He,ee,te] = betaNMF(X,r,options);
%% Display results 
set(0, 'DefaultAxesFontSize', 22);
set(0, 'DefaultLineLineWidth', 2); 
figure; 
err0 = betadivfac(X,X*ones(n,1)/n,ones(1,n),options.beta); 
mine = min(min(e),min(ee))/err0; 
semilogy(e/err0-mine); 
hold on; grid on;  
semilogy(ee/err0-mine,'-.'); 
ax.YAxis.Exponent = 3;
ytickformat('%2.0f'); 
legend('MU', 'MUe', 'Interpreter', 'latex'); 
xlabel('Iterations', 'Interpreter', 'latex'); 
ylabel('$\frac{D_{3/2}(X,WH)}{D_{3/2}(X,Xee^T/n)} - e_{min}$', ... 
            'Interpreter', 'latex'); 