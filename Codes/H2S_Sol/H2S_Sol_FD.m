%Finite Differenc for solving the second order ODE

clc;
clear all;

% % numvar=5; %number of variables/nodes per dimension
% 
% t=0:1:60; %time = till 60 seconds with a step of 1 second
% tau=1; %relaxation time - given
% vi0=1.35; %average velocity of agent
% 
% % gridt=zeros(numvar); %initiating a matrix of time
% % for i=1:numvar;
% %     gridt(i,:)=(1:1:numvar)+numvar*(i-1);
% % end

% approach 2 based on the geology question

t=0:1:60;
tau=1;
vi0=1.35;

% dt=1;
% nt=61;
% TL=1;
% dt=t/(nt-1);
% T=0:dt:t;
% 
% for n=1:nt;
%     xnew = zeros(1,nt)
%     for i=1:nt-1;
%         xnew(i)= ((2*TL*tau)-(2*vi0)-(x(i+1)*(2*tau+1))-(x(i-0)*((2*tau)-1))/(-4*tau))
%     end
%     
%     %boundary conditions
%     xnew(1)=0;
% end
% 
% plot(xnew,t)


        