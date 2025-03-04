%Defining the eqaution for finite difference
%Base Equation => d2x/dt2=(v0-dx/dt)/tau - ft*TL/m
%Using backward difference
%xi=x
%xi-1=y
%xi-2=z

clear;
% v0=1.35;
% tau=1;
% m=80;
% ft=50;
% h=1;
% TL=1;

% syms x y z v0 tau m ft h TL
% solve((x-2*y-z/h^2)-(v0-((x-y)/h)/tau)+(ft*TL/m)) %Backward Difference

%Forward Difference
%xi=x
%xi+1=y
%xi+2=z

syms x l n v0 tau m ft h TL
eqn=((x-(l(2*tau-h)-n*tau)+h*v0)/(tau+h))-(ft*TL*tau*(h^2)/m*(tau+h));
solve(eqn,x)
