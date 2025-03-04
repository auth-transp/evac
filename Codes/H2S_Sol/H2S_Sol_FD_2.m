%finite difference approach
%Matlab does not accept '0' or 'negative' indices for 'x'
%x(-2)=x(1)
%x(-1)=x(2)
%x(0)=x(3)

clear;
m=80;       %Mass of agent (kg)
TL=0.75;    %Toxic load (curently defined as constant - incorrect)
ft=12;   %Toxic force as per symptom (N)
tau=1;      %Relaxation time (1/s)
v0=1.35;    %Average velocity of agent (m/s)
h=1;        %Step size for finite difference (s)
t=1:h:64;   %Time domain for finite difference (s)

for i=4:64
    x(3)=0;        %Boundary Condition 1 - x(0)
    x(2)=(-v0*h);  %Calculated by hand using boundary condition v(0)=1.35 - x(-1)
    x(i)=(((x(i-1)*(2*tau+h))-(x(i-2)*tau)+((h^2)*v0))/(tau+h))-((ft*TL*tau*h^2)/m*(tau+h));  %Simplified Equation for finite difference
end

 
for i=4:64
    vel(3)=((x(3)-x(2))/h); 
    vel(i)=((x(i)-x(i-1))/h);          %Velocity of agent as calculated from FD
end
for i=4:64
    force(i)=(m*(v0-vel(i))/tau)-(ft*TL/m); %Mutiplying base equation by m; and using the right side
end

%Graphs

subplot(3,1,1);
plot(t,x);
title('x vs t');

subplot(3,1,2);
plot(t,vel);
title('vel vs t');

subplot(3,1,3);
plot(t,force);
title('force vs t');
