%finite difference approach

m=80;           %mass of agent (kg)
TL=1;           %Toxic load (curently defined as constant - incorrect)
ft=50;          %Toxic force as per symptom (N)
tau=1;          %relaxation time (1/s)
v0=1.35;        %Average velocity of agent (m/s)
vin=1.35;       %Intial Velocity of agent (t=0) (m/s) 
h=1;            %Step size for finite difference (s)
tin=0;          %Start Time (s)
tfin=60;        %Finish Time (s)
t=tin:h:tfin;   %Time domain for finite difference (s)


i=tin:h:tfin;       %Number of points for x
x(i)=0:length(i);   %Defining x and indices
x(0)=0;             %First Boundary Condition

%Second boundary condition is v(0)=1.35
%Therefore, dx/dt(0)=1.35
%Applying finite difference to the above formula and the finite difference equation for x(0) we get,

x(1)=((vin+(((h^2*v0/2*tau)-(h^2*50*TL/2*m))/(2*tau-h/4*tau))*(2*tau-h)*(2*h))/((2*tau-h)+((2*tau+h)*(2*h))));

