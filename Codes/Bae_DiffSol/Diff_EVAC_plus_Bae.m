% this is the attempt to solve the force equation in differential form
% using Bae's smoke term
% in Bae's smoke term, we can see replace dis by dis=x-xs
% where xs=smoke boundary (user defined)
clear;
mi=80;
nis=-1;
sis=125;
Vi=1.5;
Vinf=3;
xs=10;
tau=0.25;
vi0=1.35;
[t,x]=ode45(@(t,x) odefcn2(t,x,tau,vi0,mi,sis,xs,Vi,Vinf,nis),[0 60],[0 0]);
plot(t,x(:,2));
yyaxis right;
plot(t,x(:,1));
for i=1:length(t)
    dxdt = odefcn2(t(i),x(i,:),tau,vi0,mi,sis,xs,Vi,Vinf,nis);
    fs(i)=dxdt(2);
end