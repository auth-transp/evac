% introduced a step function in the ODE function file instead of the
% heaviside function
clear;
mi=80;
nis=-1;
sis=125;
Vi=15;
Vinf=30;
xs=20;
tau=0.25;
vi0=1.35;
[t,x]=ode45(@(t,x) odefcn3(t,x,tau,vi0,mi,sis,xs,Vi,Vinf,nis),[0 60],[0 1.35]);

for i=1:length(t)
    dxdt = odefcn3(t(i),x(i,:),tau,vi0,mi,sis,xs,Vi,Vinf,nis);
    fs(i)=mi*dxdt(2);
end
subplot(2,1,1);
plot(t,x(:,2));
yyaxis right;
plot(t,x(:,1));
subplot(2,1,2);
plot(x(:,1),fs);