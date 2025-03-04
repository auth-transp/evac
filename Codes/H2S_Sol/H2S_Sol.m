%Solution file for H2S Force

clear;
k1=10;
k2=30;
k3=60;
k4=25;
Qm=250;
tau=1;
vi0=1.35;
mi=80;
n=-1;
[t,x]=ode45(@(t,x) H2S_Func(t,x,k1,k2,k3,k4,Qm,tau,vi0,mi,n),[0 60],[0 1.35]);

for i=1:length(t)
    dxdt = H2S_Func(t(i),x(i,:),k1,k2,k3,k4,Qm,tau,vi0,mi,n);
    fs(i)=mi*dxdt(2);
end

subplot(2,1,1);
plot(t,fs)
title('fs vs t')
subplot(2,1,2);
plot(t,x(:,2))
title('v vs t')