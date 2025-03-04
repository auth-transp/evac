%nis loop present
clear;
vi00=1.35;
tau=1.0;
sis=125;
nis=-1;
m=80;
xs=20;
Vi=3;
Vinf=30;
[t,x]=ode45(@(t,x) odefcn6(t,x,vi00,tau,sis,nis,m,xs,Vi,Vinf),[0 60],[0 1.35]);

for i=1:length(t)
    dxdt = odefcn6(t(i),x(i,:),vi00,tau,sis,nis,m,xs,Vi,Vinf);
    fs(i)=m*dxdt(2);
end
subplot(3,1,1);
plot(t,x(:,2));
title('vel & x vs t');
yyaxis right;
plot(t,x(:,1));
subplot(3,1,2);
plot(x(:,1),fs);
yyaxis right;
plot(x(:,1),x(:,2));
title('fs & vel vs x');
subplot(3,1,3);
plot(t,fs);
title('fs vs t');