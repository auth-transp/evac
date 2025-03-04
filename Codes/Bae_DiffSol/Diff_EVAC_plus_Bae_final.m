%final result after removing the step functions for nis and vio%
%the agent now only moves forward%

clear;
mi=80;
nis=-1;
sis=125;
Vi=5;
Vinf=30;
xs=20;
tau=1.2;
vi0=1.35;
[t,x]=ode45(@(t,x) odefcn_final(t,x,tau,vi0,mi,sis,xs,Vi,Vinf,nis),[0 60],[0 1.35]);

for i=1:length(t)
    dxdt = odefcn_final(t(i),x(i,:),tau,vi0,mi,sis,xs,Vi,Vinf,nis);
    fs(i)=mi*dxdt(2);
end
subplot(3,1,1);
plot(t,x(:,2));
title('vel & x vs t');
% yyaxis right;
% plot(t,x(:,1));
% subplot(3,1,2);
% plot(x(:,1),fs);
% yyaxis right;
% plot(x(:,1),x(:,2));
% title('fs & vel vs x');
% subplot(3,1,3);
% plot(t,fs);
% title('fs vs t');