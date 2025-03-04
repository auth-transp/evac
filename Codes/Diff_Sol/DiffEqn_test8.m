%Using odefcn function which is defined for our system of equations%
tau=0.25;
vi0=1.35;
[t,x]=ode45(@(t,y) odefcn(t,y,tau,vi0),[0 60],[0 1.5]);
plot(t,x(:,2));