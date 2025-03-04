%trying to use the van der pol example from Chapara's book%
function yp=velocity(t,y,vi0)
vi0=1.35;
yp=[y(2); (4*(vi0-y(2)))]
[t,yp]=ode45(@velocity,[0 60],[0 0],[],vi0)
plot(t,y(:,1),'-',t,y(:,2),'--')
legend('y1','y2');
end