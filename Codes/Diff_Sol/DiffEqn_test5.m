%solving using different types of ODE functions%
syms t y tau vi0
function yp = DiffEqn_test5(t,y)
yp = [1.2*y(1)-0.6*y(1)*y(2);-0.8*y(2)+0.3*y(1)*y(2)];
    tspan = [0 20];
    y0 = [2, 1];
    [t,y] = ode45(@predprey, tspan, y0);
    plot(t,y)