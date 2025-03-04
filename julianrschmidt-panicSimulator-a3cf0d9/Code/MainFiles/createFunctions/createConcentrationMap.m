% Concentration Map function
% Using the same layout as the plotGrid function

function [CM]=createConcentrationMap(limits)
xMin = limits(1);
xMax = limits(2);
yMin = limits(3);
yMax = limits(4);
Nx = ceil(xMax-xMin);
Ny = ceil(yMax-yMin);
dx=1;
dy=1;
conc=100;

CM = zeros(Nx,Ny);
    for i=1:Nx
    CM(i)=100;
        for j=1:Ny
        CM(i,j)=100;
        end
    end
end
