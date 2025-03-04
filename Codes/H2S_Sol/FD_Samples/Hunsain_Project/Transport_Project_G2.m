% % % % Nour Basha and Husnain Manzoor
% % % % CHEN 629 - Transport Phenomena
% % % % Class Project Group 2

% % Code Initialization:
clc; % clear command window
clear all; % clear all variables from workspace
close all; % close all open figures
format short; % set number format to 5 significant figures
addpath('../'); % add relevant functions needed to save figures

% % Control Panel:
numvar=10; % number of variables/nodes per dimension. The higher numvar is, the finer the pressure grid is
savepics=0; % save figures as images (1) or not (0)

% % Given values:
North=[45:5:80,80,80]; % pressure on the north side
South=[60*ones(1,8),65,66]; % pressure on the south side
East=[68,69,70,70,72,72,73,73,75]; % pressure on the east side
XPos=0:200:1800; % respective x-position of given pressures for north and south sides
YPos=200:200:1800; % respective y-position of given pressures for east side
k=3.5*(10^-10); % permeability (m^2)
mu=2.5*(10^-3); % viscosity (Pa*s)
por=0.35; % porosity

if(savepics==1) % if the user desires to save figure to file
    foldername=sprintf('numvar=%d',numvar); % set foldername to save pics to
    mkdir(foldername); % create new directory
end

% % find best fit polynomial to boundary pressure values for north, south and east.
% % This is necessary when using a finer grid, where we would need to
% % extrapolate/predict values in between given boundary pressure values:
polynorth=polyfit(XPos,North,3); % polynomial for north side
polysouth=polyfit(XPos,South,3); % polynonmial for south side
polyeast=polyfit(YPos,East,2); % polynomial for east side

% % Plot best-fit polynomials and given boundary pressure values for each boundary:
figure('units','normalized','outerposition',[0 0 1 1]); % create new figure
subplot(2,2,1); % set current axis to first subplot
a=scatter(XPos,North,'r'); % plot given North boundary pressure values (red circles)
xlabel('X-Position'); % set x-axis label for first subplot
ylabel('Pressure (bar)'); % set y-axis label for first subplot
title('North'); % set title for first subplot
hold on; % allow for additional plots to be added to first subplot
subplot(2,2,2); % set current axis to second subplot
b=scatter(XPos,South,'r'); % plot given South boundary pressure values (red circles)
xlabel('X-Position'); % set x-axis label for second subplot
ylabel('Pressure (bar)'); % set y-axis label for second subplot
title('South'); % set title for second subplot
hold on; % allow for additional plots to be added to first subplot
subplot(2,2,3); % set current axis to third subplot
c=scatter(YPos,East,'r'); % plot given East boundary pressure values (red circles)
xlabel('Y-Position'); % set x-axis label for third subplot
ylabel('Pressure (bar)'); % set y-axis label for third subplot
title('East'); % set title for third subplot
hold on; % allow for additional plots to be added to first subplot

% % extrapolate/predict new boundary pressure values for finer grid:
XPos=linspace(XPos(1),XPos(end),numvar); % find new respective x-position of given pressures for north and south sides
YPos=linspace(YPos(1),YPos(end),numvar); % find new respective y-position of given pressures for east side
North=polyval(polynorth,XPos); % find new North boundary pressure values
South=polyval(polysouth,XPos); % find new South boundary pressure values
East=polyval(polyeast,YPos); % find new East boundary pressure values

subplot(2,2,1); % set current axis to first subplot
plot(a,XPos,North,'b'); % plot new North boundary pressure values (blue line)
grid on; % turn on grid
hold off; % disallow any additional plots for first subplot
subplot(2,2,2); % set current axis to second subplot
plot(b,XPos,South,'b'); % plot new South boundary pressure values (blue line)
grid on; % turn on grid
hold off; % disallow any additional plots for first subplot
subplot(2,2,3); % set current axis to third subplot
plot(c,YPos,East,'b'); % plot new East boundary pressure values (blue line)
grid on; % turn on grid
hold off; % disallow any additional plots for first subplot

if(savepics==1) % if the user desires to save figure to file
    filename='Polyfit Pressure Sides'; % set filename for first figure
    savefig(gcf,[filename,'.fig']); % save as .fig file
    export_fig(filename,'-jpg','-r300','-nocrop',gcf); % save as .jpg file
    close all; % close open figure
end

% % grididx is the matrix of indices of pressure nodes in desired pressure
% % grid. This is used in order to apply the finite difference method.
grididx=zeros(numvar); % initialize to zero
for i=1:numvar
    grididx(i,:)=(1:1:numvar)+numvar*(i-1); % set indices for each node
end

% % lftmat is the (left) matrix of coefficients for each unknown (i.e. not given) pressure node in the grid.
lftmat=zeros(numvar*numvar); % initialize to zero
for i=1:numvar % run through rows of pressure grid
    for j=1:numvar % run through columns of pressure grid
        currpt=grididx(i,j); % find index of current node in pressure grid
        lftmat(currpt,currpt)=-4; % default coefficient of current node in lftmat (as defined by the finite difference method)
        coeff=1; % default coefficient of neighbouring unknown pressure nodes in grid
        % % find neighbouring pressure nodes in pressure grid:
        if(i==1) % if current node is at top of grid (near north boundary)
            if(j==numvar) % if current node is at the right of grid (near east boundary)
                allpts=[grididx(i+1,j),grididx(i,j-1)]; % 2 unknown pressure nodes on the bottom and left
            elseif(j==1) % if current node is at the left of the grid (near west boundary)
                allpts=[grididx(i+1,j),grididx(i,j+1)]; % 2 unknown pressure nodes on the bottom and right
                coeff=[1,2]; % since the west side is impermeable, the right node has a coefficient of 2
            else % if current node is anywhere else near north boundary
                allpts=[grididx(i+1,j),grididx(i,j-1),grididx(i,j+1)]; % 3 unknown pressure nodes on the bottom, left, and right
            end
        elseif(i==numvar) % if current node is at the bottom of grid (near south boundary)
            if(j==numvar) % if current node is at the right of grid (near east boundary)
                allpts=[grididx(i-1,j),grididx(i,j-1)]; % 2 unknown pressure nodes on the up and left
            elseif(j==1) % if current node is at the left of the grid (near west boundary)
                allpts=[grididx(i-1,j),grididx(i,j+1)]; % 2 unknown pressure nodes on the up and right
                coeff=[1,2]; % since the west side is impermeable, the right node has a coefficient of 2
            else % if current node is anywhere else near south boundary
                allpts=[grididx(i-1,j),grididx(i,j-1),grididx(i,j+1)]; % 3 unknown pressure nodes on the up, left, and right
            end
        else % if current node is anywhere else in the grid
            if(j==numvar) % if current node is at the right of grid (near east boundary)
                allpts=[grididx(i-1,j),grididx(i,j-1),grididx(i+1,j)]; % 3 unkown pressure nodes on the up, left, and bottom
            elseif(j==1) % if current node is at the left of grid (near west boundary)
                allpts=[grididx(i-1,j),grididx(i,j+1),grididx(i+1,j)]; % 3 unknown pressure nodes on the up, right, and bottom
                coeff=[1,2,1]; % since the west side is impermeable, the right node has a coefficient of 2
            else % if current node is anywhere else in the grid
                allpts=[grididx(i-1,j),grididx(i,j-1),grididx(i,j+1),grididx(i+1,j)]; % 3 unknown pressure nodes on the up, left, right, and bottom
            end
        end
        lftmat(currpt,[allpts(:)])=coeff.*1; %#ok<NBRAK> % set coefficients in left matrix
    end
end

% % rgtmat is the (right) matrix (column vector) of coefficients for each known (i.e. given) pressure node in the grid.
rgtmat=zeros(numvar*numvar,1); % initialize to zero
for i=1:numvar % run through rows of pressure grid
    for j=1:numvar % run through columns of pressure grid
        rgtidx=grididx(i,j); % find index of current node in pressure grid
        if(i==1) % if current node is at top of grid (near north boundary)
            if(j==numvar) % if current node is at the right of grid (near east boundary)
                rgtmat(rgtidx,1)=-North(j)-East(i); % add north and east boundary pressure values
            else % if current node is anywhere else near north boundary
                rgtmat(rgtidx,1)=-North(j); % add north boundary pressure value
            end
        elseif(i==numvar) % if current node is at the bottom of grid (near south boundary)
            if(j==numvar) % if current node is at the right of grid (near east boundary)
                rgtmat(rgtidx,1)=-South(j)-East(i); % add south and east boundary pressure values
            else % if current node is anywhere else near south boundary
                rgtmat(rgtidx,1)=-South(j); % add south boundary pressure value
            end
        else % if current node is anywhere else in grid
            if(j==numvar) % if current node is at the right of grid (near east boundary)
                rgtmat(rgtidx,1)=-East(i); % add east boundary pressure value
            else % if current node is anywhere else in grid
                rgtmat(rgtidx,1)=0; % no known pressure values near current node (set to zero)
            end
        end
    end
end

PressureGrid=(reshape(lftmat\rgtmat,numvar,numvar))'; % pressure grid
[XPos1,YPos1]=meshgrid(XPos,YPos); % create meshgrid to define x- and y- position of each pressure value in PressureGrid
YPos1=flipud(YPos1); % flip YPos matrix vertically to match pressure values to their correct positon

figure('units','normalized','outerposition',[0 0 1 1]); % create new figure
contour(XPos1,YPos1,PressureGrid,75); % contour plot of PressureGrid
xlabel('X-Position'); % set x-axis label
ylabel('Y-Position'); % set y-axis label
grid on; % turn grid on
title('Pressure Grid Contour Plot'); % set title for plot
colorbar; % turn on colorbar

if(savepics==1) % if the user desires to save figure to file
    filename=[foldername,'\Pressure Grid Contour Plot']; % set filename for second figure
    savefig(gcf,[filename,'.fig']); % save as .fig file
    export_fig(filename,'-jpg','-r300','-nocrop',gcf); % save as .jpg file
    close all; % close open figure
end

% % volumetric flux vector plot:
dx=XPos(2)-XPos(1); % find change in x-positon for each pressure value
dy=YPos(2)-YPos(1); % find change in y-positon for each pressure value
[dPx,dPy]=gradient(PressureGrid,dx,dy); % find change in pressure along x- and y- positons
qx=(-k/mu)*dPx; % volumetric flux along x-positon
qy=(-k/mu)*-1*dPy; % volumetric flux along y-position

figure('units','normalized','outerposition',[0 0 1 1]); % create new figure
quiver(XPos1,YPos1,qx,qy,'LineWidth',1.5); % plot vector plot of volumetric flux
xlabel('X-Position'); % set x-axis label
ylabel('Y-Position'); % set y-axis label
grid on; % turn grid on
title('Darcy Flux Vector Plot'); % set title for plot

if(savepics==1) % if the user desires to save figure to file
    filename=[foldername,'\Darcy (Volumetric) Flux Vector Plot']; % set filename for third figure
    savefig(gcf,[filename,'.fig']); % save as .fig file
    export_fig(filename,'-jpg','-r300','-nocrop',gcf); % save as .jpg file
    close all; % close open figure
end

% % fluid velocity vector plot:
v0x=(1/por)*qx; % fluid velocity along x-position
v0y=(1/por)*qy; % fluid velocity along y-position

figure('units','normalized','outerposition',[0 0 1 1]); % create new figure
quiver(XPos1,YPos1,v0x,v0y,'LineWidth',1.5); % plot vector plot of fluid velocity
xlabel('X-Position'); % set x-axis label
ylabel('Y-Position'); % set y-axis label
grid on; % turn grid on
title('Fluid Velocity Vector Plot'); % set title for plot

if(savepics==1) % if the user desires to save figure to file
    filename=[foldername,'\Fluid Velocity Vector Plot']; % set filename for fourth figure
    savefig(gcf,[filename,'.fig']); % save as .fig file
    export_fig(filename,'-jpg','-r300','-nocrop',gcf); % save as .jpg file
    close all; % close open figure
end

% % Plot 3d Surface Plot of PressureGrid:
figure('units','normalized','outerposition',[0 0 1 1]);
surf(XPos1,YPos1,PressureGrid);
xlabel('X-Position');
ylabel('Y-Position');
zlabel('Pressure (bar)');
title('3D Surface Pressure Grid');
grid on;
colorbar;
if(savepics==1)
    filename=[foldername,'\3D Surface Pressure Grid'];
    savefig(gcf,[filename,'.fig']);
    export_fig(filename,'-jpg','-r300','-nocrop',gcf);
    close all;
end

% % Plot contour plot of PressureGrid and flux vector plots in 1 figure:
figure('units','normalized','outerposition',[0 0 1 1]);
contour(XPos1,YPos1,PressureGrid,75);
xlabel('X-Position');
ylabel('Y-Position');
grid on;
colorbar;
hold on;
quiver(XPos1,YPos1,qx,qy,'LineWidth',1.5);
hold off;
title('Pressure Grid Contour Plot and Darcy Flux Vector Plot');
if(savepics==1)
    filename=[foldername,'\Contour Plot and Darcy Flux'];
    savefig(gcf,[filename,'.fig']);
    export_fig(filename,'-jpg','-r300','-nocrop',gcf);
    close all;
end










