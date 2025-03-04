% Solving the 2-D Laplace's equation by the Finite Difference Method 
% Numerical scheme used is a second order central difference in space

%%
%Specifying parameters
P=zeros(11,11)
x=linspace(0,2000,11);                          %Number of steps in space(x)
y=linspace(0,2000,11)                           %Number of steps in space(y)       
niter=1000;                                     %Number of iterations 
dx=x(2);                                         %Width of space step(x)
dy=y(2);                                          %Width of space step(y)

%%
%Initial Conditions
k= 8*10^-10;
u=0.0035;
Poro=0.35;
%Boundary Conditions
%North
P(1,:)=100000*[88 86 84 84 84 85 86 88 90 91 95];
%West
P(:,1)=100000*[86 90 80 80 78 76 74 72 70 70 70];
%South
P(11,:)=100000*[70 70 65 65 60 60 65 86 88 90 95];
%%
%Explicit iterative scheme with C.D in space 
Pnew=P
err=Pnew-P
z=1
%Pressure Calculations
while z<niter
    for m=2:10
        for n=2:11
            if n==11
                Pnew(m,n)= (2*P(m-1,n)+P(m+1,n)+P(m,n-1))/4
                err(m,n)=abs(Pnew(m,n)-P(m,n));
            else 
                Pnew(m,n)=(P(m+1,n)+P(m-1,n)+P(m,n+1)+P(m,n-1))/4;
                err(m,n)=abs(Pnew(m,n)-P(m,n));
            end
         end
        P=Pnew;
        z=z+1
        max_error=max(max(err));
        if max_error  <1e-5
           return
        end
end
   
end
%Volumetric Flux
for j =2:10
    for i=2:10
        dPy(i,j)=(P(i+1,j)-P(i-1,j))./(400);
        qy=(-k/u).*dPy;
        dPx(i,j)=(P(i,j+1)-P(i,j-1))./(400);
        qx=(-k/u).*dPx;
    end
end
%Velocity 
Vx=qx./Poro;
Vy=qy./Poro;
V=sqrt(Vx^2+Vy^2)
Vnew=[0         0         0         0         0         0         0         0         0         0       0
         0    0.0016    0.0014    0.0014    0.0016    0.0019    0.0020    0.0017    0.0013    0.0009    0
         0    0.0014    0.0015    0.0016    0.0018    0.0021    0.0023    0.0020    0.0017    0.0015    0
         0    0.0016    0.0016    0.0016    0.0019    0.0021    0.0023    0.0019    0.0016    0.0014    0
         0    0.0016    0.0016    0.0017    0.0019    0.0022    0.0023    0.0020    0.0017    0.0015    0
         0    0.0017    0.0016    0.0017    0.0019    0.0022    0.0023    0.0021    0.0018    0.0016    0
         0    0.0017    0.0017    0.0017    0.0019    0.0022    0.0024    0.0022    0.0020    0.0017    0
         0    0.0018    0.0017    0.0017    0.0019    0.0022    0.0024    0.0024    0.0021    0.0019    0   
         0    0.0019    0.0017    0.0017    0.0018    0.0021    0.0025    0.0027    0.0024    0.0022    0
         0    0.0021    0.0018    0.0018    0.0018    0.0021    0.0026    0.0030    0.0027    0.0025    0
         0         0         0         0         0         0         0         0         0         0       0]
     
  Q=sqrt(qx^2+qy^2)/2000^2
  
  Qnew=[ 0         0         0         0         0         0         0         0         0         0    0
         0    0.1432    0.1229    0.1193    0.1371    0.1620    0.1787    0.1484    0.1155    0.0789    0
         0    0.1261    0.1295    0.1379    0.1617    0.1867    0.2015    0.1708    0.1476    0.1274    0
         0    0.1366    0.1370    0.1430    0.1653    0.1878    0.2003    0.1671    0.1440    0.1241    0
         0    0.1418    0.1413    0.1474    0.1685    0.1903    0.2024    0.1723    0.1503    0.1315    0
         0    0.1466    0.1439    0.1500    0.1695    0.1913    0.2043    0.1802    0.1589    0.1406    0
         0    0.1520    0.1454    0.1514    0.1685    0.1910    0.2066    0.1920    0.1709    0.1525    0
         0    0.1589    0.1463    0.1520    0.1653    0.1896    0.2099    0.2094    0.1878    0.1687    0
         0    0.1673    0.1471    0.1522    0.1598    0.1874    0.2155    0.2343    0.2111    0.1904    0
         0    0.1875    0.1552    0.1557    0.1554    0.1878    0.2272    0.2652    0.2393    0.2150    0
         0         0         0         0         0         0         0         0         0         0       0]
%%

%Plotting the solution
[X,Y] = meshgrid(x,y);

figure (1)
contourf(X,Y,Vnew,10)
axis ij
colorbar
xlabel('x');
ylabel('y');
title('Contour Map of Velocity')

figure (2)
contourf(X,Y,P,10)
axis ij
colorbar
xlabel('x');
ylabel('y');
title('Contour Map of Pressure')

figure (3)
contourf(X,Y,Qnew,10)
axis ij
colorbar
xlabel('x');
ylabel('y');
title('Contour Map of Volumetric Flux')


x1=[0:200:1800]
y1=[0:200:1800]

figure (5)
quiver(x1,y1,qx,qy)
axis ij
title('Vector Field of Volumetric Flux')

figure (6)
quiver(x1,y1,Vx,Vy)
axis ij
title('Vector Field of Velocity')
