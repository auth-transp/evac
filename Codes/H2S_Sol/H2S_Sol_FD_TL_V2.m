% same as v1 but with modified TL calculations

clear;
tin=0;      %Intial Time
tfin=60;    %Final Time
h=1;        %Step size for finite difference (s)
m=80;       %Mass of agent (kg)
%ft=12;      %Toxic force as per symptom (N)
tau=1;      %Relaxation time (1/s)
v0=1.35;    %Average velocity of agent (m/s)
t=(tin-2):h:(tfin);   %Time domain for finite difference (s)
vo=1;        %Initial velocity of agent (m/s)

x=zeros(1,length(t));       %Intialization
vel=zeros(1,length(t));     %Initialization
force=zeros(1,length(t));   %Initialization

for i=4:length(t)
    x(3)=0;        %Boundary Condition 1 - x(0)
    x(2)=(-vo*h);  %Calculated by hand using boundary condition v(0) = x(0)-x(-1)/h
    x(1)=x(2);     %Just so x(1) has a value, doesn't affect solution
    Fmax=m*((v0-vo)/tau); %Motivation force at t=0 calculation
        TL=cTLV5(ctf(x),t,1,SymptomsSetup('H2S')); %Toxic Load Function

        if TL(1)>0.01 && TL(2)<0.01 && TL(3)<0.01 % Only level 1
            TL=TL(1);
            if TL>1
                TL=1;
            end
        elseif TL(1)>1 && TL(2)>0.01 && TL(3)<0.01 % Only level 2
            TL=TL(2)+1;
            if TL>2
                TL=2;
            end
        elseif TL(1)>1 && TL(2)>1 && TL(3)>0.01 % Only level 3
            TL=TL(3)+2;
            if TL>3
                TL=3;
            end
        elseif TL(1)>0.01 && TL(2)>0.01 && TL(3)<0.01 % Level 1&2
            TL=TL(1)+TL(2);                               % Skip level 1  
            if TL>2
                TL=2;
            end
        elseif TL(1)>0.01 && TL(2)>0.01 && TL(3)>0.01 % Level 1,2 & 3
            TL=TL(1)+TL(2)+TL(3);                               % Skip leve1&2  
            if TL>3
                TL=3;
            end
        elseif TL(1)<0.01 && TL(2)<0.01 && TL(3)<0.01 % below 1
                    TL=0;                             % TL=0
        elseif TL(1)>1 && TL(2)>0.01 && TL(3)>0.01    % Level 2&3 
            TL=TL(1)+TL(2)+TL(3);                               % Skip Level 1&2  
            if TL>3
                TL=3;
            end
        end
        TL=TL(1);
           
    ft=TLvsF(TL,Fmax); %Response of ft wrt TL
    x(i)=(((x(i-1)*(2*tau+h))-(x(i-2)*tau)+((h^2)*v0))/(tau+h))-((ft*TL*tau*h^2)/m*(tau+h));  %Simplified Equation for finite difference
    vel(3)=vo;                              %Intial Velocity of Agent
    vel(i)=((x(i)-x(i-1))/h);               %Velocity of agent as calculated from FD
    force(i)=(m*(v0-vel(i))/tau)-(ft*TL/m); %Mutiplying base equation by m; and using the right side
        if vel(i)>=0            %if loop for making velocity only positive
            vel(i)=vel(i);
        else
            vel(i)=0;
        end
        if x(i)>=x(i-1)         %if loop for making agent only go forward
            x(i)=x(i);
        else
            x(i)=x(i-1);
        end
end

%Graphs

subplot(4,1,1);
plot(t(3:end),x(3:end));
title('x vs t');
xlabel('time(s)');
ylabel('Dist(m)');

subplot(4,1,2);
plot(t(3:end),vel(3:end));
title('vel vs t');
xlabel('time(s)');
ylabel('Vel(m/s)');

subplot(4,1,3);
plot(t(3:end),force(3:end));
title('force vs t');
xlabel('time(s)');
ylabel('Force(N)');

subplot(4,1,4);
plot(t(3:end),ctf(x(3:end)));
title('Conc vs t');
xlabel('time(s)');
ylabel('Conc(ppm)');

% subplot(5,1,5);
% plot(t,TL);
% title('TL vs t');