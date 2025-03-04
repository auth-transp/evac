%Concentration as a function of 'x' to be used with cTLV5
%Master file for ambient concentration

% function ct=ctf(x)
% clear ct;
% for i=1 : length(x)
%     if  x(i)>0 && x(i)<=6; %Flat line at 0-6 m
%         ct(i)=0;
%      elseif x(i)>6 && x(i)<12 % slow rise to 50ppm at 6-12 m
%          ct(i)=((25/3)*x(i))-50;
%      elseif x(i)>=12 && x(i)<=30 %slow fall to 20ppm at 12-30
%          ct(i)=(-(10/9)*x(i))+(160/3);
%      elseif x(i)>30 && x(i)<=40 %slow rise to 35ppm at 30-40m
%          ct(i)=((3/2)*x(i))-25;
%      elseif x(i)<=0 % 0ppm for negtive x
%          ct(i)=0;
%      else           %35ppm constant after 40m
%          ct(i)=35;
%      end
% end
% end

% function ct=ctf(x)
% clear ct;
% for i=1 : length(x)
%     if x(i)>0;
%         ct(i)=380;
%     else
%         ct(i)=0;
%     end
% end

function ct=ctf(x)
clear ct;
for i=1 : length(x)
    if  x(i)>0 && x(i)<20; %Flat line 
        ct(i)=380;
     elseif x(i)>20 && x(i)<40 % 
         ct(i)=170;
     elseif x(i)>=40 && x(i)<=60 %0
         ct(i)=140;
     elseif x(i)>60 && x(i)<=200 %
         ct(i)=85;
     elseif x(i)>200 && x(i)<=400 %
         ct(i)=(5-85)/(400-200)*(x(i)-200)+85;    
     elseif x(i)<=0 % 0ppm for negtive x
         ct(i)=0;
     else           %
         ct(i)=5;
     end
end
end