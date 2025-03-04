function ct=ct_time(t)
clear ct
% ct=zeros(1,length(t));
for i=1 : length(t)
    if t(i)>4;
        ct=((0.00007*t(i))^2)+(0.4053*t(i))+4.496;
    end
end