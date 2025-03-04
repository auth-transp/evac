%test function for F vs TL to be used with Test1

function F=f(ToxicLoad,Fmax)
for i=1:length(ToxicLoad)
if ToxicLoad(i)>=0 && ToxicLoad(i)<=1
    F(i)=Fmax*ToxicLoad(i)
elseif ToxicLoad(i)>1 && ToxicLoad(i)<=3
    F(i)=240-(120*ToxicLoad(i))
end
end
