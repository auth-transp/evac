function tl=TL(t,x)
for i=1:length(t)
    TL(i)=cTLV4(ctf(x),t,1,AgentSetup('H2S'));
end