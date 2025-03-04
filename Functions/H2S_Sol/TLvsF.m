%Function for the force behaviour under varying Toxic Load, from 0-3
%Refer #009 - Dr. Kostas in OneNote - Not correct
%Refer TLvsF in OneNote->Notes

function F=TLvsF(TL,Fmax)
if TL>=0 && TL<1
    F=-14;
elseif TL==1
    F=-14;
elseif TL>1 && TL<2
    F=3;
elseif TL==2
    F=3;
elseif TL>2 && TL<=3
    F=5.5;
end