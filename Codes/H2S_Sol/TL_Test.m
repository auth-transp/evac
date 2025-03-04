% Toxic Load test

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