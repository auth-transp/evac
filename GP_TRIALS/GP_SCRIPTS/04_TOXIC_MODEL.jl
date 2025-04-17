# 04_toxic_model.jl

function setupToxic()
    Atime = [0.0, 0.17, 0.83, 1.67, 4.17, 8.33] * 60
    Arho = zeros(3, 6)
    Arho[1, 2:6] = [4.85, 4.23, 4.17, 4.06, 3.82]
    Arho[2, 2:6] = [180.79, 157.56, 155.43, 151.37, 142.48]
    Arho[3, 2:6] = [485.62, 423.22, 417.49, 406.59, 382.71]
    Arho *= MW / 24.04
    Arho = Arho'

    taumin, taumax = 200.0, 86400.0
    Brho = zeros(7, 3)
    Balpha = zeros(7, 3)
    rhomax = zeros(1, 3)
    rhomin = zeros(1, 3)
    Btime = zeros(7, 3)

    for k in 1:3
        for b in 2:6
            Brho[b, k] = Arho[b, k]
            Btime[b, k] = Atime[b]
        end
        Btime[1, k] = taumin
        Btime[7, k] = taumax
    end

    for k in 1:3
        for b in 3:6
            if Brho[b-1,k] == Brho[b,k]
                Balpha[b,k] = 0.0
            else
                Balpha[b,k] = log(Atime[b]/Atime[b-1]) / log(Brho[b-1,k]/Brho[b,k])
            end
        end
        Balpha[2,k] = Balpha[3,k]
        Balpha[1,k] = Balpha[2,k]
        Balpha[7,k] = Balpha[6,k]
    end

    for k in 1:3
        if Balpha[3,k] == 0
            rhomax[k] = Brho[2,k]
            Brho[1,k] = rhomax[k]
        else
            rhomax[k] = Brho[2,k]*(Btime[2,k]/taumin)^(1/Balpha[2,k])
            Brho[1,k] = rhomax[k]
        end

        if Brho[5,k] == Brho[6,k]
            rhomin[k] = Brho[6,k]
            Brho[7,k] = rhomin[k]
        else
            rhomin[k] = Brho[6,k]*(Btime[6,k]/taumax)^(1/Balpha[6,k])
            Brho[7,k] = rhomin[k]
        end
    end

    for k in 1:3, b in 2:7
        if Balpha[b,k] == 0
            Btime[b,k] = Btime[b-1,k]
        end
    end

    for k in 1:3, b in 3:5
        if Balpha[b-1,k]==0 && Balpha[b,k]>0
            Balpha[b,k] = log(Btime[b,k]/Btime[b-1,k]) / log(Brho[b-1,k]/Brho[b,k])
        end
    end

    return Balpha, Btime, Brho'
end

function update_toxic_load(Ct, TLcurrent, dt)
    TL = TLcurrent
    TL_rate = 0.0
    for k in 1:3
        Cmin, Cmax = Brho[k,7], Brho[k,1]
        if Ct > Cmax
            TL_rate = 1 / Btime[1]
        elseif Ct < Cmin
            TL_rate = 0.0
        else
            for i in 2:size(Btime,1)
                if i <= size(Brho,2) && Brho[k,i-1] < Ct < Brho[k,i]
                    TL_rate = (1/Btime[i,k]) * ((Ct/Brho[k,i])^Balpha[i,k])
                end
            end
        end
        TL[k] += TL_rate * dt
    end

    TL .= min.(TL, 1.0)
    return TL
end

Balpha, Btime, Brho = setupToxic()