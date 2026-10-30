# Μοντέλο τοξικού φορτίου (Boris & Patnaik, 2014): πιστή μεταφορά των setupToxic / tl_rate / update_toxic_load
# του DStarLite - Scenario 3.jl, με τους πίνακες σε struct αντί για global μεταβλητές.
# Το test_sim.jl συγκρίνει αριθμητικά τις συναρτήσεις αυτές με τις αυθεντικές του σεναρίου.

# Πίνακες του μοντέλου: Balpha (7×3) εκθέτες power-law, Btime (7×3) χρόνοι (s), Brho (3×7) συγκεντρώσεις (mg/m^3)
struct ToxicTables
    Balpha::Matrix{Float64}
    Btime::Matrix{Float64}
    Brho::Matrix{Float64}
end

# Υπολογίζει τους πίνακες από τα όρια AEGL του H2S: για κάθε σύμπτωμα (οσμή, ερεθισμός, οίδημα)
function setup_toxic()
    Atime = [0.0, 2.0, 5.0, 8.0, 15.0, 30.0] #min
    Arho = zeros(3, 6)
    Arho[1, 2:6] = [2.87, 2.33, 2.09, 1.81, 1.55]                   # οσμή
    Arho[2, 2:6] = [202.38, 164.38, 147.74, 128.10, 109.45]         # ερεθισμός
    Arho[3, 2:6] = [286.71, 232.87, 209.30, 181.47, 155.05] #ppm    # οίδημα
    MW = 34 #Molecular weight of H2S in g/mol
    Arho *= MW / 24.04 #mg/m^3
    Arho = Arho'
    Atime = Atime * 60 #seconds
    taumin = 200.                                                   # (Boris&Patnaik, 2014) συντομότερος χρόνος έναρξης
    taumax = 86400.                                                 # μεγαλύτερος χρόνος έναρξης
    Brho = zeros(7, 3)
    Balpha = zeros(7, 3)
    rhomax = zeros(1, 3)
    rhomin = zeros(1, 3)
    Btime = zeros(7, 3)

    for k = 1:3
        for b = 2:6
            Brho[b, k] = Arho[b, k]
            Btime[b, k] = Atime[b]
        end
        Btime[1, k] = taumin
        Btime[7, k] = taumax
    end

    for k = 1:3
        for b = 3:6
            if Brho[b-1, k] == Brho[b, k]
                Balpha[b, k] = 0.0
            else
                Balpha[b, k] = log(Atime[b] / Atime[b-1]) / log(Brho[b-1, k] / Brho[b, k])
            end
        end
        Balpha[2, k] = Balpha[3, k]
        Balpha[1, k] = Balpha[2, k]
        Balpha[7, k] = Balpha[6, k]
    end
    for k = 1:3
        if Balpha[3, k] == 0
            rhomax[k] = Brho[2, k]
            Brho[1, k] = rhomax[k]
        else
            rhomax[k] = Brho[2, k] * (Btime[2, k] / taumin)^(1 / Balpha[2, k])
            Brho[1, k] = rhomax[k]
        end

        if Brho[5, k] == Brho[6, k]
            rhomin[k] = Brho[6, k]
            Brho[7, k] = rhomin[k]
        else
            rhomin[k] = Brho[6, k] * (Btime[6, k] / taumax)^(1 / Balpha[6, k])
            Brho[7, k] = rhomin[k]
        end
    end
    for k = 1:3
        for b = 2:7
            if Balpha[b, k] == 0
                Btime[b, k] = Btime[b-1, k]
            end
        end
    end
    for k = 1:3
        for b = 3:5
            if Balpha[b-1, k] == 0 && Balpha[b, k] > 0
                Balpha[b, k] = log(Btime[b, k] / Btime[b-1, k]) / log(Brho[b-1, k] / Brho[b, k])
            end
        end
    end
    return ToxicTables(Balpha, Btime, Matrix(Brho'))
end

# Ρυθμός συσσώρευσης τοξικού φορτίου dTL/dt (s^-1) για πυκνότητα rho (mg/m^3) και επίπεδο AEGL k (1, 2, 3)
function tl_rate(t::ToxicTables, rho, k)
    Balpha, Btime, Brho = t.Balpha, t.Btime, t.Brho
    if rho < Brho[k, 6]
        Balpha[7, k] == 0.0 && return 0.0                       # κατώφλι: καμία συσσώρευση
        rho >= Brho[k, 7] || return 0.0                         # rho < rhomin: καμία συσσώρευση
        return (rho / Brho[k, 6])^Balpha[6, k] / Btime[6, k]
    end
    rho >= Brho[k, 1] && return 1.0 / Btime[1, k]               # rho >= rhomax: μέγιστος ρυθμός 1/taumin

    if rho < Brho[k, 5] && Brho[k, 5] != Brho[k, 6]
        b = 5
    elseif rho < Brho[k, 4] && Brho[k, 4] != Brho[k, 5]
        b = 4
    elseif rho < Brho[k, 3] && Brho[k, 3] != Brho[k, 4]
        b = 3
    elseif rho < Brho[k, 2]
        b = 2
    else
        b = 1
    end
    return (rho / Brho[k, b+1])^Balpha[b+1, k] / Btime[b+1, k]
end

# Ενημερώνει το TL των τριών επιπέδων AEGL για έκθεση σε πυκνότητα Ct επί χρόνο dt (TL_k >= 1: έναρξη συμπτωμάτων)
function update_toxic_load(t::ToxicTables, Ct, TLcurrent, dt)
    TL = TLcurrent
    for k = 1:3
        TL[k] = TL[k] + tl_rate(t, Ct, k) * dt
    end
    return TL
end
