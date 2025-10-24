using JuMP
import PGLib
import PowerModels
import HDF5: h5open
import PythonCall
import MathOptAI
import Ipopt

include("structs.jl")

# sigmoid!
sig(x) = 1.0 / (1.0 + exp(-x))

function scale_load(load, alpha_lb, alpha_ub)
    nload = length(load)
    lb = zeros(nload)
    ub = zeros(nload)

    for ii in 1:nload
        if load[ii] > 0
            ub[ii] = load[ii]*alpha_ub
            lb[ii] = load[ii]*alpha_lb
        else
            ub[ii] = load[ii]*alpha_lb
            lb[ii] = load[ii]*alpha_ub 
        end
    end
    return lb, ub
end

function soc_voltage_bound_vectors(data, nl, fr_buses, to_buses)
    # build the reg
    pm_ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
    wr_min_pm, wr_max_pm, wi_min_pm, wi_max_pm = PowerModels.ref_calc_voltage_product_bounds(pm_ref[:buspairs])

    # now, loop over the line list, and populate the updated vectors
    wr_min = zeros(nl)
    wr_max = zeros(nl)
    wi_min = zeros(nl)
    wi_max = zeros(nl)
    for ii in 1:nl
        pair = (fr_buses[ii], to_buses[ii])
        wr_min[ii] = wr_min_pm[pair]
        wr_max[ii] = wr_max_pm[pair]
        wi_min[ii] = wi_min_pm[pair]
        wi_max[ii] = wi_max_pm[pair]
    end

    return wr_min, wr_max, wi_min, wi_max
end

function parse_PM_to_SOCGridModel(network_data; perturb=false)
    # build the ref
    ref = PowerModels.build_ref(network_data)[:it][PowerModels.pm_it_sym][:nw][PowerModels.nw_id_default]

    #  build a custom OPF objective function -- the key here is "_build_opf_cl"
    pm = PowerModels.instantiate_model(network_data, PowerModels.ACPPowerModel, PowerModels.build_opf)#, PowerModels._build_opf_cl)
    OPF_soln = PowerModels.optimize_model!(pm, optimizer=Ipopt.Optimizer)
    println(OPF_soln["objective"])

    OPF_soln["solution"]
    nb = length(OPF_soln["solution"]["bus"])
    vm_pm = zeros(nb)
    va_pm = zeros(nb)
    for ii in 1:nb
        vm_pm[ii] = OPF_soln["solution"]["bus"][string(ii)]["vm"]
        va_pm[ii] = OPF_soln["solution"]["bus"][string(ii)]["va"]
    end

    nl = length(OPF_soln["solution"]["branch"])
    pf_pm = zeros(nl)
    qf_pm = zeros(nl)
    pt_pm = zeros(nl)
    qt_pm = zeros(nl)

    for ii in 1:nl
        pf_pm[ii] = OPF_soln["solution"]["branch"][string(ii)]["pf"]
        qf_pm[ii] = OPF_soln["solution"]["branch"][string(ii)]["qf"]
        pt_pm[ii] = OPF_soln["solution"]["branch"][string(ii)]["pt"]
        qt_pm[ii] = OPF_soln["solution"]["branch"][string(ii)]["qt"]
    end

    # ============== Try to replicate
    model = Model(Ipopt.Optimizer)
    nb = length(network_data["bus"])
    ng = length(network_data["gen"])
    nl = length(network_data["branch"])

    bus_list  = collect(1:nb)
    line_list = collect(1:nl)
    gen_list  = collect(1:ng)

    # are all lines on?
    if nl != sum([network_data["branch"][string(line)]["br_status"] for line in line_list])
        @warn("some lines are off")
    end

    # are all gens on?
    if ng != sum([network_data["gen"][string(gen)]["gen_status"] for gen in gen_list])
        @warn("some gens are off")
    end

    @variable(model, vm[1:nb])
    @variable(model, va[1:nb])
    @variable(model, pg[1:ng])
    @variable(model, qg[1:ng])
    @variable(model, pto[1:nl])
    @variable(model, pfr[1:nl])
    @variable(model, qto[1:nl])
    @variable(model, qfr[1:nl])

    # set starts
    if perturb == false
        for ii in 1:nb
            set_start_value(vm[ii], vm_pm[ii])
            set_start_value(va[ii], va_pm[ii])
        end

        for ii in 1:nl
            set_start_value(pto[ii], pf_pm[ii])
            set_start_value(pfr[ii], qf_pm[ii])
            set_start_value(qto[ii], pt_pm[ii])
            set_start_value(qfr[ii], qt_pm[ii])
        end
    else
        # perturb!!
        for ii in 1:nb
            set_start_value(vm[ii], 1.0 + 0.1*randn())
            set_start_value(va[ii], 0.0 + 0.1*randn())
        end

        for ii in 1:nl
            set_start_value(pto[ii], randn())
            set_start_value(pfr[ii], randn())
            set_start_value(qto[ii], randn())
            set_start_value(qfr[ii], randn())
        end
        for ii in 1:ng
            set_start_value(pg[ii], randn())
            set_start_value(qg[ii], randn())
        end
    end

    ref_bus = 1
    for (bus,val) in network_data["bus"]
        if val["bus_type"] == 3
            ref_bus = val["bus_i"]
        end
    end

    # generator parameters
    pg_max = [network_data["gen"][string(gen)]["pmax"] for gen in gen_list]
    pg_min = [network_data["gen"][string(gen)]["pmin"] for gen in gen_list]
    qg_max = [network_data["gen"][string(gen)]["qmax"] for gen in gen_list]
    qg_min = [network_data["gen"][string(gen)]["qmin"] for gen in gen_list]

    # map gens to buses
    Eg = zeros(nb,ng)
    clin = zeros(ng)
    c0 = zeros(ng)
    ii = 1
    for (gen,val) in network_data["gen"]
        Eg[val["gen_bus"],val["index"]] = 1
        if val["cost"] == []
            clin[val["index"]] = 0.0
            c0[val["index"]] = 0.0
        elseif length(val["cost"]) == 2
            clin[val["index"]] = val["cost"][1]
            c0[val["index"]] = val["cost"][2]
        else # => length(val["cost"]) == 3
            @warn("this one has quadratic terms!")
            clin[val["index"]] = val["cost"][2]
        end

    end
    # => push!(cl, val["cost"][1])
    # => push!(c0, val["cost"][2])
    # => push!(cg_ind, val["index"])

    # network parameters
    fr_buses = [network_data["branch"][string(line)]["f_bus"] for line in line_list]
    to_buses = [network_data["branch"][string(line)]["t_bus"] for line in line_list] 
    r        = [network_data["branch"][string(line)]["br_r"]  for line in line_list] 
    x        = [network_data["branch"][string(line)]["br_x"]  for line in line_list]
    g        = real(1 ./ (r+im*x))
    b        = imag(1 ./ (r+im*x))
    ta       = [network_data["branch"][string(line)]["shift"] for line in line_list] 
    tm       = [network_data["branch"][string(line)]["tap"]   for line in line_list] 
    g_to     = [network_data["branch"][string(line)]["g_to"]  for line in line_list] 
    g_fr     = [network_data["branch"][string(line)]["g_fr"]  for line in line_list] 
    b_to     = [network_data["branch"][string(line)]["b_to"]  for line in line_list] 
    b_fr     = [network_data["branch"][string(line)]["b_fr"]  for line in line_list] 
    amax     = [network_data["branch"][string(line)]["angmax"]  for line in line_list] 
    amin     = [network_data["branch"][string(line)]["angmin"]  for line in line_list] 

    # loads
    pd = zeros(nb)
    qd = zeros(nb)
    for (load,val) in network_data["load"]
        if val["status"] == 1
            bus = val["load_bus"]
            pd[bus] += val["pd"]
            qd[bus] += val["qd"]
        end
    end

    # shunts
    gs = zeros(nb)
    bs = zeros(nb)
    for (shunt,val) in network_data["shunt"]
        if val["status"] == 1
            bus = val["shunt_bus"]
            gs[bus] += val["gs"]
            bs[bus] += val["bs"]
        end
    end

    # build the incidence matrix
    E = zeros(nl,nb)
    for ii in 1:nl
        E[ii,fr_buses[ii]] = 1.0
        E[ii,to_buses[ii]] = -1.0
    end
    Efr = (E + abs.(E))/2
    Eto = (abs.(E) - E)/2

    # constraint 1: voltage magnitudes
    vmax = [network_data["bus"][string(bus)]["vmax"] for bus in bus_list]
    vmin = [network_data["bus"][string(bus)]["vmin"] for bus in bus_list]

    # constraint 2: flow limits
    smax = [minimum([network_data["branch"][string(line)]["rate_a"];
                    network_data["branch"][string(line)]["rate_b"];
                    network_data["branch"][string(line)]["rate_b"]]) for line in line_list]

    # flows
    vm_fr = Efr*vm
    vm_to = Eto*vm
    va_fr = Efr*va
    va_to = Eto*va

    @constraint(model, pfr .== @.  (g+g_fr)*(vm_fr/tm)^2 - g*vm_fr/tm*vm_to*cos(va_fr-va_to-ta) + -b*vm_fr/tm*vm_to*sin(va_fr-va_to-ta) )
    @constraint(model, qfr .== @. -(b+b_fr)*(vm_fr/tm)^2 + b*vm_fr/tm*vm_to*cos(va_fr-va_to-ta) + -g*vm_fr/tm*vm_to*sin(va_fr-va_to-ta) )
    @constraint(model, pto .== @.  (g+g_to)*vm_to^2      - g*vm_to*vm_fr/tm*cos(va_to-va_fr+ta) + -b*vm_to*vm_fr/tm*sin(va_to-va_fr+ta) )
    @constraint(model, qto .== @. -(b+b_to)*vm_to^2      + b*vm_to*vm_fr/tm*cos(va_to-va_fr+ta) + -g*vm_to*vm_fr/tm*sin(va_to-va_fr+ta) )

    # add constraints -- ignore angle limits
    @constraint(model, va[ref_bus] == 0.0)
    @constraint(model, vmin   .<= vm .<= vmax)
    @constraint(model, pg_min .<= pg .<= pg_max)
    @constraint(model, qg_min .<= qg .<= qg_max)

    @constraint(model, pfr.^2 + qfr.^2 .<= smax.^2 )
    @constraint(model, pto.^2 + qto.^2 .<= smax.^2 )

    @constraint(model, Eg*pg .== pd - gs.*vm.^2 + Efr'*pfr + Eto'*pto)
    @constraint(model, Eg*qg .== qd - bs.*vm.^2 + Efr'*qfr + Eto'*qto)

    @objective(model, Min, clin'*pg)
    optimize!(model)

    if abs(objective_value(model) - OPF_soln["objective"]) < 1e-3
        println("Reformulated model matches the PM \u2705")
    else
        @warn("Reformulated model objective doesn't match PM objective (missing quadratic cost terms?).")
        println(objective_value(model))
        println(OPF_soln["objective"])
    end

    xtr = tm .* cos.(ta)
    xti = tm .* sin.(ta)

    # sparsify
    Efr  = sparse(Efr)
    Eto  = sparse(Eto)

    # build the power flow matrices
    Tpfr  = sparse(diagm(@. (g+g_fr)/tm^2)*Efr)
    TpRfr = sparse(diagm(@. (-g*xtr+b*xti)/tm^2))
    TpIfr = sparse(diagm(@. (-b*xtr-g*xti)/tm^2))
    Tqfr  = sparse(diagm(@. (-(b+b_fr)/tm^2))*Efr)
    TqRfr = sparse(diagm(@. -(-b*xtr-g*xti)/tm^2))
    TqIfr = sparse(diagm(@. (-g*xtr+b*xti)/tm^2))
    Tpto  = sparse(diagm(@. (g+g_to))*Eto)
    TpRto = sparse(diagm(@. (-g*xtr-b*xti)/tm^2))
    TpIto = sparse(diagm(@. -(-b*xtr+g*xti)/tm^2))
    Tqto  = sparse(diagm(@. -(b+b_to))*Eto)
    TqRto = sparse(diagm(@. -(-b*xtr+g*xti)/tm^2))
    TqIto = sparse(diagm(@. -(-g*xtr-b*xti)/tm^2))

    # for use with w_fr and w_to
    Tp_wfr  = sparse(diagm(@. (g+g_fr)/tm^2))
    Tq_wfr  = sparse(diagm(@. (-(b+b_fr)/tm^2)))
    Tp_wto  = sparse(diagm(@. (g+g_to)))
    Tq_wto  = sparse(diagm(@. -(b+b_to)))

    # grab the SOC bounds
    wr_min, wr_max, wi_min, wi_max = soc_voltage_bound_vectors(network_data, nl, fr_buses, to_buses)

    # build diagonal shunt matrices
    Bs_neg = diagm(min.(bs,0.0))
    Bs_pos = diagm(max.(bs,0.0))
    Gs     = diagm(gs)

    # now, build it
    gm = SOCGridModel(nb,nl,ng,value.(vm),value.(va),value.(pg),value.(qg),g,g_fr,g_to,b,b_fr,b_to,tm,ta,
                      xtr,xti,pd,qd,gs,bs,vmax,vmin,wr_min,wr_max,wi_min,wi_max,pg_max,pg_min,qg_max,
                      qg_min,smax,clin,fr_buses,to_buses,Eg,Efr,Eto,Tpfr,TpRfr,TpIfr,Tqfr,TqRfr,TqIfr,
                      Tpto,TpRto,TpIto,Tqto,TqRto,TqIto,Tp_wfr,Tq_wfr,Tp_wto,Tq_wto,Bs_neg,Bs_pos,Gs) 

    # output useful stuff
    return gm
end

function canonicalize_flowcuts(zl, 
                               pd0, 
                               qd0, 
                               gm::SOCGridModel; 
                               use_float64::Bool=false,
                               include_cost::Bool=false,
                               normalize_shed::Bool=false)
    "zl, pd0, qd0 can be Float64 vectors, or they can be Affine (variable) vectors"
    nl = gm.nl
    nb = gm.nb
    ng = gm.ng

    w_idx       =                     (1:nb)
    w_fr_idx    = w_idx[end]       .+ (1:nl)
    w_to_idx    = w_fr_idx[end]    .+ (1:nl)
    wr_idx      = w_to_idx[end]    .+ (1:nl)
    wi_idx      = wr_idx[end]      .+ (1:nl)
    zd_idx      = wi_idx[end]      .+ (1:nb)
    pg_idx      = zd_idx[end]      .+ (1:ng)  
    qg_idx      = pg_idx[end]      .+ (1:ng)  
    pgs_idx     = qg_idx[end]      .+ (1:nb)
    qbs_pos_idx = pgs_idx[end]     .+ (1:nb)
    qbs_neg_idx = qbs_pos_idx[end] .+ (1:nb)
    p_fr_idx    = qbs_neg_idx[end] .+ (1:nl)
    p_to_idx    = p_fr_idx[end]    .+ (1:nl)   
    q_fr_idx    = p_to_idx[end]    .+ (1:nl)   
    q_to_idx    = q_fr_idx[end]    .+ (1:nl)
    t_idx       = q_to_idx[end]     + 1

    # injection constraints
    p_fr_flowidx = 1:nl
    p_to_flowidx = (nl+1):2*nl
    q_fr_flowidx = (2*nl+1):3*nl
    q_to_flowidx = (3*nl+1):4*nl
    pjidx        = 4*nl    .+ (1:nb)
    qjidx        = 4*nl+nb .+ (1:nb)

    nvar  = t_idx[end]
    neq   = 2*nb + 4*nl
    nineq = 2*nl + 2*nl + 2*nb + 2*nb + 8*nl + 4*ng + 6*nb
    if use_float64 == true
        A = spzeros(neq, nvar)
    else
        A = SparseMatrixCSC{AffExpr, Int64}(undef, neq, nvar)
    end
    # => A = SparseMatrixCSC{AffExpr, Int64}(undef, neq, nvar)
    # => A = Matrix{AffExpr}(undef, neq, nvar)
    # => A .= 0.0
    # => A = zeros(neq, nvar)
    # if zl is embedded as a nonlinear constraint:
        # => A = Matrix{NonlinearExpr}(undef, neq, nvar)
            # =>    A = SparseMatrixCSC{NonlinearExpr, Int64}(undef, neq, nvar)
        # => A .= 0.0
    b = zeros(neq) # this stays 0

    # flow constraints
    A[p_fr_flowidx, p_fr_idx] = sparse(I,nl,nl)
    A[p_to_flowidx, p_to_idx] = sparse(I,nl,nl)
    A[q_fr_flowidx, q_fr_idx] = sparse(I,nl,nl)
    A[q_to_flowidx, q_to_idx] = sparse(I,nl,nl)
    zm                        = spzeros(nl,nl)
    Mflow = [-[gm.Tp_wfr   zm         gm.TpRfr gm.TpIfr];
             -[zm          gm.Tp_wto  gm.TpRto gm.TpIto];
             -[gm.Tq_wfr   zm         gm.TqRfr gm.TqIfr];
             -[zm          gm.Tq_wto  gm.TqRto gm.TqIto]]
    A[1:4*nl,w_fr_idx[1]:wi_idx[end]] = Mflow

    # injection constraints
    A[pjidx,pg_idx]   = -gm.Eg
    A[pjidx,zd_idx]   = spdiagm(pd0)
    A[pjidx,p_fr_idx] = gm.Efr'
    A[pjidx,p_to_idx] = gm.Eto'
    A[pjidx,pgs_idx]  = sparse(I,nb,nb)

    A[qjidx,qg_idx]   = -gm.Eg
    A[qjidx,zd_idx]   = spdiagm(qd0)
    A[qjidx,q_fr_idx] = gm.Efr'
    A[qjidx,q_to_idx] = gm.Eto'
    A[qjidx,qbs_pos_idx]  = -sparse(I,nb,nb)
    A[qjidx,qbs_neg_idx]  = -sparse(I,nb,nb)

    # inequality constraints!
    C  = spzeros(nineq, nvar)
    d  = Vector{AffExpr}(undef, nineq)
    d .= 0.0

    #C1
    idx_nd = 0
    C[idx_nd .+ (1:nl), w_fr_idx] = +sparse(I,nl,nl)
    C[idx_nd .+ (1:nl), w_idx]    = -gm.Efr
    d[idx_nd .+ (1:nl)] .= 0.0
    idx_nd += nl

    #C2
    C[idx_nd .+ (1:nl), w_to_idx] = +sparse(I,nl,nl)
    C[idx_nd .+ (1:nl), w_idx]    = -gm.Eto
    d[idx_nd .+ (1:nl)] .= 0.0
    idx_nd += nl

    #C3
    C[idx_nd .+ (1:nl), w_idx]    = gm.Efr
    C[idx_nd .+ (1:nl), w_fr_idx] = -sparse(I,nl,nl)
    d[idx_nd .+ (1:nl)] .= -gm.vmax[gm.fr_buses].^2 .* (1 .- zl)
    idx_nd += nl

    #C4
    C[idx_nd .+ (1:nl), w_idx]    = gm.Eto
    C[idx_nd .+ (1:nl), w_to_idx] = -sparse(I,nl,nl)
    d[idx_nd .+ (1:nl)] .= -gm.vmax[gm.to_buses].^2 .* (1 .- zl)
    idx_nd += nl

    #C5
    C[idx_nd .+ (1:nb), zd_idx] = -sparse(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    #C6
    C[idx_nd .+ (1:nb), zd_idx]= +sparse(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= -1.0
    idx_nd += nb

    #C7
    C[idx_nd .+ (1:nb), w_idx] = -sparse(I,nb,nb) 
    d[idx_nd .+ (1:nb)] .= gm.vmin.^2
    idx_nd += nb

    #C8
    C[idx_nd .+ (1:nb), w_idx] = +sparse(I,nb,nb) 
    d[idx_nd .+ (1:nb)] .= -gm.vmax.^2
    idx_nd += nb

    #C9
    C[idx_nd .+ (1:nl), w_fr_idx] = -sparse(I,nl,nl)
    d[idx_nd .+ (1:nl)] .= zl.*gm.vmin[gm.fr_buses].^2
    idx_nd += nl

    #C10
    C[idx_nd .+ (1:nl), w_fr_idx] = +sparse(I,nl,nl)
    d[idx_nd .+ (1:nl)] .= -zl.*gm.vmax[gm.fr_buses].^2
    idx_nd += nl

    #C11
    C[idx_nd .+ (1:nl), w_to_idx] = -sparse(I,nl,nl)
    d[idx_nd .+ (1:nl)] .= zl.*gm.vmin[gm.fr_buses].^2
    idx_nd += nl

    #C12
    C[idx_nd .+ (1:nl), w_to_idx] = +sparse(I,nl,nl)
    d[idx_nd .+ (1:nl)] .= -zl.*gm.vmax[gm.fr_buses].^2
    idx_nd += nl

    #C13
    C[idx_nd .+ (1:nl), wr_idx] = -sparse(I,nl,nl) 
    d[idx_nd .+ (1:nl)] .= zl.*gm.wr_min
    idx_nd += nl

    #C14
    C[idx_nd .+ (1:nl), wr_idx] = sparse(I,nl,nl) 
    d[idx_nd .+ (1:nl)] .= -zl.*gm.wr_max
    idx_nd += nl

    #C15
    C[idx_nd .+ (1:nl), wi_idx] = -sparse(I,nl,nl) 
    d[idx_nd .+ (1:nl)] .= zl.*gm.wi_min
    idx_nd += nl

    #C16
    C[idx_nd .+ (1:nl), wi_idx] = sparse(I,nl,nl) 
    d[idx_nd .+ (1:nl)] .= -zl.*gm.wi_max
    idx_nd += nl

    #C etc...
    C[idx_nd .+ (1:ng), pg_idx] = -sparse(I,ng,ng) 
    d[idx_nd .+ (1:ng)] .= 0.0
    idx_nd += ng

    C[idx_nd .+ (1:ng), pg_idx] = sparse(I,ng,ng) 
    d[idx_nd .+ (1:ng)] .= -gm.pg_max
    idx_nd += ng

    C[idx_nd .+ (1:ng), qg_idx] = -sparse(I,ng,ng) 
    d[idx_nd .+ (1:ng)] .= gm.qg_min
    idx_nd += ng

    C[idx_nd .+ (1:ng), qg_idx] = sparse(I,ng,ng) 
    d[idx_nd .+ (1:ng)] .= -gm.qg_max
    idx_nd += ng

    C[idx_nd .+ (1:nb), pgs_idx] = -sparse(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    # double
    C[idx_nd .+ (1:nb), pgs_idx] = sparse(I,nb,nb)
    C[idx_nd .+ (1:nb), w_idx]   = -gm.Gs
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    C[idx_nd .+ (1:nb), qbs_pos_idx] = -sparse(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    # double
    C[idx_nd .+ (1:nb), qbs_pos_idx] = sparse(I,nb,nb)
    C[idx_nd .+ (1:nb), w_idx]       = -gm.Bs_pos
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    # double
    C[idx_nd .+ (1:nb), qbs_neg_idx] = -sparse(I,nb,nb)
    C[idx_nd .+ (1:nb), w_idx]       = gm.Bs_neg
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    C[idx_nd .+ (1:nb), qbs_neg_idx] = sparse(I,nb,nb)
    d[idx_nd .+ (1:nb)] .= 0.0
    idx_nd += nb

    # linear terms in the objective
    if use_float64 == true
        H  = zeros(nvar)
    else
        H  = Vector{AffExpr}(undef, nvar)
        H .= 0.0
    end
    h = sum(pd0)
    H[zd_idx] = - pd0

    # normalize?
    if (normalize_shed == true) && (typeof(pd0) == Vector{Float64})
        println("Normalizing load shed objective by total load.")
        H = H./sum(pd0)
        h = h/sum(pd0)
    elseif (normalize_shed == true) && (typeof(pd0) == Vector{VariableRef})
        println("Use epigraph trick, in this case, to enforce normalization.")
    end

    if include_cost == true
        # this is mainly for testing and troubleshooting
        H[pg_idx] = gm.clin
    end

    # apply all RSOCs
    nrsoc = 3*nl

    m1 = Vector{Any}(undef,nrsoc)#[Vector{Any}(undef,10)for ii in 1:nrsoc]
    m2 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]
    m3 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]

    b1 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]
    b2 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]
    b3 = Vector{Any}(undef,nrsoc)#[Any for ii in 1:nrsoc]

    # flow limit f -> t
    for ii in 1:nl
        m1[ii] = zeros(1,nvar)
        b1[ii] = gm.smax[ii]^2 # zl[ii]*

        m2[ii] = zeros(1,nvar)
        b2[ii] = 0.5

        m3[ii] = zeros(2,nvar)
        m3[ii][1,p_fr_idx[ii]] = 1
        m3[ii][2,q_fr_idx[ii]] = 1
        b3[ii] = zeros(2)
    end

    # flow limit t -> f
    for ii in 1:nl
        m1[ii+nl] = zeros(1,nvar)
        b1[ii+nl] = gm.smax[ii]^2 # zl[ii]*

        m2[ii+nl] = zeros(1,nvar)
        b2[ii+nl] = 0.5

        m3[ii+nl] = zeros(2,nvar)
        m3[ii+nl][1,p_to_idx[ii]] = 1
        m3[ii+nl][2,q_to_idx[ii]] = 1
        b3[ii+nl] = zeros(2)
    end

    # RSOC on voltage
    for ii in 1:nl
        m1[ii+2*nl] = zeros(1,nvar)
        m1[ii+2*nl][w_fr_idx[ii]] = 1
        b1[ii+2*nl] = 0

        m2[ii+2*nl] = zeros(1,nvar)
        m2[ii+2*nl][w_to_idx[ii]] = 0.5
        b2[ii+2*nl] = 0

        m3[ii+2*nl] = zeros(2,nvar)
        m3[ii+2*nl][1,wr_idx[ii]] = 1
        m3[ii+2*nl][2,wi_idx[ii]] = 1
        b3[ii+2*nl] = zeros(2)
    end

    # throw into a common dict
    lp = Dict(:A => A,
              :b => b,
              :C => C,
              :d => d,
              :H => H,
              :h => h)
    soc = Dict(:m1 => m1,
               :m2 => m2,
               :m3 => m3,
               :b1 => b1,
               :b2 => b2,
               :b3 => b3)

    return lp, soc
end

function get_maxls_model(
    gm::SOCGridModel,
    bounds::Dict{Symbol, Float64},
    nn_model::String;
    tol::Float64=1e-5,
    tmax::Float64=100.0,
    #hot_start::Bool=true,
    include_cost::Bool=false,
    #flowcuts::Bool=true,
    include_QGB_shedding::Bool=false,
)
    #if hot_start == true
    #    # first, get the nominal line status
    #    zl0, logit_zl0 = line_status(gm, bounds, nn_model; high_load=true)
    #    gm_shed        = deepcopy(gm)
    #    gm_shed.pd    .= bounds[:load_scale_ub]*copy(gm.pd)
    #    gm_shed.qd    .= bounds[:load_scale_ub]*copy(gm.qd)
    #    dual_soln      = min_loadshed_soc_dual(gm_shed, zl0; flowcuts=flowcuts)
    #end

    model = Model()
    #model = Model(Ipopt.Optimizer)
    #set_optimizer_attribute(model, "max_wall_time",      tmax)
    #set_optimizer_attribute(model, "tol",                 tol) # overall convergence tolerance
    #set_optimizer_attribute(model, "acceptable_tol",      tol) # "Acceptable" convergence tolerance (relative).
    #set_optimizer_attribute(model, "max_iter",          10000)
    #set_optimizer_attribute(model, "mu_init",            1e-8)
    #set_attribute(model, "hsllib", HSL_jll.libhsl_path)
    #set_attribute(model, "linear_solver", "ma57")

    @variable(model, pd0_var[1:gm.nb])
    @variable(model, qd0_var[1:gm.nb])
    @variable(model, risk[1:gm.nl])
    @variable(model, alpha)

    # this only works if pd0 and qd0 are positive
    pd0 = copy(gm.pd)
    qd0 = copy(gm.qd)

    p_lb, p_ub = scale_load(pd0, bounds[:load_scale_lb], bounds[:load_scale_ub])
    q_lb, q_ub = scale_load(qd0, bounds[:load_scale_lb], bounds[:load_scale_ub])

    @constraint(model, p_lb              .<= pd0_var .<= p_ub)
    @constraint(model, q_lb              .<= qd0_var .<= q_ub)
    @constraint(model, bounds[:risk_lb]  .<=  risk   .<= bounds[:risk_ub])
    @constraint(model, bounds[:alpha_lb]  <=  alpha   <= bounds[:alpha_ub])

    # call the NN
    x = [risk; qd0_var; pd0_var; alpha]
    model[:x] = x

    # now, we need to normalize the nn input
    normalization_data = nn_model[1:findlast(==('_'), nn_model)]*"normalization_values.h5"
    fid   = h5open(normalization_data, "r")
    mean = read(fid, "mean")
    std  = read(fid, "std")
    close(fid)

    xn   = (x .- mean)./(std)
    predictor = MathOptAI.PytorchModel(nn_model)
    # config = Dict(:ReLU => MOAI.ReLUQuadratic(relaxation_parameter = 1e-6))
    logit_zl, formulation = MathOptAI.add_predictor(
        model,
        predictor,
        xn;
        #hessian=true,
        #vector_nonlinear_oracle = true,
    )

    @variable(model, zl[1:gm.nl])
    @constraint(model, zl .== sig.(logit_zl) )

    # now, canonicalize
    #if flowcuts == true
    #    lp, soc = canonicalize_flowcuts(zl, pd0_var, qd0_var, gm) 
    #else
    #    lp, soc = canonicalize(zl, pd0_var, qd0_var, gm) 
    #end
    lp, soc = canonicalize_flowcuts(zl, pd0_var, qd0_var, gm) 
    neq   = size(lp[:A],1)
    nineq = size(lp[:C],1)
    nvar  = size(lp[:A],2)
    nrsoc = length(soc[:m1])

    @variable(model, lambda[1:neq])
    @variable(model, mu[1:nineq], lower_bound = 0.0)
    @variable(model, s1[1:nrsoc], lower_bound = 0.0)
    @variable(model, s2[1:nrsoc], lower_bound = 0.0)
    s = Dict(ii => @variable(model, [1:size(soc[:m3][ii],1)]) for ii in 1:nrsoc)
    
    #s2 = Vector{NonlinearExpr}(undef, nvar)
    #s2 .= 0.0
    #for ii in 1:nrsoc
    #    s2[ii] = dot(s[ii],s[ii])/(2*s1[ii] + 0.0001)
    #end

    # epigraph trick!
    # =? Rather than H/p, use constrain H = p*G, and use G
    # =? Rather than h/p, use constrain h = p*g, and use g, but g = 1, since p=sum(p0), and h=sum(p0)
    #nl = gm.nl
    #nb = gm.nb
    #ng = gm.ng
    #w_idx       =                     (1:nb)
    #wr_idx      = w_idx[end]       .+ (1:nl)
    #wi_idx      = wr_idx[end]      .+ (1:nl)
    #zd_idx      = wi_idx[end]      .+ (1:nb)
    #G  = Vector{AffExpr}(undef, nvar)
    #G .= 0.0
    #@variable(model, Gvar[1:length(zd_idx)])
    #G[zd_idx] = Gvar

    # ======= WITH normalization of the load
        # => @variable(model, G[1:length(lp[:H])])
        # => @constraint(model, G*sum(pd0_var) .== lp[:H])
        # => g = 1.0
        # => 
        # => @constraint(model, [ii in 1:nrsoc],  dot(s[ii],s[ii]) <= 2*s1[ii]*s2[ii])
        # => @constraint(model, G + lp[:A]'*lambda + lp[:C]'*mu - sum(s1[ii]*soc[:m1][ii]' + s2[ii]*soc[:m2][ii]' + soc[:m3][ii]'*s[ii] for ii in 1:nrsoc) .== 0.0)
        # => obj = g + lambda'*lp[:b] + mu'*lp[:d] - sum(s1[ii]*soc[:b1][ii]  + s2[ii]*soc[:b2][ii]  + s[ii]'*soc[:b3][ii] for ii in 1:nrsoc)

    # ======= WITHOUT normalization of the load
    @constraint(model, [ii in 1:nrsoc],  dot(s[ii],s[ii]) <= 2*s1[ii]*s2[ii])
    @constraint(model, lp[:H] + lp[:A]'*lambda + lp[:C]'*mu - sum(s1[ii]*soc[:m1][ii]' + s2[ii]*soc[:m2][ii]' + soc[:m3][ii]'*s[ii] for ii in 1:nrsoc) .== 0.0)
    
    # regularize?
     # => s_vec = []
     # => s0_vec = []
     # => for ii in 1:nrsoc
     # =>     s_vec  = vcat(s_vec, s[ii])
     # =>     s0_vec = vcat(s0_vec, value.(dual_soln[:s][ii]))
     # => end
     # => lp0, _ = canonicalize(zl0, gm.pd, gm.qd, gm) 
     # => xx  = [pd0_var; qd0_var; risk; alpha; zl; logit_zl; lambda; mu; s1; s2; s_vec; G]
     # => xx0 = [gm.pd; gm.qd; 0.5*(bounds[:risk_lb] + bounds[:risk_ub])*ones(gm.nl);
     # =>       0.5*(bounds[:alpha_lb] + bounds[:alpha_ub]); zl0; logit_zl0; value.(dual_soln[:lambda]);
     # =>       value.(dual_soln[:mu]); value.(dual_soln[:s1]); value.(dual_soln[:s2]);
     # =>       s0_vec; value.(lp0[:H])]
     # => #regularization = -0.1*dot(xx-xx0,xx-xx0)
    obj = lp[:h] + lambda'*lp[:b] + mu'*lp[:d] - sum(s1[ii]*soc[:b1][ii]  + s2[ii]*soc[:b2][ii]  + s[ii]'*soc[:b3][ii] for ii in 1:nrsoc)
    model[:obj] = obj
    @objective(model, Max, obj)

    #if hot_start == true
    #    set_start_value.(pd0_var, gm_shed.pd)
    #    set_start_value.(qd0_var, gm_shed.qd)
    #    set_start_value.(risk, bounds[:risk_ub]*ones(gm.nl))
    #    set_start_value(alpha, bounds[:alpha_ub])
    #    set_start_value.(zl, zl0)
    #    set_start_value.(logit_zl, logit_zl0)
    #    set_start_value.(lambda, value.(dual_soln[:lambda]))
    #    set_start_value.(mu, value.(dual_soln[:mu]))
    #    set_start_value.(s1, value.(dual_soln[:s1]))
    #    set_start_value.(s2, value.(dual_soln[:s2]))

    #    # loop and set the soc variables
    #    for ii in 1:nrsoc
    #        set_start_value.(s[ii], value.(dual_soln[:s][ii]))
    #    end

    #    # to get H, we need to re-run the canonicalization with constant inputs
    #    # => lp0, _ = canonicalize(zl0, gm.pd, gm.qd, gm; normalize_shed=false) 
    #    # => set_start_value.(G, value.(lp0[:H])) # no need to normalize!
    #end
    return (;
        model,
        formulation,
    )
end

function get_maxls_model(
    nnfile::String;
    sample_index = 1,
)
    data = PGLib.pglib("case118")
    gm = parse_PM_to_SOCGridModel(data)
    # Hardcoded bounds from the maxmin_via_sampling function
    bounds = Dict(
        :load_scale_lb => 0.75,
        :load_scale_ub => 1.25,
        :risk_lb => 0.4,
        :risk_ub => 0.5,
        :alpha_lb => 0.4,
        :alpha_ub => 0.5,
    )
    model, formulation = get_maxls_model(gm, bounds, nnfile)
    return (; model, formulation)
end

if abspath(PROGRAM_FILE) == @__FILE__
    include("../config.jl")
    nndir = get_nn_dir()
    nnfile = joinpath(nndir, "lsv", "118_bus", "118_bus_128node.pt")
    model, formulation = get_maxls_model(nnfile)
    ipopt = JuMP.optimizer_with_attributes(
        Ipopt.Optimizer,
        "linear_solver" => "ma57",
        "tol" => 1e-5,
        "print_user_options" => "yes",
    )
    JuMP.set_optimizer(model, ipopt)
    JuMP.optimize!(model)
end
