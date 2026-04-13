# - code to estimate the simple model using expectation-maximization

# calculate the posterior for one observation
# Q is an r_I × r_θ × r_θ array
function expectation_step_n(Q,y,x_θ,x_I,π0,Π, P_I, P_θ)
    for kI in axes(Q,3), kθ₀ in axes(Q,2), kθ₁ in axes(Q,1)
        # calculate the probability of the latent variables
        q = π0[kθ₀, kI] * Π[kθ₁, kθ₀, kI]
        # calculate probability of investment measures:
        q *= P_I[x_I, kI]
        # calculate probability of skill measures:
        q *= P_θ[x_θ, kθ₀] * P_θ[y, kθ₁]
        Q[kθ₁, kθ₀, kI] = q
    end
    ll = sum(Q)
    Q ./= ll #<- normalize
    return ll
end
function expectation_step(Q,p,data)
    (;X_I,X_θ,Y) = data
    (;π0,Π,P_I,P_θ) = p
    ll = 0.
    for n in axes(Q,4)
        @views ll +=  expectation_step_n(Q[:,:,:,n],Y[n],X_θ[n],X_I[n],π0,Π,P_I,P_θ)
    end
    return ll
end

function update_P_I(Q,X_I)
    P_I = zeros(maximum(X_I),size(Q,3)) 
    for kI in axes(Q,3)
        denom = 0.
        for n in axes(Q,4)
            @views q = sum(Q[:,:,kI,n])
            P_I[X_I[n],kI] += q
            denom += q
        end
        P_I[:,kI] ./= denom
    end
    return P_I
end

function update_P_θ(Q,Y,X_θ)
    P_θ = zeros(maximum(X_θ),size(Q,2)) 
    for kθ in axes(Q,2)
        denom = 0.
        for n in axes(Q,4)
            @views q = sum(Q[:,kθ,:,n])
            P_θ[X_θ[n],kθ] += q
            denom += q
            @views q = sum(Q[kθ,:,:,n])
            P_θ[Y[n],kθ] += q
            denom += q
        end
        P_θ[:,kθ] ./= denom
    end
    return P_θ
end

function update_Π(Q)
    r1,r2,r3,_ = size(Q)
    Π_numer = sum(Q,dims=4)
    Π_denom = sum(Q,dims=[1,4])
    reshape(Π_numer ./ Π_denom,r1,r2,r3)
end


function update_π0(Q)
    r_θ,r_I,_,N = size(Q)
    reshape(sum(Q,dims=[1,4])/N,r_θ,r_I)
end

function maximization_step(Q,data)
    (;Y,X_I,X_θ) = data
    P_I = update_P_I(Q,data.X_I)
    P_θ = update_P_θ(Q,data.Y,data.X_θ)
    Π = update_Π(Q)
    π0 = update_π0(Q)
    return (;P_I,P_θ,Π,π0)
end

function expectation_maximization(data,p_init)
    err_tol = 1e-5
    max_iter = 1_000
    iter = 0
    err = Inf
    Q = zeros(size(p_init.Π)...,length(data.Y))
    while err>err_tol && iter<max_iter
        ll = expectation_step(Q,p_init,data)
        p_new = maximization_step(Q, data)
        err = maximum(abs.(p_new.Π .- p_init.Π))
        iter += 1
        p_init = p_new
        #@show err iter ll
    end
    return p_init
end

function get_hessian_Π(Q,Π)
    r_θ,_,r_I = size(Π)
    N = size(Q,4)
    Qsum = reshape(sum(Q,dims=4) / N,r_θ,r_θ,r_I)

    p_idx = LinearIndices((r_θ-1,r_θ,r_I))
    #p_inv = CartesianIndices()
    np = prod(size(p_idx))
    H = zeros(np,np)
    for kI in 1:r_I, kθ₀ in 1:r_θ, kθ₁ in 1:r_θ-1
        i_p = p_idx[kθ₁, kθ₀, kI]
        H[i_p,i_p] = -Qsum[kθ₁, kθ₀, kI] / Π[kθ₁,kθ₀,kI]^2
        for kθ₂ in 1:r_θ-1
            i_p2 = p_idx[kθ₂, kθ₀, kI]
            H[i_p, i_p2] += Qsum[r_θ, kθ₀, kI] / Π[r_θ, kθ₀, kI]^2
        end
    end
    return H
end

function get_score(Q,Π)
    r_θ,_,r_I = size(Π)
    N = size(Q,4)
    Qsum = reshape(sum(Q,dims=4) / N,r_θ,r_θ,r_I)

    p_idx = LinearIndices((r_θ-1,r_θ,r_I))
    #p_inv = CartesianIndices()
    np = prod(size(p_idx))
    dL = zeros(np,N)
    for n in 1:N
        for kI in 1:r_I, kθ₀ in 1:r_θ, kθ₁ in 1:r_θ-1
            i_p = p_idx[kθ₁, kθ₀, kI]
            dL[i_p,n] = Q[kθ₁, kθ₀, kI, n] / Π[kθ₁, kθ₀, kI] - Q[r_θ, kθ₀, kI, n] / Π[r_θ, kθ₀, kI]
        end
    end
    return dL
end