"""
    test_filters.jl

Tests for Phase 3 filters: Kalman (linear), Extended Kalman, Ensemble Kalman, Particle Filter.
"""

using Test
using Prezo
using LinearAlgebra
using Random
using Statistics

Random.seed!(42)

@testset "Filters - Phase 3" begin
    @testset "Linear Kalman Filter" begin
        # 1D random walk + noisy observation: x_t = x_{t-1} + w, z_t = x_t + v
        F = [1.0;;]
        H = [1.0;;]
        Q = [0.1^2;;]
        R = [0.5^2;;]
        kf = KalmanFilter(F, H, Q, R)
        @test kf isa StateSpaceFilter

        x0 = [0.0]
        P0 = [1.0;;]
        state0 = KalmanFilterState(x0, P0, 0)
        pred = predict(kf, state0)
        @test pred.x ≈ [0.0]
        @test pred.P[1, 1] ≈ 1.0 + 0.01
        @test pred.t == 1

        state1 = update(kf, pred, [1.0])
        @test length(state1.x) == 1
        @test state1.P[1, 1] > 0

        # Filter on synthetic data (true state = 0, observations = noise)
        obs = [0.1, -0.2, 0.0, 0.3, -0.1]
        obs_vec = [[y] for y in obs]
        filtered = filter_data(kf, obs_vec, state0)
        @test length(filtered) == 5
        @test all(s -> length(s.x) == 1 && size(s.P) == (1, 1), filtered)
    end

    @testset "Kalman smooth" begin
        F = [1.0;;]
        H = [1.0;;]
        Q = [0.01;;]
        R = [0.1;;]
        kf = KalmanFilter(F, H, Q, R)
        state0 = KalmanFilterState([0.0], [1.0;;], 0)
        obs = [[0.5], [0.6], [0.4]]
        filtered = filter_data(kf, obs, state0)
        smoothed = smooth(kf, filtered)
        @test length(smoothed) == 3
        @test smoothed[3].x ≈ filtered[3].x
        # Smoothed variance at t=1 should be <= filtered variance at t=1
        @test smoothed[1].P[1, 1] <= filtered[1].P[1, 1] + 1e-10
    end

    @testset "Extended Kalman Filter" begin
        # Nonlinear: x_t = 0.5*x_{t-1} + 0.1*x_{t-1}^2, z_t = x_t^2 (scalar state/obs)
        f(x) = [0.5 * x[1] + 0.1 * x[1]^2]
        h(x) = [x[1]^2]
        F_jac(x) = [0.5 + 0.2 * x[1];;]
        H_jac(x) = [2 * x[1];;]
        Q = [0.01;;]
        R = [0.1;;]
        ekf = ExtendedKalmanFilter(f, h, F_jac, H_jac, Q, R)
        @test ekf isa StateSpaceFilter

        state0 = KalmanFilterState([1.0], [0.1;;], 0)
        pred = predict(ekf, state0)
        @test length(pred.x) == 1
        obs = [0.8]
        state1 = update(ekf, pred, obs)
        @test length(state1.x) == 1

        obs_seq = [[0.5], [0.4], [0.6]]
        filtered = filter_data(ekf, obs_seq, state0)
        @test length(filtered) == 3
    end

    @testset "Ensemble Kalman Filter" begin
        f(x) = [0.9 * x[1]]
        h(x) = [x[1]]
        n_ens = 50
        Q = [0.01;;]
        R = [0.1;;]
        enkf = EnsembleKalmanFilter(f, h, n_ens, Q, R)
        @test enkf isa StateSpaceFilter

        initial_ens = randn(n_ens, 1) .* 0.5
        state0 = EnsembleKalmanState(initial_ens, 0)
        pred = predict(enkf, state0; rng=MersenneTwister(1))
        @test size(pred.ensemble) == (n_ens, 1)
        state1 = update(enkf, pred, [0.5]; rng=MersenneTwister(2))
        @test size(state1.ensemble) == (n_ens, 1)

        obs_seq = [[0.3], [0.2], [0.25]]
        filtered = filter_data(enkf, obs_seq, initial_ens; rng=MersenneTwister(123))
        @test length(filtered) == 3
        @test all(s -> size(s.ensemble) == (n_ens, 1), filtered)
    end

    @testset "Particle Filter" begin
        f(x, rng) = [0.9 * x[1] + 0.1 * randn(rng)]
        h(x) = [x[1]]
        R = [0.1;;]
        n_particles = 200
        pf = ParticleFilter(f, h, R, n_particles, 0.5)
        @test pf isa StateSpaceFilter

        ess = effective_sample_size(fill(1.0 / 100, 100))
        @test ess ≈ 100.0

        # Resampling returns N indices
        w = normalize!(rand(50), 1)
        idx_sys = systematic_resample(w; rng=MersenneTwister(1))
        @test length(idx_sys) == 50
        @test all(i -> 1 <= i <= 50, idx_sys)
        idx_mn = multinomial_resample(w; rng=MersenneTwister(1))
        @test length(idx_mn) == 50
        idx_str = stratified_resample(w; rng=MersenneTwister(1))
        @test length(idx_str) == 50

        initial_particles = randn(n_particles, 1) .* 0.5
        obs_seq = [[0.2], [0.15], [0.18]]
        filtered = filter_data(pf, obs_seq, initial_particles; rng=MersenneTwister(42))
        @test length(filtered) == 3
        @test all(s -> size(s.particles) == (n_particles, 1) && length(s.weights) == n_particles, filtered)
    end
end
