# GPU acceleration (CUDA) — run only when CUDA.functional()

using Test
using Prezo
using CUDA

@testset "GPU (CUDA) when available" begin
    if !CUDA.functional()
        # No GPU: skip testset entirely (no broken count)
        return
    end

    data = MarketData(100.0, 0.05, 0.2, 0.0)
    call = EuropeanCall(100.0, 1.0)
    engine_cpu = MonteCarlo(50, 10_000)
    engine_gpu = MonteCarloGPU(50, 10_000)

    @testset "asset_paths_col_gpu" begin
        paths = asset_paths_col_gpu(engine_cpu, 100.0, 0.05, 0.2, 1.0)
        @test size(paths) == (51, 10_000)
        @test paths isa CuArray
        @test all(Array(paths[1, :]) .== 100.0)
        @test all(isfinite.(Array(paths)))
    end

    @testset "MonteCarloGPU price" begin
        price_gpu = price(call, engine_gpu, data)
        @test price_gpu > 0.0
        @test isfinite(price_gpu)
        price_bs = price(call, BlackScholes(), data)
        @test 0.5 * price_bs < price_gpu < 2.0 * price_bs
    end
end
