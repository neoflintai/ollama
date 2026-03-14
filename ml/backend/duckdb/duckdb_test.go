package duckdb

import (
	"database/sql"
	"math"
	"testing"

	_ "github.com/marcboeker/go-duckdb"
	"github.com/ollama/ollama/ml"
)

func setupContext(tb testing.TB) (*Backend, ml.Context) {
	tb.Helper()
	// Open a real in-memory DuckDB for matmul tests
	db, err := sql.Open("duckdb", "")
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() { db.Close() })

	b := &Backend{db: db, tensors: make(map[string]*Tensor)}
	ctx := &Context{b: b, layer: -1}
	return b, ctx
}

func TestTensorCreation(t *testing.T) {
	_, ctx := setupContext(t)

	t.Run("Empty", func(t *testing.T) {
		tensor := ctx.Empty(ml.DTypeF32, 3, 4)
		if got := tensor.Dim(0); got != 3 {
			t.Errorf("Dim(0) = %d, want 3", got)
		}
		if got := tensor.Dim(1); got != 4 {
			t.Errorf("Dim(1) = %d, want 4", got)
		}
	})

	t.Run("FromFloats", func(t *testing.T) {
		tensor := ctx.FromFloats([]float32{1, 2, 3, 4}, 4)
		floats := tensor.Floats()
		if len(floats) != 4 || floats[0] != 1 || floats[3] != 4 {
			t.Errorf("FromFloats got %v, want [1 2 3 4]", floats)
		}
	})

	t.Run("Arange", func(t *testing.T) {
		tensor := ctx.Arange(0, 5, 1, ml.DTypeF32)
		floats := tensor.Floats()
		if len(floats) != 5 {
			t.Fatalf("Arange len = %d, want 5", len(floats))
		}
		for i := 0; i < 5; i++ {
			if floats[i] != float32(i) {
				t.Errorf("Arange[%d] = %f, want %f", i, floats[i], float32(i))
			}
		}
	})
}

func TestArithmetic(t *testing.T) {
	_, ctx := setupContext(t)

	a := ctx.FromFloats([]float32{1, 2, 3, 4}, 4)
	b := ctx.FromFloats([]float32{5, 6, 7, 8}, 4)

	t.Run("Add", func(t *testing.T) {
		c := a.Add(ctx, b)
		assertFloats(t, c.Floats(), []float32{6, 8, 10, 12}, 0)
	})

	t.Run("Sub", func(t *testing.T) {
		c := a.Sub(ctx, b)
		assertFloats(t, c.Floats(), []float32{-4, -4, -4, -4}, 0)
	})

	t.Run("Mul", func(t *testing.T) {
		c := a.Mul(ctx, b)
		assertFloats(t, c.Floats(), []float32{5, 12, 21, 32}, 0)
	})

	t.Run("Scale", func(t *testing.T) {
		c := a.Scale(ctx, 2.0)
		assertFloats(t, c.Floats(), []float32{2, 4, 6, 8}, 0)
	})
}

func TestMatmulGo(t *testing.T) {
	_, ctx := setupContext(t)

	t.Run("2x2_small", func(t *testing.T) {
		// Small matrix — uses pure Go path
		// C[i,j] = sum_k A[i*K+k] * B[j*K+k]
		a := ctx.FromFloats([]float32{1, 3, 2, 4}, 2, 2)
		b := ctx.FromFloats([]float32{5, 7, 6, 8}, 2, 2)
		c := a.Mulmat(ctx, b)
		got := c.Floats()
		want := []float32{26, 38, 30, 44}
		assertFloats(t, got, want, 1e-5)
	})

	t.Run("DotProduct", func(t *testing.T) {
		a := ctx.FromFloats([]float32{1, 2, 3}, 3)
		b := ctx.FromFloats([]float32{4, 5, 6}, 3)
		c := a.Mulmat(ctx, b)
		got := c.Floats()
		if len(got) != 1 || math.Abs(float64(got[0]-32)) > 1e-5 {
			t.Errorf("dot product = %v, want [32]", got)
		}
	})
}

func TestMatmulDuckDB(t *testing.T) {
	b, _ := setupContext(t)

	// Create matrices large enough to trigger DuckDB path (M*N*K >= 100000)
	// Use K=100, M=50, N=20 -> 100*50*20 = 100000
	K, M, N := 100, 50, 20

	aData := make([]float32, K*M)
	bData := make([]float32, K*N)
	for i := range aData {
		aData[i] = float32(i%7) * 0.1
	}
	for i := range bData {
		bData[i] = float32(i%11) * 0.1
	}

	a := &Tensor{b: b, shape: []int{K, M}, dtype: ml.DTypeF32, data: aData}
	bT := &Tensor{b: b, shape: []int{K, N}, dtype: ml.DTypeF32, data: bData}

	// Get DuckDB result
	duckResult := b.matmulDuckDB(a, bT)

	// Get pure Go result for comparison
	goResult := matmulGo(a, bT)

	duckFloats := duckResult.Floats()
	goFloats := goResult.Floats()

	if len(duckFloats) != len(goFloats) {
		t.Fatalf("DuckDB result len = %d, Go result len = %d", len(duckFloats), len(goFloats))
	}

	maxDiff := float64(0)
	for i := range duckFloats {
		diff := math.Abs(float64(duckFloats[i] - goFloats[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	// Allow small floating point differences between SQL and Go
	if maxDiff > 0.01 {
		t.Errorf("DuckDB vs Go matmul max diff = %f, want < 0.01", maxDiff)
	}
	t.Logf("DuckDB matmul: %dx%d x %dx%d -> max diff from Go: %e", K, M, K, N, maxDiff)
}

func TestSoftmax(t *testing.T) {
	_, ctx := setupContext(t)

	tensor := ctx.FromFloats([]float32{1, 2, 3}, 3)
	result := tensor.Softmax(ctx)
	got := result.Floats()

	var sum float64
	for _, v := range got {
		sum += float64(v)
	}
	if math.Abs(sum-1.0) > 1e-5 {
		t.Errorf("softmax sum = %f, want 1.0", sum)
	}
	if got[0] >= got[1] || got[1] >= got[2] {
		t.Errorf("softmax not monotonically increasing: %v", got)
	}
}

func TestRMSNorm(t *testing.T) {
	_, ctx := setupContext(t)

	tensor := ctx.FromFloats([]float32{1, 2, 3, 4}, 4)
	weight := ctx.FromFloats([]float32{1, 1, 1, 1}, 4)
	result := tensor.RMSNorm(ctx, weight, 1e-5)
	got := result.Floats()

	rms := float32(math.Sqrt(float64(1+4+9+16)/4.0 + 1e-5))
	for i, v := range got {
		expected := float32(i+1) / rms
		if math.Abs(float64(v-expected)) > 1e-4 {
			t.Errorf("RMSNorm[%d] = %f, want %f", i, v, expected)
		}
	}
}

func TestReshape(t *testing.T) {
	_, ctx := setupContext(t)

	tensor := ctx.FromFloats([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	reshaped := tensor.Reshape(ctx, 3, 2)

	if reshaped.Dim(0) != 3 || reshaped.Dim(1) != 2 {
		t.Errorf("shape = [%d, %d], want [3, 2]", reshaped.Dim(0), reshaped.Dim(1))
	}
	assertFloats(t, reshaped.Floats(), []float32{1, 2, 3, 4, 5, 6}, 0)
}

func TestActivations(t *testing.T) {
	_, ctx := setupContext(t)

	tensor := ctx.FromFloats([]float32{-1, 0, 1, 2}, 4)

	t.Run("SILU", func(t *testing.T) {
		result := tensor.SILU(ctx)
		got := result.Floats()
		if math.Abs(float64(got[1])) > 1e-5 {
			t.Errorf("SILU(0) = %f, want 0", got[1])
		}
		if got[2] <= 0 || got[3] <= 0 {
			t.Errorf("SILU should be positive for positive input, got %v", got)
		}
	})

	t.Run("RELU", func(t *testing.T) {
		result := tensor.RELU(ctx)
		assertFloats(t, result.Floats(), []float32{0, 0, 1, 2}, 1e-5)
	})

	t.Run("Sigmoid_zero", func(t *testing.T) {
		result := tensor.Sigmoid(ctx)
		got := result.Floats()
		if math.Abs(float64(got[1]-0.5)) > 1e-5 {
			t.Errorf("Sigmoid(0) = %f, want 0.5", got[1])
		}
	})
}

func TestSlice(t *testing.T) {
	_, ctx := setupContext(t)

	tensor := ctx.FromFloats([]float32{10, 20, 30, 40, 50, 60}, 6)
	sliced := tensor.Slice(ctx, 0, 1, 4, 1)
	assertFloats(t, sliced.Floats(), []float32{20, 30, 40}, 0)
}

func TestConcat(t *testing.T) {
	_, ctx := setupContext(t)

	a := ctx.FromFloats([]float32{1, 2, 3}, 3)
	b := ctx.FromFloats([]float32{4, 5, 6}, 3)
	c := a.Concat(ctx, b, 0)
	assertFloats(t, c.Floats(), []float32{1, 2, 3, 4, 5, 6}, 0)
}

func TestFloat16Conversion(t *testing.T) {
	tests := []struct {
		bits uint16
		want float32
	}{
		{0x3C00, 1.0},
		{0x4000, 2.0},
		{0x0000, 0.0},
		{0xBC00, -1.0},
		{0x3800, 0.5},
	}
	for _, tt := range tests {
		got := float16ToFloat32(tt.bits)
		if math.Abs(float64(got-tt.want)) > 1e-4 {
			t.Errorf("float16ToFloat32(0x%04X) = %f, want %f", tt.bits, got, tt.want)
		}
	}
}

func TestDequantization(t *testing.T) {
	t.Run("F32", func(t *testing.T) {
		data := float32ToBytes([]float32{1, 2, 3})
		result := dequantize(data, 0, 3)
		assertFloats(t, result, []float32{1, 2, 3}, 0)
	})

	t.Run("Q8_0", func(t *testing.T) {
		block := make([]byte, 34)
		block[0] = 0x00
		block[1] = 0x3C
		for i := 0; i < 32; i++ {
			block[2+i] = byte(int8(i))
		}
		result := dequantize(block, 8, 32)
		if len(result) != 32 {
			t.Fatalf("len = %d, want 32", len(result))
		}
		for i := 0; i < 32; i++ {
			if math.Abs(float64(result[i]-float32(i))) > 0.01 {
				t.Errorf("Q8_0[%d] = %f, want %f", i, result[i], float32(i))
			}
		}
	})
}

func assertFloats(t *testing.T, got, want []float32, eps float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > eps {
			t.Errorf("[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}
