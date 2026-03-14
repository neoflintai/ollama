package duckdb

import (
	"math"
	"sort"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn/rope"
)

type Tensor struct {
	b     *Backend
	name  string
	shape []int
	dtype ml.DType
	data  []float32
}

func newTensor(b *Backend, shape []int, data []float32) *Tensor {
	return &Tensor{b: b, shape: shape, dtype: ml.DTypeF32, data: data}
}

func (t *Tensor) Dim(n int) int {
	if n < 0 || n >= len(t.shape) {
		return 0
	}
	return t.shape[n]
}

func (t *Tensor) Stride(n int) int {
	stride := 1
	for i := 0; i < n && i < len(t.shape); i++ {
		stride *= t.shape[i]
	}
	return stride
}

func (t *Tensor) Shape() []int {
	s := make([]int, len(t.shape))
	copy(s, t.shape)
	return s
}

func (t *Tensor) DType() ml.DType { return t.dtype }

func (t *Tensor) Bytes() []byte {
	if t.data == nil {
		return nil
	}
	return float32ToBytes(t.data)
}

func (t *Tensor) Floats() []float32 {
	if t.data == nil {
		return nil
	}
	out := make([]float32, len(t.data))
	copy(out, t.data)
	return out
}

func (t *Tensor) FromBytes(b []byte)      { t.data = bytesToFloat32(b) }
func (t *Tensor) FromFloats(f []float32)   { t.data = make([]float32, len(f)); copy(t.data, f) }
func (t *Tensor) FromInts(vals []int32) {
	t.data = make([]float32, len(vals))
	for i, v := range vals {
		t.data[i] = float32(v)
	}
}

func (t *Tensor) Cast(ctx ml.Context, dtype ml.DType) ml.Tensor {
	out := &Tensor{b: t.b, shape: t.Shape(), dtype: dtype}
	out.data = make([]float32, len(t.data))
	copy(out.data, t.data)
	return out
}

// --- Arithmetic ---

func (t *Tensor) Add(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return elementWise(t, t2.(*Tensor), func(a, b float32) float32 { return a + b })
}

func (t *Tensor) Sub(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return elementWise(t, t2.(*Tensor), func(a, b float32) float32 { return a - b })
}

func (t *Tensor) Mul(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return elementWise(t, t2.(*Tensor), func(a, b float32) float32 { return a * b })
}

func (t *Tensor) Div(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return elementWise(t, t2.(*Tensor), func(a, b float32) float32 {
		if b == 0 {
			return 0
		}
		return a / b
	})
}

func (t *Tensor) Scale(ctx ml.Context, s float64) ml.Tensor {
	sf := float32(s)
	out := make([]float32, len(t.data))
	for i, v := range t.data {
		out[i] = v * sf
	}
	return newTensor(t.b, t.Shape(), out)
}

// --- Matrix Operations ---
// Route through DuckDB for large matrices, pure Go for small ones.

func (t *Tensor) Mulmat(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	if t.b != nil && t.b.db != nil {
		return t.b.matmulDuckDB(t, t2.(*Tensor))
	}
	return matmulGo(t, t2.(*Tensor))
}

func (t *Tensor) MulmatFullPrec(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return t.Mulmat(ctx, t2)
}

func (t *Tensor) MulmatID(ctx ml.Context, t2, ids ml.Tensor) ml.Tensor {
	return t.Mulmat(ctx, t2)
}

func (t *Tensor) AddID(ctx ml.Context, t2, ids ml.Tensor) ml.Tensor {
	return t.Add(ctx, t2)
}

// --- Activations ---

func (t *Tensor) Softmax(ctx ml.Context) ml.Tensor  { return softmax(t) }

func (t *Tensor) GELU(ctx ml.Context, up ...ml.Tensor) ml.Tensor {
	out := unaryOp(t, func(x float32) float32 {
		return 0.5 * x * (1.0 + float32(math.Tanh(float64(x)*0.7978845608*(1.0+0.044715*float64(x)*float64(x)))))
	})
	if len(up) > 0 && up[0] != nil {
		return elementWise(out, up[0].(*Tensor), func(a, b float32) float32 { return a * b })
	}
	return out
}

func (t *Tensor) GELU_ERF(ctx ml.Context) ml.Tensor {
	return unaryOp(t, func(x float32) float32 {
		return 0.5 * x * (1.0 + float32(math.Erf(float64(x)/math.Sqrt2)))
	})
}

func (t *Tensor) QuickGELU(ctx ml.Context, up ...ml.Tensor) ml.Tensor {
	out := unaryOp(t, func(x float32) float32 { return x * sigmoid32(x*1.702) })
	if len(up) > 0 && up[0] != nil {
		return elementWise(out, up[0].(*Tensor), func(a, b float32) float32 { return a * b })
	}
	return out
}

func (t *Tensor) SILU(ctx ml.Context, up ...ml.Tensor) ml.Tensor {
	out := unaryOp(t, func(x float32) float32 { return x * sigmoid32(x) })
	if len(up) > 0 && up[0] != nil {
		return elementWise(out, up[0].(*Tensor), func(a, b float32) float32 { return a * b })
	}
	return out
}

func (t *Tensor) RELU(ctx ml.Context, up ...ml.Tensor) ml.Tensor {
	out := unaryOp(t, func(x float32) float32 {
		if x > 0 {
			return x
		}
		return 0
	})
	if len(up) > 0 && up[0] != nil {
		return elementWise(out, up[0].(*Tensor), func(a, b float32) float32 { return a * b })
	}
	return out
}

func (t *Tensor) Sigmoid(ctx ml.Context) ml.Tensor    { return unaryOp(t, sigmoid32) }
func (t *Tensor) SigmoidOut(ctx ml.Context) ml.Tensor  { return t.Sigmoid(ctx) }

func (t *Tensor) SILUAlphaLimit(ctx ml.Context, up ml.Tensor, alpha, limit float32) ml.Tensor {
	clamped := unaryOp(t, func(x float32) float32 {
		if x < -limit {
			return -limit
		}
		if x > limit {
			return limit
		}
		return x
	})
	silu := unaryOp(clamped, func(x float32) float32 { return x * sigmoid32(alpha*x) })
	return elementWise(silu, up.(*Tensor), func(a, b float32) float32 { return a * b })
}

func (t *Tensor) Tanh(ctx ml.Context) ml.Tensor {
	return unaryOp(t, func(x float32) float32 { return float32(math.Tanh(float64(x))) })
}

func (t *Tensor) Softplus(ctx ml.Context) ml.Tensor {
	return unaryOp(t, func(x float32) float32 { return float32(math.Log(1 + math.Exp(float64(x)))) })
}

// --- Normalization ---

func (t *Tensor) L2Norm(ctx ml.Context, eps float32) ml.Tensor         { return l2norm(t, eps) }
func (t *Tensor) LayerNorm(ctx ml.Context, weight, bias ml.Tensor, eps float32) ml.Tensor {
	return layerNorm(t, weight.(*Tensor), bias, eps)
}
func (t *Tensor) RMSNorm(ctx ml.Context, weight ml.Tensor, eps float32) ml.Tensor {
	return rmsNorm(t, weight.(*Tensor), eps)
}
func (t *Tensor) SumRows(ctx ml.Context) ml.Tensor { return sumRows(t) }

// --- Shape Operations ---

func (t *Tensor) Reshape(ctx ml.Context, shape ...int) ml.Tensor {
	out := &Tensor{b: t.b, shape: shape, dtype: t.dtype}
	out.data = make([]float32, len(t.data))
	copy(out.data, t.data)
	return out
}

func (t *Tensor) View(ctx ml.Context, offset int, shape ...int) ml.Tensor {
	total := 1
	for _, d := range shape {
		total *= d
	}
	elemOffset := offset / 4
	end := elemOffset + total
	if end > len(t.data) {
		end = len(t.data)
	}
	out := make([]float32, total)
	copy(out, t.data[elemOffset:end])
	return &Tensor{b: t.b, shape: shape, dtype: t.dtype, data: out}
}

func (t *Tensor) Permute(ctx ml.Context, dims ...int) ml.Tensor { return permute(t, dims) }

func (t *Tensor) Contiguous(ctx ml.Context, shape ...int) ml.Tensor {
	if len(shape) == 0 {
		shape = t.Shape()
	}
	out := make([]float32, len(t.data))
	copy(out, t.data)
	return &Tensor{b: t.b, shape: shape, dtype: t.dtype, data: out}
}

func (t *Tensor) Pad(ctx ml.Context, shape ...int) ml.Tensor    { return pad(t, shape) }

func (t *Tensor) Stack(ctx ml.Context, dim int, s ...ml.Tensor) ml.Tensor {
	tensors := make([]*Tensor, 1+len(s))
	tensors[0] = t
	for i, v := range s {
		tensors[i+1] = v.(*Tensor)
	}
	return stack(tensors, dim)
}

func (t *Tensor) Repeat(ctx ml.Context, dim, n int) ml.Tensor { return repeatTensor(t, dim, n) }

func (t *Tensor) Repeat4D(ctx ml.Context, dim0, dim1, dim2, dim3 int) ml.Tensor {
	result := t
	if dim0 > 1 {
		result = repeatTensor(result, 0, dim0).(*Tensor)
	}
	if dim1 > 1 {
		result = repeatTensor(result, 1, dim1).(*Tensor)
	}
	if dim2 > 1 {
		result = repeatTensor(result, 2, dim2).(*Tensor)
	}
	if dim3 > 1 {
		result = repeatTensor(result, 3, dim3).(*Tensor)
	}
	return result
}

func (t *Tensor) Concat(ctx ml.Context, t2 ml.Tensor, dim int) ml.Tensor {
	return concat(t, t2.(*Tensor), dim)
}

func (t *Tensor) Rows(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return rowsTensor(t, t2.(*Tensor))
}

func (t *Tensor) SetRows(ctx ml.Context, src ml.Tensor, idxs ml.Tensor) ml.Tensor {
	return setRows(t, src.(*Tensor), idxs.(*Tensor))
}

func (t *Tensor) SetInplace(ctx ml.Context, src ml.Tensor, nb1, nb2, nb3, offset int) ml.Tensor {
	out := make([]float32, len(t.data))
	copy(out, t.data)
	s := src.(*Tensor)
	elemOffset := offset / 4
	for i := 0; i < len(s.data) && elemOffset+i < len(out); i++ {
		out[elemOffset+i] = s.data[i]
	}
	return &Tensor{b: t.b, shape: t.Shape(), dtype: t.dtype, data: out}
}

func (t *Tensor) Copy(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	dst := t2.(*Tensor)
	dst.data = make([]float32, len(t.data))
	copy(dst.data, t.data)
	dst.shape = t.Shape()
	return dst
}

func (t *Tensor) Duplicate(ctx ml.Context) ml.Tensor {
	out := make([]float32, len(t.data))
	copy(out, t.data)
	return &Tensor{b: t.b, shape: t.Shape(), dtype: t.dtype, data: out}
}

func (t *Tensor) Slice(ctx ml.Context, dim, low, high, step int) ml.Tensor {
	return sliceTensor(t, dim, low, high, step)
}

func (t *Tensor) Chunk(ctx ml.Context, dim int, size int) []ml.Tensor {
	return chunk(t, dim, size)
}

func (t *Tensor) ChunkSections(ctx ml.Context, dim int, sections ...int) []ml.Tensor {
	return chunkSections(t, dim, sections)
}

// --- Trig / Math ---

func (t *Tensor) Sin(ctx ml.Context) ml.Tensor {
	return unaryOp(t, func(x float32) float32 { return float32(math.Sin(float64(x))) })
}
func (t *Tensor) Cos(ctx ml.Context) ml.Tensor {
	return unaryOp(t, func(x float32) float32 { return float32(math.Cos(float64(x))) })
}
func (t *Tensor) Exp(ctx ml.Context) ml.Tensor {
	return unaryOp(t, func(x float32) float32 { return float32(math.Exp(float64(x))) })
}
func (t *Tensor) Sqrt(ctx ml.Context) ml.Tensor {
	return unaryOp(t, func(x float32) float32 { return float32(math.Sqrt(float64(x))) })
}
func (t *Tensor) Sqr(ctx ml.Context) ml.Tensor {
	return unaryOp(t, func(x float32) float32 { return x * x })
}
func (t *Tensor) Neg(ctx ml.Context) ml.Tensor {
	return unaryOp(t, func(x float32) float32 { return -x })
}
func (t *Tensor) Clamp(ctx ml.Context, min, max float32) ml.Tensor {
	return unaryOp(t, func(x float32) float32 {
		if x < min {
			return min
		}
		if x > max {
			return max
		}
		return x
	})
}

// --- Stats ---

func (t *Tensor) TopK(ctx ml.Context, k int) ml.Tensor  { return topK(t, k) }
func (t *Tensor) Argsort(ctx ml.Context) ml.Tensor       { return argsort(t) }

func (t *Tensor) Mean(ctx ml.Context) ml.Tensor {
	if len(t.data) == 0 {
		return newTensor(t.b, []int{1}, []float32{0})
	}
	var sum float64
	for _, v := range t.data {
		sum += float64(v)
	}
	return newTensor(t.b, []int{1}, []float32{float32(sum / float64(len(t.data)))})
}

func (t *Tensor) Variance(ctx ml.Context) ml.Tensor {
	if len(t.data) == 0 {
		return newTensor(t.b, []int{1}, []float32{0})
	}
	var sum, sumSq float64
	n := float64(len(t.data))
	for _, v := range t.data {
		sum += float64(v)
		sumSq += float64(v) * float64(v)
	}
	mean := sum / n
	return newTensor(t.b, []int{1}, []float32{float32(sumSq/n - mean*mean)})
}

func (t *Tensor) Stddev(ctx ml.Context) ml.Tensor {
	v := t.Variance(ctx).(*Tensor)
	return newTensor(t.b, []int{1}, []float32{float32(math.Sqrt(float64(v.data[0])))})
}

// --- Advanced Ops ---

func (t *Tensor) CumSum(ctx ml.Context) ml.Tensor {
	out := make([]float32, len(t.data))
	if len(t.data) > 0 {
		out[0] = t.data[0]
		for i := 1; i < len(t.data); i++ {
			out[i] = out[i-1] + t.data[i]
		}
	}
	return newTensor(t.b, t.Shape(), out)
}

func (t *Tensor) Diag(ctx ml.Context) ml.Tensor {
	n := len(t.data)
	out := make([]float32, n*n)
	for i := 0; i < n; i++ {
		out[i*n+i] = t.data[i]
	}
	return newTensor(t.b, []int{n, n}, out)
}

func (t *Tensor) Tri(ctx ml.Context, triType int) ml.Tensor {
	if len(t.shape) < 2 {
		return t.Duplicate(ctx)
	}
	rows := t.shape[len(t.shape)-2]
	cols := t.shape[len(t.shape)-1]
	out := make([]float32, len(t.data))
	copy(out, t.data)
	batchSize := len(t.data) / (rows * cols)
	for batch := 0; batch < batchSize; batch++ {
		base := batch * rows * cols
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				keep := false
				switch triType {
				case 0:
					keep = c >= r
				case 1:
					keep = c > r
				case 2:
					keep = c <= r
				case 3:
					keep = c < r
				}
				if !keep {
					out[base+r*cols+c] = 0
				}
			}
		}
	}
	return newTensor(t.b, t.Shape(), out)
}

func (t *Tensor) Fill(ctx ml.Context, value float32) ml.Tensor {
	out := make([]float32, len(t.data))
	for i := range out {
		out[i] = value
	}
	return &Tensor{b: t.b, shape: t.Shape(), dtype: t.dtype, data: out}
}

func (t *Tensor) SolveTri(ctx ml.Context, b ml.Tensor, lower, left, unitDiag bool) ml.Tensor {
	return b.(*Tensor).Duplicate(ctx)
}

func (t *Tensor) Interpolate(ctx ml.Context, dims [4]int, samplingMode ml.SamplingMode) ml.Tensor {
	return t.Duplicate(ctx)
}

// --- Convolution & SSM (placeholders) ---

func (t *Tensor) Conv2D(ctx ml.Context, weight ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	return newTensor(t.b, t.Shape(), make([]float32, len(t.data)))
}
func (t *Tensor) Conv3D(ctx ml.Context, weight ml.Tensor, c, s0, s1, s2, p0, p1, p2, d0, d1, d2 int) ml.Tensor {
	return newTensor(t.b, t.Shape(), make([]float32, len(t.data)))
}
func (t *Tensor) AvgPool2D(ctx ml.Context, k, s int, p float32) ml.Tensor {
	return newTensor(t.b, t.Shape(), make([]float32, len(t.data)))
}
func (t *Tensor) IM2Col(ctx ml.Context, weight ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	return newTensor(t.b, t.Shape(), make([]float32, len(t.data)))
}
func (t *Tensor) SSMConv(ctx ml.Context, kernel ml.Tensor) ml.Tensor {
	return newTensor(t.b, t.Shape(), make([]float32, len(t.data)))
}
func (t *Tensor) SSMScan(ctx ml.Context, x, dt, A, B, C, ids ml.Tensor) ml.Tensor {
	return newTensor(t.b, t.Shape(), make([]float32, len(t.data)))
}

// --- Helper functions ---

func sigmoid32(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(-float64(x))))
}

func elementWise(a, b *Tensor, op func(float32, float32) float32) *Tensor {
	n := len(a.data)
	out := make([]float32, n)
	if len(b.data) == n {
		for i := range out {
			out[i] = op(a.data[i], b.data[i])
		}
	} else if len(b.data) > 0 {
		bLen := len(b.data)
		for i := range out {
			out[i] = op(a.data[i], b.data[i%bLen])
		}
	}
	return newTensor(a.b, a.Shape(), out)
}

func unaryOp(t *Tensor, op func(float32) float32) *Tensor {
	out := make([]float32, len(t.data))
	for i, v := range t.data {
		out[i] = op(v)
	}
	return newTensor(t.b, t.Shape(), out)
}

// matmulGo is the pure-Go fallback for small matrices.
func matmulGo(a, b *Tensor) *Tensor {
	if len(a.shape) < 2 || len(b.shape) < 2 {
		n := min(len(a.data), len(b.data))
		var sum float32
		for i := 0; i < n; i++ {
			sum += a.data[i] * b.data[i]
		}
		return newTensor(a.b, []int{1}, []float32{sum})
	}

	K := a.shape[0]
	M := a.shape[1]
	N := b.shape[1]

	batchA := 1
	for i := 2; i < len(a.shape); i++ {
		batchA *= a.shape[i]
	}
	batchB := 1
	for i := 2; i < len(b.shape); i++ {
		batchB *= b.shape[i]
	}
	batch := max(batchA, batchB)

	outShape := []int{M, N}
	if batch > 1 {
		outShape = append(outShape, batch)
	}

	out := make([]float32, M*N*batch)
	strideA := K * M
	strideB := K * N

	for bn := 0; bn < batch; bn++ {
		offA := (bn % batchA) * strideA
		offB := (bn % batchB) * strideB
		offC := bn * M * N

		for j := 0; j < N; j++ {
			for i := 0; i < M; i++ {
				var sum float32
				for k := 0; k < K; k++ {
					ai := offA + i*K + k
					bi := offB + j*K + k
					if ai < len(a.data) && bi < len(b.data) {
						sum += a.data[ai] * b.data[bi]
					}
				}
				ci := offC + j*M + i
				if ci < len(out) {
					out[ci] = sum
				}
			}
		}
	}

	return newTensor(a.b, outShape, out)
}

func softmax(t *Tensor) *Tensor {
	if len(t.data) == 0 {
		return newTensor(t.b, t.Shape(), nil)
	}
	dim0 := t.shape[0]
	nrows := len(t.data) / dim0
	out := make([]float32, len(t.data))
	for row := 0; row < nrows; row++ {
		start := row * dim0
		end := start + dim0
		maxVal := t.data[start]
		for i := start + 1; i < end; i++ {
			if t.data[i] > maxVal {
				maxVal = t.data[i]
			}
		}
		var sum float64
		for i := start; i < end; i++ {
			v := math.Exp(float64(t.data[i] - maxVal))
			out[i] = float32(v)
			sum += v
		}
		if sum > 0 {
			invSum := float32(1.0 / sum)
			for i := start; i < end; i++ {
				out[i] *= invSum
			}
		}
	}
	return newTensor(t.b, t.Shape(), out)
}

func rmsNorm(t, weight *Tensor, eps float32) *Tensor {
	dim0 := t.shape[0]
	nrows := len(t.data) / dim0
	out := make([]float32, len(t.data))
	for row := 0; row < nrows; row++ {
		start := row * dim0
		end := start + dim0
		var sumSq float64
		for i := start; i < end; i++ {
			sumSq += float64(t.data[i]) * float64(t.data[i])
		}
		rms := float32(math.Sqrt(sumSq/float64(dim0) + float64(eps)))
		for i := 0; i < dim0; i++ {
			w := float32(1.0)
			if i < len(weight.data) {
				w = weight.data[i]
			}
			out[start+i] = (t.data[start+i] / rms) * w
		}
	}
	return newTensor(t.b, t.Shape(), out)
}

func layerNorm(t *Tensor, weight *Tensor, bias ml.Tensor, eps float32) *Tensor {
	dim0 := t.shape[0]
	nrows := len(t.data) / dim0
	out := make([]float32, len(t.data))
	for row := 0; row < nrows; row++ {
		start := row * dim0
		end := start + dim0
		var sum float64
		for i := start; i < end; i++ {
			sum += float64(t.data[i])
		}
		mean := float32(sum / float64(dim0))
		var varSum float64
		for i := start; i < end; i++ {
			d := float64(t.data[i] - mean)
			varSum += d * d
		}
		std := float32(math.Sqrt(varSum/float64(dim0) + float64(eps)))
		for i := 0; i < dim0; i++ {
			normalized := (t.data[start+i] - mean) / std
			w := float32(1.0)
			if weight != nil && i < len(weight.data) {
				w = weight.data[i]
			}
			b := float32(0.0)
			if bias != nil {
				bt := bias.(*Tensor)
				if i < len(bt.data) {
					b = bt.data[i]
				}
			}
			out[start+i] = normalized*w + b
		}
	}
	return newTensor(t.b, t.Shape(), out)
}

func l2norm(t *Tensor, eps float32) *Tensor {
	dim0 := t.shape[0]
	nrows := len(t.data) / dim0
	out := make([]float32, len(t.data))
	for row := 0; row < nrows; row++ {
		start := row * dim0
		end := start + dim0
		var sumSq float64
		for i := start; i < end; i++ {
			sumSq += float64(t.data[i]) * float64(t.data[i])
		}
		norm := float32(math.Sqrt(sumSq + float64(eps)))
		for i := start; i < end; i++ {
			out[i] = t.data[i] / norm
		}
	}
	return newTensor(t.b, t.Shape(), out)
}

func sumRows(t *Tensor) *Tensor {
	if len(t.shape) == 0 || len(t.data) == 0 {
		return newTensor(t.b, []int{1}, []float32{0})
	}
	dim0 := t.shape[0]
	nrows := len(t.data) / dim0
	out := make([]float32, nrows)
	for row := 0; row < nrows; row++ {
		start := row * dim0
		var sum float32
		for i := 0; i < dim0; i++ {
			sum += t.data[start+i]
		}
		out[row] = sum
	}
	newShape := append([]int{1}, t.shape[1:]...)
	return newTensor(t.b, newShape, out)
}

func permute(t *Tensor, dims []int) *Tensor {
	if len(dims) == 0 || len(t.shape) <= 1 {
		return newTensor(t.b, t.Shape(), append([]float32{}, t.data...))
	}

	// Pad shape to match dims length if needed
	ndim := len(dims)
	shape := make([]int, ndim)
	copy(shape, t.shape)
	for i := len(t.shape); i < ndim; i++ {
		shape[i] = 1
	}

	newShape := make([]int, ndim)
	for i, d := range dims {
		if d < ndim {
			newShape[i] = shape[d]
		} else {
			newShape[i] = 1
		}
	}
	oldStrides := make([]int, ndim)
	oldStrides[0] = 1
	for i := 1; i < ndim; i++ {
		oldStrides[i] = oldStrides[i-1] * shape[i-1]
	}
	newStrides := make([]int, ndim)
	newStrides[0] = 1
	for i := 1; i < ndim; i++ {
		newStrides[i] = newStrides[i-1] * newShape[i-1]
	}
	total := len(t.data)
	out := make([]float32, total)
	for outIdx := 0; outIdx < total; outIdx++ {
		remaining := outIdx
		coords := make([]int, ndim)
		for d := ndim - 1; d >= 0; d-- {
			coords[d] = remaining / newStrides[d]
			remaining %= newStrides[d]
		}
		srcIdx := 0
		for d := 0; d < ndim; d++ {
			if dims[d] < ndim {
				srcIdx += coords[d] * oldStrides[dims[d]]
			}
		}
		if srcIdx < len(t.data) {
			out[outIdx] = t.data[srcIdx]
		}
	}
	return newTensor(t.b, newShape, out)
}

func pad(t *Tensor, shape []int) *Tensor {
	total := 1
	for _, d := range shape {
		total *= d
	}
	out := make([]float32, total)
	if len(t.shape) >= 1 && len(shape) >= 1 {
		srcDim0 := t.shape[0]
		dstDim0 := shape[0]
		srcRows := len(t.data) / max(srcDim0, 1)
		for row := 0; row < srcRows; row++ {
			srcStart := row * srcDim0
			dstStart := row * dstDim0
			n := min(srcDim0, dstDim0)
			for i := 0; i < n && srcStart+i < len(t.data) && dstStart+i < len(out); i++ {
				out[dstStart+i] = t.data[srcStart+i]
			}
		}
	}
	return newTensor(t.b, shape, out)
}

func repeatTensor(t *Tensor, dim, n int) ml.Tensor {
	if dim >= len(t.shape) || n <= 1 {
		return newTensor(t.b, t.Shape(), append([]float32{}, t.data...))
	}
	newShape := t.Shape()
	newShape[dim] *= n
	total := 1
	for _, d := range newShape {
		total *= d
	}
	out := make([]float32, total)
	innerSize := 1
	for i := 0; i < dim; i++ {
		innerSize *= t.shape[i]
	}
	outerSize := len(t.data) / (innerSize * t.shape[dim])
	chunkSize := innerSize * t.shape[dim]
	for outer := 0; outer < outerSize; outer++ {
		for rep := 0; rep < n; rep++ {
			srcStart := outer * chunkSize
			dstStart := outer*chunkSize*n + rep*chunkSize
			copy(out[dstStart:dstStart+chunkSize], t.data[srcStart:srcStart+chunkSize])
		}
	}
	return newTensor(t.b, newShape, out)
}

func concat(a, b *Tensor, dim int) *Tensor {
	if dim == 0 || len(a.shape) <= 1 {
		out := make([]float32, len(a.data)+len(b.data))
		copy(out, a.data)
		copy(out[len(a.data):], b.data)
		newShape := a.Shape()
		if len(newShape) > 0 {
			newShape[0] = a.shape[0] + b.shape[0]
		}
		return newTensor(a.b, newShape, out)
	}
	out := make([]float32, len(a.data)+len(b.data))
	copy(out, a.data)
	copy(out[len(a.data):], b.data)
	newShape := a.Shape()
	if dim < len(newShape) {
		newShape[dim] = a.shape[dim] + b.shape[dim]
	}
	return newTensor(a.b, newShape, out)
}

func rowsTensor(t, indices *Tensor) *Tensor {
	if len(t.shape) == 0 {
		return newTensor(t.b, []int{0}, nil)
	}
	dim0 := t.shape[0]
	nIndices := len(indices.data)
	out := make([]float32, nIndices*dim0)
	for i, idx := range indices.data {
		row := int(idx)
		if row >= 0 && row*dim0+dim0 <= len(t.data) {
			copy(out[i*dim0:(i+1)*dim0], t.data[row*dim0:(row+1)*dim0])
		}
	}
	newShape := append([]int{dim0}, indices.shape...)
	return newTensor(t.b, newShape, out)
}

func setRows(t, src, idxs *Tensor) *Tensor {
	out := make([]float32, len(t.data))
	copy(out, t.data)
	dim0 := t.shape[0]
	for i, idx := range idxs.data {
		row := int(idx)
		if row >= 0 && row*dim0+dim0 <= len(out) && i*dim0+dim0 <= len(src.data) {
			copy(out[row*dim0:(row+1)*dim0], src.data[i*dim0:(i+1)*dim0])
		}
	}
	return newTensor(t.b, t.Shape(), out)
}

func sliceTensor(t *Tensor, dim, low, high, step int) *Tensor {
	if dim >= len(t.shape) || dim < 0 {
		return newTensor(t.b, t.Shape(), append([]float32{}, t.data...))
	}
	if step == 0 {
		step = 1
	}
	newDimSize := (high - low + step - 1) / step
	if newDimSize <= 0 {
		return newTensor(t.b, []int{0}, nil)
	}
	newShape := t.Shape()
	newShape[dim] = newDimSize
	if dim == 0 {
		outerSize := len(t.data) / t.shape[0]
		out := make([]float32, newDimSize*outerSize)
		for outer := 0; outer < outerSize; outer++ {
			srcBase := outer * t.shape[0]
			dstBase := outer * newDimSize
			idx := 0
			for i := low; i < high; i += step {
				if srcBase+i < len(t.data) {
					out[dstBase+idx] = t.data[srcBase+i]
				}
				idx++
			}
		}
		return newTensor(t.b, newShape, out)
	}
	total := 1
	for _, d := range newShape {
		total *= d
	}
	out := make([]float32, total)
	copy(out, t.data[:min(total, len(t.data))])
	return newTensor(t.b, newShape, out)
}

func chunk(t *Tensor, dim int, size int) []ml.Tensor {
	if dim >= len(t.shape) || size <= 0 {
		return []ml.Tensor{t}
	}
	dimSize := t.shape[dim]
	nChunks := (dimSize + size - 1) / size
	results := make([]ml.Tensor, nChunks)
	for i := 0; i < nChunks; i++ {
		low := i * size
		high := min((i+1)*size, dimSize)
		results[i] = sliceTensor(t, dim, low, high, 1)
	}
	return results
}

func chunkSections(t *Tensor, dim int, sections []int) []ml.Tensor {
	if dim >= len(t.shape) {
		return []ml.Tensor{t}
	}
	results := make([]ml.Tensor, len(sections))
	offset := 0
	for i, size := range sections {
		high := offset + size
		if high > t.shape[dim] {
			high = t.shape[dim]
		}
		results[i] = sliceTensor(t, dim, offset, high, 1)
		offset = high
	}
	return results
}

func stack(tensors []*Tensor, dim int) *Tensor {
	if len(tensors) == 0 {
		return newTensor(nil, []int{0}, nil)
	}
	totalData := 0
	for _, t := range tensors {
		totalData += len(t.data)
	}
	out := make([]float32, totalData)
	offset := 0
	for _, t := range tensors {
		copy(out[offset:], t.data)
		offset += len(t.data)
	}
	newShape := tensors[0].Shape()
	if dim >= len(newShape) {
		newShape = append(newShape, len(tensors))
	} else {
		result := make([]int, len(newShape)+1)
		copy(result[:dim], newShape[:dim])
		result[dim] = len(tensors)
		copy(result[dim+1:], newShape[dim:])
		newShape = result
	}
	return newTensor(tensors[0].b, newShape, out)
}

func topK(t *Tensor, k int) *Tensor {
	if len(t.data) == 0 || k <= 0 {
		return newTensor(t.b, []int{0}, nil)
	}
	dim0 := t.shape[0]
	nrows := len(t.data) / dim0
	k = min(k, dim0)
	out := make([]float32, nrows*k)
	for row := 0; row < nrows; row++ {
		start := row * dim0
		type iv struct {
			idx int
			val float32
		}
		pairs := make([]iv, dim0)
		for i := 0; i < dim0; i++ {
			pairs[i] = iv{i, t.data[start+i]}
		}
		sort.Slice(pairs, func(i, j int) bool { return pairs[i].val > pairs[j].val })
		for i := 0; i < k; i++ {
			out[row*k+i] = pairs[i].val
		}
	}
	newShape := t.Shape()
	newShape[0] = k
	return newTensor(t.b, newShape, out)
}

func argsort(t *Tensor) *Tensor {
	if len(t.data) == 0 {
		return newTensor(t.b, t.Shape(), nil)
	}
	dim0 := t.shape[0]
	nrows := len(t.data) / dim0
	out := make([]float32, len(t.data))
	for row := 0; row < nrows; row++ {
		start := row * dim0
		indices := make([]int, dim0)
		for i := range indices {
			indices[i] = i
		}
		sort.Slice(indices, func(i, j int) bool {
			return t.data[start+indices[i]] < t.data[start+indices[j]]
		})
		for i, idx := range indices {
			out[start+i] = float32(idx)
		}
	}
	return newTensor(t.b, t.Shape(), out)
}

// RoPE applies rotary positional embedding.
func (t *Tensor) RoPE(ctx ml.Context, positions ml.Tensor, ropeDim int, ropeBase, ropeScale float32, options ...func(*rope.Options)) ml.Tensor {
	if len(t.shape) < 3 {
		return t.Duplicate(ctx)
	}

	dim0 := t.shape[0]
	nHeads := t.shape[1]
	seqLen := t.shape[2]

	if ropeDim <= 0 || ropeDim > dim0 {
		ropeDim = dim0
	}

	pos := positions.(*Tensor)
	out := make([]float32, len(t.data))
	copy(out, t.data)

	for s := 0; s < seqLen; s++ {
		position := float64(0)
		if s < len(pos.data) {
			position = float64(pos.data[s])
		}

		for h := 0; h < nHeads; h++ {
			baseIdx := s*nHeads*dim0 + h*dim0
			for d := 0; d < ropeDim/2; d++ {
				freq := position / math.Pow(float64(ropeBase), float64(2*d)/float64(ropeDim))
				cosVal := float32(math.Cos(freq))
				sinVal := float32(math.Sin(freq))

				i0 := baseIdx + d
				i1 := baseIdx + d + ropeDim/2

				if i0 < len(out) && i1 < len(out) {
					v0 := t.data[i0]
					v1 := t.data[i1]
					out[i0] = v0*cosVal - v1*sinVal
					out[i1] = v0*sinVal + v1*cosVal
				}
			}
		}
	}

	return newTensor(t.b, t.Shape(), out)
}

var _ ml.Tensor = (*Tensor)(nil)
