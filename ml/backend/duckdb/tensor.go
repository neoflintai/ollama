package duckdb

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"sync/atomic"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn/rope"
)

// Global counter for unique tensor table names
var tensorID atomic.Int64

// Tensor represents data living IN DuckDB as a table.
// Data is only pulled to Go when explicitly needed (Floats/Bytes).
// All math ops create new DuckDB tables via SQL — no Go math loops.
type Tensor struct {
	b     *Backend
	name  string // debug name
	shape []int
	dtype ml.DType
	table string // DuckDB table name holding this tensor's data: (idx INT, val FLOAT)
	data  []float32 // lazy cache, only populated on Floats()/Bytes()
}

func nextTable() string {
	return fmt.Sprintf("_t%d", tensorID.Add(1))
}

// newTensorFromData creates a DuckDB table from Go data
func newTensorFromData(b *Backend, shape []int, data []float32) *Tensor {
	t := &Tensor{b: b, shape: shape, dtype: ml.DTypeF32, table: nextTable(), data: data}
	if b != nil && b.db != nil && len(data) > 0 {
		// Bulk insert via VALUES — build in chunks for large tensors
		b.db.Exec(fmt.Sprintf("DROP TABLE IF EXISTS %s", t.ensureTable()))
		b.db.Exec(fmt.Sprintf("CREATE TEMPORARY TABLE %s (idx INT, val FLOAT)", t.ensureTable()))

		const batch = 10000
		for start := 0; start < len(data); start += batch {
			end := start + batch
			if end > len(data) {
				end = len(data)
			}
			var vals strings.Builder
			for i := start; i < end; i++ {
				if i > start {
					vals.WriteString(",")
				}
				fmt.Fprintf(&vals, "(%d,%e)", i, data[i])
			}
			b.db.Exec(fmt.Sprintf("INSERT INTO %s VALUES %s", t.table, vals.String()))
		}
	}
	return t
}

// newTensorFromSQL creates a tensor backed by a SQL expression
// The query must produce (idx INT, val FLOAT) rows
func newTensorFromSQL(b *Backend, shape []int, query string) *Tensor {
	tbl := nextTable()
	_, err := b.db.Exec(fmt.Sprintf("CREATE TEMPORARY TABLE %s AS %s", tbl, query))
	if err != nil {
		// Fallback: return empty
		return &Tensor{b: b, shape: shape, dtype: ml.DTypeF32, table: tbl, data: make([]float32, product(shape))}
	}
	return &Tensor{b: b, shape: shape, dtype: ml.DTypeF32, table: tbl}
}

func product(shape []int) int {
	p := 1
	for _, d := range shape {
		p *= d
	}
	return p
}

// ensureTable creates a DuckDB temp table for this tensor if it doesn't exist yet.
// Called lazily before any SQL operation.
func (t *Tensor) ensureTable() string {
	if t.table != "" {
		return t.table
	}
	if t.b == nil || t.b.db == nil || len(t.data) == 0 {
		t.table = nextTable()
		t.b.db.Exec(fmt.Sprintf("CREATE TEMPORARY TABLE %s (idx INT, val FLOAT)", t.ensureTable()))
		return t.table
	}
	t.table = nextTable()
	t.b.db.Exec(fmt.Sprintf("CREATE TEMPORARY TABLE %s (idx INT, val FLOAT)", t.ensureTable()))

	// Batch insert using VALUES — faster than row-by-row
	const batch = 10000
	for start := 0; start < len(t.data); start += batch {
		end := start + batch
		if end > len(t.data) {
			end = len(t.data)
		}
		var vals strings.Builder
		for i := start; i < end; i++ {
			if i > start {
				vals.WriteString(",")
			}
			fmt.Fprintf(&vals, "(%d,%e)", i, t.data[i])
		}
		t.b.db.Exec(fmt.Sprintf("INSERT INTO %s VALUES %s", t.table, vals.String()))
	}
	return t.table
}

// materialize pulls data from DuckDB into Go slice (lazy, cached)
func (t *Tensor) materialize() []float32 {
	if t.data != nil {
		return t.data
	}
	if t.b == nil || t.b.db == nil || t.table == "" {
		return nil
	}
	n := product(t.shape)
	t.data = make([]float32, n)
	rows, err := t.b.db.Query(fmt.Sprintf("SELECT idx, val FROM %s ORDER BY idx", t.ensureTable()))
	if err != nil {
		return t.data
	}
	defer rows.Close()
	for rows.Next() {
		var idx int
		var val float64
		rows.Scan(&idx, &val)
		if idx >= 0 && idx < n {
			t.data[idx] = float32(val)
		}
	}
	return t.data
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
	data := t.materialize()
	if data == nil {
		return nil
	}
	return float32ToBytes(data)
}

func (t *Tensor) Floats() []float32 {
	data := t.materialize()
	if data == nil {
		return nil
	}
	out := make([]float32, len(data))
	copy(out, data)
	return out
}

func (t *Tensor) FromBytes(b []byte) {
	t.data = bytesToFloat32(b)
	t.syncToDB()
}

func (t *Tensor) FromFloats(f []float32) {
	t.data = make([]float32, len(f))
	copy(t.data, f)
	t.syncToDB()
}

func (t *Tensor) FromInts(vals []int32) {
	t.data = make([]float32, len(vals))
	for i, v := range vals {
		t.data[i] = float32(v)
	}
	t.syncToDB()
}

func (t *Tensor) syncToDB() {
	if t.b != nil && t.b.db != nil && t.table != "" && len(t.data) > 0 {
		t.b.db.Exec(fmt.Sprintf("DROP TABLE IF EXISTS %s", t.ensureTable()))
		tmp := newTensorFromData(t.b, t.shape, t.data)
		t.table = tmp.table
	}
}

func (t *Tensor) Cast(ctx ml.Context, dtype ml.DType) ml.Tensor {
	return newTensorFromSQL(t.b, t.Shape(), fmt.Sprintf("SELECT idx, val FROM %s", t.ensureTable()))
}

// ==================== ARITHMETIC — ALL IN DUCKDB ====================

func (t *Tensor) Add(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return binaryOp(t, t2.(*Tensor), "+")
}
func (t *Tensor) Sub(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return binaryOp(t, t2.(*Tensor), "-")
}
func (t *Tensor) Mul(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return binaryOp(t, t2.(*Tensor), "*")
}
func (t *Tensor) Div(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return binaryOp(t, t2.(*Tensor), "/")
}

func binaryOp(a, b *Tensor, op string) *Tensor {
	bLen := product(b.shape)
	aLen := product(a.shape)

	var query string
	if bLen == aLen {
		// Same size: direct join on idx
		query = fmt.Sprintf(
			"SELECT a.idx, a.val %s b.val AS val FROM %s a JOIN %s b ON a.idx = b.idx",
			op, a.ensureTable(), b.ensureTable())
	} else {
		// Broadcasting
		query = fmt.Sprintf(
			"SELECT a.idx, a.val %s b.val AS val FROM %s a JOIN %s b ON b.idx = a.idx %% %d",
			op, a.ensureTable(), b.ensureTable(), bLen)
	}
	return newTensorFromSQL(a.b, a.Shape(), query)
}

func (t *Tensor) Scale(ctx ml.Context, s float64) ml.Tensor {
	query := fmt.Sprintf("SELECT idx, val * %g AS val FROM %s", s, t.ensureTable())
	return newTensorFromSQL(t.b, t.Shape(), query)
}

// ==================== MATMUL — IN DUCKDB ====================

func (t *Tensor) Mulmat(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	return matmulSQL(t, t2.(*Tensor))
}
func (t *Tensor) MulmatFullPrec(ctx ml.Context, t2 ml.Tensor) ml.Tensor { return t.Mulmat(ctx, t2) }
func (t *Tensor) MulmatID(ctx ml.Context, t2, ids ml.Tensor) ml.Tensor  { return t.Mulmat(ctx, t2) }
func (t *Tensor) AddID(ctx ml.Context, t2, ids ml.Tensor) ml.Tensor     { return t.Add(ctx, t2) }

// matmulSQL: C = A^T * B where A is [K,M], B is [K,N] -> C is [M,N]
// C[m,n] = SUM over k of A[m*K+k] * B[n*K+k]
// All done in DuckDB — no Go loops.
func matmulSQL(a, b *Tensor) *Tensor {
	if len(a.shape) < 2 || len(b.shape) < 2 {
		// Dot product
		query := fmt.Sprintf(
			"SELECT 0 AS idx, SUM(a.val * b.val) AS val FROM %s a JOIN %s b ON a.idx = b.idx",
			a.ensureTable(), b.ensureTable())
		return newTensorFromSQL(a.b, []int{1}, query)
	}

	K := a.shape[0]
	M := a.shape[1]
	N := b.shape[1]

	// A[m,k] is at idx = m*K+k  →  m = idx/K, k = idx%K
	// B[n,k] is at idx = n*K+k  →  n = idx/K, k = idx%K
	// C[m,n] = SUM(A[m,k] * B[n,k]) → output idx = n*M+m
	query := fmt.Sprintf(`
		SELECT (b_n * %d + a_m) AS idx, SUM(a.val * b.val) AS val
		FROM (SELECT idx, val, idx // %d AS a_m, idx %% %d AS a_k FROM %s) a
		JOIN (SELECT idx, val, idx // %d AS b_n, idx %% %d AS b_k FROM %s) b
		ON a.a_k = b.b_k
		GROUP BY a_m, b_n
		ORDER BY idx
	`, M, K, K, a.ensureTable(), K, K, b.ensureTable())

	return newTensorFromSQL(a.b, []int{M, N}, query)
}

// ==================== ACTIVATIONS — IN DUCKDB ====================

func (t *Tensor) Softmax(ctx ml.Context) ml.Tensor {
	dim0 := t.shape[0]
	// Softmax per row: row = idx / dim0
	// exp(val - max_per_row) / sum_per_row
	query := fmt.Sprintf(`
		WITH rows AS (
			SELECT idx, val, idx // %d AS rid, idx %% %d AS cid FROM %s
		),
		mx AS (SELECT rid, max(val) AS m FROM rows GROUP BY rid),
		exps AS (SELECT r.idx, r.rid, exp(r.val - mx.m) AS e FROM rows r JOIN mx ON r.rid = mx.rid),
		sums AS (SELECT rid, sum(e) AS s FROM exps GROUP BY rid)
		SELECT exps.idx AS idx, exps.e / sums.s AS val FROM exps JOIN sums ON exps.rid = sums.rid
	`, dim0, dim0, t.ensureTable())
	return newTensorFromSQL(t.b, t.Shape(), query)
}

func (t *Tensor) RMSNorm(ctx ml.Context, weight ml.Tensor, eps float32) ml.Tensor {
	w := weight.(*Tensor)
	dim0 := t.shape[0]
	query := fmt.Sprintf(`
		WITH rows AS (
			SELECT idx, val, idx // %d AS rid, idx %% %d AS cid FROM %s
		),
		rms AS (SELECT rid, sqrt(avg(val * val) + %g) AS r FROM rows GROUP BY rid)
		SELECT rows.idx AS idx, (rows.val / rms.r) * w.val AS val
		FROM rows
		JOIN rms ON rows.rid = rms.rid
		JOIN %s w ON w.idx = rows.cid
	`, dim0, dim0, t.ensureTable(), eps, w.ensureTable())
	return newTensorFromSQL(t.b, t.Shape(), query)
}

func (t *Tensor) SILU(ctx ml.Context, up ...ml.Tensor) ml.Tensor {
	query := fmt.Sprintf("SELECT idx, val / (1.0 + exp(-val)) AS val FROM %s", t.ensureTable())
	out := newTensorFromSQL(t.b, t.Shape(), query)
	if len(up) > 0 && up[0] != nil {
		return binaryOp(out, up[0].(*Tensor), "*")
	}
	return out
}

func (t *Tensor) GELU(ctx ml.Context, up ...ml.Tensor) ml.Tensor {
	query := fmt.Sprintf("SELECT idx, 0.5 * val * (1.0 + tanh(val * 0.7978845608 * (1.0 + 0.044715 * val * val))) AS val FROM %s", t.ensureTable())
	out := newTensorFromSQL(t.b, t.Shape(), query)
	if len(up) > 0 && up[0] != nil {
		return binaryOp(out, up[0].(*Tensor), "*")
	}
	return out
}

func (t *Tensor) GELU_ERF(ctx ml.Context) ml.Tensor {
	// DuckDB doesn't have erf(), use tanh approximation
	query := fmt.Sprintf("SELECT idx, 0.5 * val * (1.0 + tanh(0.7978845608 * (val + 0.044715 * val * val * val))) AS val FROM %s", t.ensureTable())
	return newTensorFromSQL(t.b, t.Shape(), query)
}

func (t *Tensor) QuickGELU(ctx ml.Context, up ...ml.Tensor) ml.Tensor {
	query := fmt.Sprintf("SELECT idx, val / (1.0 + exp(-1.702 * val)) AS val FROM %s", t.ensureTable())
	out := newTensorFromSQL(t.b, t.Shape(), query)
	if len(up) > 0 && up[0] != nil {
		return binaryOp(out, up[0].(*Tensor), "*")
	}
	return out
}

func (t *Tensor) RELU(ctx ml.Context, up ...ml.Tensor) ml.Tensor {
	query := fmt.Sprintf("SELECT idx, CASE WHEN val > 0 THEN val ELSE 0 END AS val FROM %s", t.ensureTable())
	out := newTensorFromSQL(t.b, t.Shape(), query)
	if len(up) > 0 && up[0] != nil {
		return binaryOp(out, up[0].(*Tensor), "*")
	}
	return out
}

func (t *Tensor) Sigmoid(ctx ml.Context) ml.Tensor {
	return newTensorFromSQL(t.b, t.Shape(),
		fmt.Sprintf("SELECT idx, 1.0 / (1.0 + exp(-val)) AS val FROM %s", t.ensureTable()))
}
func (t *Tensor) SigmoidOut(ctx ml.Context) ml.Tensor { return t.Sigmoid(ctx) }

func (t *Tensor) SILUAlphaLimit(ctx ml.Context, up ml.Tensor, alpha, limit float32) ml.Tensor {
	query := fmt.Sprintf(`
		SELECT idx,
			CASE WHEN val < %g THEN %g WHEN val > %g THEN %g ELSE val END
			/ (1.0 + exp(-%g * CASE WHEN val < %g THEN %g WHEN val > %g THEN %g ELSE val END))
			AS val FROM %s`,
		-limit, -limit, limit, limit, alpha, -limit, -limit, limit, limit, t.ensureTable())
	out := newTensorFromSQL(t.b, t.Shape(), query)
	return binaryOp(out, up.(*Tensor), "*")
}

func (t *Tensor) Tanh(ctx ml.Context) ml.Tensor {
	return newTensorFromSQL(t.b, t.Shape(),
		fmt.Sprintf("SELECT idx, tanh(val) AS val FROM %s", t.ensureTable()))
}

func (t *Tensor) Softplus(ctx ml.Context) ml.Tensor {
	return newTensorFromSQL(t.b, t.Shape(),
		fmt.Sprintf("SELECT idx, ln(1.0 + exp(val)) AS val FROM %s", t.ensureTable()))
}

// ==================== NORMALIZATION ====================

func (t *Tensor) L2Norm(ctx ml.Context, eps float32) ml.Tensor {
	dim0 := t.shape[0]
	query := fmt.Sprintf(`
		WITH rows AS (SELECT idx, val, idx // %d AS rid, idx %% %d AS cid FROM %s),
		     norms AS (SELECT rid, sqrt(sum(val*val) + %g) AS n FROM rows GROUP BY rid)
		SELECT rows.idx AS idx, rows.val / norms.n AS val
		FROM rows JOIN norms ON rows.rid = norms.rid
	`, dim0, dim0, t.ensureTable(), eps)
	return newTensorFromSQL(t.b, t.Shape(), query)
}

func (t *Tensor) LayerNorm(ctx ml.Context, weight, bias ml.Tensor, eps float32) ml.Tensor {
	w := weight.(*Tensor)
	dim0 := t.shape[0]
	biasJoin := ""
	biasExpr := "0"
	if bias != nil {
		bt := bias.(*Tensor)
		biasJoin = fmt.Sprintf("JOIN %s bias ON bias.idx = rows.cid", bt.ensureTable())
		biasExpr = "bias.val"
	}
	query := fmt.Sprintf(`
		WITH rows AS (SELECT idx, val, idx // %d AS rid, idx %% %d AS cid FROM %s),
		     stats AS (SELECT rid, avg(val) AS mu, sqrt(avg(val*val) - avg(val)*avg(val) + %g) AS sigma FROM rows GROUP BY rid)
		SELECT rows.idx AS idx, ((rows.val - stats.mu) / stats.sigma) * w.val + %s AS val
		FROM rows JOIN stats ON rows.rid = stats.rid JOIN %s w ON w.idx = rows.cid %s
	`, dim0, dim0, t.ensureTable(), eps, biasExpr, w.table, biasJoin)
	return newTensorFromSQL(t.b, t.Shape(), query)
}

func (t *Tensor) SumRows(ctx ml.Context) ml.Tensor {
	dim0 := t.shape[0]
	nrows := product(t.shape) / dim0
	query := fmt.Sprintf(`
		SELECT idx // %d AS idx, sum(val) AS val FROM %s GROUP BY idx // %d ORDER BY idx
	`, dim0, t.ensureTable(), dim0)
	newShape := append([]int{1}, t.shape[1:]...)
	result := newTensorFromSQL(t.b, newShape, query)
	_ = nrows
	return result
}

// ==================== TRIG — IN DUCKDB ====================

func (t *Tensor) Sin(ctx ml.Context) ml.Tensor {
	return newTensorFromSQL(t.b, t.Shape(), fmt.Sprintf("SELECT idx, sin(val) AS val FROM %s", t.ensureTable()))
}
func (t *Tensor) Cos(ctx ml.Context) ml.Tensor {
	return newTensorFromSQL(t.b, t.Shape(), fmt.Sprintf("SELECT idx, cos(val) AS val FROM %s", t.ensureTable()))
}
func (t *Tensor) Exp(ctx ml.Context) ml.Tensor {
	return newTensorFromSQL(t.b, t.Shape(), fmt.Sprintf("SELECT idx, exp(val) AS val FROM %s", t.ensureTable()))
}
func (t *Tensor) Sqrt(ctx ml.Context) ml.Tensor {
	return newTensorFromSQL(t.b, t.Shape(), fmt.Sprintf("SELECT idx, sqrt(val) AS val FROM %s", t.ensureTable()))
}
func (t *Tensor) Sqr(ctx ml.Context) ml.Tensor {
	return newTensorFromSQL(t.b, t.Shape(), fmt.Sprintf("SELECT idx, val*val AS val FROM %s", t.ensureTable()))
}
func (t *Tensor) Neg(ctx ml.Context) ml.Tensor {
	return newTensorFromSQL(t.b, t.Shape(), fmt.Sprintf("SELECT idx, -val AS val FROM %s", t.ensureTable()))
}
func (t *Tensor) Clamp(ctx ml.Context, mn, mx float32) ml.Tensor {
	return newTensorFromSQL(t.b, t.Shape(),
		fmt.Sprintf("SELECT idx, GREATEST(%g, LEAST(%g, val)) AS val FROM %s", mn, mx, t.ensureTable()))
}

// ==================== STATS — IN DUCKDB ====================

func (t *Tensor) Mean(ctx ml.Context) ml.Tensor {
	return newTensorFromSQL(t.b, []int{1},
		fmt.Sprintf("SELECT 0 AS idx, avg(val) AS val FROM %s", t.ensureTable()))
}

func (t *Tensor) Variance(ctx ml.Context) ml.Tensor {
	return newTensorFromSQL(t.b, []int{1},
		fmt.Sprintf("SELECT 0 AS idx, var_pop(val) AS val FROM %s", t.ensureTable()))
}

func (t *Tensor) Stddev(ctx ml.Context) ml.Tensor {
	return newTensorFromSQL(t.b, []int{1},
		fmt.Sprintf("SELECT 0 AS idx, stddev_pop(val) AS val FROM %s", t.ensureTable()))
}

func (t *Tensor) TopK(ctx ml.Context, k int) ml.Tensor  { return topK(t, k) }
func (t *Tensor) Argsort(ctx ml.Context) ml.Tensor       { return argsort(t) }

// ==================== SHAPE OPS (no math, just index remapping) ====================

func (t *Tensor) Reshape(ctx ml.Context, shape ...int) ml.Tensor {
	// Same data, different shape — just reference same table
	return &Tensor{b: t.b, shape: shape, dtype: t.dtype, table: t.ensureTable()}
}

func (t *Tensor) View(ctx ml.Context, offset int, shape ...int) ml.Tensor {
	total := product(shape)
	elemOffset := offset / 4
	query := fmt.Sprintf(
		"SELECT idx - %d AS idx, val FROM %s WHERE idx >= %d AND idx < %d",
		elemOffset, t.ensureTable(), elemOffset, elemOffset+total)
	return newTensorFromSQL(t.b, shape, query)
}

func (t *Tensor) Permute(ctx ml.Context, dims ...int) ml.Tensor {
	// Permute requires Go-side index remapping — materialize, permute, re-upload
	data := t.materialize()
	result := permuteData(t.b, t.shape, data, dims)
	return result
}

func (t *Tensor) Contiguous(ctx ml.Context, shape ...int) ml.Tensor {
	if len(shape) == 0 {
		shape = t.Shape()
	}
	return &Tensor{b: t.b, shape: shape, dtype: t.dtype, table: t.ensureTable()}
}

func (t *Tensor) Pad(ctx ml.Context, shape ...int) ml.Tensor {
	data := t.materialize()
	return newTensorFromData(t.b, shape, padData(t.shape, shape, data))
}

func (t *Tensor) Stack(ctx ml.Context, dim int, s ...ml.Tensor) ml.Tensor {
	// Union all tables with offset indices
	parts := []string{fmt.Sprintf("SELECT idx, val FROM %s", t.ensureTable())}
	offset := product(t.shape)
	for _, st := range s {
		tt := st.(*Tensor)
		parts = append(parts, fmt.Sprintf("SELECT idx + %d AS idx, val FROM %s", offset, tt.ensureTable()))
		offset += product(tt.shape)
	}
	newShape := t.Shape()
	if dim >= len(newShape) {
		newShape = append(newShape, 1+len(s))
	} else {
		result := make([]int, len(newShape)+1)
		copy(result[:dim], newShape[:dim])
		result[dim] = 1 + len(s)
		copy(result[dim+1:], newShape[dim:])
		newShape = result
	}
	return newTensorFromSQL(t.b, newShape, strings.Join(parts, " UNION ALL "))
}

func (t *Tensor) Repeat(ctx ml.Context, dim, n int) ml.Tensor {
	data := t.materialize()
	return newTensorFromData(t.b, repeatShape(t.shape, dim, n), repeatData(t.shape, dim, n, data))
}

func (t *Tensor) Repeat4D(ctx ml.Context, d0, d1, d2, d3 int) ml.Tensor {
	result := t
	if d0 > 1 { result = result.Repeat(ctx, 0, d0).(*Tensor) }
	if d1 > 1 { result = result.Repeat(ctx, 1, d1).(*Tensor) }
	if d2 > 1 { result = result.Repeat(ctx, 2, d2).(*Tensor) }
	if d3 > 1 { result = result.Repeat(ctx, 3, d3).(*Tensor) }
	return result
}

func (t *Tensor) Concat(ctx ml.Context, t2 ml.Tensor, dim int) ml.Tensor {
	b := t2.(*Tensor)
	offset := product(t.shape)
	query := fmt.Sprintf("SELECT idx, val FROM %s UNION ALL SELECT idx + %d AS idx, val FROM %s", t.ensureTable(), offset, b.table)
	newShape := t.Shape()
	if dim < len(newShape) {
		newShape[dim] = t.shape[dim] + b.shape[dim]
	}
	return newTensorFromSQL(t.b, newShape, query)
}

func (t *Tensor) Rows(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	indices := t2.(*Tensor)
	idxData := indices.materialize()
	dim0 := t.shape[0]
	nIndices := len(idxData)
	// For each index i, copy row idxData[i] from t
	// Output idx = i*dim0 + col
	parts := make([]string, 0, nIndices)
	for i, idx := range idxData {
		row := int(idx)
		parts = append(parts, fmt.Sprintf(
			"SELECT %d * %d + (idx - %d) AS idx, val FROM %s WHERE idx >= %d AND idx < %d",
			i, dim0, row*dim0, t.ensureTable(), row*dim0, (row+1)*dim0))
	}
	if len(parts) == 0 {
		return newTensorFromData(t.b, []int{0}, nil)
	}
	newShape := append([]int{dim0}, indices.shape...)
	return newTensorFromSQL(t.b, newShape, strings.Join(parts, " UNION ALL "))
}

func (t *Tensor) SetRows(ctx ml.Context, src ml.Tensor, idxs ml.Tensor) ml.Tensor {
	data := t.materialize()
	srcData := src.(*Tensor).materialize()
	idxData := idxs.(*Tensor).materialize()
	out := make([]float32, len(data))
	copy(out, data)
	dim0 := t.shape[0]
	for i, idx := range idxData {
		row := int(idx)
		if row >= 0 && row*dim0+dim0 <= len(out) && i*dim0+dim0 <= len(srcData) {
			copy(out[row*dim0:(row+1)*dim0], srcData[i*dim0:(i+1)*dim0])
		}
	}
	return newTensorFromData(t.b, t.Shape(), out)
}

func (t *Tensor) SetInplace(ctx ml.Context, src ml.Tensor, nb1, nb2, nb3, offset int) ml.Tensor {
	data := t.materialize()
	srcData := src.(*Tensor).materialize()
	out := make([]float32, len(data))
	copy(out, data)
	elemOffset := offset / 4
	for i := 0; i < len(srcData) && elemOffset+i < len(out); i++ {
		out[elemOffset+i] = srcData[i]
	}
	return newTensorFromData(t.b, t.Shape(), out)
}

func (t *Tensor) Copy(ctx ml.Context, t2 ml.Tensor) ml.Tensor {
	dst := t2.(*Tensor)
	dst.table = t.table
	dst.shape = t.Shape()
	dst.data = nil // invalidate cache
	return dst
}

func (t *Tensor) Duplicate(ctx ml.Context) ml.Tensor {
	return newTensorFromSQL(t.b, t.Shape(), fmt.Sprintf("SELECT idx, val FROM %s", t.ensureTable()))
}

func (t *Tensor) Slice(ctx ml.Context, dim, low, high, step int) ml.Tensor {
	data := t.materialize()
	return newTensorFromData(t.b, sliceShape(t.shape, dim, low, high, step), sliceData(t.shape, dim, low, high, step, data))
}

func (t *Tensor) Chunk(ctx ml.Context, dim int, size int) []ml.Tensor {
	data := t.materialize()
	return chunkTensors(t.b, t.shape, dim, size, data)
}

func (t *Tensor) ChunkSections(ctx ml.Context, dim int, sections ...int) []ml.Tensor {
	data := t.materialize()
	return chunkSectionsTensors(t.b, t.shape, dim, sections, data)
}

// ==================== ADVANCED ====================

func (t *Tensor) CumSum(ctx ml.Context) ml.Tensor {
	return newTensorFromSQL(t.b, t.Shape(),
		fmt.Sprintf("SELECT idx, SUM(val) OVER (ORDER BY idx) AS val FROM %s", t.ensureTable()))
}

func (t *Tensor) Diag(ctx ml.Context) ml.Tensor {
	data := t.materialize()
	n := len(data)
	out := make([]float32, n*n)
	for i := 0; i < n; i++ {
		out[i*n+i] = data[i]
	}
	return newTensorFromData(t.b, []int{n, n}, out)
}

func (t *Tensor) Tri(ctx ml.Context, triType int) ml.Tensor {
	data := t.materialize()
	return newTensorFromData(t.b, t.Shape(), triData(t.shape, triType, data))
}

func (t *Tensor) Fill(ctx ml.Context, value float32) ml.Tensor {
	n := product(t.shape)
	return newTensorFromSQL(t.b, t.Shape(),
		fmt.Sprintf("SELECT generate_series AS idx, %g::FLOAT AS val FROM generate_series(0, %d)", value, n-1))
}

func (t *Tensor) SolveTri(ctx ml.Context, b ml.Tensor, lower, left, unitDiag bool) ml.Tensor {
	return b.(*Tensor).Duplicate(ctx)
}
func (t *Tensor) Interpolate(ctx ml.Context, dims [4]int, samplingMode ml.SamplingMode) ml.Tensor {
	return t.Duplicate(ctx)
}
func (t *Tensor) Conv2D(ctx ml.Context, weight ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	return t.Fill(ctx, 0)
}
func (t *Tensor) Conv3D(ctx ml.Context, weight ml.Tensor, c, s0, s1, s2, p0, p1, p2, d0, d1, d2 int) ml.Tensor {
	return t.Fill(ctx, 0)
}
func (t *Tensor) AvgPool2D(ctx ml.Context, k, s int, p float32) ml.Tensor { return t.Fill(ctx, 0) }
func (t *Tensor) IM2Col(ctx ml.Context, weight ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	return t.Fill(ctx, 0)
}
func (t *Tensor) SSMConv(ctx ml.Context, kernel ml.Tensor) ml.Tensor { return t.Fill(ctx, 0) }
func (t *Tensor) SSMScan(ctx ml.Context, x, dt, A, B, C, ids ml.Tensor) ml.Tensor {
	return t.Fill(ctx, 0)
}

// ==================== RoPE ====================

func (t *Tensor) RoPE(ctx ml.Context, positions ml.Tensor, ropeDim int, ropeBase, ropeScale float32, options ...func(*rope.Options)) ml.Tensor {
	if len(t.shape) < 3 {
		return t.Duplicate(ctx)
	}
	// RoPE requires index-dependent trig — materialize, compute, re-upload
	data := t.materialize()
	pos := positions.(*Tensor).materialize()
	result := ropeData(t.shape, data, pos, ropeDim, ropeBase)
	return newTensorFromData(t.b, t.Shape(), result)
}

// ==================== HELPER FUNCTIONS ====================

func ropeData(shape []int, data, pos []float32, ropeDim int, ropeBase float32) []float32 {
	dim0 := shape[0]
	nHeads := shape[1]
	seqLen := shape[2]
	if ropeDim <= 0 || ropeDim > dim0 {
		ropeDim = dim0
	}
	out := make([]float32, len(data))
	copy(out, data)
	for s := 0; s < seqLen; s++ {
		position := float64(0)
		if s < len(pos) {
			position = float64(pos[s])
		}
		for h := 0; h < nHeads; h++ {
			baseIdx := s*nHeads*dim0 + h*dim0
			for d := 0; d < ropeDim/2; d++ {
				freq := position / math.Pow(float64(ropeBase), float64(2*d)/float64(ropeDim))
				cos := float32(math.Cos(freq))
				sin := float32(math.Sin(freq))
				i0 := baseIdx + d
				i1 := baseIdx + d + ropeDim/2
				if i0 < len(out) && i1 < len(out) {
					v0, v1 := data[i0], data[i1]
					out[i0] = v0*cos - v1*sin
					out[i1] = v0*sin + v1*cos
				}
			}
		}
	}
	return out
}

func permuteData(b *Backend, shape []int, data []float32, dims []int) *Tensor {
	if len(dims) == 0 || len(shape) <= 1 {
		return newTensorFromData(b, shape, append([]float32{}, data...))
	}
	ndim := len(dims)
	paddedShape := make([]int, ndim)
	copy(paddedShape, shape)
	for i := len(shape); i < ndim; i++ {
		paddedShape[i] = 1
	}
	newShape := make([]int, ndim)
	for i, d := range dims {
		if d < ndim {
			newShape[i] = paddedShape[d]
		} else {
			newShape[i] = 1
		}
	}
	oldStrides := make([]int, ndim)
	oldStrides[0] = 1
	for i := 1; i < ndim; i++ {
		oldStrides[i] = oldStrides[i-1] * paddedShape[i-1]
	}
	newStrides := make([]int, ndim)
	newStrides[0] = 1
	for i := 1; i < ndim; i++ {
		newStrides[i] = newStrides[i-1] * newShape[i-1]
	}
	total := len(data)
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
		if srcIdx < len(data) {
			out[outIdx] = data[srcIdx]
		}
	}
	return newTensorFromData(b, newShape, out)
}

func padData(oldShape, newShape []int, data []float32) []float32 {
	total := product(newShape)
	out := make([]float32, total)
	if len(oldShape) >= 1 && len(newShape) >= 1 {
		srcDim0 := oldShape[0]
		dstDim0 := newShape[0]
		srcRows := len(data) / max(srcDim0, 1)
		for row := 0; row < srcRows; row++ {
			n := min(srcDim0, dstDim0)
			for i := 0; i < n && row*srcDim0+i < len(data) && row*dstDim0+i < len(out); i++ {
				out[row*dstDim0+i] = data[row*srcDim0+i]
			}
		}
	}
	return out
}

func repeatShape(shape []int, dim, n int) []int {
	s := make([]int, len(shape))
	copy(s, shape)
	if dim < len(s) {
		s[dim] *= n
	}
	return s
}

func repeatData(shape []int, dim, n int, data []float32) []float32 {
	if dim >= len(shape) || n <= 1 {
		out := make([]float32, len(data))
		copy(out, data)
		return out
	}
	newShape := repeatShape(shape, dim, n)
	total := product(newShape)
	out := make([]float32, total)
	innerSize := 1
	for i := 0; i < dim; i++ {
		innerSize *= shape[i]
	}
	outerSize := len(data) / (innerSize * shape[dim])
	chunkSize := innerSize * shape[dim]
	for outer := 0; outer < outerSize; outer++ {
		for rep := 0; rep < n; rep++ {
			copy(out[outer*chunkSize*n+rep*chunkSize:], data[outer*chunkSize:outer*chunkSize+chunkSize])
		}
	}
	return out
}

func sliceShape(shape []int, dim, low, high, step int) []int {
	if dim >= len(shape) {
		return append([]int{}, shape...)
	}
	if step == 0 {
		step = 1
	}
	s := make([]int, len(shape))
	copy(s, shape)
	s[dim] = (high - low + step - 1) / step
	return s
}

func sliceData(shape []int, dim, low, high, step int, data []float32) []float32 {
	if dim >= len(shape) || dim < 0 {
		out := make([]float32, len(data))
		copy(out, data)
		return out
	}
	if step == 0 {
		step = 1
	}
	newDimSize := (high - low + step - 1) / step
	if newDimSize <= 0 {
		return nil
	}
	if dim == 0 {
		outerSize := len(data) / shape[0]
		out := make([]float32, newDimSize*outerSize)
		for outer := 0; outer < outerSize; outer++ {
			idx := 0
			for i := low; i < high; i += step {
				if outer*shape[0]+i < len(data) {
					out[outer*newDimSize+idx] = data[outer*shape[0]+i]
				}
				idx++
			}
		}
		return out
	}
	ns := sliceShape(shape, dim, low, high, step)
	total := product(ns)
	out := make([]float32, total)
	copy(out, data[:min(total, len(data))])
	return out
}

func chunkTensors(b *Backend, shape []int, dim, size int, data []float32) []ml.Tensor {
	if dim >= len(shape) || size <= 0 {
		return []ml.Tensor{newTensorFromData(b, shape, data)}
	}
	dimSize := shape[dim]
	nChunks := (dimSize + size - 1) / size
	results := make([]ml.Tensor, nChunks)
	for i := 0; i < nChunks; i++ {
		low := i * size
		high := min((i+1)*size, dimSize)
		results[i] = newTensorFromData(b, sliceShape(shape, dim, low, high, 1), sliceData(shape, dim, low, high, 1, data))
	}
	return results
}

func chunkSectionsTensors(b *Backend, shape []int, dim int, sections []int, data []float32) []ml.Tensor {
	if dim >= len(shape) {
		return []ml.Tensor{newTensorFromData(b, shape, data)}
	}
	results := make([]ml.Tensor, len(sections))
	offset := 0
	for i, size := range sections {
		high := offset + size
		if high > shape[dim] {
			high = shape[dim]
		}
		results[i] = newTensorFromData(b, sliceShape(shape, dim, offset, high, 1), sliceData(shape, dim, offset, high, 1, data))
		offset = high
	}
	return results
}

func triData(shape []int, triType int, data []float32) []float32 {
	if len(shape) < 2 {
		out := make([]float32, len(data))
		copy(out, data)
		return out
	}
	rows := shape[len(shape)-2]
	cols := shape[len(shape)-1]
	out := make([]float32, len(data))
	copy(out, data)
	batchSize := len(data) / (rows * cols)
	for batch := 0; batch < batchSize; batch++ {
		base := batch * rows * cols
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				keep := false
				switch triType {
				case 0: keep = c >= r
				case 1: keep = c > r
				case 2: keep = c <= r
				case 3: keep = c < r
				}
				if !keep {
					out[base+r*cols+c] = 0
				}
			}
		}
	}
	return out
}

func topK(t *Tensor, k int) *Tensor {
	data := t.materialize()
	if len(data) == 0 || k <= 0 {
		return newTensorFromData(t.b, []int{0}, nil)
	}
	dim0 := t.shape[0]
	nrows := len(data) / dim0
	k = min(k, dim0)
	out := make([]float32, nrows*k)
	for row := 0; row < nrows; row++ {
		start := row * dim0
		type iv struct{ i int; v float32 }
		pairs := make([]iv, dim0)
		for i := 0; i < dim0; i++ {
			pairs[i] = iv{i, data[start+i]}
		}
		sort.Slice(pairs, func(i, j int) bool { return pairs[i].v > pairs[j].v })
		for i := 0; i < k; i++ {
			out[row*k+i] = pairs[i].v
		}
	}
	ns := t.Shape()
	ns[0] = k
	return newTensorFromData(t.b, ns, out)
}

func argsort(t *Tensor) *Tensor {
	data := t.materialize()
	if len(data) == 0 {
		return newTensorFromData(t.b, t.Shape(), nil)
	}
	dim0 := t.shape[0]
	nrows := len(data) / dim0
	out := make([]float32, len(data))
	for row := 0; row < nrows; row++ {
		start := row * dim0
		indices := make([]int, dim0)
		for i := range indices {
			indices[i] = i
		}
		sort.Slice(indices, func(i, j int) bool {
			return data[start+indices[i]] < data[start+indices[j]]
		})
		for i, idx := range indices {
			out[start+i] = float32(idx)
		}
	}
	return newTensorFromData(t.b, t.Shape(), out)
}

var _ ml.Tensor = (*Tensor)(nil)
