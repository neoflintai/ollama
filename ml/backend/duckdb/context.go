package duckdb

import (
	"math"

	"github.com/ollama/ollama/ml"
)

type Context struct {
	b     *Backend
	layer int
}

func (c *Context) Empty(dtype ml.DType, shape ...int) ml.Tensor {
	total := product(shape)
	return newTensorFromData(c.b, shape, make([]float32, total))
}

func (c *Context) Zeros(dtype ml.DType, shape ...int) ml.Tensor {
	return c.Empty(dtype, shape...)
}

func (c *Context) FromBytes(dtype ml.DType, s []byte, shape ...int) ml.Tensor {
	var data []float32
	switch dtype {
	case ml.DTypeF32:
		data = bytesToFloat32(s)
	case ml.DTypeI32:
		n := len(s) / 4
		data = make([]float32, n)
		for i := 0; i < n; i++ {
			v := int32(uint32(s[i*4]) | uint32(s[i*4+1])<<8 | uint32(s[i*4+2])<<16 | uint32(s[i*4+3])<<24)
			data[i] = float32(v)
		}
	default:
		data = bytesToFloat32(s)
	}
	return newTensorFromData(c.b, shape, data)
}

func (c *Context) FromFloats(s []float32, shape ...int) ml.Tensor {
	data := make([]float32, len(s))
	copy(data, s)
	return newTensorFromData(c.b, shape, data)
}

func (c *Context) FromInts(s []int32, shape ...int) ml.Tensor {
	data := make([]float32, len(s))
	for i, v := range s {
		data[i] = float32(v)
	}
	return newTensorFromData(c.b, shape, data)
}

func (c *Context) Arange(start, stop, step float32, dtype ml.DType) ml.Tensor {
	if step == 0 {
		step = 1
	}
	n := int(math.Ceil(float64(stop-start) / float64(step)))
	if n <= 0 {
		return newTensorFromData(c.b, []int{0}, nil)
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = start + float32(i)*step
	}
	return newTensorFromData(c.b, []int{n}, data)
}

func (c *Context) Forward(tensors ...ml.Tensor) ml.Context { return c }
func (c *Context) SetBatchSize(n int)                       {}
func (c *Context) Compute(tensors ...ml.Tensor)             {}

func (c *Context) ComputeWithNotify(fn func(), tensors ...ml.Tensor) {
	if fn != nil {
		fn()
	}
}

func (c *Context) Reserve()           {}
func (c *Context) MaxGraphNodes() int { return 1 << 20 }
func (c *Context) Close()             {}

func (c *Context) Input() ml.Context {
	return &Context{b: c.b, layer: -1}
}

func (c *Context) Layer(n int) ml.Context {
	return &Context{b: c.b, layer: n}
}

var _ ml.Context = (*Context)(nil)
