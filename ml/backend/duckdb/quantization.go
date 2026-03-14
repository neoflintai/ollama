package duckdb

import (
	"encoding/binary"
	"math"
)

// dequantize converts raw tensor bytes of the given GGUF kind to float32 values.
func dequantize(data []byte, kind uint32, numElements uint64) []float32 {
	switch kind {
	case 0: // F32
		return dequantF32(data, numElements)
	case 1: // F16
		return dequantF16(data, numElements)
	case 2: // Q4_0
		return dequantQ4_0(data, numElements)
	case 3: // Q4_1
		return dequantQ4_1(data, numElements)
	case 6: // Q5_0
		return dequantQ5_0(data, numElements)
	case 7: // Q5_1
		return dequantQ5_1(data, numElements)
	case 8: // Q8_0
		return dequantQ8_0(data, numElements)
	case 9: // Q8_1
		return dequantQ8_1(data, numElements)
	case 10: // Q2_K
		return dequantQ2K(data, numElements)
	case 11: // Q3_K
		return dequantQ3K(data, numElements)
	case 12: // Q4_K
		return dequantQ4K(data, numElements)
	case 13: // Q5_K
		return dequantQ5K(data, numElements)
	case 14: // Q6_K
		return dequantQ6K(data, numElements)
	case 30: // BF16
		return dequantBF16(data, numElements)
	case 26: // I8
		return dequantI8(data, numElements)
	case 28: // I32
		return dequantI32(data, numElements)
	default:
		out := make([]float32, numElements)
		if uint64(len(data)) >= numElements*4 {
			return dequantF32(data, numElements)
		}
		return out
	}
}

func dequantF32(data []byte, n uint64) []float32 {
	out := make([]float32, n)
	for i := uint64(0); i < n && i*4+4 <= uint64(len(data)); i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out
}

func dequantF16(data []byte, n uint64) []float32 {
	out := make([]float32, n)
	for i := uint64(0); i < n && i*2+2 <= uint64(len(data)); i++ {
		bits := binary.LittleEndian.Uint16(data[i*2:])
		out[i] = float16ToFloat32(bits)
	}
	return out
}

func dequantBF16(data []byte, n uint64) []float32 {
	out := make([]float32, n)
	for i := uint64(0); i < n && i*2+2 <= uint64(len(data)); i++ {
		bits := uint32(binary.LittleEndian.Uint16(data[i*2:])) << 16
		out[i] = math.Float32frombits(bits)
	}
	return out
}

func dequantI8(data []byte, n uint64) []float32 {
	out := make([]float32, n)
	for i := uint64(0); i < n && i < uint64(len(data)); i++ {
		out[i] = float32(int8(data[i]))
	}
	return out
}

func dequantI32(data []byte, n uint64) []float32 {
	out := make([]float32, n)
	for i := uint64(0); i < n && i*4+4 <= uint64(len(data)); i++ {
		v := int32(binary.LittleEndian.Uint32(data[i*4:]))
		out[i] = float32(v)
	}
	return out
}

func dequantQ4_0(data []byte, n uint64) []float32 {
	const blockSize = 32
	const bytesPerBlock = 2 + 16
	out := make([]float32, n)
	nblocks := n / blockSize

	for block := uint64(0); block < nblocks; block++ {
		off := block * bytesPerBlock
		if off+bytesPerBlock > uint64(len(data)) {
			break
		}
		scale := float16ToFloat32(binary.LittleEndian.Uint16(data[off:]))
		for j := 0; j < 16; j++ {
			b := data[off+2+uint64(j)]
			x0 := float32(int(b&0x0F) - 8)
			x1 := float32(int(b>>4) - 8)
			idx := block*blockSize + uint64(j*2)
			if idx < n {
				out[idx] = x0 * scale
			}
			if idx+1 < n {
				out[idx+1] = x1 * scale
			}
		}
	}
	return out
}

func dequantQ4_1(data []byte, n uint64) []float32 {
	const blockSize = 32
	const bytesPerBlock = 2 + 2 + 16
	out := make([]float32, n)
	nblocks := n / blockSize

	for block := uint64(0); block < nblocks; block++ {
		off := block * bytesPerBlock
		if off+bytesPerBlock > uint64(len(data)) {
			break
		}
		scale := float16ToFloat32(binary.LittleEndian.Uint16(data[off:]))
		min := float16ToFloat32(binary.LittleEndian.Uint16(data[off+2:]))
		for j := 0; j < 16; j++ {
			b := data[off+4+uint64(j)]
			x0 := float32(b & 0x0F)
			x1 := float32(b >> 4)
			idx := block*blockSize + uint64(j*2)
			if idx < n {
				out[idx] = x0*scale + min
			}
			if idx+1 < n {
				out[idx+1] = x1*scale + min
			}
		}
	}
	return out
}

func dequantQ5_0(data []byte, n uint64) []float32 {
	const blockSize = 32
	const bytesPerBlock = 2 + 4 + 16
	out := make([]float32, n)
	nblocks := n / blockSize

	for block := uint64(0); block < nblocks; block++ {
		off := block * bytesPerBlock
		if off+bytesPerBlock > uint64(len(data)) {
			break
		}
		scale := float16ToFloat32(binary.LittleEndian.Uint16(data[off:]))
		qh := binary.LittleEndian.Uint32(data[off+2:])
		for j := 0; j < 16; j++ {
			b := data[off+6+uint64(j)]
			x0l := int(b & 0x0F)
			x1l := int(b >> 4)
			x0h := int((qh >> uint(j*2)) & 1)
			x1h := int((qh >> uint(j*2+1)) & 1)
			x0 := float32((x0l | (x0h << 4)) - 16)
			x1 := float32((x1l | (x1h << 4)) - 16)
			idx := block*blockSize + uint64(j*2)
			if idx < n {
				out[idx] = x0 * scale
			}
			if idx+1 < n {
				out[idx+1] = x1 * scale
			}
		}
	}
	return out
}

func dequantQ5_1(data []byte, n uint64) []float32 {
	const blockSize = 32
	const bytesPerBlock = 2 + 2 + 4 + 16
	out := make([]float32, n)
	nblocks := n / blockSize

	for block := uint64(0); block < nblocks; block++ {
		off := block * bytesPerBlock
		if off+bytesPerBlock > uint64(len(data)) {
			break
		}
		scale := float16ToFloat32(binary.LittleEndian.Uint16(data[off:]))
		min := float16ToFloat32(binary.LittleEndian.Uint16(data[off+2:]))
		qh := binary.LittleEndian.Uint32(data[off+4:])
		for j := 0; j < 16; j++ {
			b := data[off+8+uint64(j)]
			x0l := int(b & 0x0F)
			x1l := int(b >> 4)
			x0h := int((qh >> uint(j*2)) & 1)
			x1h := int((qh >> uint(j*2+1)) & 1)
			x0 := float32(x0l | (x0h << 4))
			x1 := float32(x1l | (x1h << 4))
			idx := block*blockSize + uint64(j*2)
			if idx < n {
				out[idx] = x0*scale + min
			}
			if idx+1 < n {
				out[idx+1] = x1*scale + min
			}
		}
	}
	return out
}

func dequantQ8_0(data []byte, n uint64) []float32 {
	const blockSize = 32
	const bytesPerBlock = 2 + 32
	out := make([]float32, n)
	nblocks := n / blockSize

	for block := uint64(0); block < nblocks; block++ {
		off := block * bytesPerBlock
		if off+bytesPerBlock > uint64(len(data)) {
			break
		}
		scale := float16ToFloat32(binary.LittleEndian.Uint16(data[off:]))
		for j := 0; j < 32; j++ {
			q := int8(data[off+2+uint64(j)])
			idx := block*blockSize + uint64(j)
			if idx < n {
				out[idx] = float32(q) * scale
			}
		}
	}
	return out
}

func dequantQ8_1(data []byte, n uint64) []float32 {
	const blockSize = 32
	const bytesPerBlock = 2 + 2 + 32
	out := make([]float32, n)
	nblocks := n / blockSize

	for block := uint64(0); block < nblocks; block++ {
		off := block * bytesPerBlock
		if off+bytesPerBlock > uint64(len(data)) {
			break
		}
		scale := float16ToFloat32(binary.LittleEndian.Uint16(data[off:]))
		for j := 0; j < 32; j++ {
			q := int8(data[off+4+uint64(j)])
			idx := block*blockSize + uint64(j)
			if idx < n {
				out[idx] = float32(q) * scale
			}
		}
	}
	return out
}

func dequantQ2K(data []byte, n uint64) []float32 {
	const blockSize = 256
	const bytesPerBlock = 256/16 + 256/4 + 2 + 2
	out := make([]float32, n)
	nblocks := n / blockSize

	for block := uint64(0); block < nblocks; block++ {
		off := block * bytesPerBlock
		if off+bytesPerBlock > uint64(len(data)) {
			break
		}
		scales := data[off : off+16]
		qs := data[off+16 : off+16+64]
		d := float16ToFloat32(binary.LittleEndian.Uint16(data[off+80:]))
		dmin := float16ToFloat32(binary.LittleEndian.Uint16(data[off+82:]))

		for j := 0; j < 256; j++ {
			scaleIdx := j / 16
			sc := scales[scaleIdx]
			scaleVal := float32(sc&0xF) * d
			minVal := float32(sc>>4) * dmin
			qByte := qs[j/4]
			shift := uint((j % 4) * 2)
			q := int((qByte >> shift) & 3)
			idx := block*blockSize + uint64(j)
			if idx < n {
				out[idx] = scaleVal*float32(q) - minVal
			}
		}
	}
	return out
}

func dequantQ3K(data []byte, n uint64) []float32 {
	const blockSize = 256
	const hmaskBytes = 32
	const qsBytes = 64
	const scalesBytes = 12
	const bytesPerBlock = hmaskBytes + qsBytes + scalesBytes + 2
	out := make([]float32, n)
	nblocks := n / blockSize

	for block := uint64(0); block < nblocks; block++ {
		off := block * bytesPerBlock
		if off+bytesPerBlock > uint64(len(data)) {
			break
		}
		hmask := data[off : off+32]
		qs := data[off+32 : off+96]
		rawScales := data[off+96 : off+108]
		d := float16ToFloat32(binary.LittleEndian.Uint16(data[off+108:]))

		var scales [16]int8
		for i := 0; i < 8; i++ {
			scales[i] = int8((rawScales[i] & 0xF) - 8)
		}
		for i := 0; i < 4; i++ {
			scales[8+i] = int8((rawScales[i] >> 4) | ((rawScales[8+i] & 0xF) << 4))
		}
		for i := 0; i < 4; i++ {
			scales[12+i] = int8((rawScales[4+i] >> 4) | ((rawScales[8+i] >> 4) << 4))
		}

		for j := 0; j < 256; j++ {
			qByte := qs[j/4]
			shift := uint((j % 4) * 2)
			q := int((qByte >> shift) & 3)
			hBit := int((hmask[j/8] >> uint(j%8)) & 1)
			q |= hBit << 2
			sc := float32(scales[j/16])
			idx := block*blockSize + uint64(j)
			if idx < n {
				out[idx] = d * sc * (float32(q) - 4)
			}
		}
	}
	return out
}

func dequantQ4K(data []byte, n uint64) []float32 {
	const blockSize = 256
	const bytesPerBlock = 2 + 2 + 12 + 128
	out := make([]float32, n)
	nblocks := n / blockSize

	for block := uint64(0); block < nblocks; block++ {
		off := block * bytesPerBlock
		if off+bytesPerBlock > uint64(len(data)) {
			break
		}
		d := float16ToFloat32(binary.LittleEndian.Uint16(data[off:]))
		dmin := float16ToFloat32(binary.LittleEndian.Uint16(data[off+2:]))
		rawScales := data[off+4 : off+16]
		qs := data[off+16 : off+144]

		var scales [8]uint8
		var mins [8]uint8
		for i := 0; i < 4; i++ {
			scales[i] = rawScales[i] & 63
			mins[i] = rawScales[i+4] & 63
		}
		for i := 0; i < 4; i++ {
			scales[4+i] = (rawScales[i] >> 6) | ((rawScales[i+8] & 0xF) << 2)
			mins[4+i] = (rawScales[i+4] >> 6) | ((rawScales[i+8] >> 4) << 2)
		}

		for j := 0; j < 256; j++ {
			subblock := j / 32
			sc := float32(scales[subblock]) * d
			mn := float32(mins[subblock]) * dmin
			qByte := qs[j/2]
			var q int
			if j%2 == 0 {
				q = int(qByte & 0xF)
			} else {
				q = int(qByte >> 4)
			}
			idx := block*blockSize + uint64(j)
			if idx < n {
				out[idx] = sc*float32(q) - mn
			}
		}
	}
	return out
}

func dequantQ5K(data []byte, n uint64) []float32 {
	const blockSize = 256
	const bytesPerBlock = 2 + 2 + 12 + 32 + 128
	out := make([]float32, n)
	nblocks := n / blockSize

	for block := uint64(0); block < nblocks; block++ {
		off := block * bytesPerBlock
		if off+bytesPerBlock > uint64(len(data)) {
			break
		}
		d := float16ToFloat32(binary.LittleEndian.Uint16(data[off:]))
		dmin := float16ToFloat32(binary.LittleEndian.Uint16(data[off+2:]))
		rawScales := data[off+4 : off+16]
		qh := data[off+16 : off+48]
		qs := data[off+48 : off+176]

		var scales [8]uint8
		var mins [8]uint8
		for i := 0; i < 4; i++ {
			scales[i] = rawScales[i] & 63
			mins[i] = rawScales[i+4] & 63
		}
		for i := 0; i < 4; i++ {
			scales[4+i] = (rawScales[i] >> 6) | ((rawScales[i+8] & 0xF) << 2)
			mins[4+i] = (rawScales[i+4] >> 6) | ((rawScales[i+8] >> 4) << 2)
		}

		for j := 0; j < 256; j++ {
			subblock := j / 32
			sc := float32(scales[subblock]) * d
			mn := float32(mins[subblock]) * dmin
			qByte := qs[j/2]
			var q int
			if j%2 == 0 {
				q = int(qByte & 0xF)
			} else {
				q = int(qByte >> 4)
			}
			hBit := int((qh[j/8] >> uint(j%8)) & 1)
			q |= hBit << 4
			idx := block*blockSize + uint64(j)
			if idx < n {
				out[idx] = sc*float32(q) - mn
			}
		}
	}
	return out
}

func dequantQ6K(data []byte, n uint64) []float32 {
	const blockSize = 256
	const bytesPerBlock = 128 + 64 + 16 + 2
	out := make([]float32, n)
	nblocks := n / blockSize

	for block := uint64(0); block < nblocks; block++ {
		off := block * bytesPerBlock
		if off+bytesPerBlock > uint64(len(data)) {
			break
		}
		ql := data[off : off+128]
		qh := data[off+128 : off+192]
		scales := data[off+192 : off+208]
		d := float16ToFloat32(binary.LittleEndian.Uint16(data[off+208:]))

		for j := 0; j < 256; j++ {
			qlByte := ql[j/2]
			var qlo int
			if j%2 == 0 {
				qlo = int(qlByte & 0xF)
			} else {
				qlo = int(qlByte >> 4)
			}
			qhByte := qh[j/4]
			shift := uint((j % 4) * 2)
			qhi := int((qhByte >> shift) & 3)
			q := qlo | (qhi << 4)
			sc := int8(scales[j/16])
			idx := block*blockSize + uint64(j)
			if idx < n {
				out[idx] = d * float32(sc) * (float32(q) - 32)
			}
		}
	}
	return out
}

func float16ToFloat32(bits uint16) float32 {
	sign := uint32(bits>>15) & 1
	exp := uint32(bits>>10) & 0x1F
	frac := uint32(bits) & 0x3FF

	switch {
	case exp == 0:
		if frac == 0 {
			return math.Float32frombits(sign << 31)
		}
		for frac&0x400 == 0 {
			frac <<= 1
			exp--
		}
		exp++
		frac &= 0x3FF
		return math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (frac << 13))
	case exp == 0x1F:
		if frac == 0 {
			return math.Float32frombits((sign << 31) | 0x7F800000)
		}
		return math.Float32frombits((sign << 31) | 0x7FC00000)
	default:
		return math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (frac << 13))
	}
}
