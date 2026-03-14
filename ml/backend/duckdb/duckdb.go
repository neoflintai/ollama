package duckdb

import (
	"context"
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	_ "github.com/marcboeker/go-duckdb"
	"github.com/ollama/ollama/fs"
	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
	"golang.org/x/sync/errgroup"
)

type Backend struct {
	db        *sql.DB
	dbPath    string
	modelPath string
	meta      *fsggml.GGML
	tensors   map[string]*Tensor
	mu        sync.RWMutex
}

func New(modelPath string, params ml.BackendParams) (ml.Backend, error) {
	r, err := os.Open(modelPath)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	meta, err := fsggml.Decode(r, -1)
	if err != nil {
		return nil, err
	}

	slog.Info("duckdb backend",
		"architecture", meta.KV().Architecture(),
		"file_type", meta.KV().FileType(),
		"num_tensors", len(meta.Tensors().Items()),
	)

	dbPath := strings.TrimSuffix(modelPath, filepath.Ext(modelPath)) + ".duckdb"

	db, err := sql.Open("duckdb", dbPath)
	if err != nil {
		return nil, fmt.Errorf("duckdb open: %w", err)
	}

	// DuckDB optimizations for large blob/array workloads
	for _, pragma := range []string{
		"SET threads TO " + fmt.Sprint(runtime.GOMAXPROCS(0)),
		"SET memory_limit = '4GB'",
	} {
		if _, err := db.Exec(pragma); err != nil {
			slog.Warn("duckdb pragma", "sql", pragma, "error", err)
		}
	}

	if _, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS tensors (
			name  VARCHAR PRIMARY KEY,
			dtype INTEGER NOT NULL,
			shape VARCHAR NOT NULL,
			rows  INTEGER NOT NULL,
			cols  INTEGER NOT NULL,
			data  BLOB    NOT NULL
		);

		CREATE TABLE IF NOT EXISTS metadata (
			key   VARCHAR PRIMARY KEY,
			value VARCHAR NOT NULL
		);
	`); err != nil {
		db.Close()
		return nil, fmt.Errorf("duckdb schema: %w", err)
	}

	b := &Backend{
		db:        db,
		dbPath:    dbPath,
		modelPath: modelPath,
		meta:      meta,
		tensors:   make(map[string]*Tensor),
	}

	// Import and load all tensors eagerly so Get() works immediately
	if err := b.Load(context.Background(), nil); err != nil {
		db.Close()
		return nil, fmt.Errorf("duckdb load: %w", err)
	}

	return b, nil
}

func (b *Backend) Close() {
	if b.db != nil {
		b.db.Close()
	}
}

func (b *Backend) Config() fs.Config {
	return b.meta.KV()
}

func (b *Backend) BackendMemory() ml.BackendMemory {
	return ml.BackendMemory{}
}

func (b *Backend) BackendDevices() []ml.DeviceInfo {
	return []ml.DeviceInfo{{
		DeviceID:    ml.DeviceID{ID: "duckdb-cpu-0", Library: "duckdb"},
		Name:        "duckdb-cpu",
		Description: "DuckDB columnar CPU backend",
		TotalMemory: 16 * 1024 * 1024 * 1024, // report 16GB so scheduler doesn't reject
		FreeMemory:  16 * 1024 * 1024 * 1024,
	}}
}

func (b *Backend) Get(name string) ml.Tensor {
	b.mu.RLock()
	if t, ok := b.tensors[name]; ok {
		b.mu.RUnlock()
		return t
	}
	b.mu.RUnlock()

	t, err := b.loadTensorFromDB(name)
	if err != nil || t == nil {
		return nil
	}

	b.mu.Lock()
	b.tensors[name] = t
	b.mu.Unlock()
	return t
}

func (b *Backend) NewContext() ml.Context {
	return &Context{b: b, layer: -1}
}

func (b *Backend) NewContextSize(n int) ml.Context {
	return &Context{b: b, layer: -1}
}

func (b *Backend) Load(ctx context.Context, progress func(float32)) error {
	var count int
	if err := b.db.QueryRow("SELECT COUNT(*) FROM tensors").Scan(&count); err != nil {
		return err
	}

	expectedCount := len(b.meta.Tensors().Items())
	if count >= expectedCount {
		slog.Info("duckdb: tensors already imported", "count", count)
		return b.loadAllTensorsFromDB(progress)
	}

	slog.Info("duckdb: importing tensors from GGUF", "total", expectedCount)
	return b.importFromGGUF(ctx, progress)
}

func (b *Backend) importFromGGUF(ctx context.Context, progress func(float32)) error {
	tensors := b.meta.Tensors()
	items := tensors.Items()
	total := len(items)

	tx, err := b.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	stmt, err := tx.Prepare("INSERT OR REPLACE INTO tensors (name, dtype, shape, rows, cols, data) VALUES (?, ?, ?, ?, ?, ?)")
	if err != nil {
		return err
	}
	defer stmt.Close()

	var mu sync.Mutex
	var done int

	g, ctx := errgroup.WithContext(ctx)
	g.SetLimit(runtime.GOMAXPROCS(0))

	for _, t := range items {
		g.Go(func() error {
			if err := ctx.Err(); err != nil {
				return err
			}

			file, err := os.Open(b.modelPath)
			if err != nil {
				return err
			}
			defer file.Close()

			offset := tensors.Offset + t.Offset
			sr := io.NewSectionReader(file, int64(offset), int64(t.Size()))

			raw := make([]byte, t.Size())
			if _, err := io.ReadFull(sr, raw); err != nil {
				return fmt.Errorf("read tensor %s: %w", t.Name, err)
			}

			numElements := int64(1)
			shape := make([]int, len(t.Shape))
			for i, d := range t.Shape {
				numElements *= int64(d)
				shape[i] = int(d)
			}

			f32data := dequantize(raw, t.Kind, uint64(numElements))
			blob := float32ToBytes(f32data)
			shapeJSON, _ := json.Marshal(shape)

			// Calculate rows/cols for matmul optimization
			rows := 1
			cols := 1
			if len(shape) >= 1 {
				cols = shape[0]
			}
			if len(shape) >= 2 {
				rows = shape[1]
			}

			mu.Lock()
			_, err = stmt.Exec(t.Name, int(ml.DTypeF32), string(shapeJSON), rows, cols, blob)
			done++
			current := done
			mu.Unlock()

			if err != nil {
				return fmt.Errorf("insert tensor %s: %w", t.Name, err)
			}

			if progress != nil {
				progress(float32(current) / float32(total))
			}

			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}

	if err := tx.Commit(); err != nil {
		return err
	}

	slog.Info("duckdb: import complete", "tensors", total)
	return b.loadAllTensorsFromDB(nil)
}

func (b *Backend) loadAllTensorsFromDB(progress func(float32)) error {
	rows, err := b.db.Query("SELECT name, dtype, shape, data FROM tensors")
	if err != nil {
		return err
	}
	defer rows.Close()

	var count int
	for rows.Next() {
		var name string
		var dtype int
		var shapeJSON string
		var data []byte

		if err := rows.Scan(&name, &dtype, &shapeJSON, &data); err != nil {
			return err
		}

		var shape []int
		if err := json.Unmarshal([]byte(shapeJSON), &shape); err != nil {
			return fmt.Errorf("parse shape for %s: %w", name, err)
		}

		t := newTensorFromData(b, shape, bytesToFloat32(data))
		t.name = name
		t.dtype = ml.DType(dtype)

		b.mu.Lock()
		b.tensors[name] = t
		b.mu.Unlock()
		count++
	}

	slog.Info("duckdb: loaded tensors into cache", "count", count)
	return nil
}

func (b *Backend) loadTensorFromDB(name string) (*Tensor, error) {
	var dtype int
	var shapeJSON string
	var data []byte

	err := b.db.QueryRow("SELECT dtype, shape, data FROM tensors WHERE name = ?", name).
		Scan(&dtype, &shapeJSON, &data)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	var shape []int
	if err := json.Unmarshal([]byte(shapeJSON), &shape); err != nil {
		return nil, err
	}

	t := newTensorFromData(b, shape, bytesToFloat32(data))
	t.name = name
	t.dtype = ml.DType(dtype)
	return t, nil
}

func float32ToBytes(f []float32) []byte {
	buf := make([]byte, len(f)*4)
	for i, v := range f {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

func bytesToFloat32(b []byte) []float32 {
	if len(b) == 0 {
		return nil
	}
	f := make([]float32, len(b)/4)
	for i := range f {
		f[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return f
}
