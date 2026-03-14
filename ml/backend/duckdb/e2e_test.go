package duckdb

import (
	"context"
	"os"
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestE2ELoadModel(t *testing.T) {
	modelPath := os.Getenv("OLLAMA_TEST_MODEL")
	if modelPath == "" {
		modelPath = os.ExpandEnv("$HOME/.ollama/models/blobs/sha256-f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57")
	}

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("model file not found, run: ollama pull smollm2:135m")
	}

	os.Remove(modelPath + ".duckdb")
	os.Remove(modelPath + ".duckdb.wal")
	defer os.Remove(modelPath + ".duckdb")
	defer os.Remove(modelPath + ".duckdb.wal")

	t.Log("Creating DuckDB backend...")
	backend, err := New(modelPath, ml.BackendParams{AllocMemory: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer backend.Close()

	b := backend.(*Backend)
	t.Logf("Model architecture: %s", b.Config().Architecture())

	t.Log("Loading/importing tensors...")
	err = b.Load(context.Background(), func(p float32) {
		if int(p*100)%25 == 0 {
			t.Logf("  progress: %.0f%%", p*100)
		}
	})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	t.Logf("Loaded %d tensors", len(b.tensors))
	if len(b.tensors) == 0 {
		t.Fatal("no tensors loaded")
	}

	var firstName string
	var firstTensor *Tensor
	for name, tensor := range b.tensors {
		firstName = name
		firstTensor = tensor
		break
	}

	t.Logf("Sample tensor %q: shape=%v len=%d", firstName, firstTensor.shape, len(firstTensor.data))

	ctx := b.NewContext()
	sliced := firstTensor.Slice(ctx, 0, 0, min(64, firstTensor.Dim(0)), 1)
	scaled := sliced.Scale(ctx, 0.5)
	floats := scaled.Floats()

	allZero := true
	for _, v := range floats {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("all values zero after ops")
	}

	// Test matmul with real weights
	if firstTensor.Dim(0) >= 2 && len(firstTensor.shape) >= 2 && firstTensor.Dim(1) >= 2 {
		small := firstTensor.Slice(ctx, 0, 0, min(32, firstTensor.Dim(0)), 1)
		result := small.Mulmat(ctx, small)
		t.Logf("Matmul result shape=%v", result.Shape())
	}

	t.Log("DuckDB e2e PASSED")
}
