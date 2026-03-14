package duckdb

import "github.com/ollama/ollama/ml"

func init() {
	ml.RegisterBackend("duckdb", New)
}
