package sqlite

import "github.com/ollama/ollama/ml"

func init() {
	ml.RegisterBackend("sqlite", New)
}
