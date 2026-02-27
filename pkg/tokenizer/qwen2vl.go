package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
)

// Qwen2VLTokenizer is a BPE tokenizer for Qwen2-VL models
// It loads the vocabulary from the model's vocab.json file
type Qwen2VLTokenizer struct {
	vocabSize   int
	vocabPath   string
	// idToToken maps token IDs to their string representation
	idToToken   map[uint32]string
	// tokenToID maps token strings to their IDs
	tokenToID   map[string]uint32
	once        sync.Once
	initErr     error
}

// NewQwen2VLTokenizer creates a new Qwen2-VL tokenizer
// modelPath should point to the directory containing vocab.json
func NewQwen2VLTokenizer(modelPath string, vocabSize int) *Qwen2VLTokenizer {
	return &Qwen2VLTokenizer{
		vocabSize: vocabSize,
		vocabPath: modelPath + "/vocab.json",
		idToToken: make(map[uint32]string),
		tokenToID: make(map[string]uint32),
	}
}

// Load loads the vocabulary from the vocab.json file
// This is safe to call multiple times (uses sync.Once)
func (t *Qwen2VLTokenizer) Load() error {
	t.once.Do(func() {
		data, err := os.ReadFile(t.vocabPath)
		if err != nil {
			t.initErr = fmt.Errorf("failed to read vocab.json: %w", err)
			return
		}

		// vocab.json format: {"token": id, "another": id, ...}
		var vocab map[string]uint32
		if err := json.Unmarshal(data, &vocab); err != nil {
			t.initErr = fmt.Errorf("failed to parse vocab.json: %w", err)
			return
		}

		// Build bidirectional maps
		for token, id := range vocab {
			t.idToToken[id] = token
			t.tokenToID[token] = id
		}

		// Update vocab size to actual size
		actualSize := len(t.idToToken)
		if actualSize != t.vocabSize {
			// Update to actual vocab size
			t.vocabSize = actualSize
		}
	})
	return t.initErr
}

// Decode converts token IDs back to text
// For most tokens, this is a direct lookup from the vocab
func (t *Qwen2VLTokenizer) Decode(tokens []uint32) (string, error) {
	if err := t.Load(); err != nil {
		return "", err
	}

	if len(tokens) == 0 {
		return "", nil
	}

	var result strings.Builder
	result.Grow(len(tokens) * 4) // Pre-allocate average 4 bytes per token

	for _, token := range tokens {
		if token >= uint32(t.vocabSize) {
			return "", fmt.Errorf("token ID %d out of range (vocab size: %d)", token, t.vocabSize)
		}
		tokenStr, ok := t.idToToken[token]
		if !ok {
			return "", fmt.Errorf("token ID %d not found in vocabulary", token)
		}
		result.WriteString(tokenStr)
	}

	return result.String(), nil
}

// DecodeSingle decodes a single token ID to text
func (t *Qwen2VLTokenizer) DecodeSingle(token uint32) (string, error) {
	if err := t.Load(); err != nil {
		return "", err
	}

	if token >= uint32(t.vocabSize) {
		return "", fmt.Errorf("token ID %d out of range (vocab size: %d)", token, t.vocabSize)
	}

	tokenStr, ok := t.idToToken[token]
	if !ok {
		return "", fmt.Errorf("token ID %d not found in vocabulary", token)
	}

	return tokenStr, nil
}

// Encode converts text to token IDs
// This is a simplified implementation that does character-level encoding
// Full BPE encoding would require loading merges.txt and implementing the algorithm
func (t *Qwen2VLTokenizer) Encode(text string) ([]uint32, error) {
	if err := t.Load(); err != nil {
		return nil, err
	}

	if text == "" {
		return nil, fmt.Errorf("empty text")
	}

	// For now, use a simple character-based encoding
	// TODO: Implement proper BPE encoding using merges.txt
	tokens := make([]uint32, 0, len(text))
	for _, ch := range text {
		tokenStr := string(ch)
		if id, ok := t.tokenToID[tokenStr]; ok {
			tokens = append(tokens, id)
		} else {
			// Fall back to byte-level encoding for unknown characters
			for _, b := range []byte(tokenStr) {
				tokens = append(tokens, uint32(b))
			}
		}
	}

	return tokens, nil
}

// VocabSize returns the vocabulary size
func (t *Qwen2VLTokenizer) VocabSize() int {
	return t.vocabSize
}
