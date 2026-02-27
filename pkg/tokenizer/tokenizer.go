package tokenizer

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
)

// Tokenizer converts text and images to token sequences
// This is a placeholder implementation - production would use HuggingFace tokenizers
type Tokenizer struct {
	vocabSize int
	// decoder is an optional custom decoder function
	// If set, it will be used instead of the default Decode implementation
	decoder func([]uint32) (string, error)
	// Add actual tokenizer state here
}

// NewTokenizer creates a new tokenizer instance
func NewTokenizer(vocabSize int) *Tokenizer {
	return &Tokenizer{
		vocabSize: vocabSize,
	}
}

// SetDecoder sets a custom decoder function
func (t *Tokenizer) SetDecoder(decoder func([]uint32) (string, error)) {
	t.decoder = decoder
}

// EncodeText converts text to token IDs
// Returns slice of uint32 token IDs
func (t *Tokenizer) EncodeText(text string) ([]uint32, error) {
	if text == "" {
		return nil, fmt.Errorf("empty text")
	}

	// Placeholder: simple character-level tokenization
	// Production would use BPE/WordPiece/SentencePiece
	tokens := make([]uint32, len(text))
	for i, ch := range text {
		tokens[i] = uint32(ch) % uint32(t.vocabSize)
	}

	return tokens, nil
}

// EncodeImage converts an image (base64 encoded) to token IDs
// This is a placeholder - production would use vision encoder (CLIP, SigLIP, etc.)
func (t *Tokenizer) EncodeImage(imageBase64 string) ([]uint32, error) {
	if imageBase64 == "" {
		return nil, fmt.Errorf("empty image data")
	}

	// Decode base64 to verify format
	_, err := base64.StdEncoding.DecodeString(imageBase64)
	if err != nil {
		return nil, fmt.Errorf("invalid base64 image: %w", err)
	}

	// Placeholder: Generate fake image tokens
	// Production would run image through vision encoder
	// Vision encoder outputs would be embedded into token space
	const imageTokensPerPatch = 1
	numPatches := 256 // 16x16 patches for example

	tokens := make([]uint32, numPatches*imageTokensPerPatch)
	for i := range tokens {
		tokens[i] = uint32(i) % uint32(t.vocabSize)
	}

	return tokens, nil
}

// Decode converts token IDs back to text
func (t *Tokenizer) Decode(tokens []uint32) (string, error) {
	// Use custom decoder if set
	if t.decoder != nil {
		return t.decoder(tokens)
	}

	if len(tokens) == 0 {
		return "", fmt.Errorf("empty tokens")
	}

	// Placeholder: Convert tokens back to characters
	result := make([]rune, len(tokens))
	for i, token := range tokens {
		result[i] = rune(token)
	}

	return string(result), nil
}

// VocabSize returns the tokenizer vocabulary size
func (t *Tokenizer) VocabSize() int {
	return t.vocabSize
}

// ChatMessage represents a single message in a chat conversation
type ChatMessage struct {
	Role    string `json:"role"`    // "system", "user", "assistant"
	Content string `json:"content"` // Message content
	// Optional: Image for multimodal input
	Image string `json:"image,omitempty"` // Base64 encoded image
}

// ChatRequest represents a chat completion request
type ChatRequest struct {
	Messages    []ChatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
}

// TokenizeChatRequest converts a chat request to token sequence
// Handles special tokens for chat format
func (t *Tokenizer) TokenizeChatRequest(req *ChatRequest) ([]uint32, error) {
	if len(req.Messages) == 0 {
		return nil, fmt.Errorf("no messages in request")
	}

	var allTokens []uint32

	for _, msg := range req.Messages {
		// Add role-specific special tokens
		roleTokens, err := t.encodeRole(msg.Role)
		if err != nil {
			return nil, err
		}
		allTokens = append(allTokens, roleTokens...)

		// Tokenize content
		contentTokens, err := t.EncodeText(msg.Content)
		if err != nil {
			return nil, fmt.Errorf("failed to encode content: %w", err)
		}
		allTokens = append(allTokens, contentTokens...)

		// Handle image if present
		if msg.Image != "" {
			imageTokens, err := t.EncodeImage(msg.Image)
			if err != nil {
				return nil, fmt.Errorf("failed to encode image: %w", err)
			}
			allTokens = append(allTokens, imageTokens...)
		}
	}

	// Add assistant response prefix
	allTokens = append(allTokens, t.getAssistantPrefix()...)

	return allTokens, nil
}

// encodeRole adds role-specific special tokens
func (t *Tokenizer) encodeRole(role string) ([]uint32, error) {
	// Special token IDs for chat format
	// Production would use model-specific special tokens
	switch role {
	case "system":
		return []uint32{1000, 1001}, nil // <|system|>\n
	case "user":
		return []uint32{1002, 1001}, nil // <|user|>\n
	case "assistant":
		return []uint32{1003, 1001}, nil // <|assistant|>\n
	default:
		return nil, fmt.Errorf("unknown role: %s", role)
	}
}

// getAssistantPrefix returns tokens that prompt assistant response
func (t *Tokenizer) getAssistantPrefix() []uint32 {
	return []uint32{1003, 1001} // <|assistant|>\n
}

// TokenizerConfig holds tokenizer configuration
type TokenizerConfig struct {
	VocabSize      int    `json:"vocab_size"`
	ModelPath      string `json:"model_path"`
	Type           string `json:"type"` // "bpe", "wordpiece", "sentencepiece"
	SpecialTokens  map[string]int `json:"special_tokens"`
}

// LoadConfig loads tokenizer configuration from JSON
func LoadConfig(r io.Reader) (*TokenizerConfig, error) {
	var config TokenizerConfig
	decoder := json.NewDecoder(r)
	if err := decoder.Decode(&config); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}
	return &config, nil
}

// NewTokenizerFromConfig creates a tokenizer from configuration
func NewTokenizerFromConfig(config *TokenizerConfig) (*Tokenizer, error) {
	if config.VocabSize <= 0 {
		return nil, fmt.Errorf("invalid vocab size: %d", config.VocabSize)
	}

	return &Tokenizer{
		vocabSize: config.VocabSize,
	}, nil
}
