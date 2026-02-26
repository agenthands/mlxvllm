package tokenizer

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestNewTokenizer(t *testing.T) {
	tok := NewTokenizer(32000)
	if tok == nil {
		t.Fatal("Expected non-nil tokenizer")
	}

	if tok.VocabSize() != 32000 {
		t.Errorf("Expected vocab size 32000, got %d", tok.VocabSize())
	}
}

func TestEncodeText(t *testing.T) {
	tok := NewTokenizer(1000)

	text := "Hello, world!"
	tokens, err := tok.EncodeText(text)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if len(tokens) != len(text) {
		t.Errorf("Expected %d tokens, got %d", len(text), len(tokens))
	}
}

func TestEncodeTextEmpty(t *testing.T) {
	tok := NewTokenizer(1000)

	_, err := tok.EncodeText("")
	if err == nil {
		t.Error("Expected error for empty text")
	}
}

func TestEncodeImage(t *testing.T) {
	tok := NewTokenizer(1000)

	// Valid base64 image (1x1 PNG)
	imageBase64 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

	tokens, err := tok.EncodeImage(imageBase64)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	// Should generate image tokens
	if len(tokens) == 0 {
		t.Error("Expected tokens for image")
	}

	// Each token should be within vocab size
	for _, token := range tokens {
		if int(token) >= tok.VocabSize() {
			t.Errorf("Token %d exceeds vocab size %d", token, tok.VocabSize())
		}
	}
}

func TestEncodeImageInvalidBase64(t *testing.T) {
	tok := NewTokenizer(1000)

	_, err := tok.EncodeImage("not-valid-base64!!!")
	if err == nil {
		t.Error("Expected error for invalid base64")
	}
}

func TestEncodeImageEmpty(t *testing.T) {
	tok := NewTokenizer(1000)

	_, err := tok.EncodeImage("")
	if err == nil {
		t.Error("Expected error for empty image data")
	}
}

func TestDecode(t *testing.T) {
	tok := NewTokenizer(1000)

	tokens := []uint32{72, 101, 108, 108, 111} // "Hello"
	text, err := tok.Decode(tokens)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if text != "Hello" {
		t.Errorf("Expected 'Hello', got '%s'", text)
	}
}

func TestDecodeEmpty(t *testing.T) {
	tok := NewTokenizer(1000)

	_, err := tok.Decode([]uint32{})
	if err == nil {
		t.Error("Expected error for empty tokens")
	}
}

func TestTokenizeChatRequest(t *testing.T) {
	tok := NewTokenizer(10000)

	req := &ChatRequest{
		Messages: []ChatMessage{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "Hello!"},
		},
	}

	tokens, err := tok.TokenizeChatRequest(req)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	// Should have tokens for both messages
	if len(tokens) == 0 {
		t.Error("Expected tokens from chat request")
	}

	// Should contain special role tokens (1000-1003 range)
	hasSpecialTokens := false
	for _, token := range tokens {
		if token >= 1000 && token <= 1003 {
			hasSpecialTokens = true
			break
		}
	}
	if !hasSpecialTokens {
		t.Error("Expected special role tokens in output")
	}
}

func TestTokenizeChatRequestEmpty(t *testing.T) {
	tok := NewTokenizer(10000)

	req := &ChatRequest{
		Messages: []ChatMessage{},
	}

	_, err := tok.TokenizeChatRequest(req)
	if err == nil {
		t.Error("Expected error for empty messages")
	}
}

func TestTokenizeChatRequestWithImage(t *testing.T) {
	tok := NewTokenizer(10000)

	// Valid 1x1 PNG base64
	imageBase64 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

	req := &ChatRequest{
		Messages: []ChatMessage{
			{
				Role:    "user",
				Content: "What's in this image?",
				Image:   imageBase64,
			},
		},
	}

	tokens, err := tok.TokenizeChatRequest(req)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	// Should have more tokens than text-only (image tokens added)
	textOnlyReq := &ChatRequest{
		Messages: []ChatMessage{
			{Role: "user", Content: "What's in this image?"},
		},
	}
	textOnlyTokens, _ := tok.TokenizeChatRequest(textOnlyReq)

	if len(tokens) <= len(textOnlyTokens) {
		t.Errorf("Expected more tokens with image, got %d vs %d", len(tokens), len(textOnlyTokens))
	}
}

func TestTokenizeChatRequestInvalidRole(t *testing.T) {
	tok := NewTokenizer(10000)

	req := &ChatRequest{
		Messages: []ChatMessage{
			{Role: "invalid_role", Content: "Test"},
		},
	}

	_, err := tok.TokenizeChatRequest(req)
	if err == nil {
		t.Error("Expected error for invalid role")
	}
}

func TestEncodeRole(t *testing.T) {
	tok := NewTokenizer(10000)

	tests := []struct {
		role         string
		expectError  bool
		expectTokens int
	}{
		{"system", false, 2},
		{"user", false, 2},
		{"assistant", false, 2},
		{"invalid", true, 0},
	}

	for _, tt := range tests {
		t.Run(tt.role, func(t *testing.T) {
			tokens, err := tok.encodeRole(tt.role)

			if tt.expectError {
				if err == nil {
					t.Error("Expected error for invalid role")
				}
				return
			}

			if err != nil {
				t.Fatalf("Expected no error, got %v", err)
			}

			if len(tokens) != tt.expectTokens {
				t.Errorf("Expected %d tokens, got %d", tt.expectTokens, len(tokens))
			}
		})
	}
}

func TestLoadConfig(t *testing.T) {
	configJSON := `{
		"vocab_size": 32000,
		"model_path": "/models/tokenizer.json",
		"type": "bpe",
		"special_tokens": {
			"pad": 0,
			"eos": 2,
			"bos": 1
		}
	}`

	r := strings.NewReader(configJSON)
	config, err := LoadConfig(r)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if config.VocabSize != 32000 {
		t.Errorf("Expected vocab size 32000, got %d", config.VocabSize)
	}

	if config.Type != "bpe" {
		t.Errorf("Expected type 'bpe', got '%s'", config.Type)
	}

	if config.ModelPath != "/models/tokenizer.json" {
		t.Errorf("Expected model path '/models/tokenizer.json', got '%s'", config.ModelPath)
	}

	if len(config.SpecialTokens) != 3 {
		t.Errorf("Expected 3 special tokens, got %d", len(config.SpecialTokens))
	}
}

func TestNewTokenizerFromConfig(t *testing.T) {
	config := &TokenizerConfig{
		VocabSize: 32000,
		ModelPath: "/models/tokenizer.json",
		Type:      "bpe",
	}

	tok, err := NewTokenizerFromConfig(config)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}

	if tok.VocabSize() != 32000 {
		t.Errorf("Expected vocab size 32000, got %d", tok.VocabSize())
	}
}

func TestNewTokenizerFromConfigInvalid(t *testing.T) {
	config := &TokenizerConfig{
		VocabSize: 0, // Invalid
	}

	_, err := NewTokenizerFromConfig(config)
	if err == nil {
		t.Error("Expected error for invalid vocab size")
	}
}

func TestVocabSize(t *testing.T) {
	tok := NewTokenizer(50000)
	if tok.VocabSize() != 50000 {
		t.Errorf("Expected vocab size 50000, got %d", tok.VocabSize())
	}
}

func TestChatMessageJSON(t *testing.T) {
	msg := ChatMessage{
		Role:    "user",
		Content: "Hello",
		Image:   "base64data",
	}

	// Verify it can be marshaled to JSON
	_, err := json.Marshal(msg)
	if err != nil {
		t.Errorf("Failed to marshal ChatMessage: %v", err)
	}
}

func TestChatRequestJSON(t *testing.T) {
	req := ChatRequest{
		Messages: []ChatMessage{
			{Role: "user", Content: "Hello"},
		},
		MaxTokens:   100,
		Temperature: 0.7,
	}

	// Verify it can be marshaled to JSON
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Failed to marshal ChatRequest: %v", err)
	}

	// Verify it can be unmarshaled back
	var req2 ChatRequest
	if err := json.Unmarshal(data, &req2); err != nil {
		t.Fatalf("Failed to unmarshal ChatRequest: %v", err)
	}

	if req2.MaxTokens != req.MaxTokens {
		t.Errorf("Expected MaxTokens %d, got %d", req.MaxTokens, req2.MaxTokens)
	}
}
