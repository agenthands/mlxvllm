package radix

import (
	"testing"
)

func TestNewNode(t *testing.T) {
	node := NewNode([]uint32{1, 2, 3}, nil)

	if len(node.Tokens) != 3 {
		t.Errorf("Expected 3 tokens, got %d", len(node.Tokens))
	}

	if node.Tokens[0] != 1 || node.Tokens[1] != 2 || node.Tokens[2] != 3 {
		t.Errorf("Tokens mismatch: got %v", node.Tokens)
	}

	if node.refCount.Load() != 0 {
		t.Errorf("Expected refCount 0, got %d", node.refCount.Load())
	}

	if node.ready == nil {
		t.Error("Expected ready channel to be initialized")
	}

	if node.Children == nil {
		t.Error("Expected Children map to be initialized")
	}

	if node.Parent != nil {
		t.Error("Expected Parent to be nil")
	}
}

func TestNodeWithParent(t *testing.T) {
	parent := NewNode([]uint32{99}, nil)
	child := NewNode([]uint32{1, 2}, parent)

	if child.Parent != parent {
		t.Error("Expected child Parent to point to parent node")
	}
}

func TestNodeWait(t *testing.T) {
	node := NewNode([]uint32{1}, nil)

	// Not finalized yet - Wait() should block
	done := make(chan struct{})
	go func() {
		node.Wait()
		close(done)
	}()

	select {
	case <-done:
		t.Error("Wait should block when node not ready")
	default:
		// Expected - Wait is blocking
	}

	// Finalize the node
	FinalizeNode(node, 123)

	// Now wait should unblock
	err := node.Wait()
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	if node.CacheHandle != 123 {
		t.Errorf("Expected CacheHandle 123, got %d", node.CacheHandle)
	}

	// Wait should be idempotent after finalization
	node.Wait() // Should not block
}

func TestNodeWaitAfterPoison(t *testing.T) {
	node := NewNode([]uint32{1}, nil)

	poisonErr := TestError("computation failed")
	PoisonNode(node, poisonErr)

	err := node.Wait()
	if err == nil {
		t.Error("Expected error from poisoned node")
	}
	if err != poisonErr {
		t.Errorf("Expected poison error, got %v", err)
	}
}

func TestFinalizeNode(t *testing.T) {
	node := NewNode([]uint32{42}, nil)

	FinalizeNode(node, 999)

	if node.CacheHandle != 999 {
		t.Errorf("Expected CacheHandle 999, got %d", node.CacheHandle)
	}

	if node.err != nil {
		t.Errorf("Expected no error, got %v", node.err)
	}
}

func TestPoisonNode(t *testing.T) {
	node := NewNode([]uint32{42}, nil)

	expectedErr := TestError("test error")
	PoisonNode(node, expectedErr)

	if node.err != expectedErr {
		t.Errorf("Expected error %v, got %v", expectedErr, node.err)
	}
}

func TestIsReady(t *testing.T) {
	node := NewNode([]uint32{1}, nil)

	if node.IsReady() {
		t.Error("Expected IsReady to return false for new node")
	}

	FinalizeNode(node, 123)

	if !node.IsReady() {
		t.Error("Expected IsReady to return true after finalization")
	}
}

func TestIsReadyAfterPoison(t *testing.T) {
	node := NewNode([]uint32{1}, nil)

	if node.IsReady() {
		t.Error("Expected IsReady to return false for new node")
	}

	PoisonNode(node, TestError("error"))

	if !node.IsReady() {
		t.Error("Expected IsReady to return true after poisoning")
	}
}

func TestRefCount(t *testing.T) {
	node := NewNode([]uint32{1}, nil)

	if node.refCount.Load() != 0 {
		t.Errorf("Expected initial refCount 0, got %d", node.refCount.Load())
	}

	node.refCount.Add(1)
	if node.refCount.Load() != 1 {
		t.Errorf("Expected refCount 1, got %d", node.refCount.Load())
	}

	node.refCount.Add(-1)
	if node.refCount.Load() != 0 {
		t.Errorf("Expected refCount 0 after decrement, got %d", node.refCount.Load())
	}
}

// TestError is a simple error type for testing
type TestError string

func (e TestError) Error() string {
	return string(e)
}
